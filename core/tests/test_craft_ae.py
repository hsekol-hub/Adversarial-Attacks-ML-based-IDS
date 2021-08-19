import os
import json
import joblib
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients


# ----------------------------------------Prepare result frames---------------------------------------------------------
def rev_dict(d_):  # reverse a dictionary (key, value) pair
    return dict(zip(d_.values(), d_.keys()))


def tranform_columns(frame, encoder):
    label_encoder = {v: k for k, v in encoder['Label'].items()}
    frame.protocol_type.replace(rev_dict(encoder['protocol_type']), inplace=True)
    frame.service.replace(rev_dict(encoder['service']), inplace=True)
    frame.flag.replace(rev_dict(encoder['flag']), inplace=True)
    frame.true_label.replace(label_encoder, inplace=True)
    frame.orig_pred.replace(label_encoder, inplace=True)
    frame.adv_pred.replace(label_encoder, inplace=True)
    return frame


# ------------------------------------------Utility Functions-----------------------------------------------------------

def get_weights(weights):  # weights: from the utility files (generated from random forest classifier)
    t1 = pd.read_csv(os.path.join(cwd, 'processed_data', 'train.csv'))
    t2 = pd.read_csv(os.path.join(cwd, 'processed_data', 'test.csv'))
    df = pd.concat([t1, t2])

    cat_columns = [col for col in df.columns if df[col].dtype == 'object']
    encoder_keys = list(encoder.keys())

    for col1, col2 in zip(cat_columns, encoder_keys):  # encode only categorical features
        assert col1 == col2
        df[col1].replace(encoder[col2], inplace=True)

    cor = df.corr()
    cor = abs(cor['Label'])
    cor_weights = cor[:-1]
    # One of these have to be used and should be selected based on test results
    cor_weights = cor_weights / np.linalg.norm(cor_weights, ord=2)  # TODO
    weights = weights / np.linalg.norm(weights, ord=2)  # TODO

    print('Weights: {}\t|\tCor Weights: {}'.format(sum(weights), sum(cor_weights)))
    print('=' * 100)
    return cor_weights, weights


def sec2pri_constraint(xprime, mask, lb, ub, true_label, encoder, test_set):
    '''
    Based on current values of xprime, there will be cases when we have two or more features correlated.
    Their updated values should obey the functional relationships and not only depending on gradient updates to reduce
    the final loss value. This is done with holding a relation between the secondary and primary features.
    Core Idea:
        1. For a given class label
        2. Fetch the primary features and subset the whole test set based on these values. (stored in utils)
        3. The obtained subset will have all plausible values present for that specific class and above values in primary features.
        4. Fetch the [lower, upper] bounds of all secondary features.
        5. return the updated [lb, ub] as they need to be clipped again to make the attack look more realistic and guarantee functional coherence.
    Note: We will not update values in [lb, ub] where we do not want any changes to occur (based on mask vector).
    Parameters
    ----------
    xprime: x_tensor + (mask * r); where r is the perturbation obtained based on gradient values
    mask: Mask vector which has a equal length of featureNames
    Returns
    -------
    Updated lower and upper bound (now based on the values present in the primary features).
    '''

    label_encoder = {v: k for k, v in encoder['Label'].items()}
    _ = xprime.cpu().detach().numpy()  # ndarray(1,40) | scaled
    xt_it = scaler.inverse_transform(_)[0]
    label_subset = test_set[
        test_set.Label == label_encoder[true_label]]  # subset records from test_set for the current class label
    label_constraint = corr_constraint[label_encoder[true_label]]
    cols_idx_map = dict(zip(idx_cols_map.values(), idx_cols_map.keys()))
    lb, ub, mask = lb.reshape(-1), ub.reshape(-1), mask.reshape(-1)
    for sec_col, pri_cols in label_constraint.items():  # for each secondary column fetch all its primary dependencies
        sec_col_idx, pri_cols_idx = int(cols_idx_map[sec_col]), [int(cols_idx_map[p]) for p in pri_cols]
        # print('Secondary: ',sec_col)
        # print(sec_col_idx, pri_cols_idx)
        if mask[sec_col_idx] != 0:  # do not updated those features bounds where mask vector has value == 0
            pri_values = xt_it[pri_cols_idx]  # IMPORTANT; fetch values from primary features  of current sample
            subset = label_subset  # RESET subset to above label_subset on each iter for secondary column
            assert len(subset) > 0
            for pri_col, val in zip(pri_cols, pri_values):
                # print('Primary Column:', pri_col, val)
                temp = subset[subset[pri_col] <= val]  # We further subset one at a time for each primary value.
                subset = temp if len(temp) > 0 else subset

            lb[sec_col_idx], ub[sec_col_idx] = min(subset[sec_col]), max(
                subset[sec_col])  # Update the specific secondary feature with min and max
            # print('_'*100)
    lb, ub, mask = lb.reshape(-1, num_features), ub.reshape(-1, num_features), mask.reshape(-1, num_features)
    return lb, ub


def check_multiplicity(model, x_prime, multiplicity):
    '''
    Ensures discrete features are coherent and do not have decimal values.
    We also check if the rounded x_prime is adversarial in nature. If NOT we not round off the values and let the
    algorithm work on original x_prime.
    Note: It is only in the end that we update the rounded xprime when it was able to fool the model.

    Returns
    -------
    Rectified xprime: incase the attack was possible; else original xprime (not a valid sample)
    '''
    _ = x_prime.cpu().detach().numpy()
    xt_it = scaler.inverse_transform(_)[0]
    multiplicity = multiplicity.reshape(-1)
    #  round the values if they have to be discrete.
    _ = np.array([round(elem) if m == 'discrete' else elem for m, elem in zip(multiplicity, xt_it)]).reshape(-1,
                                                                                                             num_features)
    x = scaler.transform(_)
    x = torch.tensor(x, device=device)
    op = model(x)
    _, adv_pred = torch.max(torch.log_softmax(op, dim=1), dim=1)

    if adv_pred.item() != 0:
        return x_prime, False
    else:
        return x, True  # attack was a success


def sample_test_accuracy(ts_x, ts_y, num_class):
    '''
    Sample only malicious traffic examples from our overall test dataset;
    so recompute the base test accuracy.
    '''

    def multi_acc(y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc, y_pred_tags

    print('Evaluating sample test dataset...')
    test_dataset = ClassifierDataset(torch.from_numpy(ts_x).float(),
                                     torch.from_numpy(ts_y).long())
    test_loader = DataLoader(dataset=test_dataset, batch_size=128)

    test_acc = 0
    test_preds = []
    confusion_matrix = np.zeros((num_class, num_class))
    with torch.no_grad():
        model.eval()

        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            acc, y_pred_tags = multi_acc(y_pred, y_batch)
            test_preds.append(y_pred_tags)
            test_acc += acc
            for t, p in zip(y_batch.view(-1), y_pred_tags.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    test_acc = test_acc.item() / len(test_loader)
    print('Test Accuracy: ', test_acc)
    print('=' * 100)
    return test_acc, test_preds[0].numpy(), confusion_matrix


def load_utility_files(num_features, num_class):
    '''
    Loads all the utility objects created so far
    '''

    model_state_path = os.path.join(cwd, 'model.pth')
    model = MultiLabelClassifier(num_features, num_class, num_features * 2)
    model.load_state_dict(torch.load(model_state_path))
    model = model.to(device)

    with open(os.path.join(cwd, 'utility_files', 'categorical_encoder.json'), 'r') as file:
        encoder = json.load(file)

    with open(os.path.join(cwd, 'utility_files', 'feature_importance.json'), 'r') as file:
        weights = json.load(file)

    with open(os.path.join(cwd, 'utility_files', 'perturbation_space.json'), 'r') as file:
        p_space = json.load(file)

    with open(os.path.join(cwd, 'utility_files', 'corr_constraints.json'), 'r') as file:
        corr_constraint = json.load(file)

    with open(os.path.join(cwd, 'utility_files', 'idx_cols_map.json'), 'r') as file:
        idx_cols_map = json.load(file)

    file_path = os.path.join(cwd, 'utility_files', 'scaler.pkl')
    scaler = joblib.load(file_path)

    weights = np.array(list(weights.values()))

    return model, encoder, weights, p_space, scaler, corr_constraint, idx_cols_map


def encode_scale_ts(ts):
    cat_columns = [col for col in ts.columns if ts[col].dtype == 'object']
    encoder_keys = list(encoder.keys())

    for col1, col2 in zip(cat_columns, encoder_keys):  # encode only categorical features
        assert col1 == col2
        ts[col1].replace(encoder[col2], inplace=True)
    ts_x, ts_y = ts.iloc[:, :-1], ts['Label']
    ts_x = scaler.transform(ts_x)
    ts_x, ts_y = np.array(ts_x), np.array(ts_y)
    return ts_x, ts_y


if __name__ == '__main__':

    n_samples = 25
    print('Working on first {} test samples...'.format(n_samples))
    time.sleep(5)
    ts = ts[ts['Label'] != 'normal']  # First attempt is to make malicious --> normal (traffic)
    ts = ts.iloc[:n_samples]
    ts = ts.reset_index(drop=True)
    ts.to_csv(os.path.join(cwd, 'adv', 'test_sample.csv'), index=False)

    # Load utility objects
    model, encoder, weights, p_space, scaler, corr_constraint, idx_cols_map = load_utility_files(num_features,
                                                                                                 num_class)
    ts_x, ts_y = encode_scale_ts(ts)  # Encode categorical features and scale value as model is trained on scaled set.
    orig_test_acc, test_preds, cm = sample_test_accuracy(ts_x, ts_y, num_class)
    ts_x = pd.DataFrame(ts_x).reset_index(drop=True)
    ts_y = pd.DataFrame(ts_y).reset_index(drop=True)
    ts = pd.concat([ts_x, ts_y], axis=1)
    cols = feature_names.copy()
    cols.append('Label')
    ts.columns = cols  # scaled encoded test set.
    print('Confusion Matrix for sample test dataset\n', cm)
    print('=' * 100)
    del ts_x, ts_y, cm, cols

    # ------------------------------------------------------------------------------------------------------------------
    cor_weights, weights = get_weights(weights)  # two weight vectors [based on correlation, and random forest]; need to
    # check which provides better results.
    mask = get_masks(feature_names)  # We do not want to change all the features

    # ------------------------------------------Adv Block---------------------------------------------------------------

    model = model.double()
    extra_cols = ['true_label', 'orig_pred', 'adv_pred', 'imperceptibility', 'perturbed_norm', 'n_iters']
    # Imperceptibility: weighted r norm (as mentioned in the paper)
    # perturbed_norm: l2 norm (x_prime - n_tensor); not sure if this is a good metric | need discussion.
    # n_iters: Best iteration at which least perceptible x_prime was/wasn't found.
    # orig_pred: what model predicts initially to the true x_tensor.

    results = np.zeros((len(ts), num_features + len(extra_cols)))  # final frame to hold the x_prime
    print('Calculating adversarial counterparts ...')
    for ind, row in ts.iterrows():  # find adversarial counterpart for each row

        alpha = 0.01  # small step size for updating r
        best_iter = -1  # iteration at which least perceptible x_prime was observed
        best_norm_weighted = torch.tensor(np.inf)  # hold the lowest l2 norm value for x_prime; initially infinite
        best_adv_flag = False  # sets True if adversarial example was found
        loop_i, max_iters = 0, 1000
        lambda_ = torch.tensor([5])  # hyper parameter to control the perceptibility loss
        iter_frame = pd.DataFrame()  # to check how the x_tensor is changing over each iteration

        x_tensor = torch.tensor(row[feature_names], device=device)
        x_tensor.unsqueeze_(0)
        true_label = int(torch.tensor(row['Label'], device=device).item())
        target_pred = Variable(torch.tensor([0], dtype=torch.long, requires_grad=False, device=device))
        r = Variable(torch.tensor(1e-4 * np.ones(x_tensor.shape), device=device), requires_grad=True)
        v = torch.tensor(np.array(weights), device=device)  # weight vector

        # ------------------------------------------Loss Functions------------------------------------------------------

        cse = nn.CrossEntropyLoss()
        perceptibility = lambda v, r, m: torch.sqrt(torch.sum((v * r * m) ** 2))  # Weighted perturbation L2 norm
        l2_norm = lambda x, y: torch.sqrt(torch.sum(torch.abs(x - y) ** 2))  # to find l2 norm of (x_prime - x_tensor)

        x_prime = x_tensor + (r * mask)
        best_pert_x = x_prime
        op = model(x_prime)
        _, orig_pred = torch.max(torch.log_softmax(op, dim=1), dim=1)

        while loop_i <= max_iters:
            loop_i += 1
            zero_gradients(r)
            cse_loss = cse(op, target_pred)  # Cross Entropy loss between prediction and target label
            per_loss = perceptibility(v, r, mask)  # Weighted perturbation l2-norm; measures the perceptibility of
            # perturbation weighted by feature importance vector
            loss = cse_loss + lambda_ * per_loss
            # print(cse_loss.item(), per_loss)  # should decrease
            loss.backward(retain_graph=True)
            grad = r.grad.data.clone()
            grad *= alpha  # Minimize the loss with small steps.
            r = r.detach().clone() - grad  # we go in negative direction
            r.requires_grad = True
            x_prime = x_tensor + (r * mask)  # we want to mask vectors we do not wish to change

            # ----------------------------------------------------------------------------------------------------------
            #  Validate x_prime against constraint space and functionality checks
            multiplicity, lb, ub = fetch_bounds(true_label, feature_names, num_features, encoder)
            x_prime = clip(x_prime, lb, ub)  # Tensor(1,40)
            # lb, ub = sec2pri_constraint(x_prime, mask, lb, ub, true_label, encoder, ts_copy)  # TODO further work needs to be done here
            # xprime = clip(x_prime, lb, ub)  # Tensor(1,40)
            x_prime, adv_flag = check_multiplicity(model, x_prime, multiplicity)

            # ----------------------------------------------------------------------------------------------------------
            op = model(x_prime)
            _, adv_pred = torch.max(torch.log_softmax(op, dim=1), dim=1)

            # Question: Which among the below two metrics make more sense to our case. Their values happen to be
            # varying by large numbers.
            current_norm_weighted = per_loss
            perturbed_norm = l2_norm(x_prime, x_tensor)  # Calculation value slight different from torch.linalg.norm(r)
            # due to the multiplicity factor.

            # Consider the x_prime with least perceptibility score
            if adv_flag and (current_norm_weighted < best_norm_weighted):
                best_adv_flag = True  # We atleast have an adv example in max_iters
                best_iter = loop_i
                best_norm_weighted = current_norm_weighted
                best_pert_x = x_prime.squeeze(0).detach().cpu()

            # ----------------------------------------------------------------------------------------------------------
            # Only used to manually check the updates on x_prime for each new test sample.
            name = 'Iter_' + str(loop_i)
            iter_frame = track(iter_frame, x_prime, scaler, name,
                               [true_label, orig_pred.item(), adv_pred.item(), current_norm_weighted.item(),
                                perturbed_norm.item(), loop_i])

        # ------------------------------------------Outside While loop -------------------------------------------------
        # After max_iters add the perturbed x_prime to our final results
        if best_adv_flag:
            # Note: If best adversarial example is found in iter: 1; it is likely that model classified it wrongly
            # as normal traffic without any perturbation. We should avoid such cases from our evaluation.
            print('Best Adversarial example found in iter: {}'.format(best_iter))
            results[ind] = np.concatenate((best_pert_x,
                                           [true_label, orig_pred.item(), adv_pred.item(), best_norm_weighted.item(),
                                            perturbed_norm.item(), best_iter]), axis=0)
        else:
            print('Adversarial example not found in {} attempts'.format(max_iters))
            results[ind] = np.concatenate((x_prime.squeeze(0).detach().cpu(),
                                           [true_label, orig_pred.item(), adv_pred.item(), current_norm_weighted.item(),
                                            perturbed_norm.item(), loop_i]), axis=0)

    # ------------------------------------------Results-----------------------------------------------------------------
    # Used only if we want to track the changes in the last sample from above for loop

    name = 'Original tensor'
    iter_frame = track(iter_frame, x_tensor, scaler, name,
                       [true_label, orig_pred.item(), adv_pred.item(), -1, -1, -1])  # only x_tensor value matters
    name = 'Mask'
    iter_frame = track(iter_frame, x_tensor, scaler, name,
                       [true_label, orig_pred.item(), adv_pred.item(), -1, -1, loop_i])  # only x_tensor value matters

    columns = list(feature_names) + extra_cols
    iter_frame.columns = columns
    iter_frame = tranform_columns(iter_frame, encoder)  # utility function to decode values

    # Convert the results into a final DataFrame object
    adv_set = scaler.inverse_transform(results[:, :-len(extra_cols)])
    adv_set = np.concatenate([adv_set, results[:, -len(extra_cols):]], axis=1)
    adv_set = pd.DataFrame(data=adv_set, columns=columns)
    adv_set = tranform_columns(adv_set, encoder)
    print('*' * 100)
    print('Adversarial Prediction Labels:\n', adv_set.adv_pred.value_counts())
    print('_' * 100)
    print('Original Prediction Labels:\n', adv_set.orig_pred.value_counts())

    adv_test_accuracy = (adv_set.true_label == adv_set.adv_pred).sum() / len(adv_set)
    orig_test_acc = (adv_set.true_label == adv_set.orig_pred).sum() / len(adv_set)

    print('*' * 100)
    print('Original test accuracy: {}\t| Adversarial test accuracy: {}'.format(orig_test_acc, adv_test_accuracy))
    print('*' * 100)

    adv_set.to_csv(os.path.join(cwd, 'adv', 'adv_samples.csv'), index=False)

    # ------------------------------Plot Imperceptibility & perturbed norm ---------------------------------------------
    plot_frame = iter_frame.iloc[:-2]
    x_range = list(range(0, max_iters + 1))
    x_tick_labels = list(range(1, max_iters + 1, int(max_iters / 5)))
    plt.plot(plot_frame['imperceptibility'], label='Imperceptibility')
    plt.plot(plot_frame['perturbed_norm'], label='Perturbation L2 Norm')
    plt.xlabel('No. of Iterations')
    plt.xticks(x_tick_labels)
    plt.ylabel('L2 Norm Value')
    plt.title('Best Iter: {} | Imperceptibility: {} | Pertubed Norm: {}'.format(best_iter,
                                                                                round(best_norm_weighted.item(), 4),
                                                                                round(perturbed_norm.item(), 4)))
    plt.legend()
    plt.savefig(os.path.join(cwd, 'plots', 'imperceptibility_plot.png'))
    plt.show()
    plt.close()