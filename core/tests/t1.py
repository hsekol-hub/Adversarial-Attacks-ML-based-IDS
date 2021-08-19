print('Modifying {} features'.format(mask.sum()))
result_cols = list(test.columns) + ['orig_pred', 'adv_pred', 'n_iters']
results = np.zeros((len(test), len(result_cols)))

for ind, (x_tensors, orig_labels) in enumerate(data_loader):

    print(ind)
    # -----------------------------------------Initializations------------------------------------
    epsilon = 1e-5
    alpha = 0.001
    overshoot = 0.02
    loop_i, max_iters = 0, 5000
    total_perturb = torch.zeros(x_tensors.shape)
    # target_labels = torch.tensor([0 if label != 0 else torch.randint(1, torch.max(orig_labels), (1,))\
    #                               for label in orig_labels], dtype=torch.long)


    # -----------------------------------------------------------------------------
    x_tensors.requires_grad_(True)
    latent_output = model(x_tensors)
    original_preds, target_preds = custom_softmax(latent_output)  # accuracy verified and it is v. good ~78%
    target_labels = torch.tensor([0 if l1 != 0 else l2 for l1, l2 in zip(orig_labels, target_preds)], dtype=torch.long)

    empty_tensors = lambda x: torch.tensor([x]).new_full((x_tensors.shape[0],), x)
    external_grad = torch.ones_like(latent_output[:, 0])  # column value doesn't matter

    best_pert_tensor = x_tensors.squeeze(0).detach().clone()
    best_iter, best_adv_pred = empty_tensors(np.inf), empty_tensors(-1)
    existing_indices = []
    # -----------------------------------------------------------------------------
    # print('Original', orig_labels)

    while loop_i < max_iters:

        # Source Class: considered here is the max logit value for each class label being predicted on each iteration
        max_logit_indices = torch.argmax(latent_output, dim=1)
        latent_output[torch.arange(len(latent_output)), max_logit_indices].backward(gradient=external_grad, retain_graph=True)
        # print(latent_output)
        # print('Max logit indices', max_logit_indices)
        grad_orig = x_tensors.grad.data.detach().clone() + epsilon # in-order to avoid nan when src and target latent output values are almost same
        zero_gradients(x_tensors)
        # Target Class
        latent_output[torch.arange(len(latent_output)), target_labels].backward(gradient=external_grad, retain_graph=True)
        grad_target = x_tensors.grad.data.detach().clone()

        net_grad = torch.mul((grad_target - grad_orig), mask)  # verified: element-wise multiplication or broadcasting
        net_latent_output = abs(latent_output[:, 0] - latent_output[:, 1])
        # -----------------------------------------------------------------------------
        perturb = net_latent_output / np.linalg.norm(net_grad.flatten())  # verified: scalar value per sample
        perturb = perturb.detach().clone()
        perturb = (net_grad.T * (perturb + 1e-6)).T
        # perturb = ((perturb + 1e-4) * net_grad) / np.linalg.norm(net_grad)
        perturb = alpha * perturb / np.linalg.norm(perturb)
        perturb /= (rf_weights + 1e-6)
        total_perturb = total_perturb + perturb
        x_tensors = x_tensors.detach().clone() + (((1 + overshoot) * total_perturb) * mask)

        # ---------------------------------------------Functionality Constraints--------------------------------

        x_tensors = clip_tensor(x_tensors, [lower_bounds, upper_bounds], scaler, orig_labels)  # Validated and Working
        x_tensors = check_multiplicity(x_tensors, multiplicity, mask, model, scaler, target_labels)
        x_tensors = clip_tensor(x_tensors, [lower_bounds, upper_bounds], scaler, orig_labels)  # Clip again in the end no matter what
        x_tensors.requires_grad_(True)
        latent_output = model(x_tensors)
        adv_pred, _ = custom_softmax(latent_output)

        # -----------------------------------------------------------------------------
        update_indices = list(np.where(adv_pred == target_labels)[0])
        update_indices = list(set(update_indices) - set(existing_indices))
        if len(update_indices) > 0:
            print('{}   {}'.format('_' * 70, loop_i + 1))
            print('orig', orig_labels)
            print('adv ', adv_pred)
            print('targ', target_labels)
            best_pert_tensor[update_indices] = x_tensors[update_indices]
            best_iter[update_indices] = loop_i
            best_adv_pred[update_indices] = adv_pred[update_indices]
            existing_indices += update_indices
            existing_indices = list(set(existing_indices))
            print('Adversarial found for: {} total: {}'.format(update_indices, len(existing_indices)))
        loop_i += 1

    print('orig', orig_labels.numpy())
    print('adv ', best_adv_pred.numpy())
    print('targ', target_labels.numpy())

    x_prime = np.c_[best_pert_tensor.detach().clone(),
                    orig_labels, original_preds, best_adv_pred, best_iter]
    results[ind * batch_size: (ind+1) * batch_size] = x_prime

    success_rate = (best_adv_pred == target_labels).sum() / len(x_prime) * 100
    print('Success rate', success_rate)



test_sample = np.empty_like(test)
for ind, (x_batch, y_batch) in enumerate(data_loader):
    batch = np.c_[x_batch.numpy(), y_batch.numpy()]
    test_sample[ind * batch_size:(ind + 1) * batch_size] = batch

def decode(frame, encoder, flag):

    def rev_dict(dictionary):
        return dict(zip(dictionary.values(), dictionary.keys()))

    for key in encoder.keys():
        if key == 'Label':
            label_encoder = {v: k for k, v in encoder[key].items()}
            frame.Label.replace(label_encoder, inplace=True)
            if flag:
                frame.orig_pred.replace(label_encoder, inplace=True)
                frame.adv_pred.replace(label_encoder, inplace=True)
        else:
            frame[key].replace(rev_dict(encoder[key]), inplace=True)
    return frame

def prepare_results(np_array, cols, flag=True):

    frame = scaler.inverse_transform(np_array[:, :len(test.columns) - 1])
    frame = np.round(frame, 4)
    frame = np.concatenate([frame, np_array[:, len(test.columns) - 1:]], axis=1)
    frame = pd.DataFrame(data=frame, columns=cols)
    frame = decode(frame, encoder, flag)
    return frame


adv_set = prepare_results(results, result_cols)
test_sample = prepare_results(test_sample, test.columns, flag=False)

print('*' * 100)
adv_acc = ((adv_set.Label == adv_set.adv_pred).sum() / len(results)) * 100
orig_acc = ((adv_set.Label == adv_set.orig_pred).sum() / len(results)) * 100

print('Accuracy for {} test samples: \n--Adversarial: {}%\n--Original {}%'.format(len(test_sample), adv_acc, orig_acc))
print('*' * 100)
print('Original Prediction Labels:\n', adv_set.Label.value_counts())
print('_' * 100)
print('Adversarial Prediction Labels:\n', adv_set.adv_pred.value_counts())

test_sample.to_csv(os.path.join(results_dir, 'adv', 'deepfool_test_samples.csv'), index=False)
adv_set.to_csv(os.path.join(results_dir, 'adv', 'deepfool_adv_samples.csv'), index=False)