import pandas as pd
print('Modifying {} features'.format(mask.sum()))
result_cols = list(test.columns) + ['orig_pred', 'adv_pred', 'perceptibility', 'pertubation_norm', 'best_iter']
results = np.zeros((len(test), len(result_cols)))


for ind, (x_tensors, orig_labels) in enumerate(data_loader):
    print('Batch no: ',ind)
    # -----------------------------------------Function Declarations------------------------------------
    empty_tensors = lambda x: torch.tensor([x]).new_full((x_tensors.shape[0],), x)
    perturb_l2_norm = lambda x, y: torch.sqrt(torch.sum(torch.abs(x - y) ** 2, dim=1))  # l2 norm of (x_prime - x_tensor)
    cse = torch.nn.CrossEntropyLoss()
    perciptibility = lambda r, v, m: torch.sqrt(torch.sum(r * v * m, dim=1) ** 2)  # Weighted perturbation L2 norm

    # -----------------------------------------Initializations------------------------------------

    alpha = 0.001
    lambda_ = 4.5
    max_iters = 5000
    loop_i = 0
    best_iter, best_adv_pred = empty_tensors(np.inf), empty_tensors(-1)
    best_perceptibility, best_perturbed_norm = empty_tensors(np.inf), empty_tensors(np.inf)

    latent_output = model(x_tensors)
    original_preds, target_preds = custom_softmax(latent_output)  # accuracy verified and it is v. good ~78%
    target_labels = torch.tensor([0 if l1 != 0 else l2 for l1, l2 in zip(orig_labels, target_preds)], dtype=torch.long)
    # target_labels = torch.tensor([l2 if l1 == 1 else 1 for l1, l2 in zip(orig_labels, target_preds)], dtype=torch.long)

    pertub_noise = torch.rand_like(x_tensors) * 1e-2
    pertub_noise.requires_grad_(True)
    x_prime = torch.add(x_tensors, torch.mul(pertub_noise, mask))
    best_pert_tensor = x_prime.squeeze(0).detach().clone()
    existing_indices = []

    while loop_i <= max_iters:

        zero_gradients(pertub_noise)
        latent_output = model(x_prime)
        cse_loss = cse(latent_output, target_labels)
        perceptibility_loss = perciptibility(rf_weights, pertub_noise, mask)
        total_loss = cse_loss + lambda_ * torch.sum(perceptibility_loss)


        total_loss.backward(retain_graph=True)
        grad = pertub_noise.grad.data.detach().clone()
        grad *= alpha
        pertub_noise = pertub_noise.detach().clone() - grad
        # pertub_noise /= np.linalg.norm(pertub_noise)
        # pertub_noise /= ((1-rf_weights) + 1e-6)
        # pertub_noise *= (1-rf_weights)

        pertub_noise.requires_grad_(True)
        x_prime = torch.add(x_tensors, torch.mul(pertub_noise, mask))
        current_perturbed_norm = perturb_l2_norm(best_pert_tensor, x_tensors)

        if loop_i % 100 == 0:
            # print(target_labels)
            print('Total loss: {} |\t CSE Loss: {}\t Percep.Loss: {}\t L2 norm: {}'.\
                  format(round(total_loss.item(), 4),
                         round(cse_loss.item(), 4),
                         round(torch.sum(perceptibility_loss).item(), 4),
                         round(torch.sum(current_perturbed_norm).item(), 4)))

        x_prime = clip_tensor(x_prime, [lower_bounds, upper_bounds], scaler, orig_labels)  # Validated and Working
        x_prime = check_multiplicity(x_prime, multiplicity, mask, model, scaler, target_labels)
        x_prime = clip_tensor(x_prime, [lower_bounds, upper_bounds], scaler, orig_labels)  # Validated and Working

        latent_output = model(x_prime)
        adv_pred, _ = custom_softmax(latent_output)
        # print('Adv preds:', adv_pred, torch.sum(x_prime))
        update_indices = list(np.where(adv_pred == target_labels)[0])

        if len(update_indices) > 0:
            # row_indices = np.where(current_perturbed_norm < best_perturbed_norm)[0]
            existing_indices += update_indices
            existing_indices = list(set(existing_indices))
            # row_indices = np.where((perceptibility_loss < best_perceptibility))[0]
            row_indices = np.where((current_perturbed_norm < best_perturbed_norm))[0]
            row_indices = [ind for ind in row_indices if ind in existing_indices]

            if len(row_indices) > 0:
                print(row_indices)
                # print('{}   {}'.format('_' * 70, loop_i + 1))
                # print('orig', orig_labels)
                # print('adv ', adv_pred)
                # print('targ', target_labels)
                # print('Updating index:', update_indices)
                best_iter[row_indices] = loop_i
                best_adv_pred[row_indices] = adv_pred[row_indices]
                best_pert_tensor[row_indices] = x_prime[row_indices]
                best_perceptibility[row_indices] = perceptibility_loss[row_indices]
                best_perturbed_norm[row_indices] = current_perturbed_norm[row_indices]

                # print('Updating Indices: {}\t total success: {}'.format(row_indices, len(existing_indices)))
        loop_i += 1

    print('orig', orig_labels.numpy())
    print('adv ', best_adv_pred.numpy())
    print('targ', target_labels.numpy())
    x_prime = np.c_[best_pert_tensor.detach().clone(),
                    orig_labels.detach().clone(),
                    original_preds.detach().clone(),
                    best_adv_pred.detach().clone(),
                    best_perceptibility.detach().clone(),
                    best_perturbed_norm.detach().clone(),
                    best_iter.detach().clone()]
    results[ind * batch_size: (ind + 1) * batch_size] = x_prime

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
    frame = np.round(frame, 3)  # cautionary measure
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

test_sample.to_csv(os.path.join(results_dir, 'adv', 'lowprofool_test_samples.csv'), index=False)
adv_set.to_csv(os.path.join(results_dir, 'adv', 'lowprofool_adv_samples.csv'), index=False)