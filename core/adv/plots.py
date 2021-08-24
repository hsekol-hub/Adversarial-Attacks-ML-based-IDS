import json
import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def encode(data, encoder, adv_flag=False):
    cat_cols = list(encoder.keys())
    for col in cat_cols:
        data[col] = data[col].map(encoder[col])

    if adv_flag:
        label_encoder = encoder['Label']
        data.adv_pred.replace(label_encoder, inplace=True)
        data.orig_pred.replace(label_encoder, inplace=True)
        data.target_pred.replace(label_encoder, inplace=True)
    return data


def load_utils(algo, rf_weights, utils_dir):

    dict_ = {1: 'lowprofool',
             2: 'deepfool'}
    algo = dict_[algo]

    scaler = joblib.load(os.path.join(utils_dir, 'scaler.pkl'))
    with open(os.path.join(utils_dir, 'encoder.json'), 'r') as file:
        encoder = json.load(file)

    with open(os.path.join(utils_dir, 'mask.json'), 'r') as file:
        mask = json.load(file)
    mask = list(mask.values())

    if rf_weights:
        with open(os.path.join(utils_dir, 'rf_feature_importance.json'), 'r') as file:
            weights = json.load(file)
        weights = list(weights.values())
        weights = weights / np.linalg.norm(weights)
    else:
        with open(os.path.join(utils_dir, 'cor_weights.json'), 'r') as file:
            weights = json.load(file)
        weights = list(weights.values())

    test = pd.read_csv(os.path.join(results_dir, 'adv', algo + '_test_samples.csv'))
    adv = pd.read_csv(os.path.join(results_dir, 'adv', algo + '_adv_samples.csv'))
    test = encode(test, encoder)
    adv = encode(adv, encoder, True)
    adv.adv_pred = adv.adv_pred.astype('float64').astype('int')

    return algo, scaler, encoder, mask, weights, test, adv


#  ----------------------------------------------------------------- Base loading

dataset, algo = 1, 1
dict_ = {1: 'NSL-KDD',
         2: 'CICDDoS',
         3: 'CICIDS'}
dataset = dict_[dataset]

cwd = os.path.dirname(__file__)
utils_dir = os.path.join(cwd[:-8], 'utils', dataset)
results_dir = os.path.join(cwd[:-8], 'results', dataset)

rf_weights = True
algo, scaler, encoder, mask, weights, test, adv = load_utils(algo, rf_weights, utils_dir)




#  ----------------------------------------------------------------- Confusion Matrices

encoder = encoder['Label']
encoder = {v: k for k, v in encoder.items()}
labels = encoder.values()
y_true, y_pred, adv_pred = np.array(adv.Label), np.array(adv.orig_pred), np.array(adv.adv_pred)

ind = np.where(adv_pred == -1.0)
adv_pred[ind] = y_true[ind]
adv_pred = adv_pred.astype('int')
cm = confusion_matrix(y_true, y_pred).astype('int')
adv_cm = confusion_matrix(y_true, adv_pred).astype('int')


def confusion_matrix_plot():

    grid_kws = {'width_ratios': [.7, .7], 'wspace': .1}
    fig, ax = plt.subplots(1, 2, constrained_layout=True,
                           gridspec_kw=grid_kws)

    ax[0].get_shared_y_axes().join(ax[1])
    g1 = sns.heatmap(data=cm, ax=ax[0], annot=True, fmt='d', cmap='Purples',
                     xticklabels=labels, yticklabels=labels,
                     linewidths=0.6, cbar=False)
    g1.set_ylabel('True Label', fontsize=12)
    g1.set_xlabel('Original Prediction', fontsize=12)

    g2 = sns.heatmap(data=adv_cm, ax=ax[1],
                          annot=True, fmt='d', cmap='Purples',
                          linewidths=0.6,
                          xticklabels=labels, yticklabels=False, cbar=False)
    g2.set_xlabel('Adversarial Prediction', fontsize=12)

    flag = True
    for _ in [ax[0], ax[1]]:
        _.set_xticklabels(_.get_xticklabels(), rotation=90, fontsize=10)
        if flag:
            _.set_yticklabels(_.get_yticklabels(), rotation=0, fontsize=10)
            flag = False
    # plt.suptitle(f'{dataset} - {algo} algorithm', fontsize=14)
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(results_dir, 'plots',  algo+'_' + dataset +'_cm.png'))
    plt.show()
    plt.close()

confusion_matrix_plot()



#  ----------------------------------------------------------------- Perturbation per column VS weights

threshold = 0
feature_cols = test.columns[:-1]
adv = adv.dropna()
s_adv = adv[(adv.target_pred == adv.adv_pred) | ((adv.Label == 0) & (adv.adv_pred != -1))]
s_test = test.loc[s_adv.index]

# m_iters = np.round(np.median(s_adv.best_iter))
# m_perceptibility = np.round(np.median(s_adv.perceptibility))
# m_p_distance = np.round(np.median(s_adv.perturbation_norm))

f_i = s_test[feature_cols].values
f_i_prime = s_adv[feature_cols].values
f_i = scaler.transform(f_i)
f_i_prime = scaler.transform(f_i_prime)

diff_matrix = np.subtract(f_i, f_i_prime)
diff_vector = np.mean(diff_matrix, axis=0)
diff_mask_idx = [1 if abs(value) > threshold else 0 for value in diff_vector]
perturbed_feats = mask and diff_mask_idx
perturbed_feats_idx = np.where(np.array(perturbed_feats) != 0)
total_perturbed = len(perturbed_feats_idx[0])

# Re-calculate only for features which have differences

diff_vector = np.mean(diff_matrix, axis=0)[perturbed_feats_idx]
f_i_vector = np.mean(f_i, axis=0)[perturbed_feats_idx]
f_i_prime_vector = np.mean(f_i_prime, axis=0)[perturbed_feats_idx]
weight_vector = np.array(weights)[perturbed_feats_idx]
feature_cols = feature_cols[perturbed_feats_idx]
total_samples = f_i.shape[0]

def bar_plot():

    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    ax[0].plot(feature_cols, f_i_vector, label='original', color='green')
    ax[0].plot(feature_cols, f_i_prime_vector, label='adversarial', color='blue')
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.1, 1.05),
              ncol=3, fancybox=True, shadow=True, fontsize=8)
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel('Feature values (0-1)')
    ax[0].grid(zorder=5)
    ax[1].bar(feature_cols, diff_vector, label='difference', color='red', width=0.4)
    ax[1].plot(feature_cols, weight_vector, label='Weights', color='purple')
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.1, 1.05),
              ncol=3, fancybox=True, shadow=True, fontsize=8)
    ax[1].set_ylabel('Difference')
    ax[1].grid(zorder=5, color='gray')
    plt.xticks(rotation=90, fontsize=7)
    plt.suptitle(f'C-{algo} algorithm', fontsize=12)
    plt.savefig(os.path.join(results_dir, 'plots', algo +'_' + dataset + '_feats_perturbed.png'))
    plt.show()
    plt.close()

bar_plot()

#  ---------------------------------------------------- Highest & lowest perceptibility

def heatmap_1d(adv_percep, orig_percep, feature_cols):

    adv_conf, orig_conf = np.round(adv_percep.adv_confidence * 100), np.round(adv_percep.orig_confidence * 100)

    diff = np.abs(adv_percep[feature_cols].values[0] - orig_percep[feature_cols].values[0])
    adv_percep = list(adv_percep[feature_cols].values[0])
    orig_percep = list(orig_percep[feature_cols].values[0])

    diff = np.expand_dims(diff, 0)
    adv_percep = np.expand_dims(adv_percep, 0)
    orig_percep = np.expand_dims(orig_percep, 0)
    adv_percep = scaler.transform(adv_percep)
    orig_percep = scaler.transform(orig_percep)
    diff = scaler.transform(diff)

    return adv_percep, orig_percep, diff, adv_conf, orig_conf


def plot_heatmap(adv_percep, diff, orig_percep, title, filename):

    grid_kws = {'height_ratios': [1, 1, 1], 'hspace': 0.01}
    fig, axn = plt.subplots(nrows=3, sharex=True, gridspec_kw=grid_kws)
    cbar_ax = fig.add_axes([0.92, .3, .01, .5])

    for (i, ax), df in zip(enumerate(axn.flat), [adv_percep, diff, orig_percep]):
        sns.heatmap(df, ax=ax,
                    cbar=i==0, yticklabels=False, xticklabels=i==2,
                    vmin=0, vmax=0.5, cmap='gist_gray_r',
                    cbar_ax=None if i else cbar_ax)

    axn[0].set_ylabel('Original', fontsize=10)
    axn[1].set_ylabel('Perturbation', fontsize=10)
    axn[2].set_ylabel('Adversarial', fontsize=10)
    axn[2].set_xlabel('Features', fontsize=12)
    axn[2].set_aspect(2, adjustable='box', share=True)
    axn[2].set_xticklabels(axn[2].get_xticklabels(), rotation=90, fontsize=8)
    plt.suptitle(title, fontsize=12)
    plt.savefig(os.path.join(results_dir, 'plots', filename +'_' + dataset + '_perceptible.png'))
    plt.show()
    plt.close()



if algo == 'lowprofool':
    feature_cols = test.columns[:-1]
    # Highest Perceptible Sample
    # s_adv = s_adv[s_adv.adv_confidence>0.5]
    h_adv_percep = s_adv[(s_adv.perceptibility.max() == s_adv.perceptibility)]
    h_original = s_test.loc[h_adv_percep.index]
    adv_pred, orig_pred = s_adv.loc[h_adv_percep.index].adv_pred.values[0], s_adv.loc[h_adv_percep.index].Label.values[0]
    adv_percep, orig_percep, diff, adv_conf, orig_conf = heatmap_1d(h_adv_percep, h_original, feature_cols)
    title = f'Highest Original pred: {encoder[orig_pred]}-{orig_conf.values[0]} % ---> Adversarial pred: {encoder[adv_pred]}-{adv_conf.values[0]} %'
    title = f'Original pred: {encoder[orig_pred]} ---> Adversarial pred: {encoder[adv_pred]}'

    plot_heatmap(adv_percep, diff, orig_percep, title, 'highest')

    # Lowest Perceptible Sample
    l_adv_percep = s_adv[s_adv.perceptibility.min() == s_adv.perceptibility]
    l_original = s_test.loc[l_adv_percep.index]
    adv_pred, orig_pred = s_adv.loc[l_adv_percep.index].adv_pred.values[0], s_adv.loc[l_adv_percep.index].Label.values[0]
    adv_percep, orig_percep, diff, adv_conf, orig_conf = heatmap_1d(l_adv_percep, l_original, feature_cols)
    title = f'Original pred: {encoder[orig_pred]}-{orig_conf.values[0]} % ---> Adversarial pred: {encoder[adv_pred]}-{adv_conf.values[0]} %'
    title = f'Original pred: {encoder[orig_pred]} ---> Adversarial pred: {encoder[adv_pred]} %'

    plot_heatmap(adv_percep, diff, orig_percep, title, 'lowest')


print('Hold')