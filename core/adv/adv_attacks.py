import os
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)

from core.adv.adv_utils import ThreatModel, AdversarialUtilities
from core.adv.lowprofool import run as lowprofool
from core.adv.deepfool import run as deepfool
from core.models.metrics import CustomMetrics, SkLearnMetrics


def evaluate_adv_set(test, adv, weights, scaler, algo='lowprofool'):

    # Compute the baseline metrics
    metrics = SkLearnMetrics('Adversarial Setting')
    y_true, adv_pred = adv.Label, adv.adv_pred
    adv_pred = [a if a != -1 else o for a, o in zip(adv_pred, y_true)]
    acc = metrics.accuracy(y_true, adv_pred)
    pre = metrics.precision(y_true, adv_pred)
    re = metrics.recall(y_true, adv_pred)
    f_support = metrics.f1(y_true, adv_pred)

    # Compute adversarial metrics
    metrics = CustomMetrics()
    adv.adv_pred = adv.adv_pred.astype('float64').astype('int')
    s_adv = adv[(adv.target_pred == adv.adv_pred) | ((adv.Label == 0) & (adv.adv_pred != -1))]
    s_test = test.loc[s_adv.index]
    feature_cols = test.columns[:-1]

    f_i = s_test[feature_cols].values
    f_i_prime = s_adv[feature_cols].values
    f_i = scaler.transform(f_i)
    f_i_prime = scaler.transform(f_i_prime)

    sr = metrics.success_rate(adv)
    ascore = metrics.attack_score(f_i, f_i_prime, weights, s_adv)

    if algo == 'lowprofool':
        pd = metrics.perturb_distance(s_adv)
        percep = metrics.perceptibility(f_i, f_i_prime, weights)
    else:
        percep = '-1'
        pd = metrics.perturb_distance2(f_i, f_i_prime)

    print('Attack Score: {}\nPerceptibility: {}\nPerturbation Distance: {}\nSucess Rate: {}'.\
          format(ascore, percep, pd, sr))
    return acc, pre, re, f_support, sr, pd, percep, ascore

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


def craft_ae(config, n_samples = None):

    tmObj = ThreatModel(config)
    advUtilsObj = AdversarialUtilities(config)

    print('_________________________________\n'
          'Select an attack algorithm:\n'
          '1. Lowprofool\n'
          '2. Deepfool\n'
          '_________________________________\n'
          'Enter choice in the above range separated by space as delimiter:...\n')
    # selection = list(input())
    selection = [1]
    selection = list(reversed(sorted([int(c) for c in selection if c != ' '])))
    dict_ = {1: 'lowprofool',
             2: 'deepfool'}

    n_samples = 256
    for algo in selection:
        algo = dict_[algo]
        threat_model = tmObj.define(algo)
        model, data, data_loader, scaler, weights, encoder, perturbation_space, mask\
            = advUtilsObj.load_utils(threat_model=threat_model,
                                     n_samples=n_samples)
        model.eval()
        multiplicity, lower_bounds, upper_bounds = advUtilsObj.fetch_bounds(encoder)

        if algo == 'lowprofool':
            test_set, adv_set = lowprofool(advUtilsObj, threat_model, model, data_loader, data, scaler, weights,
                                              encoder, mask, multiplicity, lower_bounds, upper_bounds)
        elif algo == 'deepfool':
            test_set, adv_set = deepfool(advUtilsObj, threat_model, model, data_loader, data, scaler, weights,
                                            encoder, mask, multiplicity, lower_bounds, upper_bounds)
        test_set.to_csv(os.path.join(tmObj.results_dir, 'adv', algo + '_test_samples.csv'), index=False)
        adv_set.to_csv(os.path.join(tmObj.results_dir, 'adv', algo + '_adv_samples.csv'), index=False)

        # --------------------------------------------------- Generate Reports
        test_set = encode(test_set, encoder)
        adv_set = encode(adv_set, encoder, adv_flag=True)
        if algo == 'lowprofool':
            acc, pre, re, f_support, sr, p_dist, percep, a_score = evaluate_adv_set(test_set, adv_set, weights.numpy(),
                                                                                    scaler)
        else:
            acc, pre, re, f_support, sr, p_dist, percep, a_score = evaluate_adv_set(test_set, adv_set, weights.numpy(),
                                                                                    scaler, algo)

        uniq_labels = test_set.Label.unique()
        reports_pth = os.path.join(advUtilsObj.root_dir, 'results', algo+'_reports.csv')
        reports = pd.DataFrame([advUtilsObj.dataset, n_samples, len(uniq_labels), algo,
                        acc, pre, re, f_support, sr, p_dist, percep, a_score, 'NA',
                        str(tmObj.adv_config['capability']), str(tmObj.adv_config[algo]), uniq_labels]).T

        reports.columns = ['Dataset', '#Samples', '#Labels', 'Algorithm',
                           'Accuracy', 'Precision', 'Recall', 'F1 Score',
                           'Success Rate', 'Perturb distance', 'Perceptibility',
                           'Attack Score', 'Transfer Success', 'Capability', 'Setting', 'Labels']

        if not os.path.exists(reports_pth):
            reports.to_csv(reports_pth, mode='w', header=True, index=False)
            print('New report file generated')
        else:
            reports.to_csv(reports_pth, mode='a', header=False, index=False)
            print('Test results appended to existing report file.')

        print('Done')