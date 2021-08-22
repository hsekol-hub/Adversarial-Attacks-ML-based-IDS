import os
import pickle
import json
import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from core.models.metrics import CustomMetrics, SkLearnMetrics

def encode(data, utils_dir):
    with open(os.path.join(utils_dir, 'encoder.json'), 'r') as file:
        encoder = json.load(file)
    cat_cols = list(encoder.keys())
    for col in cat_cols:
        data[col] = data[col].map(encoder[col])
    return data


def scale(data, utils_dir):
    scaler = joblib.load(os.path.join(utils_dir, 'scaler.pkl'))
    x, y = data.iloc[:, :-1], data['Label']
    x = scaler.transform(x)
    x = pd.DataFrame(x, columns=data.columns[:-1]).reset_index(drop=True)
    data = pd.concat([x, y], axis=1)
    return data


def gen_eval_metrics(yts, ypred, clf_name='Transferability'):

    metricsObj = SkLearnMetrics(clf_name)
    acc = metricsObj.accuracy(yts, ypred)
    pre = metricsObj.precision(yts, ypred)
    re = metricsObj.recall(yts, ypred)
    f_support = metricsObj.f1(yts, ypred)

if __name__=='__main__':

    dataset, algo = 1, 1
    dict_ = {1: 'NSL-KDD',
             2: 'CICDDoS',
             3: 'CICIDS'}
    dataset = dict_[dataset]
    dict_ = {1: 'lowprofool',
             2: 'deepfool'}
    algo = dict_[algo]

    cwd = os.path.dirname(__file__)
    utils_dir = os.path.join(cwd[:-8], 'utils', dataset)
    results_dir = os.path.join(cwd[:-8], 'results', dataset)
    adv_path_ext = algo + '_adv_samples.csv'
    test_path_ext = algo + '_test_samples.csv'
    test = pd.read_csv(os.path.join(results_dir, 'adv', test_path_ext))
    adv = pd.read_csv(os.path.join(results_dir, 'adv', adv_path_ext))
    adv, test = encode(adv, utils_dir), encode(test, utils_dir)
    adv, test = scale(adv, utils_dir), scale(test, utils_dir)
    columns = test.columns
    xat, yat = adv[columns[:-1]], adv['Label']


    gb_path = os.path.join(results_dir, 'models', 'gradient_boost.sav')
    svc_path = os.path.join(results_dir, 'models', 'linear_svc.sav')
    nb_path = os.path.join(results_dir, 'models', 'naive_bayes.sav')
    rf_path = os.path.join(results_dir, 'models', 'random_forest.sav')
    gb = pickle.load(open(gb_path, 'rb'))
    svc = pickle.load(open(svc_path, 'rb'))
    nb = pickle.load(open(nb_path, 'rb'))
    rf = pickle.load(open(rf_path, 'rb'))

    models = {'random_forest': gb,
              'svc': svc,
              'naive_bayes': nb,
              'gradient_boost': gb}

    for name, model in models.items():

        y_pred = model.predict(xat)
        gen_eval_metrics(yat, y_pred, clf_name=name)