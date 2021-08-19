import torch
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,\
    roc_auc_score

class SkLearnMetrics(object):

    def __init__(self, name):
        self.classifier = name
        logging.info(f'Training {name} model...')

    def accuracy(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        print('accuracy score: {:.3f}'.format(acc))
        logging.info('accuracy score: {:.3f}'.format(acc))
        return np.round(acc, 3)

    def precision(self, y_true, y_pred):
        pre = precision_score(y_true, y_pred, average='weighted')
        print('precision score: {:.3f}'.format(pre))
        logging.info('precision score: {:.3f}'.format(pre))
        return np.round(pre, 3)

    def recall(self, y_true, y_pred):
        re = recall_score(y_true, y_pred, average='weighted')
        print('recall score: {:.3f}'.format(re))
        logging.info('recall score: {:.3f}'.format(re))
        return np.round(re, 3)

    def f1(self, y_true, y_pred):
        f_support = f1_score(y_true, y_pred, average='weighted')
        print('F-Score score: {:.3f}'.format(f_support))
        logging.info('F-Score score: {:.3f}'.format(f_support))
        return np.round(f_support, 3)

    def auroc(self, y_true, y_pred):
        roc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
        print('AU-ROC score: {:.3f}'.format(roc))
        logging.info('AU-ROC score: {:.3f}'.format(roc))
        return np.round(roc, 3)

class CustomMetrics(object):

    def __init__(self):
        pass


    def accuracy(self, y_pred, y_true):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_true).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return np.round(acc, 3), y_pred_tags

    def precision(self, y_pred, y_true):

        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        pre = precision_score(y_true, y_pred_tags, average='weighted')
        return np.round(pre, 3)

    def recall(self, y_pred, y_true):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        re = recall_score(y_true, y_pred_tags, average='weighted')
        return np.round(re, 3)

    def f1(self, y_pred, y_true):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        f_support = f1_score(y_true, y_pred_tags, average='weighted')
        return np.round(f_support, 3)

    def auroc(self, y_pred, y_true):
        y_pred_softmax = torch.softmax(y_pred, dim=1)/100
        # _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        roc = roc_auc_score(y_true, y_pred_softmax, average='weighted', multi_class='ovr')
        return np.round(roc, 3)

    def custom_softmax(self, latent_output):
        y_pred = torch.log_softmax(latent_output, axis=1)
        _, top3 = torch.topk(y_pred, 3)
        y_pred = top3[:, 0]

        y2_pred, y3_pred = top3[:, 1].numpy(), top3[:, 2].numpy()
        yt_pred = [y2 if y2 != 0 else y3 for y2, y3 in zip(y2_pred, y3_pred)]

        # get the confidence value here
        y_pred_ = torch.softmax(latent_output, axis=1)
        y_pred_confd, _ = torch.topk(y_pred_, 3)
        y_confidence = y_pred_confd[:, 0]

        return y_pred, yt_pred, y_confidence


    def success_rate(self, adv):
        sr = ((adv.target_pred == adv.adv_pred) | ((adv.Label == 0) & (adv.adv_pred != 0))).sum() / len(adv)
        return round(sr, 4)

    def perceptibility(self, f_i, f_i_prime, weights):
        _ = (f_i - f_i_prime) ** 2
        _ = _ * weights
        percep = np.sum(_)
        return np.round(percep, 2)

    def perturb_distance(self, adv):
        perturb_norm = np.round(np.mean(adv.perturbation_norm), 3)
        return perturb_norm

    def attack_score(self, f_i, f_i_prime, weights, adv):
        delta_c = adv.adv_confidence - (1 - adv.orig_confidence)
        denom = np.sqrt(((f_i - f_i_prime) ** 2)) * weights
        denom = np.sum(denom, 1)
        _ = delta_c / denom
        ascore = np.round(np.median(_), 2)
        return ascore

    def perturb_distance2(self, f_i, f_i_prime):

        f_i, f_i_prime = torch.tensor(f_i), torch.tensor(f_i_prime)
        perturb_l2_norm = lambda x, y: torch.sqrt(torch.sum(torch.abs(x - y) ** 2, dim=1))
        perturb_norm = perturb_l2_norm(f_i, f_i_prime)
        perturb_norm = np.round(torch.mean(perturb_norm), 3)
        return perturb_norm.numpy()

    def transfer_success(self):
        pass