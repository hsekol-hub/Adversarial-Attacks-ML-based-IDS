import os
import pandas as pd
import numpy as np
import yaml
import logging
import json
import joblib
import torch

from torch.utils.data import DataLoader
from core.models.model_utils import ModelUtilities, Network, ClassifierDataset

class ConstraintSpace(ModelUtilities):

    def __init__(self, config: dict):
        super().__init__(config)
        pass

    def rev_dict(self, dictionary):  # reverse a dictionary (key, value) pair
        return dict(zip(dictionary.values(), dictionary.keys()))

    def generate_perturb_space(self, data):
        # creates a perturbation space for each label of the dataset.
        # load the encoder object
        with open(os.path.join(self.utils_dir, 'encoder.json'), 'r') as file:
            encoder = json.load(file)

        dictionary = self.rev_dict(encoder['Label'])
        data['Label'] = data['Label'].map(dictionary)
        print('-- Calculating Perturbation Space... --')
        perturbation_space = {}
        for label in data['Label'].unique():
            subset = data[data['Label'] == label]  # bin subset for current class label
            helper_dict = {}
            label_cols = ['Label']
            for col in subset.columns:  # iterate on each feature to calculate bounds
                if col not in label_cols:  # except the label itself
                    multiplicity = 'categorical' if subset[col].dtype == 'object' else 'numerical'
                    lb, ub = min(subset[col]), max(subset[col])
                    if multiplicity == 'numerical':  # append the multiplicity label for each feature
                        _ = [1 if value % 1 == 0 else 0 for value in subset[col]]
                        multiplicity = 'discrete' if sum(_) == len(subset) else 'continuous'
                        lb, ub = min(subset[col]), max(subset[col])
                    helper_dict[col] = (multiplicity, lb, ub)
                perturbation_space[label] = helper_dict
        perturbation_space = dict(sorted(perturbation_space.items()))
        with open(os.path.join(self.utils_dir, 'perturbation_space.json'), 'w') as file:
            json.dump(perturbation_space, file, indent=4)
        print('-- Perturbation space saved in utils directory... --')
        logging.info('-- Perturbation space saved in utils directory... --')

    def generate_cor_weights(self, data):
        # generate a feature importance vector based on Pearson's coefficient formulation.
        cor = data.corr()
        cor = abs(cor['Label'])
        cor_weights = cor[:-1]
        cor_weights = cor_weights / np.linalg.norm(cor_weights, ord=2)
        weights_dict = {k:v for k, v in zip(data.columns[:-1], cor_weights)}
        with open(os.path.join(self.utils_dir, 'cor_feature_importance.json'), 'w') as file:
            json.dump(weights_dict, file, indent=4)
        print('-- co-relation weight vector saved in utils directory')
        logging.info('-- co-relation weight vector saved in utils directory')

    def generate_masks(self, data, threshold=None, cor_weights=False):
        # creates a mask vector. 1 denotes perturbation os possible and 0 means those features can't be manipulated.
        with open(os.path.join(self.root_dir, 'config', 'ip_perturb_config.yaml'), 'r') as stream:
            pertub_config = yaml.safe_load(stream)

        feature_names = list(data.columns)
        feature_names.remove('Label')
        masking_config = pertub_config[self.dataset]
        functional_features = [] if masking_config['functional_features'] is None else masking_config['functional_features']
        non_perturbed_features = [] if masking_config['non_perturbed_features'] is None else masking_config['non_perturbed_features']

        # -----------------------------------------------------------------------------
        if threshold is not None:  # idea is to not perturb which are significant. threshold sets a cut-off in percentage
            if cor_weights:
                print('Considering co-relation weight vector')
                with open(os.path.join(self.utils_dir, 'cor_feature_importance.json'), 'r') as file:
                    weight_vector = json.load(file)
            else:
                print('Considering random forest weight vector')
                with open(os.path.join(self.utils_dir, 'rf_feature_importance.json'), 'r') as file:
                    weight_vector = json.load(file)

            threshold_features, weights = list(weight_vector.keys()), list(weight_vector.values())
            threshold_features = [feature for feature, weight in zip(threshold_features, weights) if weight > threshold]
            non_perturbed_features += threshold_features
        # -----------------------------------------------------------------------------

        binary_check = set([0, 1])  # we do not want to manipulate the binary features as well.
        binary_features = [col for col in feature_names if set(data[col].unique()) == binary_check]
        non_perturbed_features = list(set(non_perturbed_features) - set(binary_features) - set(functional_features))
        # a list of all features which are to be manipulated
        pertubed_features = list(
            set(feature_names) - set(functional_features) - set(non_perturbed_features) - set(binary_features))
        mask = [1 if col in pertubed_features else 0 for col in feature_names]
        mask_dict = {k: v for k, v in zip(data.columns[:-1], mask)}
        # save this inffile information in a .YAML in the config directory
        masking_config['binary_features'] = binary_features
        masking_config['perturbed_features'] = pertubed_features
        masking_config['non_perturbed_features'] = non_perturbed_features
        pertub_config[self.dataset] = masking_config

        # -----------------------------------------------------------------------------
        with open(os.path.join(self.utils_dir, 'mask.json'), 'w') as file:
            json.dump(mask_dict, file, indent=4)  # save the masking information; to be used by the adversary

        with open(os.path.join(self.root_dir, 'config', 'tmp', 'op_perturb_config.yaml'), 'w') as stream:
            yaml.dump(pertub_config, stream)
        print('** Total features being perturbed: {}'.format(len(pertubed_features)))
        logging.info('** Total features being perturbed: {}'.format(len(pertubed_features)))


# Define our threat model
class ThreatModel(ModelUtilities):

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.adv_config = config['adv_config']

    def adversary_goals(self, threat_model: dict) -> dict:
        threat_model['target'] = self.adv_config['goals']
        return threat_model

    def adversary_knowledge(self, threat_model: dict) -> dict:

        threat_model['knowledge'] = self.adv_config['knowledge']
        train = pd.read_csv(os.path.join(self.processed_data_dir, 'train.csv'))
        test = pd.read_csv(os.path.join(self.processed_data_dir, 'test.csv'))
        data = test if threat_model['knowledge'] == 'test' else pd.concat([train, test])
        data = self.encode(data)
        return threat_model, data

    def adversary_capabilities(self, algorithm: str, threat_model: dict) -> dict:

        threat_model['threshold'] = self.adv_config['capability']['perturb_threshold_features']
        threat_model['cor_weights'] = self.adv_config['capability']['cor_weights']
        threat_model['batch_size'] = self.adv_config['capability']['batch_size']

        if algorithm == 'lowprofool':
            threat_model['max_iters'] = self.adv_config['lowprofool']['max_iters']
            threat_model['alpha'] = self.adv_config['lowprofool']['alpha']
            threat_model['lambda'] = self.adv_config['lowprofool']['lambda']
        elif algorithm == 'deepfool':
            threat_model['max_iters'] = self.adv_config['deepfool']['max_iters']
            threat_model['alpha'] = self.adv_config['deepfool']['alpha']
            threat_model['overshoot'] = self.adv_config['deepfool']['overshoot']
            threat_model['epsilon'] = self.adv_config['deepfool']['epsilon']
        return threat_model


    def define(self, algorithm: str) -> dict:  # defines the threat model based on [goals, knowledge, capabilities]

        print('_'*120)
        logging.info('_'*120)
        print('-- Initialising threat model for {}... --'.format(algorithm))
        logging.info('-- Initialising threat model for {}... --'.format(algorithm))

        threat_model = dict()
        threat_model = self.adversary_goals(threat_model)
        threat_model, data = self.adversary_knowledge(threat_model)
        threat_model = self.adversary_capabilities(algorithm, threat_model)

        csObj = ConstraintSpace(self.config)
        csObj.generate_perturb_space(data=data)
        csObj.generate_masks(data=data, threshold=threat_model['threshold'], cor_weights=threat_model['cor_weights'])
        return threat_model


class AdversarialUtilities(ThreatModel):
    def __init__(self, config: dict):
        super().__init__(config)
        pass
        # self.threat_model, self.data = self.define()

    def load_utils(self, threat_model: dict, n_samples=None):  # load all utility object

        model = Network(self.n_features, self.n_class, self.n_hidden)
        model.load_state_dict(torch.load(os.path.join(self.results_dir, 'models', 'dnn_model.pth')))

        scaler = joblib.load(os.path.join(self.utils_dir, 'scaler.pkl'))
        if threat_model['cor_weights']:
            with open(os.path.join(self.utils_dir, 'cor_feature_importance.json'), 'r') as file:
                weights = json.load(file)
        else:
            with open(os.path.join(self.utils_dir, 'rf_feature_importance.json'), 'r') as file:
                weights = json.load(file)
                
        with open(os.path.join(self.utils_dir, 'encoder.json'), 'r') as file:
            encoder = json.load(file)
            
        with open(os.path.join(self.utils_dir, 'perturbation_space.json'), 'r') as file:
            perturbation_space = json.load(file)

        with open(os.path.join(self.utils_dir, 'mask.json'), 'r') as file:
            mask = json.load(file)
            
        weights = torch.tensor(list(weights.values()))
        mask = torch.tensor(list(mask.values()))
        
        data = pd.read_csv(os.path.join(self.processed_data_dir, 'test.csv'))
        data = self.encode(data)
        data = self.scale(data)

        # -----------------------------------------------------------------------------
        if n_samples is not None:  # relevant for the ablation study in Section 5
            # data = data[data.Label != 1].reset_index(drop=True)
            # data = data[data.Label != 2].reset_index(drop=True)
            # data = data[data.Label != 0].reset_index(drop=True)
            # data = data[data.Label == 6].reset_index(drop=True)
            data = data.iloc[:n_samples]

        x, y = data.iloc[:, :-1].values, data['Label'].values
        dataset = ClassifierDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
        data_loader = DataLoader(dataset=dataset, batch_size=threat_model['batch_size'],
                                 shuffle=False)  # smaller batch size --> better adversarial attack
        # -----------------------------------------------------------------------------
        return model, data, data_loader, scaler, weights, encoder, perturbation_space, mask


    def fetch_bounds(self, encoder):

        with open(os.path.join(self.utils_dir, 'perturbation_space.json'), 'r') as file:
            perturbation_space = json.load(file)
        multiplicity, lower_bounds, upper_bounds = {}, {}, {}
        for label in perturbation_space.keys():
            encoded_label = (encoder['Label'][label])
            values = perturbation_space[label]
            multiplicity[encoded_label] = [values[col][0] for col in values.keys()]
            lower_bounds[encoded_label] = [values[col][1] for col in values.keys()]
            upper_bounds[encoded_label] = [values[col][2] for col in values.keys()]

        multiplicity = np.array([values for label, values in sorted(multiplicity.items())])
        lower_bounds = torch.tensor([values for label, values in sorted(lower_bounds.items())])
        upper_bounds = torch.tensor([values for label, values in sorted(upper_bounds.items())])

        return multiplicity[0], lower_bounds, upper_bounds

    def clip_tensor(self, x_prime, bounds, scaler, true_label):  # input coherence
        lower_bounds = torch.tensor(np.array(bounds[0][true_label]))
        upper_bounds = torch.tensor(np.array(bounds[1][true_label]))
        lower_bounds = scaler.transform(lower_bounds)
        upper_bounds = scaler.transform(upper_bounds)
        lower_bounds, upper_bounds = torch.tensor(lower_bounds, dtype=torch.float), torch.tensor(upper_bounds,
                                                                                                 dtype=torch.float)
        clipped = torch.max(torch.min(x_prime, upper_bounds), lower_bounds)
        return clipped

    def check_multiplicity(self, x_prime, multiplicity, mask, model, scaler, target_pred):

        x_prime_it = x_prime.detach().clone()
        x_prime_it = scaler.inverse_transform(x_prime_it)
        x_prime_it = np.round(x_prime_it, 3)  # cautionary measure
        x_prime_mult_it = x_prime_it.copy()
        rounding_indices = np.where([a and b for a, b in zip(multiplicity == 'discrete', mask == 1)])[0]
        x_prime_mult_it[:, rounding_indices] = np.round(x_prime_it[:, rounding_indices])
        x_prime_mult_it = scaler.transform(x_prime_mult_it)
        x_prime_mult_it = torch.tensor(x_prime_mult_it, dtype=torch.float32)
        latent_output = model(x_prime_mult_it)
        _, adv_pred = torch.max(torch.log_softmax(latent_output, dim=1), dim=1)
        # do not return multiplied where adv. prediction did not match target pred
        row_indices = np.where(adv_pred != target_pred)[0]
        x_prime_mult_it[row_indices, :] = x_prime[row_indices, :]

        return x_prime_mult_it


    def cicids_recompute_features(self, x_prime):
        x_prime = x_prime.squeeze(0)
        feature_names = self.test.columns[:-1]
        assert len(x_prime) == len(feature_names)
        x_prime = pd.Series(x_prime, index=feature_names)
        zero_check = lambda num, den: num / den if den > 0 else 0
        x_prime['FwdPacketLengthMean'] = zero_check(x_prime['TotalLengthofFwdPackets'], x_prime['TotalFwdPackets'])
        x_prime['BwdPacketLengthMean'] = zero_check(x_prime['TotalLengthofBwdPackets'], x_prime['TotalBackwardPackets'])
        x_prime['FlowBytes/s'] = zero_check((x_prime['TotalLengthofFwdPackets'] + x_prime['TotalLengthofBwdPackets']),
                                      x_prime['FlowDuration']) * 1e+6
        x_prime['FlowPackets/s'] = zero_check((x_prime['TotalFwdPackets'] + x_prime['TotalBackwardPackets']), x_prime['FlowDuration']) * 1e+6
        x_prime['FwdPackets/s'] = zero_check(x_prime['TotalFwdPackets'], x_prime['FlowDuration']) * 1e+6
        x_prime['BwdPackets/s'] = zero_check(x_prime['TotalBackwardPackets'], x_prime['FlowDuration']) * 1e+6
        x_prime['MinPacketLength'] = min(x_prime['FwdPacketLengthMin'], x_prime['BwdPacketLengthMin'])
        x_prime['MaxPacketLength'] = max(x_prime['FwdPacketLengthMax'], x_prime['BwdPacketLengthMax'])
        x_prime['AvgFwdSegmentSize'] = x_prime['FwdPacketLengthMean']
        x_prime['AvgBwdSegmentSize'] = x_prime['BwdPacketLengthMean']
        x_prime = np.array(x_prime)
        x_prime = x_prime.reshape(-1, len(x_prime))
        return x_prime