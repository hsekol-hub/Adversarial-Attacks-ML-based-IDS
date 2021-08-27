import os
import glob
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.utils import Directories
from core.viz import plot_class_dist


class DataHandling(object):

    def __init__(self):
        pass

    def drop_unique_cols(self, train, test, add_cols: list):
        '''
        Drops unique columns as a part of data-preprocessing; as they won't make any difference in model predictions.
        Unique columns are automatically decided based on the logic below.
        :param train: train set
        :param test: test set
        :param add_cols: any additional columns which need to be dropped. Needs to passed a an entry in list object
        :return:
        '''

        df = pd.concat([train, test])
        unique_cols = [col for col in df.columns if len(df[col].unique()) == 1]
        df = df.drop(unique_cols, axis=1)
        logging.info('**Dropped {} columns with unique values: {}'.format(len(unique_cols), ' '.join(unique_cols)))
        print('**Dropped {} columns with unique values: {}'.format(len(unique_cols), ' '.join(unique_cols)))
        df = df.drop(add_cols, axis=1)
        logging.info('**Dropped {} additional columns: {}'.format(len(add_cols), ', '.join(add_cols)))
        print('**Dropped {} additional columns: {}'.format(len(add_cols), ', '.join(add_cols)))
        train, test = df.iloc[:len(train)], df.iloc[len(train):]
        return train, test


    def rename_class_labels(self, df, dataset: str):
        '''
        Preprocessing of class labels; some labels are binned under a common cluster of labels.
        for CICDDoS: train & test sets had different naming conventions for nameing hte labels
        :param df: whole dataset combined: pandas DataFrame()
        :param dataset: name of the dataset
        :return: clean dataframe
        '''

        if dataset == 'CICDDoS':
            print('Class labels renamed for CICDDoS')
            df.Label = df.Label.replace('DrDoS_DNS', 'DNS')
            df.Label = df.Label.replace('DrDoS_LDAP', 'LDAP')
            df.Label = df.Label.replace('DrDoS_MSSQL', 'MSSQL')
            df.Label = df.Label.replace('DrDoS_NTP', 'NTP')
            df.Label = df.Label.replace('DrDoS_NetBIOS', 'NetBIOS')
            df.Label = df.Label.replace('DrDoS_SNMP', 'SNMP')
            df.Label = df.Label.replace('DrDoS_SSDP', 'SSDP')
            df.Label = df.Label.replace('DrDoS_UDP', 'UDP')
            df.Label = df.Label.replace('UDP-lag', 'UDPLag')
        elif dataset == 'CICIDS':
            df.Label = df.Label.replace('DoS Hulk', 'DoS')
            df.Label = df.Label.replace('DoS GoldenEye', 'DoS')
            df.Label = df.Label.replace('DoS slowloris', 'DoS')
            df.Label = df.Label.replace('DoS Slowhttptest', 'DoS')
            df.Label = df.Label.replace('Web Attack � Brute Force', 'Web Attack')
            df.Label = df.Label.replace('Web Attack � XSS', 'Web Attack')
            df.Label = df.Label.replace('Web Attack � Sql Injection', 'Web Attack')
            df.Label = df.Label.replace('Heartbleed', 'Others')
            df.Label = df.Label.replace('Infiltration', 'Others')
            df.Label = df.Label.replace('FTP-Patator', 'Patator')
            df.Label = df.Label.replace('SSH-Patator', 'Patator')
        return df


class FeatureEngineering(Directories):

    def __init__(self, config):
        super().__init__(config)
        self.dhObj = DataHandling()

    def _sample_cicddos(self, path: str, tr_samples: int, ts_samples: int):
        '''
        In CICDDoS undersampling was required as count of benign samples are limited to ~110K.
        Equal number of samples for all other labels are picked from both sets.
        :param path: directory of the raw files.
        :param tr_samples: count of samples from training set.
        :param ts_samples: count of samples from test set.
        :return: None; saves the samples as individual <label_name>.csv files.
        '''

        logging.info('+++++ Sampling CICDDoS2019 dataset +++++')
        print('+++++ Sampling CICDDoS2019 dataset +++++')
        def helper_function(data, label: str, n_samples: int):
            # Clean data for CICDDoS2019 by removing negative entries.
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(inplace=True)
            data = data[data[' Fwd Header Length'] >= 0]
            data = data[data[' Bwd Header Length'] >= 0]
            data = data[data[' min_seg_size_forward'] >= 0]

            if len(data) < n_samples:  # consider all examples of this label
                return data
            else:
                if label == 'Benign':  # consider all benign samples as they are less
                    return data
                else:
                    data = data.sample(n=n_samples).reset_index(drop=True)
                    return data

        samples_path = os.path.join(path, 'samples')
        if not os.path.exists(samples_path):  # Create sample directory the first time
            os.makedirs(samples_path)
            os.makedirs(os.path.join(samples_path, 'train'))
            os.makedirs(os.path.join(samples_path, 'test'))
            logging.info('***** CICDDoS samples directory created for the first time *****')

        # Consider class labels present in both datasets
        class_labels = ['Benign', 'LDAP', 'MSSQL', 'NetBIOS', 'Syn', 'UDPLag','UDP']

        for label in class_labels:
            print('Sampling on label: {}...'.format(label))
            filename = str(label) + '.csv'
            train = pd.read_csv(os.path.join(path, 'training', filename))
            test = pd.read_csv(os.path.join(path, 'testing', filename))
            train = helper_function(train, label, tr_samples)  # data cleaning & sampling
            test = helper_function(test, label, ts_samples)

            train.to_csv(os.path.join(samples_path, 'train', filename), index=False)
            test.to_csv(os.path.join(samples_path, 'test', filename), index=False)
            logging.info(f'***** {label} sampled in both directories\tTotal length: {len(train)} | {len(test)} *****')
            print(f'***** {label} sampled in both directories\tTotal length: {len(train)} | {len(test)} *****')



    def _nsl_kdd(self, path, add_cols):

        cols = pd.read_csv(os.path.join(path, 'Field Names.csv'), header=None)
        cols = cols.append([['label', 'Symbolic'], ['difficulty', 'Symbolic']])
        cols.columns = ['Name', 'Type']

        train = pd.read_csv(os.path.join(path, 'KDDTrain+.csv'), header=None)
        train.columns = cols['Name'].T

        test = pd.read_csv(os.path.join(path, 'KDDTest+.csv'), header=None)
        test.columns = cols['Name'].T

        with open(os.path.join(path, 'label_map.json'), 'r') as file:
            label_map = json.load(file)

        train['Label'] = train.label.map(label_map)
        test['Label'] = test.label.map(label_map)
        train, test = self.dhObj.drop_unique_cols(train, test, add_cols)

        return train, test


    def _cicddos(self, path: str, sample_data_flag: bool, add_cols: list):
        '''
        Reads CICDDoS dataset and performs basic preprocessing steps.

        :param path: raw dataset directory path
        :param sample_data_flag: if True samples from the raw dataset in defined ratios
        :param add_cols: additional columns to be dropped [dataset specific]
        :return: train and test sets.
        '''

        def helper_function(pth):
            df = pd.DataFrame()  # holds records from all individual files as [train, test]
            for fPath in glob.glob(pth):
                subset = pd.read_csv(fPath)
                df = df.append(subset)

            column_names = [c.replace(' ', '') for c in df.columns]
            df.columns = column_names
            df = self.dhObj.rename_class_labels(df, 'CICDDoS')
            return df

        if sample_data_flag:
            self._sample_cicddos(path, tr_samples=60000, ts_samples=60000)

        tr_path = os.path.join(path, 'samples', 'train', '*csv')
        ts_path = os.path.join(path, 'samples', 'test', '*csv')
        train = helper_function(tr_path)
        test = helper_function(ts_path)
        train, test = self.dhObj.drop_unique_cols(train, test, add_cols)
        return train, test

    def _cicids(self, path: str, add_cols: list):
        '''
        Reads CICIDS dataset and performs basic preprocessing steps.
        :param path: raw dataset directory path
        :param add_cols: additional columns to be dropped [dataset specific]
        :return: train and test sets.
        '''

        def helper_function(path):
            df = pd.DataFrame()
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".csv"):
                        print(f'Reading {file}...')
                        subset = pd.read_csv(os.path.join(root, file))
                        df = pd.concat([df, subset], ignore_index=True)

            column_names = [c.replace(' ', '') for c in df.columns]
            df.columns = column_names
            df = self.dhObj.rename_class_labels(df, 'CICIDS')
            return df

        df = helper_function(path)
        Y = df.pop('Label').to_frame()
        X = df
        x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
        train = pd.concat([x_train, y_train], axis=1)
        test = pd.concat([x_test, y_test], axis=1)
        train, test = self.dhObj.drop_unique_cols(train, test, add_cols)
        return train, test

    def process_raw_data(self, sample_data_flag=False, save_plots_flag=False):
        """
        Loads raw dataset does basic preprocessing steps and saves results as [train, test]
        Args:
            dataset: user selected dataset to be working on
            config: configuration file

        Returns:
        processed [train, test] dataframe objects
        """
        print('_' * 120)
        logging.info('++ Data Preparation for {} dataset ++'.format(self.dataset))
        print('++ Data Preparation for {} dataset ++'.format(self.dataset))
        train, test = None, None

        if self.dataset == 'NSL-KDD':
            add_cols = ['difficulty', 'label']
            train, test = self._nsl_kdd(self.raw_data_dir, add_cols)
        elif self.dataset == 'CICDDoS':
            add_cols = ['Unnamed:0', 'FlowID', 'SourceIP', 'DestinationIP', 'Timestamp', 'SimillarHTTP',
                        'FwdHeaderLength.1', 'FwdPacketLengthMean', 'BwdPacketLengthMean', 'FlowBytes/s',
                        'FlowPackets/s', 'FwdPackets/s', 'BwdPackets/s', 'MinPacketLength', 'MaxPacketLength',
                        'AvgFwdSegmentSize', 'AvgBwdSegmentSize']
            train, test = self._cicddos(self.raw_data_dir, sample_data_flag, add_cols)
        elif self.dataset == 'CICIDS':
            add_cols = ['DestinationPort', 'FwdHeaderLength.1', 'FwdPacketLengthMean', 'BwdPacketLengthMean', 'FlowBytes/s',
                        'FlowPackets/s', 'FwdPackets/s', 'BwdPackets/s', 'MinPacketLength', 'MaxPacketLength',
                        'AvgFwdSegmentSize', 'AvgBwdSegmentSize']
            train, test = self._cicids(self.raw_data_dir, add_cols)

        train, test = train.sample(frac=1), test.sample(frac=1)
        if 'normal' in train['Label'].unique():
            print('Normal values replaced with Benign keyword')
            train = train.replace('normal', 'Benign')
            test = test.replace('normal', 'Benign')
        elif 'BENIGN' in train['Label'].unique():
            print('BENIGN values replaced with Benign keyword')
            train = train.replace('BENIGN', 'Benign')
            test = test.replace('BENIGN', 'Benign')

        train.to_csv(os.path.join(self.processed_data_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.processed_data_dir, 'test.csv'), index=False)

        print(pd.concat([train, test]).Label.value_counts())
        if save_plots_flag:
            plot_class_dist(train, test, self.results_dir, dataset_name=self.dataset)

        logging.info('** Processed file saved in: {} **'.format(self.processed_data_dir))
        print('** Processed file saved in: {} **'.format(self.processed_data_dir))
        print('_' * 120)