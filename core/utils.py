import os
import joblib
import json
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Directories(object):

    def __init__(self, config):  # constructor
        self.dataset = config['dataset']
        self.root_dir = config['dir_config']['root_dir']
        self.raw_data_dir = config['dir_config']['raw_data_dir']
        self.processed_data_dir = config['dir_config']['processed_data_dir']
        self.logs_dir = config['dir_config']['logs']
        self.results_dir = config['dir_config']['results']
        self.utils_dir = config['dir_config']['utils']

    def make_dirs(self, dir=None):  # Create a directory the first time

        dirs = [self.logs_dir, self.results_dir, self.utils_dir, self.processed_data_dir] if dir is None else [dir]
        for dir in dirs:
            if not os.path.exists(dir):
                print(f'***** {dir} directory created for the first time *****')
                logging.warning(f'***** {dir} directory created for the first time *****')
                os.makedirs(dir)


class Utilities(Directories):

    def __init__(self, config):
        super().__init__(config)
        pass

    def _encoder(self, train, test):
        '''
        Creates a label encoder object based on iterating on all features. scikit-learn is not adopted for this process
        Note: this is not coupled with other entities
        :param train: training set
        :param test: test set
        :return: dumps a pickle object for the encoder in respective utility directory. Later used by other modules of
        the pipeline.
        '''
        df = pd.concat([train, test])
        encoder = {}
        for col in df.columns:
            if df[col].dtype == 'object':  # all dtypes with 'object' are categorical features
                _ = {value: idx for idx, value in enumerate(sorted(df[col].unique()))}
                encoder[col] = _

        with open(os.path.join(self.utils_dir, 'encoder.json'), 'w') as file:
            json.dump(encoder, file, indent=4)
        logging.info('** Categorical features encoded ordinally and saved in JSON')
        print('** Categorical features encoded ordinally and saved in JSON')

    def _scaler(self, train, test):
        '''
        Creates a scaler object from sciki-learn library. MinMaxScaler performed better than StandardScaler.
        Note: this is not coupled with other entities
        :param train: training set
        :param test: test set
        :return: dumps a pickle object for the scaler in respective utility directory. Later used by other modules of
        the pipeline.
        '''
        xtr, ytr = train.iloc[:, :-1], train['Label']
        xts, yts = test.iloc[:, :-1], test['Label']
        features = pd.concat([xtr, xts])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(features)
        pth = os.path.join(self.utils_dir, 'scaler.pkl')
        joblib.dump(scaler, pth)
        logging.info('** MinMaxScaler object saved for the overall dataset')
        print('** MinMaxScaler object saved for the overall dataset')

    def generate(self):
        '''
        Creates a encoder pickle object and saves it in the respective utils directory.
        :return: further updates the self object with the scaled version.
        '''

        def helper_function(data):
            with open(os.path.join(self.utils_dir, 'encoder.json'), 'r') as file:
                encoder = json.load(file)
            cat_cols = list(encoder.keys())
            for col in cat_cols:
                data[col] = data[col].map(encoder[col])
            return data

        train = pd.read_csv(os.path.join(self.processed_data_dir, 'train.csv'))
        test = pd.read_csv(os.path.join(self.processed_data_dir, 'test.csv'))
        self._encoder(train, test)
        train, test = helper_function(train), helper_function(test)
        self._scaler(train, test)


class DataLoader(Utilities):

    def __init__(self, config):
        super().__init__(config)
        self.model_dir = os.path.join(self.results_dir, 'models')

    def encode(self, data):  # encode the categorical features of the dataframe
        with open(os.path.join(self.utils_dir, 'encoder.json'), 'r') as file:
            encoder = json.load(file)
        cat_cols = list(encoder.keys())
        for col in cat_cols:
            data[col] = data[col].map(encoder[col])
        return data

    def scale(self, data):  # scales the features of the dataframe

        scaler = joblib.load(os.path.join(self.utils_dir, 'scaler.pkl'))
        x, y = data.iloc[:, :-1], data['Label']
        x = scaler.transform(x)
        x = pd.DataFrame(x, columns=data.columns[:-1]).reset_index(drop=True)
        data = pd.concat([x, y], axis=1)
        return data

    def load_data(self, scaled_flag=True) -> list:
        '''
        Reads csv format processed data and returns ndarray as features and labels
        :param scaled_flag: if scaling of the dataset is required
        :return: features and labels of train and test set.
        '''
        train = pd.read_csv(os.path.join(self.processed_data_dir, 'train.csv'))
        test = pd.read_csv(os.path.join(self.processed_data_dir, 'test.csv'))
        train, test = self.encode(train), self.encode(test)

        if scaled_flag:
            train, test = self.scale(train), self.scale(test)
            logging.info('** Using scaled dataset **')
            print('** Using scaled dataset **')

        xtr, ytr = train.drop('Label', axis=1), train['Label']
        xts, yts = test.drop('Label', axis=1), test['Label']

        return xtr, ytr, xts, yts