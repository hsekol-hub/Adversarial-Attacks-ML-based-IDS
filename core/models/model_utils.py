import os
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from core.utils import DataLoader as CustomDataLoader

class ClassifierDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)  # TODO

class Network(nn.Module):  # Define neural network

    def __init__(self, num_features, num_class, n_hidden):

        super(Network, self).__init__()
        self.layer_1 = nn.Linear(num_features, n_hidden)
        self.layer_2 = nn.Linear(n_hidden, int(n_hidden / 2))
        self.layer_3 = nn.Linear(int(n_hidden / 2), int(n_hidden / 4))
        self.layer_out = nn.Linear(int(n_hidden / 4), num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm_1 = nn.BatchNorm1d(n_hidden)
        self.batchnorm_2 = nn.BatchNorm1d(int(n_hidden / 2))
        self.batchnorm_3 = nn.BatchNorm1d(int(n_hidden / 4))


    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.batchnorm_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_3(x)
        x = self.batchnorm_3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

class ModelUtilities(CustomDataLoader):

    def __init__(self, config):  # [train, test] available in the object
        super().__init__(config)
        self.batch_size = config['model_config']['batch_size']
        self.n_epochs = config['model_config']['n_epochs']
        self.lr = config['model_config']['lr']

        self.train = pd.read_csv(os.path.join(self.processed_data_dir, 'train.csv'))
        self.test = pd.read_csv(os.path.join(self.processed_data_dir, 'test.csv'))
        self.train, self.test = self.encode(self.train), self.encode(self.test)

        self.n_class = len(self.train['Label'].unique())
        self.n_features = len(self.train.columns) - 1
        self.n_hidden = self.n_features * 2

    def train_val_test_split(self, train, test):
        # Split dataset in train, validation and test sets.
        print('-- Splitting dataset in train test and validation sets... --')
        xtr, ytr = train.iloc[:, :-1], train['Label']
        xts, yts = test.iloc[:, :-1], test['Label']
        xtr, xvl, ytr, yvl = train_test_split(xtr, ytr, test_size=0.2, stratify=ytr, random_state=123)
        xtr, ytr = np.array(xtr), np.array(ytr)
        xvl, yvl = np.array(xvl), np.array(yvl)
        xts, yts = np.array(xts), np.array(yts)
        
        assert len(xtr) + len(xvl) == len(train)
        total_len = len(xtr) + len(xts) + len(xvl)
        print('Dataset Samples:\n Train: {}% | Test: {}% | Validation: {}%'.format(round(len(xtr) / total_len * 100),
                                                                                   round(len(xts) / total_len * 100),
                                                                                   round(len(xvl) / total_len * 100)))

        return xtr, ytr, xvl, yvl, xts, yts

    def load_dn(self, scaled_flag=True):
        '''
        Prepater DataLoader object on dataset for trainig neural network
        :param scaled_flag: if dataset has to be scaled
        :return: DataLoader object
        '''
        logging.info(f'**** Model Hyper parameters: *****')
        logging.info('_'*100)
        print('_'*100)
        logging.info('-- Batch Size: {}\t Epochs: {}\t lr: {}\t labels: {}\t features: {} --'.\
                     format(self.batch_size, self.n_epochs, self.lr, self.n_class, self.n_features))
        print('-- Batch Size: {}\t Epochs: {}\t lr: {}\t labels: {}\t features: {} --'. \
                     format(self.batch_size, self.n_epochs, self.lr, self.n_class, self.n_features))

        if scaled_flag:
            train, test = self.scale(self.train), self.scale(self.test)
            xtr, ytr, xvl, yvl, xts, yts = self.train_val_test_split(train, test)
        else:
            xtr, ytr, xvl, yvl, xts, yts = self.train_val_test_split(self.train, self.test)

        train_dataset = ClassifierDataset(torch.from_numpy(xtr).float(), torch.from_numpy(ytr).long())
        val_dataset = ClassifierDataset(torch.from_numpy(xvl).float(), torch.from_numpy(yvl).long())
        test_dataset = ClassifierDataset(torch.from_numpy(xts).float(), torch.from_numpy(yts).long())

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=int(self.batch_size / 2),
                                 shuffle=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=int(self.batch_size * 0.5),
                                shuffle=True)
        return train_loader, test_loader, val_loader