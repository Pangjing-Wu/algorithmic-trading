import sys
from typing import List

import torch
import numpy as np

from .profile import volume_profile


class SupervisedData(object):

    def __init__(self, X, y):
        self.__X = X if torch.is_tensor(X) else torch.tensor(X)
        self.__y = y if torch.is_tensor(y) else torch.tensor(y)
        self.__X = self.__X.float()
        self.__y = self.__y.float().reshape(-1, 1)

    def __len__(self):
        return len(self.__X)

    def __str__(self):
        return "%s" % dict(x=self.__X, y=self.__y)

    @property
    def X(self):
        return self.__X
    
    @property
    def y(self):
        return self.__y


class VolumeProfileDataset(object):
    ''' get volume profile dataset

    Arguments:
    ----------
        dataset: tickdata dataset, e.g.: {CSVDataset/H2Dataset}.
    '''

    def __init__(self, dataset, split:List[float], 
                 time_range:List[int], interval:int,
                 history_length=20):
        self.__check_split(split)
        self.__dataset    = dataset
        self.__split      = split
        self.__time_range = time_range
        self.__interval   = interval
        self.__hist_len   = history_length
        self.__build_dataset()

    @property
    def X_len(self):
        return self.__hist_len

    @property
    def y_len(self):
        return 1

    @property
    def n_tranche(self):
        return self.__n_tranche

    @property
    def train_set(self):
        return self.__train_set

    @property
    def train_date(self):
        return self.__train_date

    @property
    def valid_set(self):
        return self.__valid_set

    @property
    def valid_date(self):
        return self.__valid_date

    @property
    def test_set(self):
        return self.__test_set

    @property
    def test_date(self):
        return self.__test_date

    def __build_dataset(self):
        date = list()
        dataset = list()
        for data in self.__dataset:
            date.append(data.date)
            profile = volume_profile(data.trade, self.__time_range, self.__interval)
            dataset.append(profile['volume'].values)
        self.__n_tranche = profile.shape[0]
        X, y = self.__rolling_generate(np.array(dataset), self.__hist_len)
        X, y = torch.from_numpy(X), torch.from_numpy(y)
        n_train = int(X.shape[0] * self.__split[0] // self.__n_tranche * self.__n_tranche)
        n_valid = int(X.shape[0] * self.__split[1] // self.__n_tranche * self.__n_tranche + n_train)
        self.__train_set = SupervisedData(X[:n_train], y[:n_train])
        self.__valid_set = SupervisedData(X[n_train:n_valid], y[n_train:n_valid])
        self.__test_set  = SupervisedData(X[n_valid:], y[n_valid:])
        date = date[self.__hist_len:]
        n_train = int(n_train // self.__n_tranche)
        n_valid = int(n_valid // self.__n_tranche)
        self.__train_date = date[:n_train]
        self.__valid_date = date[n_train:n_valid]
        self.__test_date  = date[n_valid:]

    def __check_split(self, split:List[float]):
        if len(split) != 3:
            raise ValueError('the length of split must be 3.')
        if sum(split) != 1:
            raise ValueError('the sum of split must be 1.')

    def __rolling_generate(self, array:np.array, n=int):
        X, y = list(), list()
        for i in range(n, array.shape[0]):
            for j in range(array.shape[1]):
                X.append(array[i-n:i, j])
                y.append(array[i, j])
        return np.array(X), np.array(y)