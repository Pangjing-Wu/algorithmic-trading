import sys
from typing import List, Tuple

import torch
import numpy as np


from ..m2t.macro.profile import volume_profile, get_tranche_time


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
        '''shape: [i_tranche].X[n_sample, n_features]
                  [i_tranche].y[n_sample, 1]
        '''
        return self.__train_set

    @property
    def train_date(self):
        return self.__train_date

    @property
    def test_set(self):
        '''shape: [i_tranche].X[n_sample, n_features]
                  [i_tranche].y[n_sample, 1]
        '''
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
        X = torch.from_numpy(X).reshape(-1, self.__n_tranche, self.__hist_len)
        y = torch.from_numpy(y).reshape(-1, self.__n_tranche)
        n_train = int(X.shape[0] * self.__split[0])
        self.__train_set = [SupervisedData(X[:n_train, i], y[:n_train, i]) for i in range(self.__n_tranche)]
        self.__test_set  = [SupervisedData(X[n_train:, i], y[n_train:, i]) for i in range(self.__n_tranche)]
        date = date[self.__hist_len:]
        self.__train_date = date[:n_train]
        self.__test_date  = date[n_train:]

    def __check_split(self, split:List[float]):
        if len(split) != 2:
            raise ValueError('the length of split must be 2.')
        if sum(split) != 1:
            raise ValueError('the sum of split must be 1.')

    def __rolling_generate(self, array:np.array, n=int):
        X, y = list(), list()
        for i in range(array.shape[1]):
            for j in range(n, array.shape[0]):
                X.append(array[j-n:j, i])
                y.append(array[j, i])
        return np.array(X), np.array(y)


class TrancheDataset(object):
    ''' get tranche dataset

    Arguments:
    ----------
        dataset: tickdata dataset, e.g.: {CSVDataset/H2Dataset}.
        i_tranche: int, tranche id, start from 1.
    '''

    def __init__(self, dataset, split:List[float], time_range:List[int],
                 interval:int, drop_length:int, i_tranche=None):
        self.__check_split(split)
        self.__times   = self.__get_tranche_time(time_range, interval)
        self.__n       = len(self.__times)
        self.__split   = split
        self.__dates   = dataset.dates[drop_length:]
        self.__dataset = dataset[drop_length:]
        self.set_tranche(i_tranche)

    @property
    def n(self):
        return self.__n

    @property
    def time(self):
        return self.__time

    @property
    def train_set(self)->list:
        return self.__train_set

    @property
    def test_set(self)->list:
        return self.__test_set

    def set_tranche(self, i:int):
        '''
        Arguments:
        ----------
        i_tranche: int, tranche id, start from 1.
        '''
        if i == None:
            self.__time = (self.__times[0][0], self.__times[-1][-1])
        else:
            self.__check_tranche(i)
            self.__time = self.__times[i-1]
            self.__build_dataset()
        return self

    def __build_dataset(self):
        n = len(self.__dataset)
        n_train = int(n * self.__split[0])
        self.__train_set = [data.between(*self.__time) for data in self.__dataset[:n_train]]
        self.__test_set  = [data.between(*self.__time) for data in self.__dataset[n_train:]]

    def __get_tranche_time(self, time_range:List[int], interval:int)->Tuple[int]:
        times = list()
        for i in range(0, len(time_range), 2):
            if interval > 0:
                series = list(range(time_range[i], time_range[i+1], interval))
            elif interval == 0:
                series = [time_range[0]]
            else:
                raise ValueError('interval must not be negative.')
            if series[-1] != time_range[i+1]:
                series += [time_range[i+1]]
            for i, j in zip(series[:-1], series[1:]):
                times.append((i,j))
        return tuple(times)

    def __check_split(self, split:List[float]):
        if len(split) != 2:
            raise ValueError('the length of split must be 2.')
        if sum(split) != 1:
            raise ValueError('the sum of split must be 1.')

    def __check_tranche(self, i):
        if i < 1 or i > len(self.__times):
            raise ValueError('tranche id not in range.')

    def __check_pair_in_list(self, x):
        if len(x) < 2 and len(x) % 2 != 0:
            raise ValueError("argument should contain 2 or multiples of 2 elements.")