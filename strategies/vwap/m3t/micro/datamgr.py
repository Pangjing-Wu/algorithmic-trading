import sys
from typing import List, Tuple

import torch
import numpy as np


class TrancheDataset(object):
    ''' get tranche dataset

    Arguments:
    ----------
        dataset: tickdata dataset, e.g.: {CSVDataset/H2Dataset}.
    '''

    def __init__(self, dataset, split:List[float], 
                 time:Tuple[int], history_length=20):
        self.__check_split(split)
        self.__time    = time
        self.__split   = split
        self.__dates   = dataset.dates[history_length:]
        self.__dataset = dataset[history_length:]
        self.__build_dataset()

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
        n = len(self.__dataset)
        n_train = int(n * self.__split[0])
        n_valid = int(n * self.__split[1] + n_train)
        self.__train_set  = [data.between(*self.__time) for data in self.__dataset[:n_train]]
        self.__valid_set  = [data.between(*self.__time) for data in self.__dataset[n_train:n_valid]]
        self.__test_set   = [data.between(*self.__time) for data in self.__dataset[n_valid:]]
        self.__train_date = self.__dates[:n_train]
        self.__valid_date = self.__dates[n_train:n_valid]
        self.__test_date  = self.__dates[n_valid:]

    def __check_split(self, split:List[float]):
        if len(split) != 3:
            raise ValueError('the length of split must be 3.')
        if sum(split) != 1:
            raise ValueError('the sum of split must be 1.')