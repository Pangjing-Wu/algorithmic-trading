import sys
from typing import List, Tuple

import torch
import numpy as np

from ..macro.profile import get_tranche_time


class TrancheDataset(object):
    ''' get tranche dataset

    Arguments:
    ----------
        dataset: tickdata dataset, e.g.: {CSVDataset/H2Dataset}.
        i_tranche: int, tranche id, start from 1.
    '''

    def __init__(self, dataset, split:List[float], i_tranche:int,
                 time_range:List[int], interval=int, drop_length=0):
        self.__check_split(split)
        self.__times   = get_tranche_time(time_range, interval)
        self.__n       = len(self.__times)
        self.__split   = split
        self.__dates   = dataset.dates[drop_length:]
        self.__dataset = dataset[drop_length:]
        self.__build_dataset()
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
    def train_date(self)->list:
        return self.__train_date

    @property
    def test_set(self)->list:
        return self.__test_set

    @property
    def test_date(self)->list:
        return self.__test_date

    def set_tranche(self, i:int):
        '''
        Arguments:
        ----------
        i_tranche: int, tranche id, start from 1.
        '''
        self.__check_tranche(i)
        self.__time = self.__times[i-1]

    def __build_dataset(self):
        n = len(self.__dataset)
        n_train = int(n * self.__split[0])
        self.__train_set  = [data.between(*self.__time) for data in self.__dataset[:n_train]]
        self.__test_set   = [data.between(*self.__time) for data in self.__dataset[n_train:]]
        self.__train_date = self.__dates[:n_train]
        self.__test_date  = self.__dates[n_train:]

    def __check_split(self, split:List[float]):
        if len(split) != 2:
            raise ValueError('the length of split must be 3.')
        if sum(split) != 1:
            raise ValueError('the sum of split must be 1.')

    def __check_tranche(self, i):
        if i < 1 or i > len(self.__times):
            raise ValueError('tranche id not in range.')