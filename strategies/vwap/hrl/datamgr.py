import sys
from typing import List, Tuple

import torch
import numpy as np


class TrancheDataset(object):
    ''' get tranche dataset

    Arguments:
    ----------
        dataset: tickdata dataset, e.g.: {CSVDataset/H2Dataset}.
        i_tranche: int, tranche id, start from 1.
    '''

    def __init__(self, dataset, split:List[float], time_range:List[int], interval=int, drop_length=0):
        self.__check_split(split)
        self.__check_pair_in_list(time_range)
        self.__times   = self.__get_tranche_time(time_range, interval)
        self.__split   = split
        self.__dataset = dataset[drop_length:]
        self.__build_dataset()

    @property
    def train_set(self)->list:
        return self.__train_set

    @property
    def test_set(self)->list:
        return self.__test_set

    def __build_dataset(self):
        n = len(self.__dataset)
        n_train = int(n * self.__split[0])
        self.__train_set = list()
        self.__test_set  = list()
        for time in self.__times:
            self.__train_set += [data.between(*time) for data in self.__dataset[:n_train]]
            self.__test_set  += [data.between(*time) for data in self.__dataset[n_train:]]

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
            raise ValueError('the length of split must be 3.')
        if sum(split) != 1:
            raise ValueError('the sum of split must be 1.')

    def __check_tranche(self, i):
        if i < 1 or i > len(self.__times):
            raise ValueError('tranche id not in range.')

    def __check_pair_in_list(self, x):
        if len(x) < 2 and len(x) % 2 != 0:
            raise ValueError("argument should contain 2 or multiples of 2 elements.")



