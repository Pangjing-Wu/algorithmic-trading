import sys
from typing import List, Tuple

import torch
import numpy as np


rolling = lambda arr, window: np.array([arr[i:i+window] for i in range(arr.shape[0] - window +1)])


def dataloader(tranche, window):
    '''
    returns:
    --------
    X: shape = (C, N, H, W)
        C: channel [price, volume] = 2
        N: batch num
        H: window size
        W: feature size
    '''

    ret = dict()
    for name, dataset in zip(['train', 'test'], [tranche.train_set, tranche.test_set]):    
        price  = list()
        volume = list()
        change = list()
        for data in dataset:
            price  += rolling(data.quote.filter(regex=r'^ask|bid').values, window).tolist()[:-1]
            volume += rolling(data.quote.filter(regex=r'size').values, window).tolist()[:-1]
            change += np.where((data.quote['ask1'][window:].values + 
                               data.quote['bid1'][window:].values - 
                               data.quote['ask1'][window-1:-1].values - 
                               data.quote['bid1'][window-1:-1].values) > 0,
                               1, 0).tolist()
        X = np.array([price, volume])
        y = np.array(change)
        ret[name] = dict(X=X, y=y)
    return ret

    
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