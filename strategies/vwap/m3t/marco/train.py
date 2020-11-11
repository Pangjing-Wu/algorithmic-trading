import abc
import sys
from typing import List, Union

import pandas as pd

sys.path.append('./')
from utils.argcheck import *


class BasicMacro(abc.ABC):
    
    def __init__(self, time_range:List[int], interval:int):
        self._tranche_time = _seperate_tranche(time_range, interval)

    def __len__(self):
        return len(self._tranche_time) - 1
    
    @property
    def n(self):
        return self.__len__()

    def train(self):
        "看看能不能由多个模块串起来"
        self._module1()
        self._module2()
        self._save()

    def _seperate_tranche(self, time_range:List[int], interval:int):
        interval    = check_postive_int(interval)
        time_range  = check_list_pair(time_range)
        time_slices = list()
        for i in range(0, len(time_range), 2):
            time_slice = list(range(time_range[i], time_range[i+1], interval))
            if time_slice[-1] != time_range[i+1]:
                time_slice.append(time_range[i+1])
            time_slices.append(time_slice)
        return time_slices

    @abc.abstractmethod
    def _module1(self):
        pass

    @abc.abstractmethod
    def _module2(self):
        pass

    @abc.abstractmethod
    def _save(self):
        pass


class MacroBaseline(object):

    def __init__(self, time_range:List[int], interval=0):
        super().__init__(time_range, interval)

    def train(self, trades: List[pd.DataFrame], savedir:str):
        volumes = dict(start=list(), end=list(), volume=list())
        for i in range(0, len(self._time_range), 2):
            if self._interval > 0:
                time_slices = list(range(self._time_range[i], self._time_range[i+1], self._interval))
            elif self._interval == 0:
                time_slices = [trades[0]['time'].iloc[0]]
            else:
                raise KeyError('interval must not be negative.')
            if time_slices[-1] != self._time_range[i+1]:
                time_slices.append(self._time_range[i+1])
            for j in range(len(time_slices) - 1):
                t0 = time_slices[j]
                t1 = time_slices[j+1]
                volume = 0
                for trade in trades:
                    index = (trade['time'] >= t0) & (trade['time'] < t1)
                    volume += trade[index]['size'].sum()
                volumes['start'].append(t0)
                volumes['end'].append(t1)
                volumes['volume'].append(int(volume / len(trades)))
        return pd.DataFrame(volumes)


class Dnn(object):

    def __init__(self, model):
        pass

    def train(self, trades, savedir):
        pass