import abc
import sys

import numpy as np
import pandas as pd

from exchange.order import ClientOrder
from utils.errors import *


dotmut = lambda x, y: sum([a * b for a, b in zip(x, y)])


class BasicTranche(abc.ABC):

    def __init__(self, tickdata, task:pd.Series,
                 exchange:callable, level:int, side:str):
        '''
        Arguments:
        ---------
            task: pd.Series, keys are 'start', 'end, 'goal'.
        '''
        self._data = tickdata.between(task['start'], task['end'])
        self._task = task
        self._side = side
        self._init = False
        self._final = False
        self._time = self._data.quote.timeseries
        self._exchange = exchange.reset()
        self._level_space = self.__int2level(level)

    def __len__(self):
        return len(self._time)
    
    @abc.abstractproperty
    def observation_space_n(self):
        pass

    @abc.abstractmethod
    def _state(self):
        pass

    @property
    def action_space(self):
        return list(range(len(self._level_space) + 1))

    @property
    def action_space_n(self):
        return len(self._level_space) + 1

    @property
    def current_time(self):
        return self._t

    @property
    def filled(self):
        return self._filled

    @property
    def vwap(self):
        price = self._filled['price']
        size  = self._filled['size']
        vwap  = dotmut(price, size)
        vwap  = vwap / sum(size) if sum(size) != 0 else 0
        return vwap

    @property
    def market_vwap(self):
        trade = self._data.trade. between(self._time[0], self._t)
        price = trade['price']
        size  = trade['size']
        vwap  = dotmut(price, size)
        vwap  = vwap / sum(size) if sum(size) != 0 else 0
        return vwap

    @property
    def task(self):
        task = self._task.to_dict()
        task['filled'] = sum(self._filled['size'])
        return task

    def is_final(self):
        return self._final

    def metrics(self):
        ret = dict(
            vwap=round(self.vwap, 5),
            market_vwap=round(self.market_vwap, 5)
            )
        return ret

    def reset(self):
        self._init = True
        self._final = False
        self._exchange.reset()
        self._iter = iter(self._time)
        self._t = next(self._iter)
        self._filled = {'time': [], 'price':[], 'size':[]}
        return self._state()

    def step(self, action)->tuple:
        if self._init == False:
            raise NotInitiateError
        if self._final == True:
            raise EnvTerminatedError
        if action < len(self._level_space):
            self._exchange.issue(2)
            self._exchange.issue(1, self.__action2order(action))
        elif action == len(self._level_space):
            self._exchange.issue(0)
        else:
            raise ValueError('unknown action.')
        order = self._exchange.step(self._t)
        if order != None:
            filled = order.filled[order.filled['time'] == self._t]
            self._filled['time']  += filled['time'].values.tolist()
            self._filled['size']  += filled['size'].values.tolist()
            self._filled['price'] += filled['price'].values.tolist()
        self._t = next(self._iter)
        if self._t == self._time[-1] or sum(self._filled['size']) == self._task['goal']:
            self._final = True
        if self._final == True and sum(self._filled['size']) < self._task['goal']:
            market_order_level = 'bid1' if self._side == 'buy' else 'ask1'
            self._filled['time'].append((self._t))
            self._filled['size'].append(self._task['goal'] - sum(self._filled['size']))
            self._filled['price'].append(self._data.quote.get(self._t)[market_order_level].iloc[0])
        if self._final:
            reward = self.market_vwap - self.vwap
            reward = reward if self._side == 'buy' else -reward
        else:
            reward = 0.
        state = None if self._final else self._state()
        return (state, reward, self._final)

    def __action2order(self, action:int):
        time  = self._t
        side  = self._side
        level = self._level_space[action]
        price = self._data.quote.get(time)[level].iloc[0]
        order = ClientOrder(time=time, side=side, price=price, size=100)
        return order

    def __int2level(self, level:int):
        if level not in range(1,11):
            raise ValueError('level must be in range(1,10)')
        level_space = list()
        for i in range(level):
            level_space.append('bid%d' % (i+1))
            level_space.append('ask%d' % (i+1))
        return level_space


# 2.0.0/2.5.0 version env
class HistoricalTranche(BasicTranche):
    
    def __init__(self, tickdata, task:pd.Series,
                 exchange:callable, level:int, side:str,
                 quote_length=1):
        super().__init__(tickdata=tickdata, task=task,
                         exchange=exchange, level=level, side=side)
        self._quote_length = quote_length

    @property
    def observation_space_n(self)->int:
        ''' intrinsic + extrinsic
        '''
        n = 2 + 2 * len(self._data.quote.level) * self._quote_length
        return n

    def _state(self)->np.array:
        t = self._t
        history = np.array([])
        for _ in range(self._quote_length):
            quote = self._data.quote.get(t)
            history = np.r_[history, np.log2([max(0.01, s) for s in quote.price.values])]
            history = np.r_[history, np.log10([max(0.01, s) for s in quote.size.values])]
            try:
                t = self._data.quote.pre_time_of(t)
            except IndexError:
                break
        padnum  = 2 * len(self._data.quote.level) * self._quote_length - len(history)
        history = np.pad(history, (padnum, 0))
        time_ratio  = (self._t - self._task['start']) / (self._task['end'] - self._task['start'])
        filled_ratio = sum(self._filled['size']) / self._task['goal']
        state = [time_ratio, filled_ratio, *history]
        return np.array(state, dtype=np.float32)


# version 3.0.0
class RecurrentTranche(BasicTranche):
    
    def __init__(self, tickdata, task:pd.Series,
                 exchange:callable, level:int, side:str,
                 quote_length:int):
        super().__init__(tickdata=tickdata, task=task,
                         exchange=exchange, level=level, side=side)
        self._quote_length = quote_length

    @property
    def observation_space_n(self)->tuple:
        ''' (intrinsic state, extrinsic state)
        '''
        n = (2, 2 * len(self._data.quote.level))
        return n

    def _state(self)->tuple:
        t = self._t
        history = np.array([])
        for _ in range(self._quote_length):
            quote = self._data.quote.get(t)
            history = np.r_[history, np.log2([max(0.01, s) for s in quote.price.values])]
            history = np.r_[history, np.log10([max(0.01, s) for s in quote.size.values])]
            try:
                t = self._data.quote.pre_time_of(t)
            except IndexError:
                break
        padnum  = 2 * len(self._data.quote.level) * self._quote_length - len(history)
        history = np.pad(history, (padnum, 0))
        history = history.reshape(self._quote_length, 2 * len(self._data.quote.level))
        time_ratio  = (self._t - self._task['start']) / (self._task['end'] - self._task['start'])
        filled_ratio = sum(self._filled['size']) / self._task['goal']
        return (np.array([time_ratio, filled_ratio], dtype=np.float32), np.array(history, dtype=np.float32))