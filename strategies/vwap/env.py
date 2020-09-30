import abc
import sys

import numpy as np
import pandas as pd

from utils.errors import *


argmax = lambda a: [i for i, val in enumerate(a) if (val == max(a))][0]
dotmut = lambda x, y: sum([a * b for a, b in zip(x, y)])


class BasicTranche(abc.ABC):

    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level:int, side:str):
        self._data = tickdata
        self._task = task
        self._engine = transaction_engine
        self._level_space = self._int2level(level)
        self._side = side
        self._init = False
        self._final = False
        self._time = [t for t in self._data.quote_timeseries if t >= task['start'] and t < task['end']]

    def __len__(self):
        return len(self._time)
    
    @abc.abstractproperty
    def observation_space_n(self):
        pass

    @property
    def action_space(self):
        return list(range(len(self._level_space) + 1))

    @abc.abstractmethod
    def _state(self):
        pass

    @abc.abstractmethod
    def _reward(self):
        pass
    
    @abc.abstractmethod
    def _postprocess(self):
        pass

    @property
    def action_space_n(self):
        return len(self._level_space) + 1

    @property
    def current_time(self):
        return self._t

    @property
    def volume_profile(self):
        return self._volume_profile

    @property
    def vwap(self):
        price = self._filled['price']
        size  = self._filled['size']
        vwap  = dotmut(price, size)
        vwap  = vwap / sum(size) if sum(size) != 0 else 0
        return vwap

    @property
    def market_vwap(self):
        trade = self._data.get_trade_between(self._time[0], self._t)
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
        self._iter = iter(self._time)
        self._t = next(self._iter)
        self._filled = {'time': [], 'price':[], 'size':[]}
        self._order = {'time': 0, 'side': self._side, 'price': 0, 'size': 0, 'pos': -1}
        return self._state()

    def step(self, action):
        if self._init == False:
            raise NotInitiateError
        if self._final == True:
            raise EnvTerminatedError
        if action < len(self._level_space):
            self._order = self._action2order(action)
        else:
            self._order['time'] = self._t
        self._order, filled = self._engine(self._order)
        if sum(filled['size']) % 100 == 0:
            self._filled['price'] += filled['price']
            self._filled['size']  += filled['size']
        self._t = next(self._iter)
        if self._t == self._time[-1] or sum(self._filled['size']) == self._task['goal']:
            self._final = True
        self._postprocess()
        return (self._state(), self._reward(), self._final)

    def _action2order(self, action:int):
        time  = self._t
        side  = self._side
        level = self._level_space[action]
        price = self._data.get_quote(time)[level].iloc[0]
        order = dict(time=time, side=side, price=price, size=100, pos=-1)
        return order

    def _int2level(self, level:int):
        if level not in range(1,11):
            raise KeyError('level must be in range(1,10)')
        level_space = list()
        for i in range(level):
            level_space.append('bid%d' % (i+1))
            level_space.append('ask%d' % (i+1))
        return level_space


class BasicHardConstrainTranche(BasicTranche, abc.ABC):

    def _reward(self):
        if self._final:
            reward = self.market_vwap - self.vwap
            reward = reward if self._side == 'buy' else -reward
        else:
            reward = 0.0
        return reward
    
    def _postprocess(self):
        if self._final == True and sum(self._filled['size']) < self._task['goal']:
            market_order_level = 'bid1' if self._side == 'buy' else 'ask1'
            self._filled['price'].append(self._data.get_quote(self._t)[market_order_level].iloc[0])
            self._filled['size'].append(self._task['goal'] - sum(self._filled['size']))

# 2.0.0 version env
class HardConstrainTranche(BasicHardConstrainTranche):
    
    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level:int, side:str):
        super().__init__(tickdata=tickdata, task=task, transaction_engine=transaction_engine, level=level, side=side)

    @property
    def observation_space_n(self):
        return 7

    def _state(self):
        prices = self._data.get_quote(self._t)[self._level_space].iloc[0].values.tolist()
        state  = [self._t / 1000, self._task['start'] / 1000, self._task['end'] / 1000,
                  self._task['goal'], sum(self._filled['size']), *prices]
        return np.array(state)


# 2.5.0 version env
class HistoricalHardConstrainTranche(BasicHardConstrainTranche):
    
    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level:int, side:str, historical_quote_num:int):
        super().__init__(tickdata=tickdata, task=task, transaction_engine=transaction_engine, level=level, side=side)
        self._historical_quote_num = historical_quote_num

    @property
    def observation_space_n(self):
        n = 5 + 40 * self._historical_quote_num
        return n

    def _state(self):
        quote  = self._data.get_quote(self._t)
        quotes = self._data.quote_board(quote).values.flatten()
        for _ in range(self._historical_quote_num - 1):
            quote = self._data.pre_quote(quote)
            if quote is not None:
                quotes = np.r_[self._data.quote_board(quote).values.flatten(), quotes]
            else:
                break
        padnum = 40 * self._historical_quote_num - len(quotes)
        quotes = np.pad(quotes, (padnum, 0))
        state  = [
            self._t / 1000,
            self._task['start'] / 1000, self._task['end'] / 1000,
            self._task['goal'], sum(self._filled['size']),
            *quotes
            ]
        return np.array(state, dtype=np.float32)


class RecurrentHardConstrainTranche(BasicHardConstrainTranche):
    
    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level:int, side:str, historical_quote_num:int):
        super().__init__(tickdata=tickdata, task=task, transaction_engine=transaction_engine, level=level, side=side)
        self._historical_quote_num = historical_quote_num

    @property
    def observation_space_n(self):
        n = (5, 40)
        return n

    def _state(self):
        quote  = self._data.get_quote(self._t)
        quotes = [self._data.quote_board(quote).values.flatten().tolist()]
        for _ in range(self._historical_quote_num - 1):
            quote = self._data.pre_quote(quote)
            if quote is not None:
                quotes.append(self._data.quote_board(quote).values.flatten().tolist())
            else:
                break
        padnum = self._historical_quote_num - len(quotes)
        quotes = np.pad(quotes, ((padnum,0), (0,0)))
        state  = [self._t / 1000, self._task['start'] / 1000, self._task['end'] / 1000, self._task['goal'], sum(self._filled['size'])]
        state  = (state, quotes)
        return np.array(state, dtype=np.float32)


class SemiHardConstrainTranche(BasicTranche):
    
    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level:int, side:str):
        super().__init__(tickdata=tickdata, task=task, transaction_engine=transaction_engine, level=level, side=side)

    def _state(self):
        prices = self._data.quote_board(self._t).loc[self._level_space, 'price'].values.tolist()
        state  = [
            self._t / 1000,
            self._task['start'] / 1000, self._task['end'] / 1000,
            self._task['goal'], sum(self._filled['size']),
            *prices
            ]
        return np.array(state, dtype=np.float32)

    def _reward(self):
        if self._final:
            if sum(self._filled['size']) < self._task['goal']:
                reward = -999.0
            else:
                reward = self.vwap - self.market_vwap
        else:
            reward = 0.0
        return reward
    
    def _postprocess(self):
        pass


class GenerateTranches(object):

    def __init__(self, env, goal, volume_profile, **env_kwargs):
        ratio = [v / sum(volume_profile['volume']) for v in volume_profile['volume']]
        subgoals = [int(goal * r // 100 * 100) for r in ratio]
        subgoals[argmax(subgoals)] += goal - sum(subgoals)
        self._subtasks = pd.DataFrame(dict(start=volume_profile['start'], end=volume_profile['end'], goal=subgoals))
        self._envs = [env(task=self._subtasks.loc[i], **env_kwargs) for i in self._subtasks.index]
    
    def __iter__(self):
        for env in self._envs:
            yield env

    def __getitem__(self, item):
        return self._envs[item]

    def __len__(self):
        return len(self._envs)

    @property
    def observation_space_n(self):
        return self._envs[0].observation_space_n

    @property
    def action_space_n(self):
        return self._envs[0].action_space_n