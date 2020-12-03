import abc
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
from exchange.order import ClientOrder


def dotmut(x, y): return sum([a * b for a, b in zip(x, y)])


Subgoal = namedtuple('Subgoal', ['step', 'ratio'])

SUBGOALS = [
    Subgoal(500, 0.10), Subgoal(500, 0.12), Subgoal(500, 0.14),
    Subgoal(600, 0.10), Subgoal(600, 0.12), Subgoal(600, 0.14),
    Subgoal(700, 0.10), Subgoal(700, 0.12), Subgoal(700, 0.14),
]


class BasicTranche(abc.ABC):

    def __init__(self, tickdata, goal:int, subgoals:list,
                 exchange:callable, level: int, side:str,
                 reward:str, unit_size:int):
        '''
        Arguments:
        ---------
            goal: int, execution goal.
            subgoals: set, set of subgoal namedtuple.
        '''
        self._data = tickdata
        self._goal = goal
        self._side = side
        self._init = False
        self._final = False
        self._subgoals = subgoals
        self._unit_size = unit_size
        self._time = self._data.quote.timeseries
        self._reward_type = reward
        self._exchange = exchange.reset()
        self._level_space = self.__int2level(level)

    def __len__(self):
        return len(self._time)

    @abc.abstractproperty
    def observation_space_n(self):
        pass

    @abc.abstractmethod
    def extrinsic_state(self):
        pass

    @abc.abstractmethod
    def _intrinsic_state(self):
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
        return self._total_filled

    @property
    def goal_space(self):
        return self._subgoals

    @property
    def goal_space_n(self):
        return len(self._subgoals)

    @property
    def extrinsic_reward(self):
        return self._extrinsic_reward()

    @property
    def vwap(self):
        price = self._total_filled['price']
        size = self._total_filled['size']
        vwap = dotmut(price, size)
        vwap = vwap / sum(size) if sum(size) != 0 else 0
        return vwap

    @property
    def market_vwap(self):
        trade = self._data.trade.between(self._time[0], self._t)
        price = trade['price']
        size = trade['size']
        vwap = dotmut(price, size)
        vwap = vwap / sum(size) if sum(size) != 0 else 0
        return vwap

    @property
    def final(self):
        return self._final

    @property
    def subfinal(self):
        return self._subfinal

    def metrics(self):
        ret = dict(
            vwap=round(self.vwap, 5),
            market_vwap=round(self.market_vwap, 5)
        )
        return ret

    def reset(self):
        self._init     = True
        self._final    = False
        self._filled   = None
        self._subgoal  = None
        self._subfinal = True
        self._iter     = iter(self._time)
        self._t        = next(self._iter)
        self._force_transaction = False
        self._total_filled = dict(time=[], price=[], size=[])
        self._exchange.reset()
        return self.extrinsic_state()

    def step(self, action) -> tuple:
        if self._init == False:
            raise RuntimeError(
                "environment has not initiated, run reset first.")
        if self._final == True:
            raise RuntimeError("environment is terminated, run reset first.")
        if self._subfinal == True:
            raise RuntimeError(
                "subgoal has terminated, agent should update new subgoal.")
        if action < len(self._level_space):
            self._exchange.issue(2)
            self._exchange.issue(1, self.__action2order(action))
        elif action == len(self._level_space):
            self._exchange.issue(0)
        else:
            raise ValueError('unknown action.')
        order = self._exchange.step(self._t)
        if order != None:
            self._filled = order.filled[order.filled['time'] == self._t]
            time = self._filled['time'].values.tolist()
            size = self._filled['size'].values.tolist()
            price = self._filled['price'].values.tolist()
            self._total_filled['time'] += time
            self._total_filled['size'] += size
            self._total_filled['price'] += price
            self._subfilled['time'] += time
            self._subfilled['size'] += size
            self._subfilled['price'] += price
        else:
            self._filled = None
        self._t = next(self._iter)
        self._substep += 1
        if self._t == self._time[-1] or sum(self._total_filled['size']) == self._goal:
            self._final = True
            self._subfinal = True
        if self._substep == self._subgoal['step'] or sum(self._subfilled['size']) == self._subgoal['size']:
            self._subfinal = True
        if self._final == True and sum(self._total_filled['size']) < self._goal:
            self._force_transaction = True
            market_order_level = 'bid1' if self._side == 'buy' else 'ask1'
            time = self._t
            size = self._goal - sum(self._total_filled['size'])
            price = self._data.quote.get(self._t)[market_order_level].iloc[0]
            self._total_filled['time'].append(time)
            self._total_filled['size'].append(size)
            self._total_filled['price'].append(price)
            self._subfilled['time'].append(time)
            self._subfilled['size'].append(size)
            self._subfilled['price'].append(price)
        state = None if self._final else self._intrinsic_state()
        reward = self._intrinsic_reward()
        return (state, reward)

    def update_subgoal(self, i):
        self.__reset_subgoal()
        self._subgoal = dict(
            step=self._subgoals[i].step,
            size=int(self._subgoals[i].ratio * self._goal)
        )
        return self._intrinsic_state()

    def _extrinsic_reward(self):
        if self._subfinal:
            if sum(self._subfilled['size']) < self._subgoal['size'] and self.__blame_extrinsic():
                reward = -999.
            elif self._force_transaction and self.__blame_extrinsic():
                reward = -999.
            else:
                trade = self._data.trade.between(self._substart, self._t)
                if not trade.empty and sum(self._subfilled['size']) > 0:
                    vwap = dotmut(self._subfilled['price'], self._subfilled['size'])
                    vwap = vwap / sum(self._subfilled['size'])
                    market_vwap = dotmut(trade['price'], trade['size']) / sum(trade['size'])
                    reward = 10000 * (market_vwap - vwap)
                    reward = reward if self._side == 'buy' else -reward
                else:
                    reward = 0.
        else:
            reward = None
        return reward

    def _intrinsic_reward(self):
        if self._reward_type == 'sparse':
            if self._subfinal:
                if sum(self._subfilled['size']) < self._subgoal['size'] and not self.__blame_extrinsic():
                    reward = -999.
                elif self._force_transaction and not self.__blame_extrinsic():
                    reward = -999.
                else:
                    trade = self._data.trade.between(self._substart, self._t)
                    if not trade.empty and sum(self._subfilled['size']) > 0:
                        vwap = dotmut(self._subfilled['price'], self._subfilled['size'])
                        vwap = vwap / sum(self._subfilled['size'])
                        market_vwap = dotmut(trade['price'], trade['size']) / sum(trade['size'])
                        reward = 10000 * (market_vwap - vwap)
                        reward = reward if self._side == 'buy' else -reward
                    else:
                        reward = 0.
            else:
                reward = 0.
        elif self._reward_type == 'dense':
            pre_t = self._time[max(self._time.index(self._t)-1, 0)]
            trade = self._data.trade.between(pre_t, self._t)
            if self._subfinal and sum(self._subfilled['size']) < self._subgoal['size'] and not self.__blame_extrinsic():
                reward = -999.
            elif self._subfinal and self._force_transaction and not self.__blame_extrinsic():
                reward = -999.
            elif not trade.empty and self._filled is not None and self._filled.shape[0] > 0:
                vwap = dotmut(self._filled['price'].values, self._filled['size'].values)
                vwap = vwap / sum(self._filled['size'])
                market_vwap = dotmut(trade['price'], trade['size']) / sum(trade['size'])
                reward = 100 * (market_vwap - vwap)
                reward = reward if self._side == 'buy' else -reward
            else:
                reward = 0.
        else:
            raise ValueError('unknown reward type.')
        return reward

    def __reset_subgoal(self):
        self._subfinal = False
        self._substep = 0
        self._substart = self._t
        self._subfilled = dict(time=[], price=[], size=[])
    
    def __blame_extrinsic(self):
        sub_volume = self._data.trade.between(self._substart, self._t)['size'].sum()
        return self._subgoal['size'] > min(self._substep * 100 * 0.3, sub_volume * 0.05)

    def __action2order(self, action: int):
        time = self._t
        side = self._side
        level = self._level_space[action]
        price = self._data.quote.get(time)[level].iloc[0]
        order = ClientOrder(time=time, side=side,
                            price=price, size=self._unit_size)
        return order

    def __int2level(self, level: int):
        if level not in range(1, 11):
            raise ValueError('level must be in range(1,10)')
        level_space = list()
        for i in range(level):
            level_space.append('bid%d' % (i+1))
            level_space.append('ask%d' % (i+1))
        return level_space


# 2.0.0/2.5.0 version env
class HistoricalTranche(BasicTranche):

    def __init__(self, tickdata, goal:int, subgoals:list,
                 exchange: callable, level: int, side: str,
                 quote_length=1, reward='sparse', unit_size=100):
        super().__init__(tickdata=tickdata, goal=goal, subgoals=subgoals,
                         exchange=exchange, level=level, side=side,
                         reward=reward, unit_size=unit_size)
        self._quote_length = quote_length

    @property
    def observation_space_n(self)->int:
        ''' intrinsic + extrinsic
        '''
        n = 2 + 2 * len(self._data.quote.level) * self._quote_length
        return n

    def _intrinsic_state(self)->np.array:
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
        padnum = 2 * len(self._data.quote.level) * self._quote_length - len(history)
        history = np.pad(history, (padnum, 0))
        time_ratio = self._substep / self._subgoal['step']
        filled_ratio = sum(self._subfilled['size']) / self._subgoal['size']
        state = [time_ratio, filled_ratio, *history]
        return np.array(state, dtype=np.float32)

    def extrinsic_state(self)->np.array:
        time_ratio = (self._t - self._time[0]) / (self._time[-1] - self._time[0])
        filled_ratio = sum(self._total_filled['size']) / self._goal
        if self._t == self._time[0]:
            trade_volume = 8
        else:
            trade_volume = np.log10(self._data.trade.between(self._substart, self._t)['size'].sum() + 1e-10)
        state = [time_ratio, filled_ratio, trade_volume]
        return np.array(state, dtype=np.float32)


# version 3.0.0
class RecurrentTranche(BasicTranche):

    def __init__(self, tickdata, goal:int, subgoals:list,
                 exchange: callable, level: int, side: str,
                 quote_length=1, reward='sparse', unit_size=100):
        super().__init__(tickdata=tickdata, goal=goal, subgoals=subgoals,
                         exchange=exchange, level=level, side=side,
                         reward=reward, unit_size=unit_size)
        self._quote_length = quote_length

    @property
    def observation_space_n(self)->tuple:
        ''' (intrinsic state, extrinsic state)
        '''
        n = (2, 2 * len(self._data.quote.level))
        return n

    def _intrinsic_state(self)->tuple:
        t = self._t
        history = np.array([])
        for _ in range(self._quote_length):
            quote = self._data.quote.get(t)
            history = np.r_[history, np.log2(
                [max(0.01, s) for s in quote.price.values])]
            history = np.r_[history, np.log10(
                [max(0.01, s) for s in quote.size.values])]
            try:
                t = self._data.quote.pre_time_of(t)
            except IndexError:
                break
        padnum = 2 * len(self._data.quote.level) * self._quote_length - len(history)
        history = np.pad(history, (padnum, 0))
        history = history.reshape(self._quote_length, 2 * len(self._data.quote.level))
        time_ratio = self._substep / self._subgoal['step']
        filled_ratio = sum(self._subfilled['size']) / self._subgoal['size']
        return (np.array([time_ratio, filled_ratio], dtype=np.float32), np.array(history, dtype=np.float32))
    
    def extrinsic_state(self)->np.array:
        time_ratio = (self._t - self._time[0]) / (self._time[-1] - self._time[0])
        filled_ratio = sum(self._total_filled['size']) / self._goal
        if self._t == self._time[0]:
            trade_volume = 8
        else:
            trade_volume = np.log10(self._data.trade.between(self._substart, self._t)['size'].sum() + 1e-10)
        state = [time_ratio, filled_ratio, trade_volume]
        return np.array(state, dtype=np.float32)