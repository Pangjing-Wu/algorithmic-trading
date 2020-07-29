from math import floor
from tqdm import tqdm
import sys
sys.path.append('./')

import pandas as pd

from src.utils.statastic import group_trade_volume_by_time
from src.utils.errors import *


Inf = 0x7EEEEEEE
argmax = lambda a: [i for i, val in enumerate(a) if (val == max(a))][0]

class VWAPEnv(object):

    def __init__(self, tickdata, goal:int, time:list, interval:int,
                 hist_trade:pd.DataFrame, transaction_engine):

        if len(time) < 2 and len(time) % 2 != 0:
            raise KeyError("argument time should have 2 or multiples of 2 elements.")

        self._data = tickdata
        self._goal = goal
        self._interval = interval
        self._engine = transaction_engine
        self._hist_volume = group_trade_volume_by_time(hist_trade, time, interval)
        
        subgoals = [floor(goal * v / sum(self._hist_volume['volume'])) for v in self._hist_volume['volume']]
        subgoals[argmax(subgoals)] += self._goal - sum(subgoals)
        subtasks = dict(
            start=self._hist_volume['start'],
            end=self._hist_volume['end'],
            goal=subgoals,
            filled=0
            )
        self._subtasks = pd.DataFrame(subtasks)

        self._time_list = []
        for i in range(0, len(time), 2):
            self._time_list += [t for t in self._data.quote_timeseries if t >= time[i] and t < time[i+1]]

        self._level_space = ['bid1', 'ask1']
        self._level_space_n = len(self._level_space)

        self._init = False
        self._final = False

    @property
    def level_space(self):
        return self._level_space

    @property
    def level_space_n(self):
        return self._level_space_n

    @property
    def history_volume(self):
        return self._hist_volume

    @property
    def subtasks(self):
        return self._subtasks

    @property
    def vwap(self):
        price = self._filled['price']
        size  = self._filled['size']
        vwap  = sum(map(lambda x, y: x * y, price, size))
        vwap = vwap / sum(size) if sum(size) != 0 else 0
        return vwap

    @property
    def market_vwap(self):
        trade = self._data.get_trade_between(self._time_list[0], self._t)
        price = trade['price']
        size  = trade['size']
        vwap  = sum(map(lambda x, y: x * y, price, size))
        vwap = vwap / sum(size) if sum(size) != 0 else 0
        return vwap

    def is_final(self):
        return self._final

    def reset(self):
        self._init = True
        self._final = False
        self._iter = iter(self._time_list)
        self._subtasks['filled'] = 0
        self._filled = {'time': [], 'price':[], 'size':[]}
        self._order = {'time': 0, 'side': 0, 'price': 0, 'size': 0, 'pos': -1}
        self._tqdm = None
        self._tqdm_id = []
        return 1

    def step(self, action):
        '''
        Argument:
        ---------
        action: list, tuple, or array like,
            forms as [side, level, size], where side in {0=buy, 1=sell}.
        Returns:
        --------
        schedule: float, process schedule of current subtask.

        final: bool, final signal.

        info: str, detailed transaction information of simulated orders.
        '''
        if self._init == False:
            raise NotInitiateError
        if self._final == True:
            raise EnvTerminatedError

        self._t = next(self._iter)
        info = 'At time %s, ' % self._t

        time_range = (self._subtasks['start'] <= self._t) & (self._subtasks['end'] > self._t)
        index = self._subtasks[time_range].index[0]
        task  = self._subtasks.loc[index]

        if action[2] > 0:
            self._order = self._create_new_order(action)
        else:
            self._order['time'] = self._t
        info += 'issued order %s to exchange, ' % self._order

        self._order, filled = self._engine(self._order)
        task['filled'] += sum(filled['size'])
        info += 'filled %s.\n' % filled

        for key in self._filled:
            self._filled[key] += filled[key]
        
        schedule_ratio = self._schedule_ratio(task)

        if self._t == self._time_list[-1]:
            self._final = True

        if self._subtasks['filled'].sum() == self._subtasks['goal'].sum():
            self._final = True

        self._update_tqdm(task, sum(filled['size']))

        info += 'current subtask is:\n%s\n'% task

        return (schedule_ratio, self._final, info)

    def _create_new_order(self, action):
        '''
        Argument:
        ---------
        action: list, tuple, or array like,
            forms as [side, level, size], where side: int, 0=buy, 1=sell.
        Return:
        -------
        order: dict, keys are formed by
               ('side', 'price', 'size', 'pos').
        '''
        time  = self._t
        side  = 'buy' if action[0] == 0 else 'sell'
        level = self._level_space[action[1]]
        price = self._data.get_quote(time)[level].iloc[0]
        order = dict(time=time, side=side, price=price, size=action[2], pos=-1)
        return order

    def _get_subtask(self):
        condition = (self._subtasks['start'] <= self._t) & (self._subtasks['end'] > self._t)
        index = self._subtasks[condition].index[0]
        return self._subtasks.loc[index]
        
    def _schedule_ratio(self, task):
        if task['goal'] == 0:
            return Inf
        filled_ratio = task['filled'] / task['goal']
        time_ratio = (self._t - task['start']) / (task['end'] - task['start'])
        if filled_ratio == 1:
            return Inf
        if time_ratio == 0:
            return 1
        else:
            return filled_ratio / time_ratio
            
    def _update_tqdm(self, task, increment):
        if task.name not in self._tqdm_id:
            if len(self._tqdm_id) > 0:
                self._tqdm.close()
            self._tqdm_id.append(task.name)
            self._tqdm = tqdm(desc='task %s' % task.name, total=task['goal'])

        self._tqdm.set_postfix(vwap=self.vwap, market_vwap=self.market_vwap)
        self._tqdm.update(increment)

        if self._final == True:
            self._tqdm.close()
        

class BasicAgent(object):
    
    def __init__(self, side: str, threshold: int=0.1):
        self._side = 0 if side == 'buy' else 1
        self._threshold = threshold
        self._level = [0, 1] if side == 'buy' else [1, 0]   # level[0] is benifit, level[1] is cost.

    def action(self, state):
        if state < 1.05:
            return [self._side, self._level[1], 1]
        elif state > 1.05 + self._threshold:
            return [self._side, self._level[0], 0]
        else:
            return [self._side, self._level[0], 1]