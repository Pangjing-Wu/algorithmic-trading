import sys
sys.path.append('./')

from tqdm import tqdm
import pandas as pd

from src.utils.statastic import group_trade_volume_by_time
from src.utils.errors import *


INF = 0x7EEEEEEE
argmax = lambda a: [i for i, val in enumerate(a) if (val == max(a))][0]


class VWAPEnv(object):

    def __init__(self, tickdata, goal:int, time:list, interval:int,
                 hist_trade:pd.DataFrame, transaction_engine, level_space):

        if len(time) < 2 and len(time) % 2 != 0:
            raise KeyError("argument time should have 2 or multiples of 2 elements.")

        self._data = tickdata
        self._goal = goal
        self._interval = interval
        self._engine = transaction_engine

        self._volume_profile = group_trade_volume_by_time(hist_trade, time, interval)
        self._subtasks = self._destribute_subtask()

        self._level_space = level_space
        self._level_space_n = len(self._level_space)

        self._init = False
        self._final = False

        self._time_list = []
        for i in range(0, len(time), 2):
            self._time_list += [t for t in self._data.quote_timeseries if t >= time[i] and t < time[i+1]]

    @property
    def current_time(self):
        return self._t
    
    @property
    def level_space(self):
        return self._level_space

    @property
    def level_space_n(self):
        return self._level_space_n

    @property
    def subtasks(self):
        return self._subtasks

    @property
    def volume_profile(self):
        return self._volume_profile

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
        self._t = next(self._iter)
        self._subtasks['filled'] = 0
        self._filled = {'time': [], 'price':[], 'size':[]}
        self._order = {'time': 0, 'side': 0, 'price': 0, 'size': 0, 'pos': -1}
        self._tqdm = None
        self._tqdm_id = []
        task  = self.subtasks.loc[0]
        state = [task['start']] + task.values.tolist()
        return state

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

        time_range = (self._subtasks['start'] <= self._t) & (self._subtasks['end'] > self._t)
        index = self._subtasks[time_range].index[0]
        task  = self._subtasks.loc[index]

        if action[2] > 0:
            self._order = self._action2order(action)
        else:
            self._order['time'] = self._t

        self._order, filled = self._engine(self._order)

        if sum(filled['size']) % 100 == 0:
            task['filled'] += sum(filled['size'])
            self._filled['price'] += filled['price']
            self._filled['size']  += filled['size']
            self._update_tqdm(task, sum(filled['size']))

        self._t = next(self._iter)

        if self._t == self._time_list[-1]:
            self._final = True
        if self._subtasks['filled'].sum() == self._goal:
            self._final = True

        state = [self._t] + task.values.tolist()
        
        return (state, self._final)

    def _action2order(self, action):
        '''
        Argument:
        ---------
        action: list, tuple, or array like,
            forms as [side, level, size], where side: int, 0=buy, 1=sell.
        Return:
        -------
        order: dict,
            keys are ('side', 'price', 'size', 'pos').
        '''
        time  = self._t
        side  = 'buy' if action[0] == 0 else 'sell'
        level = self._level_space[action[1]]
        price = self._data.get_quote(time)[level].iloc[0]
        order = dict(time=time, side=side, price=price, size=action[2], pos=-1)
        return order

    def _destribute_subtask(self):
        ratio = [v / sum(self._volume_profile['volume']) for v in self._volume_profile['volume']]
        subgoals = [int(self._goal * r // 100 * 100) for r in ratio]
        subgoals[argmax(subgoals)] += self._goal - sum(subgoals)
        subtasks = dict(
            start=self._volume_profile['start'],
            end=self._volume_profile['end'],
            goal=subgoals,
            filled=0
            )
        return pd.DataFrame(subtasks)
            
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