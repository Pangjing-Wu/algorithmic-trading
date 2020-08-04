import sys
sys.path.append('./')

from tqdm import tqdm
import pandas as pd

from src.utils.errors import *


argmax = lambda a: [i for i, val in enumerate(a) if (val == max(a))][0]


class TrancheEnv(object):

    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level_space:list, verbose=0):
        self._data = tickdata
        self._task = task
        self._engine = transaction_engine
        self._level_space = level_space
        self._init = False
        self._final = False
        self._time = [t for t in self._data.quote_timeseries if t >= task['start'] and t < task['end']]
        self._verbose = verbose
    
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
    def observation_space_n(self):
        return 5

    @property
    def volume_profile(self):
        return self._volume_profile

    @property
    def vwap(self):
        price = self._filled['price']
        size  = self._filled['size']
        vwap  = sum(map(lambda x, y: x * y, price, size))
        vwap  = vwap / sum(size) if sum(size) != 0 else 0
        return vwap

    @property
    def market_vwap(self):
        trade = self._data.get_trade_between(self._time[0], self._t)
        price = trade['price']
        size  = trade['size']
        vwap  = sum(map(lambda x, y: x * y, price, size))
        vwap  = vwap / sum(size) if sum(size) != 0 else 0
        return vwap

    def is_final(self):
        return self._final

    def reset(self):
        self._init = True
        self._final = False
        self._iter = iter(self._time)
        self._t = next(self._iter)
        self._filled = {'time': [], 'price':[], 'size':[]}
        self._order = {'time': 0, 'side': 'buy', 'price': 0, 'size': 0, 'pos': -1}
        state = [self._t, self._task['start'], self._task['end'], self._task['goal'], sum(self._filled['size'])]
        if verbose > 0:
            self._tqdm = tqdm(desc='task %s' % self._task.name, total=self._task['goal'])
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

        if action[2] > 0:
            self._order = self._action2order(action)
        else:
            self._order['time'] = self._t

        self._order, filled = self._engine(self._order)

        if sum(filled['size']) % 100 == 0:
            self._filled['price'] += filled['price']
            self._filled['size']  += filled['size']
            if self._verbose > 0:
                self._tqdm.set_postfix(vwap=self.vwap, market_vwap=self.market_vwap)
                self._tqdm.update(sum(filled['size']))

        self._t = next(self._iter)

        if self._t == self._time[-1]:
            self._final = True
            if self._verbose > 0:
                self._tqdm.close()

        if self._final and sum(self._filled['size']) != self._task['goal']:
            reward = -999
        else:
            reward = self.vwap - self.market_vwap

        state = [self._t, self._task['start'], self._task['end'], self._task['goal'], sum(self._filled['size'])]
        
        return (state, reward, self._final)

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


class GenerateTranches(object):

    def __init__(self, goal, volume_profile, **env_kwargs):
        ratio = [v / sum(volume_profile['volume']) for v in volume_profile['volume']]
        subgoals = [int(goal * r // 100 * 100) for r in ratio]
        subgoals[argmax(subgoals)] += goal - sum(subgoals)
        self._subtasks = pd.DataFrame(dict(start=volume_profile['start'], end=volume_profile['end'], goal=subgoals))
        self._envs = [TrancheEnv(task=self._subtasks.loc[i], **env_kwargs) for i in self._subtasks.index]
    
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