import abc
import sys
sys.path.append('./')

from tqdm import tqdm
import pandas as pd

from utils.errors import *


argmax = lambda a: [i for i, val in enumerate(a) if (val == max(a))][0]
dotmut = lambda x, y: sum([a * b for a, b in zip(x, y)])

class BasicTranche(object):

    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level_space:list, verbose=0):
        self._data = tickdata
        self._task = task
        self._engine = transaction_engine
        self._level_space = level_space
        self._init = False
        self._final = False
        self._time = [t for t in self._data.quote_timeseries if t >= task['start'] and t < task['end']]
        self._verbose = verbose

    def __len__(self):
        return len(self._time)
    
    @abc.abstractproperty
    @property
    def observation_space_n(self):
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
        self._order = {'time': 0, 'side': 'buy', 'price': 0, 'size': 0, 'pos': -1}
        if self._verbose > 0:
            self._tqdm = tqdm(desc='task %s' % self._task.name, total=self._task['goal'])

    @abc.abstractmethod
    def step(self):
        pass

    def _action2order(self, action):
        '''
        Argument:
        ---------
        action: list, tuple, or array likes [side, level, size], where side: int, 0=buy, 1=sell.
        
        Return:
        -------
        order: dict, keys are ('side', 'price', 'size', 'pos').
        '''
        time  = self._t
        side  = 'buy' if action[0] == 0 else 'sell'
        level = self._level_space[action[1]]
        price = self._data.get_quote(time)[level].iloc[0]
        order = dict(time=time, side=side, price=price, size=action[2], pos=-1)
        return order


class HardConstrainTranche(BasicTranche):
    
    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level_space:list, verbose=0):
        super().__init__(tickdata, task, transaction_engine, level_space, verbose)

    @property
    def observation_space_n(self):
        return 7
        
    def reset(self):
        super().reset()
        prices = self._data.quote_board(self._t).loc[self._level_space, 'price'].values.tolist()
        state  = [
            self._t / 1000,
            self._task['start'] / 1000, self._task['end'] / 1000,
            self._task['goal'], sum(self._filled['size']),
            *prices
            ]
        return state

    def step(self, action):
        '''
        Argument:
        ---------
        action: list, tuple, or array likes [side, level, size], where side in {0=buy, 1=sell}.
        
        Returns:
        --------
        state: list, state of next step.

        reward: int, reward of current action.

        final: bool, final signal.
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
                self._tqdm.set_postfix(**self.metrics)
                self._tqdm.update(sum(filled['size']))

        self._t = next(self._iter)

        if self._t == self._time[-1]:
            self._final = True
        if sum(self._filled['size']) == self._task['goal']:
            self._final = True

        # if filled rate is less than 100%, issue remain orders as market order.
        if self._final and sum(self._filled['size']) < self._task['goal']:
            market_order_level = 'ask1' if action[0] == 0 else 'bid1'
            self._filled['price'].append(self._data.get_quote(self._t)[market_order_level].iloc[0])
            self._filled['size'].append(self._task['goal'] - sum(self._filled['size']))
            if self._verbose > 0:
                self._tqdm.set_postfix(**self.metrics)
                self._tqdm.update(self._task['goal'] - sum(self._filled['size']))

        if self._final == True and self._verbose > 0:
            self._tqdm.close()

        reward = self.vwap - self.market_vwap
        prices = self._data.get_quote(self._t)[self._level_space].iloc[0].values.tolist()
        state  = [
            self._t / 1000,
            self._task['start'] / 1000, self._task['end'] / 1000,
            self._task['goal'], sum(self._filled['size']),
            *prices
            ]

        return (state, reward, self._final)


class SemiHardConstrainTranche(BasicTranche):
    
    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level_space:list, verbose=0):
        super().__init__(tickdata, task, transaction_engine, level_space, verbose)

    def reset(self):
        super().reset()
        state = [self._t / 1000, self._task['start'] / 1000, self._task['end'] / 1000, self._task['goal'], sum(self._filled['size'])]
        return state

    def step(self, action):
        '''
        Argument:
        ---------
        action: list, tuple, or array likes [side, level, size], where side in {0=buy, 1=sell}.
        
        Returns:
        --------
        state: list, state of next step.

        reward: int, reward of current action.

        final: bool, final signal.
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
        if sum(self._filled['size']) == self._task['goal']:
            self._final = True
            
        if self._final == True and self._verbose > 0:
            self._tqdm.close()

        if self._final and sum(self._filled['size']) < self._task['goal']:
            reward = -999
        else:
            reward = self.vwap - self.market_vwap

        state = [self._t / 1000, self._task['start'] / 1000, self._task['end'] / 1000, self._task['goal'], sum(self._filled['size'])]

        return (state, reward, self._final)


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