import sys
sys.path.append('./')

import numpy as np

from src.utils.errors import *

# TODO improve logger function.

class AlgorithmicTradingEnv(object):
    '''
    arguments:
    ----------
    data: TickData, tick data consist of quote and trade.
    transaction_engine: callable, transaction engine.
    total_volume: the volume of total orders.
    reward_function: callable, tansaction cost reward function.
    max_level: int, the number of level in quote data. 
    '''

    def __init__(
            self,
            tickdata,
            transaction_engine: callable,
            total_volume: int,
            reward_function: callable or str,
            max_level=5
    ):
        self._data = tickdata
        self._transaction_engine = transaction_engine
        self._total_volume = total_volume
        self._level_space = list(range(max_level * 2))
        self._level_space_n = len(self._level_space)
        self._reward_function = reward_function
        self._time = self._data.quote_timeseries
        self._init = False
        self._final = False
        
    @property
    def level_space(self):
        return self._level_space

    @property
    def level_space_n(self):
        return self._level_space_n

    @property
    def current_time(self):
        try:
            return self._time[self._i]
        except AttributeError:
            raise NotInitiateError

    @property
    def trade_results(self):
        try:
            return self._simulated_all_trade
        except AttributeError:
            raise NotInitiateError

    @property
    def res_volume(self):
        return self._res_volume

    def reset(self)->np.array:
        self._init = True
        self._final = False
        self._i = 0
        self._res_volume = self._total_volume
        self._order = {'time': 0, 'side': 0, 'price': 0, 'size': 0, 'pos': -1}
        self._simulated_all_trade = {'price': [], 'size': []}
        env_s = self._data.get_quote(self._time[0]).drop('time', axis=1)
        env_s = env_s.values.reshape(-1)
        agt_s = [self._res_volume, 0, 0]
        s_0   = np.append(env_s, agt_s, axis=0)
        return s_0

    def step(self, action):
        '''
        argument:
        ---------
        action: list, tuple, or array like,
                forms as [side, level, size], where side in {0=buy, 1=sell}.
        returns:
        --------
        next_s: np.array, next state of environment and agent.
        reward: float, environment rewards.
        final: bool, final signal.
        info: str, detailed transaction information of simulated orders.
        '''
    
        # raise exception if not initiate or reach final.
        if self._init == False:
            raise NotInitiateError
        if self._final == True:
            raise EnvTerminatedError

        if action[1] not in self._level_space:
            raise KeyError("action[1] should be in %s." % self._level_space)

        # get current timestamp.
        t = self._time[self._i]
        info = 'At %s ms, ' % t
        # issue an new order if the size of action great than 0.
        if action[2] > 0:
            self._order = self._create_new_order(action) 
            info += 'issue an order %s; ' % self._order
        # else execute remaining order.
        else:
            # update order's timestamp if there is a order remaining.
            if self._order['size'] > 0:
                self._order['time'] = t
                info += 'execute remaining order; '
            else:
                info += 'no order issued.'
        # transaction matching by transaction engine.
        if self._order['size'] > 0:
            self._order, traded = self._transaction_engine(self._order)
            self._simulated_all_trade['price'] += traded['price']
            self._simulated_all_trade['size']  += traded['size']
            self._res_volume -= sum(traded['size'])
            info += 'after matching, %s hand(s) were traded at %s and ' \
                    '%s hand(s) waited to trade at %s; total.' % (
                        sum(traded['size']), sum(traded['price']),
                        self._order['size'], self._order['price']
                    )
        # give a final signal
        if t == self._time[-2]:
            self._final = True
        elif self._res_volume == 0:
            self._final = True
        elif self._res_volume < 0:
            # NOTE verify if there is self._res_volume < 0
            print('[WARN] residual volume less than 0.')
            self._final = True
        else:
            self._final = False
        # calculate trasaction cost as reward.
        if self._final == True:
            # if order completed.
            if self._res_volume == 0:
                if self._reward_function == 'vwap':
                    reward = self._vwap(self._simulated_all_trade)
                elif self._reward_function == 'twap':
                    reward = self._twap(self._simulated_all_trade)
                else:
                    reward = self._reward_function(self._simulated_all_trade)
            # if order not completed.
            else:
                reward = -999.
        else:
            reward = 0.
        # go to next step.
        self._i += 1
        env_s = self._data.next_quote(t).drop('time', axis=1).values.reshape(-1)
        agt_s = [self._res_volume]
        next_s = np.append(env_s, agt_s, axis=0)
        return (next_s, reward, self._final, info)

    # transition function
    # -------------------
    def _action2level(self, action_level:int) -> str:
        max_level = int(self._level_space_n / 2)
        if action_level < max_level:
            level = 'bid' + str(max_level - action_level)
        else:
            level = 'ask' + str(action_level - max_level + 1)
        return level

    def _create_new_order(self, action) -> dict:
        '''
        argument:
        ---------
        action: list, tuple, or array like,
                forms as [side, level, size], where
                side: int, 0=buy, 1=sell.
        return:
        -------
        order: dict, keys are formed by
               ('side', 'price', 'size', 'pos').
        '''

        side = 'buy' if action[0] == 0 else 'sell'
        time  = self._time[self._i]
        level = self._action2level(action[1])
        price = self._data.get_quote(time)[level].iloc[0]
        order = {'time': time, 'side': side, 'price': price,
                 'size': action[2], 'pos': -1}
        return order

    # reward function
    # ---------------
    def _vwap(self, trade:dict):
        vwap  = sum(map(lambda x, y: x * y, trade['price'], trade['size']))
        vwap /= sum(trade['size'])
        return vwap

    def _twap(self, trade):
        pass

    # arugument check
    # ---------------
    def _action_check(self, action):
        if type(action) not in [tuple, list, np.array]:
            raise KeyError("action type should be tuple, list, "\
                           " or numpy.array.")
        if len(action) != 3:
            raise KeyError("action should contain 3 elements, "\
                           "but got %d." % len(action))
        if (action[0]) not in [0, 1]:
            raise KeyError("transaction side of action must be 0 or 1.")
        if type(action[1]) not in [int, np.int32, np.int64]:
            raise TypeError("transaction level type of action must be int.")
        if action[1] < 0 or action[1] >= self.level_space_n:
            raise ValueError('action level cross-border.')
        if type(action[2]) not in [int, np.int32, np.int64]:
            raise TypeError("transaction size type of action must be int.")

class EnvError(Exception):

    def __init__(self, *args):
        self.args = args

class NotInitiateError(EnvError):

    def __init__(self):
        super().__init__(self)
        self.errorinfo = 'please run AlgorithmicTradingEnv.reset()' \
                         'to initiate first.'
    
    def __str__(self):
        return self.errorinfo

class EnvTerminatedError(EnvError):

    def __init__(self):
        super().__init__(self)
        self.errorinfo = 'environment is terminated, please run ' \
                         'AlgorithmicTradingEnv.reset() to reset.'
    
    def __str__(self):
        return self.errorinfo
