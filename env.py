import numpy as np

from tickdata import TickData
from transaction import transaction_matching

# TODO improve logger function.

class AlgorithmTrader(object):

    def __init__(
            self,
            td: TickData,
            total_volume: int,
            reward_function: callable or str,
            wait_t=0,
            max_level=5
    ):
        self._id = td
        self._res_volume = total_volume
        self._wait_t = wait_t # TODO add wait_t mechanism
        self._level_space = list(range(max_level * 2))
        self._level_space_n = len(self._level_space)
        self._reward_function = reward_function
        self._iime = self._id.quote_timeseries
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
            return self._iime[self._i]
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
        return self.res_volume

    def reset(self)->np.array:
        self._init = True
        self._final = False
        self._i = 0
        self._simulated_all_trade = {'price': [], 'size': []}
        env_s = self._id.quote(self._iime[0]).drop('time', axis=1)
        env_s = env_s.values.reshape(-1)
        agt_s = [self._res_volume, 0, 0]
        s0 = np.append(env_s, agt_s, axis=0)
        return s0

    def step(self, action):
        '''
        argument:
        ---------
        action: list, tuple, or array like,
                forms as [direction, level, size], where
                dirction: int, 0=buy, 1=sell.
        returns:
        --------
        next_s: np.array, next state of environment and agent.
        reward: float, environment rewards.
        signal: bool, final signal.
        info: str, transaction information.
        '''
        # raise exception if not initiate or reach final.
        if self._init == False:
            raise NotInitiateError
        if self._final == True:
            raise EnvTerminatedError
        # get current timestamp.
        t = self._id.quote_timeseries[self._i]
        info = 'At %s ms, ' % t
        # load quote and trade.
        quote = self._id.quote_board(t)
        trade = self._id.get_trade_between(t)
        # issue an order if the size of action great than 0.
        if action[-1] > 0:
            order = self._action2order(action) 
            info += 'issue an order %s; ' % order
        else:
            info += 'execute remaining order; '
        # transaction matching
        order, traded = transaction_matching(quote, trade, order)
        self._res_volume -= sum(traded['size'])
        info += 'after matching, %s hand(s) were traded at %s and' \
                '%s hand(s) waited to trade at %s; total.' % (
                    sum(traded['size']), sum(traded['price']),
                    order['size'], order['price']
                )
        # give a final signal
        if t == self._id.quote_timeseries[-2]:
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
                    reward = self._iwap(self._simulated_all_trade)
                else:
                    reward = self._reward_function(self._simulated_all_trade)
            # if order not completed.
            else:
                reward = -999.
        else:
            reward = 0.
        # go to next step.
        env_s = self._id.next_quote(t).drop('time', axis=1).values.reshape(-1)
        agt_s = [self._res_volume] + traded['price'] + traded['size']
        next_s = np.append(env_s, agt_s, axis=0)
        return (next_s, reward, self._final, info)

    # transform function
    # ------------------
    def _action2level(self, action_level:int)->str:
        if action_level not in self.level_space:
            raise ValueError('action level cross-border.')
        max_level = int(self._level_space_n / 2)
        if action_level < max_level:
            level = 'bid' + str(max_level - action_level)
            return level
        else:
            level = 'ask' + str(action_level - max_level + 1)
            return level

    def _action2order(self, action)->dict:
        '''
        argument:
        ---------
        action: list, tuple, or array like,
                forms as [direction, level, size], where
                dirction: int, 0=buy, 1=sell.
        return:
        -------
        order: dict, keys are formed by
               ('direction', 'price', 'size', 'pos').
        '''
        if action[0] == 0:
            direction = 'buy'
        elif action[0] == 1:
            direction = 'sell'
        else:
            raise KeyError('the transaction direction in action must be 0 or 1')
        level = self._action2level(action[1])
        price = self._id.get_quote(self._iime[self._i])[level]
        order = {'direction':direction, 'price':price, 'size': action[2], 'pos': -1}
        return order

    # reward function
    # ---------------
    def _vwap(self, trade:dict):
        vwap  = sum(map(lambda x, y: x * y, trade['price'], trade['size']))
        vwap /= sum(trade['size'])
        return vwap

    def _twap(self, trade):
        pass


class EnvError(Exception):

    def __init__(self, *args):
        self.args = args

class NotInitiateError(EnvError):

    def __init__(self):
        super().__init__(self)
        self.errorinfo = 'please run AlgorithmTrader.reset()' \
                         'to initiate first.'
    
    def __str__(self):
        return self.errorinfo

class EnvTerminatedError(EnvError):

    def __init__(self):
        super().__init__(self)
        self.errorinfo = 'environment is terminated, please run '\
                         'AlgorithmTrader.reset() to reset.'
    
    def __str__(self):
        return self.errorinfo