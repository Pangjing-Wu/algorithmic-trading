import numpy as np

from tickdata import TickData, dataset


class AlgorithmTrader(object):

    def __init__(
            self,
            td: TickData,
            strategy_direction: ['buy', 'sell'],
            volume: int,
            reward_function: callable or str,
            wait_t,
            max_level=5  # TODO rewrite corresponding codes.
    ):
        self._n = 0
        self._td = td
        self._t = self._td.quote_timeseries[0]
        self._simulated_quote = {'level': None, 'price': None, 'size': None}
        self._simulated_all_trade = {'level': [], 'price': [], 'size': []}
        self._res_volume = volume
        self._wait_t = wait_t
        self._wait_signal=0
        self._action_space = max_level * 2
        self._reward_function = reward_function


    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        self._n = 0
        self._t = self._td.quote_timeseries[0]
        self._simulated_quote = {'level': None, 'price': None, 'size': None}
        self._simulated_all_trade = {'level': [], 'price': [], 'size': []}

    def step(self, action) -> '(next_s, reward, signal, info)':
        self._t = self._td.trade_timeseries[self._n]
        quote = self._td.get_quote(self._t)
        trade = self._td.get_trade(self._t)
        # Issue an order;
        level = self._action_map(action[0])
        if action[1] != 0:
            self._simulated_quote['level'] = level
            self._simulated_quote['price'] = quote[level][0]
            self._simulated_quote['size'] = action[1]
        simulated_trade = self._transaction_matching(quote, trade)
        self._res_volume -= sum(simulated_trade['size'])
        # Give a final signal
        if self._t == self._td.trade_timeseries[-2]:
            signal = True
        elif self._res_volume == 0:
            signal = True
        # TODO what to do if self._res_volume < 0
        elif self._res_volume < 0:
            signal = True
        else:
            signal = False
        # Calculate reward: trasaction cost, margin conditions
        # Order completed.
        if signal == True:
            if self._res_volume == 0:
                if self._reward_function == 'vwap':
                    reward = self._vwap(self._simulated_all_trade)
                elif self._reward_function == 'twap':
                    reward = self._twap(self._simulated_all_trade)
                else:
                    reward = self._reward_function(self._simulated_all_trade)
            # Order not completed.
            elif self._res_volume > 0:
                reward = -999
            # TODO what to do if self._res_volume < 0
            else:
                reward = -99
        else:
            reward = 0
        # conclude info
        info = 'At %s ms, %s hand(s) were traded at level %s and' \
               '%s hand(s) waited to trade at level %s.' % (
                   self._t, self._simulated_trade['size'], self._simulated_trade['level'],
                   self._simulated_quote['level'], self._simulated_quote['size']
               )
        # go to next step.
        env_s = self._td.next_quote(self._t).drop('time', axis=1).values.reshape(-1)
        agt_s = [self._res_volume] + simulated_trade['price'] + simulated_trade['size']
        next_s = np.append(env_s, agt_s, axis=0)
        self._n += 1
        return (next_s, reward, signal, info)

    def _action2level(self, action: int) ->'(kind,newlevel)':  #返回这个ACTION的level：从数字0变成kind'bid',newlevel'5'
        max_level = self.action_space / 2
        if action < max_level:
            newlevel = max_level - action
            kind='bid'
            return (kind,newlevel)
        else:
            newlevel = action - max_level + 1
            kind='ask'
            return (kind,newlevel)

    def _level2price(self, kind:str,newlevel) -> str:  #从 kind'bid',newlevel'5' 变成'bid5'
        price_tag = kind+newlevel
        return price_tag

    def _action2level2(self,kind:str,newlevel:int)->'level':  #从kind'bid',newlevel'5'变成数字0
        max_level = self.action_space / 2
        if kind=='bid':
            level=max_level-newlevel
        else:
            level=max_level+newlevel-1
        return(level)

    def _level2size(self, kind:str,newlevel) -> str:     #返回B_size_5
        size_tag = kind[0] + 'size' + newlevel
        return size_tag




    def _transaction_matching(self,quote,trade,action,simulated_quote,strategy_direction) -> 'simulated_trade':
        simulated_trade = {'level': [], 'price': [], 'size': []}
        if self._simulated_quote['size'] != 0:
            (kind, newlevel) = self._action2level(action)
            if strategy_direction=='buy':
               if kind=='ask':  # 如果levelkind是'ask',则现在可以直接成交，只考虑quote中的数量
                    i=1 #真实level的数值
                    ii=0 #现在的simulated_trade里有几个值
                    while i<=newlevel:
                        size_tag=self._level2size(kind,i)
                        price_tag = self._level2price(kind, i)
                        if self._td.quote[self._n][size_tag]<=simulated_quote['size']:                  #如果ask1的量小于等于提交的订单量
                            level2=self._action2level2(self,kind,i)
                            simulated_trade['level'].append(level2)
                            simulated_trade['price'].append(quote(self._t)[price_tag])
                            simulated_trade['size'].append(quote(self._t)[size_tag])
                            simulated_quote['size'] -= simulated_trade['size'][ii]
                            i=i+1
                            ii=ii+1
                        else:
                            level2 = self._action2level2(self, kind, i)
                            simulated_trade['level'].append(level2)                             #如果大于需求的量，则全部成交
                            simulated_trade['price'].append(quote(self._t)[price_tag])
                            simulated_trade['size'].append(simulated_quote['size'])
                            simulated_quote['size'] -= simulated_trade['size'][ii]
                            break
               else:  # level kind,是 bid,则需要排队并且考虑trade
                    wait_t=self._wait_t
                    if action['size'] != 0:
                        self._wait_signal = 0
                    else :
                        self._wait_signal= self._wait_signal +1
                    if self._wait_signal >= wait_t:
                        trade_sum=self.td.trade_sum(trade)
                        if trade_sum['price'][0]<=simulated_quote['price'] :   #说明存在比我提交的价格低的值成交了
                           i=0
                           ii=0
                           while trade_sum['price'][i]<=simulated_quote['price']:
                               if trade_sum['size'][i]<=simulated_quote['size']:
                                   simulated_trade['level'].append(trade_sum['level'][ii])
                                   simulated_trade['price'].append(trade_sum['price'][ii])
                                   simulated_trade['size'].append(trade_sum['size'][ii])
                                   simulated_quote['size'] -= simulated_trade['size'][ii]
                                   i = i + 1
                                   ii = ii + 1
                               else:
                                   simulated_trade['level'].append(trade_sum['level'][ii])
                                   simulated_trade['price'].append(trade_sum['price'][ii])
                                   simulated_trade['size'].append(simulated_quote['size'])
                                   simulated_quote['size'] -= simulated_trade['size'][ii]
                                   break
                        else:
                            simulated_trade['level'][0]=0
                            simulated_trade['price'][0] = 0
                            simulated_trade['size'][0] = 0
                    else:
                        simulated_trade['level'][0] = 0
                        simulated_trade['price'][0] = 0
                        simulated_trade['size'][0] = 0

            # 说明有订单是要卖
            else:
                if kind == 'bid':  # 如果levelkind是'ask',则现在可以直接成交，只考虑quote中的数量
                    i = 1
                    ii = 0
                    while i <= newlevel:
                        size_tag = self._level2size(kind, i)
                        price_tag = self._level2price(kind, i)
                        if self._td.quote[self._n][size_tag] < simulated_quote['size']:  # 如果bid1的量小于提交的订单量
                            level2 = self._action2level2(self, kind, i)
                            simulated_trade['level'].append(level2)
                            simulated_trade['price'].append(quote(self._t)[price_tag])
                            simulated_trade['size'].append(quote(self._t)[size_tag])
                            simulated_quote['size'] -= simulated_trade['size'][ii]
                            i = i + 1
                            ii = ii + 1
                        else:
                            level2 = self._action2level2(self, kind, i)
                            simulated_trade['level'] .append(level2)
                            simulated_trade['price'].append(quote(self._t)[price_tag])
                            simulated_trade['size'].append(simulated_quote['size'])
                            simulated_quote['size'] -= simulated_trade['size'][ii]
                            break
                else:  # level kind,是 ask,则需要排队并且考虑trade
                    wait_t = self._wait_t
                    if action['size'] != 0:
                        self._wait_signal = 0
                    else:
                        self._wait_signal = self._wait_signal + 1
                    if self._wait_signal == wait_t:
                        trade_sum = self.td.trade_sum(trade)
                        if trade_sum['price'].iloc[-1] >= self._simulated_quote['price']:
                            i = len(trade_sum['price'])-1
                            ii = 0
                            while trade_sum['price'][i] >= simulated_quote['price']:
                                if trade_sum['size'][i] < simulated_quote['size']:
                                    simulated_trade['level'].append(trade_sum['level'][ii])
                                    simulated_trade['price'].append(trade_sum['price'][ii])
                                    simulated_trade['size'].append(trade_sum['size'][ii])
                                    simulated_quote['size'] -= simulated_trade['size'][ii]
                                    i = i - 1
                                    ii = ii + 1
                                else:
                                    simulated_trade['level'].append(trade_sum['level'][ii])
                                    simulated_trade['price'].append(trade_sum['price'][ii])
                                    simulated_trade['size'].append(simulated_quote['size'])
                                    simulated_quote['size'] -= simulated_trade['size'][ii]
                                    break
                        else:
                            simulated_trade['level'][0] = 0
                            simulated_trade['price'][0] = 0
                            simulated_trade['size'][0] = 0
                    else:
                        simulated_trade['level'][0] = 0
                        simulated_trade['price'][0] = 0
                        simulated_trade['size'][0] = 0
        self._simulated_all_trade['level'].append(simulated_trade['level'])
        self._simulated_all_trade['price'].append(simulated_trade['price'])
        self._simulated_all_trade['size'].append(simulated_trade['size'])
        return(simulated_trade)



    def _vwap(self, trade):
        total_volume = sum(trade['size'])
        vwap = sum(map(lambda x, y: x * y, trade['price'], trade['size'])) / total_volume
        return vwap

    def _twap(self, trade):
        pass


data = dataset('000001', 'dbdir', 'cra001', 'cra001')
trader1 = AlgorithmTrader(data, strategy_direction='sell', wait_t=3)