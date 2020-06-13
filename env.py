import numpy as np

from tickdata import TickData


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
        simulated_trade = self._transaction_matching(quote, trade)   #进行matching之后的输出给到simulated_trade
        self._res_volume -= sum(simulated_trade['size'])     #
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

    def _action2level(self, action: int) ->'(kind,newlevel)':  #返回这个ACTION的level BID几或者ASK几
        max_level = self.action_space / 2
        if action < max_level:
            newlevel = max_level - action
            kind='bid'
            return (kind,newlevel)
        else:
            newlevel = action - max_level + 1
            kind='ask'
            return (kind,newlevel)

    def _action2level2(self,kind:str,newlevel:int)->'level':
        max_level = self.action_space / 2
        if kind=='bid':
            level=max_level-newlevel
        else:
            level=max_level+newlevel-1
        return(level)


    def _level2size(self, kind:str,newlevel) -> str:     #返回B_size_5
        size_tag = kind[0] + 'size' + newlevel
        return size_tag

    def _level2price(self, kind:str,newlevel) -> str:     #返回B_size_5
        price_tag = kind+newlevel
        return price_tag

'''怎么从TRADER那获得strategy_direction，在这里把它作为输入可行吗？...如果改成real_quote的话是不是前面的slef._td，self._n等用不了？，'''
    def _transaction_matching(self,action,strategy_direction,quote,trade ) -> 'simulated_trade':
        # NOTE @wupj 这次matching中发生的trade要记录在simulated_trade中，并追加到simulated_all_trade中
        simulated_trade = {'level': [], 'price': [], 'size': []}
        if self._simulated_quote['size'] != 0:
            (kind, newlevel) = self._action2level(action)
            if strategy_direction=='buy':
               if kind=='ask':  # 如果levelkind是'ask',则现在可以直接成交，只考虑quote中的数量
                    i=1
                    ii=0
                    while i<=newlevel:
                        size_tag=self._level2size(kind,i)
                        price_tag = self._level2price(kind, i)
                        if self._td.quote[self._n][size_tag]<=quote['size']:                  #如果ask1的量小于等于提交的订单量
                            level2=self._action2level2(self,kind,i)
                            simulated_trade['level'][ii]=level2
                            simulated_trade['price'][ii]=self._td.quote[self._n]['price_tag']  #这样查找real quote 里的 price对吗？？？？？？？？？？？
                            simulated_trade['size'][ii]=self._td.quote[self._n]['size_tag']
                            quote['size'] -= simulated_trade['size'][ii]
                            i=i+1
                            ii=ii+1
                        else:
                            level2 = self._action2level2(self, kind, i)
                            simulated_trade['level'][ii] = level2                               #如果大于需求的量，则全部成交
                            simulated_trade['price'][ii] = self._td.quote[self._n]['price_tag']
                            simulated_trade['size'][ii] =quote['size']
                            quote['size'] -= simulated_trade['size'][ii]
                            break
               else:  # level kind,是 bid,则需要排队并且考虑trade
                    wait_t=self._wait_t
                    if action['size'] != 0:
                        self._wait_signal = 0
                    else :
                        self._wait_signal= self._wait_signal +1
                    if self._wait_signal == wait_t:
                        trade_sum=self.td.trade_sum(trade)
                        if trade_sum['price'][0]<=quote['price'] :   #说明存在比我提交的价格低的值呗成交了
                           i=0
                           ii=0
                           while trade_sum['price'][i]<=quote['price']:
                               if trade_sum['size'][i]<=quote['size']:
                                   simulated_trade['level'][ii] = trade_sum['level'][ii]      #能不能在trade_sum里面多一列level
                                   simulated_trade['price'][ii] = trade_sum['price'][ii]
                                   simulated_trade['size'][ii] = trade_sum['size'][ii]
                                   quote['size'] -= simulated_trade['size'][ii]
                                   i = i + 1
                                   ii = ii + 1
                               else:
                                   simulated_trade['level'][ii] = trade_sum['level'][ii]  # 能不能在trade_sum里面多一列level
                                   simulated_trade['price'][ii] = trade_sum['price'][ii]
                                   simulated_trade['size'][ii] = quote['size']
                                   quote['size'] -= simulated_trade['size'][ii]
                                   break
                        else:
                            simulated_trade['level'][0]=0   #一点也没成交 simulated_trade 0 0
                            simulated_trade['price'][0] = 0
                            simulated_trade['size'][0] = 0
                    else:
                        #一点也没成交 simulated_trade 0 0
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
                        if self._td.quote[self._n][size_tag] < quote['size']:  # 如果bid1的量小于等于提交的订单量
                            level2 = self._action2level2(self, kind, i)
                            simulated_trade['level'][ii] = level2
                            simulated_trade['price'][ii] = self._td.quote[self._n]['price_tag']
                            simulated_trade['size'][ii] = self._td.quote[self._n]['size_tag']
                            quote['size'] -= simulated_trade['size'][ii]
                            i = i + 1
                            ii = ii + 1
                        else:
                            level2 = self._action2level2(self, kind, i)
                            simulated_trade['level'][ii] = level2
                            simulated_trade['price'][ii] = self._td.quote[self._n]['price_tag']
                            simulated_trade['size'][ii] = quote['size']
                            quote['size'] -= simulated_trade['size'][ii]
                            break
                else:  # level kind,是 ask,则需要排队并且考虑trade
                    wait_t = self._wait_t
                    if action['size'] != 0:
                        self._wait_signal = 0
                    else:
                        self._wait_signal = self._wait_signal + 1
                    if self._wait_signal == wait_t:
                        trade_sum = self.td.trade_sum(trade)
                        if trade_sum['price'][len(trade_sum['price'])] >= self._simulated_quote['price']:  # 可不可以这么用len?????
                            i = len(trade_sum['price'])
                            ii = 0
                            while trade_sum['price'][i] >= quote['price']:
                                if trade_sum['size'][i] < quote['size']:
                                    simulated_trade['level'][ii] = trade_sum['level'][ii]  # 能不能在trade_sum里面多一列lavel
                                    simulated_trade['price'][ii] = trade_sum['price'][ii]
                                    simulated_trade['size'][ii] = trade_sum['size'][ii]
                                    quote['size'] -= simulated_trade['size'][ii]
                                    i = i - 1
                                    ii = ii + 1
                                else:
                                    simulated_trade['level'][ii] = trade_sum['level'][ii]  # 能不能在trade_sum里面多一列level????????
                                    simulated_trade['price'][ii] = trade_sum['price'][ii]
                                    simulated_trade['size'][ii] = quote['size']
                                    quote['size'] -= simulated_trade['size'][ii]
                                    break
                        else:
                            simulated_trade['level'][0] = 0  # 一点也没成交 simulated_trade 0 0
                            simulated_trade['price'][0] = 0
                            simulated_trade['size'][0] = 0
                    else:
                        # 一点也没成交 simulated_trade 0 0
                        simulated_trade['level'][0] = 0
                        simulated_trade['price'][0] = 0
                        simulated_trade['size'][0] = 0
        '''返回列表？？怎么把simulated_trade 加到simulated_all_trade 中'''
        self._simulated_all_trade=self._simulated_all_trade+simulated_trade
        return(simulated_trade)



    def _vwap(self, trade):
        map(lambda x, y: x * y / sum(y), trade['price'], trade['size'])

    def _twap(self, trade):
        pass
