import numpy as np

from tickdata import TickData


class AlgorithmTrader(object):

    def __init__(
        self,
        td:TickData,
        strategy_direction:['buy', 'sell'],
        volume:int,
        reward_function:callable or str,
        wait_t,
        max_level=5 # TODO rewrite corresponding codes.
        ):
        self._n  = 0
        self._td = td
        self._t  = self._td.quote_timeseries[0]
        self._simulated_quote     = {'level': None, 'price':None, 'size': None}
        self._simulated_all_trade = {'level': [], 'price':[], 'size': []}
        self._res_volume = volume
        self._wait_t     = wait_t
        self._action_space    = max_level * 2
        self._reward_function = reward_function
    
    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        self._n = 0
        self._t = self._td.quote_timeseries[0]
        self._simulated_quote     = {'level': None, 'price':None, 'size': None}
        self._simulated_all_trade = {'level': [], 'price':[], 'size': []}

    def step(self, action)->'(next_s, reward, signal, info)':
        self._t = self._td.trade_timeseries[self._n]
        quote = self._td.get_quote(self._t)
        trade = self._td.get_trade(self._t)
        # Issue an order;
        level = self._action_map(action[0])
        if action[1] != 0:
            self._simulated_quote['level'] = level
            self._simulated_quote['price'] = quote[level][0]
            self._simulated_quote['size']  = action[1]
        simulated_trade = self._transaction_matching(quote, trade)
        self._res_volume -= sum(simulated_trade['size'])
        # Give a final signal
        if self._t == self._td.trade_timeseries[-2]:
            signal = True
        elif self._res_volume == 0:
            signal = True
        # TODO what to do if self._res_volume < 0
        elif self._res_volume < 0:
            signal =True
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

    def _action2level(self, action:int)->str:
        max_level = self.action_space / 2
        if action <  max_level:
            level = max_level - action
            return 'bid%d' % level
        else:
            level = action - max_level + 1
            return 'ask%d' % level

    def _level2size(self, level:str)->str:
        size_tag = level[0] +'size' + level[-1]
        return size_tag

    def _transaction_matching(self, quote, trade,)->'simulated_trade':
        # NOTE @wupj 这次matching中发生的trade要记录在simulated_trade中，并追加到simulated_all_trade中
        simulated_trade = {'level': [], 'price':[], 'size': []}
        m = np.size(self._simulated_quote, 0)
        m2= np.size(self._simulated_trade, 0)
        if self._simulated_quote['size']!=0:                                                       #说明有订单在等待成交
             if  self._simulated_quote[m-1][2]>0:
                 if self._simulated_quote[m-1][1]>=ask[1]:                                         #如果买的价格高于现在市场上卖的最低价
                     if self._simulated_quote[m-1][2] <= self._real_quote[n][4]:                           #并且ASK1的量足够
                         OK_signal = 1                                                      #直接全部成交
                     else:
                         # TODO complete warning log.
                         warning_signal=1                                                    #报警
                 else:                                                                       #先判断simulated_quote[m-1][1]=BID几
                     if action[n][1] != 0:
                         wait_signal=0
                     if wait_signal == 3:
                         aa = np.argwhere(bid == simulated_quote[m-1][1])                    # 先判断simulated_quote[m-1][1]=bid几
                         bb = np.argwhere(real_trade2 == simulated_quote[m-1][0])
                         cc = np.zeros([1, 6])
                         for i in bb:
                             if real_trade2[i][1]==bid[aa]:
                                 cc[aa]=real_trade2[i][2]
                             if real_trade2[i][1]<bid[aa]:
                                 aa2 = np.argwhere(ask == real_trade2[i][1])
                                 cc[aa2]=real_trade2[i][2]
                         k=simulated_quote[m-1][2]
                         cc2=np.cumsum(cc, axis=0)
                         if cc2>=simulated_quote[m-1][2] :
                             OK_signal = 1                                                     #全部在当前价位交易
                         else :
                             if cc2>0:
                                 k=k-cc2                                                       #成交了cc2的量没成交k的量
                                 OK_signal = 2
                             else:
                                 OK_signal = 0
             else:                                                                            #说明有订单是要卖
                 if simulated_quote[m-1][1]<=bid[1]:
                     if simulated_quote[m-1][2] <= real_quote[n][1]:
                         OK_signal = 1
                     else:
                         warning_signal=1                                                     #报警
                 else:
                     if action[n][1] != 0:
                         wait_signal=0
                     if wait_signal==3:                                                        #已经等待三个时钟周期，可以进行交易
                         aa = np.argwhere(ask == simulated_quote[m-1][1])                      #先判断simulated_quote[m-1][1]=ask几

                         bb = np.argwhere( real_trade2 == simulated_quote[m-1][0])
                         cc =np.zeros([1,6])
                         for i in bb:
                             if real_trade2[i][1]==ask[aa]:
                                 cc[aa]=real_trade2[i][2]
                             if real_trade2[i][1]>ask[aa]:
                                 aa2 = np.argwhere(ask == real_trade2[i][1])
                                 cc[aa2]=real_trade2[i][2]
                         k=simulated_quote[m-1][2]
                         cc2=np.cumsum(cc, axis=0)
                         if cc2>=simulated_quote[m-1][2] :
                             OK_signal = 1                                                     #全部在当前价位交易
                         else :
                             if cc2>0:
                                 k=k-cc2                                                       #成交了cc2的量没成交k的量
                                 OK_signal = 2
                             else:
                                 OK_signal = 0
                     else:
                         OK_signal = 0                                                         #无成交
                         wait_signal += wait_signal                                            #还需要等待几个周期才可以进行交易
        else:
            OK_signal=0

        if OK_signal==1:                                                                       #如果全部成交
             simulated_trade =np.insert(simulated_trade,m2,values=simulated_quote[m-1],axis=0)
             quote2=np.arry(simulated_quote[m-1][0],simulated_quote[m-1][1],0)
             simulated_quote=np.insert(simulated_quote ,m-1,values=quote2,axis=0)
             a1=simulated_trade[m2][1]
             a2=simulated_trade[m2][2]
             b1 = simulated_quote[m][1]
             b2 = simulated_quote[m][2]
        elif OK_signal==2:                                                                     #如果成交了一部分
             trade2 = np.arry(simulated_quote[m-1][0], simulated_quote[m-1][1], cc2)
             simulated_trade =np.insert(simulated_trade,m2,values=trade2,axis=0)                #trade里面显示成交的量，quote里增加保留的量
             quote2 = np.arry(simulated_quote[m-1][0], simulated_quote[m-1][1], k)
             simulated_quote = np.insert(simulated_quote, m-1, values=quote2, axis=0)
             a1 = simulated_trade[m2][1]
             a2 = simulated_trade[m2][2]
             b1 = simulated_quote[m][1]
             b2 = simulated_quote[m][2]
        elif OK_signal==0:
             a1 = 0
             a2 = 0
             b1 = simulated_quote[m-1][1]
             b2 = simulated_quote[m-1][2]
        pass

    def _vwap(self, trade):
        map(lambda x,y: x * y/ sum(y), trade['price'], trade['size'])

    def _twap(self, trade):
        pass
