import numpy as np
import pandas as pd
from tickdata import dataset
from tickdata import TickData

class AlgorithmTrader(object):

    def __init__(
            self,
            td:TickData,

            wait_t,
            max_level=5  # TODO rewrite corresponding codes.
    ):


        self._n = 0
        self._td = td
        self.simulated_quote = {'level': None, 'price': None, 'size': None}
        self._simulated_all_trade = {'level': [], 'price': [], 'size': []}
        self.wait_t = wait_t
        self._wait_signal=0
        self._action_space = max_level * 2


    def _action2level(self, action: int) ->'(kind,newlevel)':  #返回这个ACTION的level：从数字0变成kind'bid',newlevel'5'
        max_level = self._action_space / 2
        if action < max_level:
            newlevel = max_level - action
            kind='bid'
            return (kind,newlevel)
        else:
            newlevel = action - max_level + 1
            kind='ask'
            return (kind,newlevel)

    def _level2price(self, kind:str,newlevel) -> str:  #从 kind'bid',newlevel'5' 变成'bid5'
        price_tag = kind+str(newlevel)
        return price_tag

    def _action2level2(self,kind:str,newlevel:int)->'level':  #从kind'bid',newlevel'5'变成数字0
        max_level = self._action_space / 2
        if kind=='bid':
            level=max_level-newlevel
        else:
            level=max_level+newlevel-1
        return(level)

    def _level2size(self, kind:str,newlevel) -> str:     #返回B_size_5
        size_tag = kind[0] + 'size' + str(newlevel)
        return size_tag

#给撮合引擎的是价格不能是0123456



    def _transaction_matching(self,action,simulated_quote,strategy_direction) -> 'simulated_trade':
        self._n = 0
        self._t = self._td.quote_timeseries[self._n]
        quote = self._td.get_quote(self._t)
        trade = self._td.get_trade_between(self._t )
        simulated_trade = {'price': [], 'size': []}

        if simulated_quote['size'] != 0:
            (kind,newlevel) = self._action2level(simulated_quote['level'])
            print(kind,newlevel)
            if strategy_direction=='buy':
               if kind=='ask':  # 如果levelkind是'ask',则现在可以直接成交，只考虑quote中的数量
                    i=1 #真实level的数值
                    ii=0 #现在的simulated_trade里有几个值
                    while i<=newlevel:
                        size_tag=self._level2size(kind,i)
                        price_tag = self._level2price(kind, i)
                        if quote[size_tag].iloc[0]<=simulated_quote['size']:                  #如果ask1的量小于等于提交的订单量
                            #level2=self._action2level2(self,kind,i)
                            #simulated_trade['level'].append(level2)
                            simulated_trade['price'].append(quote[price_tag].iloc[0])
                            simulated_trade['size'].append(quote[size_tag].iloc[0])
                            self.simulated_quote['size'] -= simulated_trade['size'][ii]
                            i=i+1
                            ii=ii+1
                        else:
                            #level2 = self._action2level2(self, kind, i)
                            #simulated_trade['level'].append(level2)                             #如果大于需求的量，则全部成交
                            simulated_trade['price'].append(quote[price_tag].iloc[0])
                            simulated_trade['size'].append(self.simulated_quote['size'])
                            self.simulated_quote['size'] -= simulated_trade['size'][ii]
                            break
               else:  # level kind,是 bid,则需要排队并且考虑trade
                    wait_t=self.wait_t
                    if action['size'] != 0:
                        self._wait_signal = 0
                    else :
                        self._wait_signal= self._wait_signal +1
                    if self._wait_signal >= wait_t:
                        trade_sum=self._td.trade_sum(trade)
                        if trade_sum['price'][0]<=self.simulated_quote['price'] :   #说明存在比我提交的价格低的值成交了
                           i=0
                           ii=0
                           while float(trade_sum['price'][i])<=float(self.simulated_quote['price']):
                               if trade_sum['size'][i]<=self.simulated_quote['size']:
                                  # simulated_trade['level'].append(trade_sum['level'][ii])
                                   simulated_trade['price'].append(trade_sum['price'][ii])
                                   simulated_trade['size'].append(trade_sum['size'][ii])
                                   self.simulated_quote['size'] -= simulated_trade['size'][ii]
                                   i = i + 1
                                   ii = ii + 1
                               else:
                                   #simulated_trade['level'].append(trade_sum['level'][ii])
                                   simulated_trade['price'].append(trade_sum['price'][ii])
                                   simulated_trade['size'].append(self.simulated_quote['size'])
                                   self.simulated_quote['size'] -= simulated_trade['size'][ii]
                                   break
                        else:
                            #simulated_trade['level'][0]=0
                            simulated_trade['price'] = 0
                            simulated_trade['size'][0] = 0
                    else:
                        #simulated_trade['level'][0] = 0
                        simulated_trade['price']= 0
                        simulated_trade['size']= 0

            # 说明有订单是要卖
            else:
                if kind == 'bid':  # 如果levelkind是'bid',则现在可以直接成交，只考虑quote中的数量
                    i = 1
                    ii = 0
                    while i <= newlevel:
                        size_tag = self._level2size(kind, i)
                        price_tag = self._level2price(kind, i)
                        if quote[size_tag].iloc[0] < self.simulated_quote['size']:  # 如果bid1的量小于提交的订单量
                            #level2 = self._action2level2(self, kind, i)
                            #simulated_trade['level'].append(level2)
                            simulated_trade['price'].append(quote[price_tag].iloc[0] )
                            simulated_trade['size'].append(quote[size_tag].iloc[0])
                            self.simulated_quote['size'] -= simulated_trade['size'][ii]
                            i = i + 1
                            ii = ii + 1
                        else:
                            #level2 = self._action2level2(self, kind, i)
                            #simulated_trade['level'] .append(level2)
                            simulated_trade['price'].append(quote[price_tag].iloc[0])
                            simulated_trade['size'].append(self.simulated_quote['size'])
                            self.simulated_quote['size'] -= simulated_trade['size'][ii]
                            break
                else:  # level kind,是 ask,则需要排队并且考虑trade
                    wait_t = self.wait_t
                    if action['size'] != 0:
                        self._wait_signal = 0
                    else:
                        self._wait_signal = self._wait_signal + 1
                    if self._wait_signal == wait_t:
                        trade_sum = self._td.trade_sum(trade)
                        if trade_sum['price'].iloc[-1] >= self.simulated_quote['price']:
                            i = len(trade_sum['price'])-1
                            ii = 0
                            while (float(trade_sum['price'][i]) >= float(self.simulated_quote['price'])):
                                if trade_sum['size'][i] < self.simulated_quote['size']:
                                    #simulated_trade['level'].append(trade_sum['level'][ii])
                                    simulated_trade['price'].append(trade_sum['price'][i])
                                    simulated_trade['size'].append(trade_sum['size'][i])
                                    self.simulated_quote['size'] -= simulated_trade['size'][ii]
                                    i = i - 1
                                    ii = ii + 1
                                else:
                                   # simulated_trade['level'].append(trade_sum['level'][ii])
                                    simulated_trade['price'].append(trade_sum['price'][i])
                                    simulated_trade['size'].append(self.simulated_quote['size'])
                                    self.simulated_quote['size'] -= simulated_trade['size'][ii]
                                    break
                        else:
                            #simulated_trade['level'][0] = 0
                            simulated_trade['price'] = 0
                            simulated_trade['size'] = 0
                    else:
                        #simulated_trade['level'][0] = 0
                        simulated_trade['price'] = 0
                        simulated_trade['size'] = 0
        self._n=self._n+1
        #self._simulated_all_trade['level'].append(simulated_trade['level'])
        self._simulated_all_trade['price'].append(simulated_trade['price'])
        self._simulated_all_trade['size'].append(simulated_trade['size'])
        return(simulated_trade)


dbdir = 'F:\\融创课题\\数据百度云\\201406(2)\\20140630'
trader1=AlgorithmTrader(td=dataset('000001', dbdir, 'cra001', 'cra001'),wait_t=3)
action1={'level':4 , 'price':9.800,'size':80000}
trader1.simulated_quote={'level':4 , 'price':9.800,'size':80000}
print(trader1._transaction_matching(action=action1,simulated_quote=trader1.simulated_quote,strategy_direction='buy'))
print(trader1.simulated_quote)

action1={'level':0 , 'price':0,'size':0}
print(trader1._transaction_matching(action=action1,simulated_quote=trader1.simulated_quote,strategy_direction='buy'))
print(trader1.simulated_quote)

action1={'level':0 , 'price':0,'size':0}
print(trader1._transaction_matching(action=action1,simulated_quote=trader1.simulated_quote,strategy_direction='buy'))
print(trader1.simulated_quote)

action1={'level':0 , 'price':0,'size':0}
print(trader1._transaction_matching(action=action1,simulated_quote=trader1.simulated_quote,strategy_direction='buy'))
print(trader1.simulated_quote)


