import numpy as np
import pandas as pd
import json
import sys
#sys.path.append()
sys.path.append('D:\\python\\my程序\\have_wait_t_20200728\\algorithmic-trading-master\\test\\utils')
'''
同时发两笔
"order1": {"time": 34200000, "side": "buy", "price": 9.96, "size": 1, "pos": -1}
"order2": {"time": 34200000, "side": "sell", "price": 9.97, "size": 1, "pos": -1}
如果order1在t1被成交了，设置一个机制，过多久order2需要降价到9.96卖出。
等待时间是n笔trade的时间
order1和order2均成交了才可以发下一个order1'和order2'
'''

from src.datasource.datatype import TickData
from src.exchange.stock import GeneralExchange
from dataloader import load_tickdata, load_case


class Marketmaking(object):
    def __init__(self,
                 tikedata,
                 transaction_engine:callable
                 ):
        self._data=tikedata
        self._transaction_engine = transaction_engine
        self._time = self._data.quote_timeseries
        self._final = False


    def reset(self) -> np.array:
        self._i = 0
        self._final = False
        t = self._time[self._i]
        price1 = float(self._data.get_quote(t)['bid1'])
        print(price1)
        price2 = float(self._data.get_quote(t)['ask1'])
        print(price2)
        self._order1 = {"time": t, "side": "buy", "price": 9.95, "size": 1, "pos": -1}

    def makeorder(self,):
        self.reset()
        self._order = self._order1
        print(self._order)
        orderd,traded=self._transaction_engine(self._order)

        self._order = {'time': self._order['time'], 'side': self._order['side'], 'price': self._order['price'], 'size': 1, 'pos': -1}
        print(traded)
        sum_i = 0
        while(self._final== False):
            # 如果成交了，就要换order
            if traded['size']==[]:
                change_orderflag = 2
            else:
               if traded['size'][0] == 1:
                   change_orderflag = 1
               else:
                   change_orderflag = 0

            t = self._time[self._i]
            price1 = float(self._data.get_quote(t)['bid1'])
            price2 = float(self._data.get_quote(t)['ask1'])
            self._order1 = {'time': t, 'side': "buy", 'price': price1, 'size': 1, 'pos': -1}
            self._order12 = {'time': t, 'side': "buy", 'price': price2, 'size': 1, 'pos': -1}
            self._order2 = {'time': t, 'side': "sell", 'price': price2, 'size': 1, 'pos': -1}
            self._order22 = {'time': t, 'side': "sell", 'price': price1, 'size': 1, 'pos': -1}

            self._i += 1
            t = self._time[self._i]
            nprice1 = float(self._data.get_quote(t)['bid1'])
            nprice2 = float(self._data.get_quote(t)['ask1'])
            print(nprice1)
            print(nprice2)
            self._norder1 = {'time': t, 'side': "buy", 'price': nprice1, 'size': 1, 'pos': -1}
            self._norder12 = {'time': t, 'side': "buy", 'price': nprice2, 'size': 1, 'pos': -1}
            self._norder2 = {'time': t, 'side': "sell", 'price': nprice2, 'size': 1, 'pos': -1}
            self._norder22 = {'time': t, 'side': "sell", 'price': nprice1, 'size': 1, 'pos': -1}

            # 如果成交了，就要换order,如果没成交，等待三秒
            print("change_orderflag=",change_orderflag)
            if (change_orderflag==1):
                if ((self._order==self._order1) or (self._order==self._order12)):
                    self._norder = self._norder2
                else:
                    self._norder = self._norder1
            else:
                if(change_orderflag==2):
                   if ((self._order['price']!=nprice1) and(self._order['price']!=nprice2)):
                       if self._order['price']==price1:
                           pp=nprice1
                       else:
                           if self._order['price']==price2:
                               pp = nprice2
                       self._norder = {'time': t, 'side': self._order['side'], 'price': pp, 'size': 1,'pos': -1}
                   else:
                       if (self._order==self._order1 or self._order==self._order2):
                          sum_i +=1
                          print("sum_i=",sum_i)
                          if sum_i==3:
                             if (self._order == self._order1):
                                 self._norder=self._norder12
                             else:
                                 self._norder=self._norder22
                             sum_i=0
                          else:
                             self._norder={'time':t, 'side':self._norder['side'], 'price':self._norder['price'], 'size': 0, 'pos': -1}
                       else:
                          sum_i = 0
                          self._norder={'time':t, 'side':self._order['side'], 'price':self._order['price'], 'size': 1, 'pos': -1}

            print('self._norder',self._norder)
            orderk, traded = self._transaction_engine(self._norder)
            print(traded)
            self._order={'time': t, 'side':self._norder['side'], 'price':self._norder['price'], 'size': 1, 'pos': -1}
            print('self._order',self._order)

            if t == self._time[-2]:
                self._final = True



quote, trade = load_tickdata(stock='000001', time='20140704')
data = TickData(quote, trade)
print(data.get_quote(34200000))
print(data.quote_timeseries[0:10])
exchange = GeneralExchange(data, 3)
MM=Marketmaking(tikedata=data, transaction_engine=exchange.transaction_engine)
MM.makeorder()





