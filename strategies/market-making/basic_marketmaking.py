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

from datasource.datatype import TickData
from exchange.stock import GeneralExchange
from dataloader import load_tickdata, load_case


class Marketmaking(object):
    def __init__(self,
                 tikedata,
                 transaction_engine:callable
                 ):
        self._data=tikedata
        self._transaction_engine = transaction_engine
        #self._transaction_engine2 = transaction_engine
        self._time = self._data.quote_timeseries
        self._final = False
        self._size=100
        self._traded_csv=pd.DataFrame(columns=('time', 'price', 'size', 'side'))
        self._num=0


    def reset(self) -> np.array:
        self._i = 0
        self._final = False
        t = self._time[self._i]
        price1 = float(self._data.get_quote(t)['bid1'])
        print(price1)
        price2 = float(self._data.get_quote(t)['ask1'])
        print(price2)
        self._order1 = {"time": t, "side": "buy", "price": price1, "size": self._size, "pos": -1}
        self._order2 = {"time": t, "side": "sell", "price": price2, "size": self._size, "pos": -1}


    def whitch_case(self,tradeda,tradedb):

       if (tradeda == 0) and (tradedb == 0):  # 都成交
           case = 0
       elif (tradeda == 0) and (tradedb != 0):  # 只有a成交
           case = 1
       elif (tradeda != 0) and (tradedb == 0):  # 只有a成交
           case = 2
       elif (tradeda != 0) and (tradedb != 0):  # 都不成交
           case = 3
       else:
           print("warning")

       return case



    def makeorder(self,):
        #从头开始
        self.reset()
        self._ordera = self._order1
        self._orderb = self._order2
        sum_i = 0

        while(self._final== False):

            orderda, tradeda = self._transaction_engine(self._ordera)
            orderdb, tradedb = self._transaction_engine(self._orderb)
            #写数据
            if tradeda['size']:
               if int(tradeda['size'][0])==100:
                   self._traded_csv=self._traded_csv.append({'time':int(tradeda['time'][0]),'price':float(tradeda['price'][0]),'size':int(tradeda['size'][0]),'side':'buy'},ignore_index=True)
            if tradedb['size']:
               if int(tradedb['size'][0])==100:
                   self._traded_csv=self._traded_csv.append({'time':int(tradedb['time'][0]),'price':float(tradedb['price'][0]),'size':int(tradedb['size'][0]),'side':'sell'},ignore_index=True)

            # 判断本次交易属于哪种情况
            case= self.whitch_case(tradeda=self._ordera['size'],tradedb=self._orderb["size"])

            if (case == 0) or (case == 3) :#如果双方都成交，或者双方都不成交；下一次就重新下单
                self._i += 1
                t = self._time[self._i]
                price1 = float(self._data.get_quote(t)['bid1'][self._i])
                price2 = float(self._data.get_quote(t)['ask1'][self._i])
                self.next_ordera = {"time": t, "side": "buy", "price": price1, "size": self._size, "pos": -1}
                self.next_orderb = {"time": t, "side": "sell", "price": price2, "size": self._size, "pos": -1}
            elif case == 1:#如果a成交了，b没成交 b就要等待三个quote时间
                self._i += 1
                t = self._time[self._i]
                price1 = float(self._data.get_quote(t)['bid1'][self._i])
                price2 = float(self._data.get_quote(t)['ask1'][self._i])
                self.next_ordera = {"time": t, "side": "buy", "price": price1, "size": 0, "pos": -1}
                if sum_i !=3 :
                   self.next_orderb = {"time": t, "side": "sell", "price":self._orderb['price'], "size": self._orderb['size'], "pos": self._orderb['pos']}
                   sum_i = sum_i+1
                else:
                   self.next_orderb = {"time": t, "side": "sell", "price": price1, "size": self._size, "pos": -1}
                   sum_i = 0
            elif case == 2:
                self._i += 1
                t = self._time[self._i]
                price1 = float(self._data.get_quote(t)['bid1'][self._i])
                price2 = float(self._data.get_quote(t)['ask1'][self._i])
                self.next_orderb = {"time": t, "side": "sell", "price": price2, "size": 0, "pos": -1}
                if sum_i !=3 :
                   self.next_ordera = {"time": t, "side": "buy", "price":self._ordera['price'], "size": self._ordera['size'], "pos": self._ordera['pos']}
                   sum_i = sum_i+1
                else:
                    self.next_ordera = {"time": t, "side": "buy", "price": price2, "size": self._size, "pos": -1}
                    sum_i = 0

            self._ordera=self.next_ordera
            self._orderb = self.next_orderb

            if t == self._time[-2]:
                self._final = True

        return self._traded_csv


quote, trade = load_tickdata(stock='000001', time='20140704')
data = TickData(quote, trade)
print(data.get_quote(34200000))
print(data.quote_timeseries[0:10])
exchange = GeneralExchange(data, 3)
MM=Marketmaking(tikedata=data, transaction_engine=exchange.transaction_engine)
traded_csv=MM.makeorder()
traded_csv.to_csv(r'D:\\python\\my程序\\have_wait_t_20200728\\algorithmic-trading-master\\test\\results\\traded.csv')