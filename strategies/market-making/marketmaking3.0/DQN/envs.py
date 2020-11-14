import abc
import sys

import numpy as np
import pandas as pd

#状态+动作+收益的合集？
class environment():
    
    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level:int, side:str, historical_quote_num:int):
        super().__init__(tickdata=tickdata, task=task, transaction_engine=transaction_engine, level=level, side=side)
        self._historical_quote_num = historical_quote_num

    @property
    def observation_space_n(self):
        n = (5, 40)
        return n

    def _state(self):
        quote  = self._data.get_quote(self._t)
        quotes = [self._data.quote_board(quote).values.flatten().tolist()]
        for _ in range(self._historical_quote_num - 1):
            quote = self._data.pre_quote(quote)
            if quote is not None:
                quotes.append(self._data.quote_board(quote).values.flatten().tolist())
            else:
                break
        padnum = self._historical_quote_num - len(quotes)
        quotes = np.pad(quotes, ((padnum,0), (0,0)))
        state  = [[self._t / 1000, self._task['start'] / 1000, self._task['end'] / 1000, self._task['goal'], sum(self._filled['size'])]]
        state  = [np.array(state, dtype=np.float32), np.array(quotes, dtype=np.float32)]
        return np.array(state, dtype=object)






class env():
    
    def __init__(self, tickdata, task:pd.Series, transaction_engine:callable, level:int, side:str, historical_quote_num:int):
        self._data = tickdata
        self._task = task
        self._engine = transaction_engine
        self._side = side
        self._init = False
        self._final = False
        self._time = [t for t in self._data.quote_timeseries if t >= task['start'] and t < task['end']]     
        
    
    def _state(self):#state是下一条QUOTE数据需要修改
        quote = self._data.get_quote(self._t).iloc[0].values.tolist()
        state  = quote
        return np.array(state, dtype=np.float32)

    def reset(self):
        self._init = True
        self._final = False
        self._iter = iter(self._time)
        self._t = next(self._iter)
        self._filled = {'time': [], 'price':[], 'size':[]}
        self._order = {'time': 0, 'side': 'sell', 'price': 0, 'size': 0, 'pos': -1}
        return self._state()


    def _action2order(self, action:int):
        order = pd.DataFrame(columns=('time','side','price','size','pos' ))
        time  = self._t
        
        if action == 0:
            order.loc[0,'time']  = self._t
            order.loc[0,'size']  = 0
            order.loc[0,'price'] = self._data.get_quote(time)['ask1'].iloc[0]
            order.loc[0,'side'] = 'sell'
            order.loc[0,'pos'] = -1

        if action == 1:
            order.loc[0,'time']  = self._t
            order.loc[0,'side']  = 'sell'
            order.loc[0,'price'] = self._data.get_quote(time)['ask1'].iloc[0]
            order.loc[0,'pos']  = -1
            order.loc[0,'size']  = 100

            order.loc[1,'time']  = self._t
            order.loc[1,'side']  = 'buy'
            order.loc[1,'price'] = self._data.get_quote(time)['bid1'].iloc[0]
            order.loc[1,'pos']  = -1
            order.loc[1,'size']  = 100

        if action == 2:
            order.loc[0,'time']  = self._t
            order.loc[0,'side']  = 'sell'
            order.loc[0,'price'] = self._data.get_quote(time)['ask1'].iloc[0]
            order.loc[0,'pos']  = -1
            order.loc[0,'size']  = 100
        
        if action == 3:
            order.loc[0,'time']  = self._t
            order.loc[0,'side']  = 'buy'
            order.loc[0,'price'] = self._data.get_quote(time)['bid1'].iloc[0]
            order.loc[0,'pos']  = -1
            order.loc[0,'size']  = 100

        return order    

    def step(self, action):
        if self._init == False:
            raise NotInitiateError
        if self._final == True:
            raise EnvTerminatedError
        else:
            self._order = self._action2order(action)
        
        self._reward=0
        i=0
        while i<len(self._order):
            self._order1 = self._order.loc[i].to_dict()
            self._order, filled = self._engine(self._order1)
            if self._order.loc[i,'side']=='buy':
                self._reward = self._reward-filled['price']*filled['size']
            else:
                self._reward = self._reward+filled['price']*filled['size']
            i=i+1    
     
        state = None if self._final else self._state()  #state是下一条QUOTE数据
        return (state, self._reward, self._final)    