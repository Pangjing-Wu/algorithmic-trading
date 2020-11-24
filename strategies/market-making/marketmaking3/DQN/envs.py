import abc
import sys

import numpy as np
import pandas as pd

#状态+动作+收益的合集？


class Env():
    
    def __init__(
        self, 
        tickdata,
        #task:pd.Series,
        transaction_engine:callable,
        number:int#是第几个quote
    ):
        self._data = tickdata
        #self._task = task
        self._engine = transaction_engine
        #self._level_space = self._int2level(level)#int2level是什么
        #self._side = side
        self._init = False
        self._final = False
        self.number=number
        #quotequote=self._data._quote
        #print(self._data._quote)
        self._t = [self._data._quote.loc[number-4,'time'],self._data._quote.loc[number-3,'time'],self._data._quote.loc[number-2,'time'],self._data._quote.loc[number-1,'time'],self._data._quote.loc[number,'time'],self._data._quote.loc[number+1,'time']]

    #加一个actionspace
    @property
    def action_space(self):
        space=[0,1,2,3]
        return space 

    @property
    def observation_space_n(self):
        return 3 

    @property
    def action_space_n(self):
        return 4  

    ##state也要改一下，要包括resorder
    def _state(self):#state是下一条QUOTE数据需要修改
        i=1
        quote=[]
        while i<5:
            quote += self._data.get_quote(int(self._t[i])).iloc[0].values.tolist()
            i=i+1
        #print(quote)    
        state=[self._money,self._position]
        for q in quote:
            state.append(q)
        return np.array(state, dtype=np.float32)

    def reset(self):#只有每天早上reset
        time  = int(self._data._quote.loc[self.number,'time'])
        self._init = True
        self._final = False
        #self._tnow = self._data._quote.loc[self.number,'time']
        self._resorder = pd.DataFrame(columns=('time','side','price','size','pos' ))
        #self._resorder.loc[0]=[time,'sell',self._data.get_quote(time)['ask1'].iloc[0],0,-1]
        self._filled = {'time': [], 'price':[], 'size':[]}
        self._money=100000
        self._position=100  #持仓多少股票
        return  (self._state(),self._money,self._position,self._resorder)

    def _action2order(self, action:int):
        order = pd.DataFrame(columns=('time','side','price','size','pos' ))
        time  = int(self._data._quote.loc[self.number,'time'])
        
        if action == 0:
            order.loc[0,'time']  = time
            order.loc[0,'size']  = 0
            order.loc[0,'price'] = self._data.get_quote(time)['ask1'].iloc[0]
            order.loc[0,'side'] = 'sell'
            order.loc[0,'pos'] = -1

        if action == 1:
            order.loc[0,'time']  = time
            order.loc[0,'side']  = 'sell'
            order.loc[0,'price'] = self._data.get_quote(time)['ask1'].iloc[0]
            order.loc[0,'pos']  = -1
            order.loc[0,'size']  = 100

            order.loc[1,'time']  = time
            order.loc[1,'side']  = 'buy'
            order.loc[1,'price'] = self._data.get_quote(time)['bid1'].iloc[0]
            order.loc[1,'pos']  = -1
            order.loc[1,'size']  = 100

        if action == 2:
            order.loc[0,'time']  = time
            order.loc[0,'side']  = 'sell'
            order.loc[0,'price'] = self._data.get_quote(time)['ask1'].iloc[0]
            order.loc[0,'pos']  = -1
            order.loc[0,'size']  = 100
        
        if action == 3:
            order.loc[0,'time']  = time
            order.loc[0,'side']  = 'buy'
            order.loc[0,'price'] = self._data.get_quote(time)['bid1'].iloc[0]
            order.loc[0,'pos']  = -1
            order.loc[0,'size']  = 100

        if action == 4:
            order.loc[0,'time']  = time
            order.loc[0,'side']  = 'sell'
            order.loc[0,'price'] = self._data.get_quote(time)['bid1'].iloc[0]
            order.loc[0,'pos']  = -1
            order.loc[0,'size']  = 100

        if action == 5:
            order.loc[0,'time']  = time
            order.loc[0,'side']  = 'buy'
            order.loc[0,'price'] = self._data.get_quote(time)['ask1'].iloc[0]
            order.loc[0,'pos']  = -1
            order.loc[0,'size']  = 100       

        return order    

    def step(self, action,money,position,resorder):
        

        self._position=position
        self._money=money
        self._resorder=resorder
        if action==1 or action==4 or action==5 :
            self._resorder.drop(self._resorder.index,inplace=True)
        self._order = self._resorder  
        self._order = pd.concat([self._order,self._action2order(action)],axis=0,ignore_index=True) 
        #print(self._order)
        self._resorder.drop(self._resorder.index,inplace=True)
        #print(self._resorder)
        #time=int(self._data._quote.loc[self.number,'time'])
        buy_num=0
        sell_num=0
        self._reward=0
        i=0
        while i<len(self._order):
            self._order1 = self._order.loc[i].to_dict()
            self._order11, filled = self._engine(self._order1)
            #print(self._order11,filled)
            if self._order11.get('size') !=0:
                append1={'time':self._order11.get('time'),'side':self._order11.get('side'),'price':self._order11.get('price'),'size':self._order11.get('size'),'pos':self._order11.get('pos')}
                self._resorder=self._resorder.append(append1,ignore_index=True)
                #print(self._resorder)
            if filled.get('size') ==[100]:
                if self._order1.get('side')=='sell':
                    sell_num=sell_num+1
                elif  self._order1.get('side')=='buy':
                    buy_num=buy_num+1 
            if buy_num==1 and sell_num==1:
                self._reward=self._reward+1   

            if filled.get('time')!=[] and filled.get('size')!=[0]:
                if self._order11['side']=='buy':
                    self._money=self._money-filled['price'][0]*filled['size'][0]
                    self._position=self._position+filled['size'][0]
                else:
                    self._money=self._money+filled['price'][0]*filled['size'][0]
                    self._position=self._position-filled['size'][0]              
            i=i+1 
        
        if self.number+2==len(self._data._quote):
            self._final=True   
        state = None if self._final else self._state()  #state是下一条QUOTE数据
        #print(self._position)

        return (state, self._reward, self._final,self._money,self._position,self._resorder,sell_num,buy_num)  

    