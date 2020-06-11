import numpy as np
real_trade = np.zeros([1, 3])
real_trade2 = np.zeros([1, 3])  #'''需要提前把trade订单数据做处理，把两个QUOTE时间之间发生的统一价位的手数累加起来,时间都记录为Quote的时间,按照价格大小排,作为real_trade2'''
real_quote = np.zeros([1, 11])
simulated_trade = np.zeros([1, 3])
simulated_quote = np.zeros([1, 3])
ask = np.zeros([1,6])
bid = np.zeros([1,6])

from dataloader import TickData

class AlgorithmTrader(object):

    def __init__(self, data:TickData, strategy_direction,
        volume:int, reward_function:callable or str,
        *args:'[arguments to describe environment]',):
        pass

    def reset(self):                                #如果以一天为一个训练单位，每天重复
        # TODO reset invironment.
        global n
        global simulated_quote
        global simulated_trade
        n=0                                        #第几步变到第0步
        simulated_trade = np.zeros([1, 3])         #simulated book 清空
        simulated_quote = np.zeros([1, 3])
        pass

    def step(self, action)->'(next_s, reward, signal, info)':
        # TODO environment step:
        # Issue an order;t_n为现在的时刻 action[n-1][2]
        global n
        global simulated_quote
        t_n=real_quote[n][0]
        m=np.size(simulated_quote ,0)
        if action[n][1]!=0:
            action2=np.arry([t_n,action[n][0],action[n][1]])
            simulated_quote=np.insert(simulated_quote ,m,values=action2,axis=0)           #加一行并记录t_n和action[0][n]和action[1][n]
        # Matching transactions and Refresh the real/simulated order book;
        self._transaction_matching()
        # Calculate reward;计算奖励
        # Give a final signal;
        m3= np.size(real_quote,0)
        if n+2>m3:
            signal=1
        else:
            signal=0
        # go to next step.
        info='在时间{tn}成交了{a1}手价格为{a2}的股票，剩余提交订单为{b1}手价格为{b2}的股票。'.format(tn=t_n,a1=a1,a2=a2,b1=b1,b2=b2)
        if signal==1:
            self.reset()                             #调用rest函数
            self.step()                              #nextstep函数
        else:
            n=n+1
            self.step()



    def _transaction_matching(self, action,args)->'(a1, a2, b1, b2)':  
        global simulated_quote
        global simulated_trade
        m = np.size(simulated_quote, 0)
        m2= np.size(simulated_trade,0)
        if simulated_quote[m-1][2]!=0:                                                       #说明有订单在等待成交
             if  simulated_quote[m-1][2]>0:
                 if simulated_quote[m-1][1]>=ask[1]:                                         #如果买的价格高于现在市场上卖的最低价
                     if simulated_quote[m-1][2]<=real_quote[n][4]:                           #并且ASK1的量足够
                         OK_signal=1                                                         #直接全部成交
                     else:
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

    def _vwap(self, price, volumn):
        pass

    def _twap(self, price, volumn):
        pass
