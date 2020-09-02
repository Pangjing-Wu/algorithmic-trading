import numpy as np
import pandas as pd
import json
import sys
sys.path.append('D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\datasource')
sys.path.append('D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\test\\utils')
print(sys.path)
from datatype.tickdata import TickData
#from exchange.stock import GeneralExchange
from dataloader import load_tickdata, load_case

quote, trade = load_tickdata(stock='000001',date='20140704')
data = TickData(quote, trade)
#print(data.get_quote(34200000))
#print(data.get_quote(34210000))
#print(data.get_trade_between(34200000,34200500))
#trade.to_csv(r'D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\traded.csv')
'''时间：9：30--11：30（34200--41400） 13：00--15：00 （46800--54000）
以10分钟为单位计算技术指标 
变动率指标ROC_1=（最近quote价格-1个时间间隔前的quote价）/1个时间间隔前的quote价
标准差SD(最近10个quote价格)
心理线PSL=N日内的上涨次数/N×100%
相对强弱指标N日RSI =N日内收盘涨幅的平均值/(N日内收盘涨幅均值+N日内收盘跌幅均值) ×100
IV=上一个间隔是un“Up”=1还是“Down”=0(布尔值)
'''
#dIndex = quote.duplicated('time')
# quote = quote.drop_duplicates()
key=quote[quote.time.duplicated(False)]
print(key.index)
k=[]
for i in key.index:
    if (i % 2) == 0:
        k.append(i)
print(k)
quote.drop(quote.index[k], inplace=True)
quote.reset_index(drop=True, inplace=True)

def time(st,et):
    quote_time = quote[(quote['time'] >= st)&(quote['time'] <= et)]
    if len(quote_time)<1:
        start_quote_time = 0
        end_quote_time = 0
    else:
    #print(quote_time)
       start_quote_time = quote_time.iloc[0]['time']
       end_quote_time =quote_time.iloc[-1]['time']
    return int(start_quote_time),int(end_quote_time)

def ROC(end_quote_time,n):#变动率指标ROC_1=（最近quote价格-1个时间间隔前的quote价）/1个时间间隔前的quote价
    df=pd.DataFrame(quote)
    price=float((data.get_quote(end_quote_time)['bid1']+data.get_quote(end_quote_time)['ask1'])/2)
    index = int(df[df.time == end_quote_time].index.tolist()[0])
    k=index-n
    if k>=0:
        price2=float((quote.loc[k]['bid1']+quote.loc[k]['ask1'])/2)
        if price2==0:
            roc=0
        else:
            roc=(price-price2)/price2
    else:
        roc=0
    return(roc)

def average_price(st,et):
    quote2 = quote[(quote['time'] >= st) & (quote['time'] <= et)][['bid1','ask1']]
    df=pd.DataFrame(quote2)
    average=df.mean(axis=1)
    return average

def SD(st,et):
    average=average_price(st,et)
    sd=np.var(average)
    return sd

def PSL(st,et):
    average = average_price(st, et)
    dex=average.index
    n=0
    m=len(dex)
    for i in dex:
        if i<dex[-1]:
           if average[i+1]>average[i]:
               n=n+1
    if (n==0):
        psl=0
    else:
        psl=n/m
    return psl

def updown(st,et):
    average = average_price(st, et)
    dex = average.index
    if len(dex)<=2:
        updown=0
    else:
        if (average[dex[-1]])>(average[dex[-2]]):
           updown=1
        else:
            if (average[dex[-1]])==(average[dex[-2]]):
                updown=0
            else:
                updown=-1

    return updown

TI=pd.DataFrame(columns=('start_time','end_time','price','ROC_1','ROC_2','ROC_3','ROC_4','ROC_5', 'SD', 'PSL','IV'))
i=0;
while  i<1440:
    #计算时间
    if i<720:
        st=34200000+i*10000
        et=34200000+(i+1)*10000
    else :
        st=46800000+(i-720)*10000
        et = 46800000 + (i-719) * 10000
    print(st,et)
    TI=TI.append({'start_time':st,'end_time':et},ignore_index=True)
    start_quote_time,end_quote_time=time(st,et)
    print(start_quote_time,end_quote_time)
    if (start_quote_time!=0):
       average = average_price(st, et)
       TI['price'][i] = np.mean(average)
#变动率指标ROC_1=（最近quote价格-1个时间间隔前的quote价）/1个时间间隔前的quote价
       TI['ROC_1'][i]=ROC(end_quote_time,1)
       TI['ROC_2'][i] = ROC(end_quote_time, 2)
       TI['ROC_3'][i] = ROC(end_quote_time, 3)
       TI['ROC_4'][i] = ROC(end_quote_time, 4)
       TI['ROC_5'][i] = ROC(end_quote_time, 5)

#标准差SD(最近时间段内quote价格的标准差)
       TI['SD'][i] = SD(st,et)

#心理线PSL=N日内的上涨次数/N×100%
       TI['PSL'][i]=PSL(st,et)

#IV=上一个间隔是un“Up”=1还是“Down”=0(布尔值)
       TI['IV'][i]=updown(st,et)
    else:
        TI['price'][i] = 0
        TI['ROC_1'][i] = 0
        TI['ROC_2'][i] = 0
        TI['ROC_3'][i] = 0
        TI['ROC_4'][i] = 0
        TI['ROC_5'][i] = 0
        TI['SD'][i] = 0
        TI['PSL'][i] = 0
        TI['IV'][i] = 0

    i=i+1

TI.to_csv(r'D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\TI.csv')