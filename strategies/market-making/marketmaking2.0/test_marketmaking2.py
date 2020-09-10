import numpy as np
import pandas as pd
import json
import sys
import os
#sys.path.append()
sys.path.append('D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master\\datasource')
sys.path.append('D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master')
sys.path.append('D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\exchange')
from source.h2 import H2Connection
from datatype.tickdata import TickData
#from exchange.stock import GeneralExchange
from marketmaking2 import Marketmaking
from stock import GeneralExchange

def load(stock, dbdir, user, psw)->TickData:
    h2 = H2Connection(dbdir, user, psw)
    QUOTE_COLS = ["time", "bid1", "bsize1", "ask1", "asize1", "bid2", "bsize2", "ask2",
        "asize2", "bid3", "bsize3", "ask3", "asize3", "bid4", "bsize4", "ask4", "asize4",
        "bid5", "bsize5", "ask5", "asize5", "bid6", "bsize6", "ask6", "asize6", "bid7",
        "bsize7", "ask7", "asize7", "bid8", "bsize8", "ask8", "asize8", "bid9", "bsize9",
        "ask9", "asize9", "bid10", "bsize10", "ask10", "asize10"]
    TRADE_COLS = ["time", "price", "size"]
    TIMES = [34200000, 41400000, 46800000, 54000000]
    if h2.status:
        sql = "select %s from %s where time between %s and %s or time between %s and %s"
        quote = h2.query(sql % (','.join(QUOTE_COLS), 'quote_' + stock, *TIMES))
        trade = h2.query(sql % (','.join(TRADE_COLS), 'trade_' + stock, *TIMES))
        quote.columns = QUOTE_COLS
        trade.columns = TRADE_COLS
    else:
        raise ConnectionError("cannot connect to H2 service, please strat H2 service first.")
    return quote, trade


user='cra001'
password='cra001'
i=0

list_date=[20140609,20140610,20140611,20140612]#,20140613,20140616,20140617,20140618,20140619,20140620,20140623,20140624,20140625,20140626,20140627,20140630]
list_id=[600000]#,600010,600015,600016,600018,600028,600030,600036,600048,600050,600089,600104,600109,600111,600150,600196,600256,600332,600372,600406,600518,600519,600585]#,600637,600690,600703,600832,600837,600887,600999,601006,601088,601118,601166,601169,601288,601299,601318,601328,601398,601601,601628,601668,601688,601766,601818,601857,601901,601989,601998]
while i<len(list_date):
    date=list_date[i]
    print(date)
    your_db_dir='F:\\融创课题\\数据百度云\\201406(2)\\%s'%(date)
    tempdir = 'D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\marketmaking2.0\\test\\' + str(date)
    i2 = 0
    while i2<len(list_id):
        id=str(list_id[i2])
        quote,trade = load(id, your_db_dir, user, password)
        quote = quote.apply(pd.to_numeric, errors='ignore')
        trade = trade.apply(pd.to_numeric, errors='ignore')
        data = TickData(quote, trade)
        exchange = GeneralExchange(data, 3)
        csvplace = 'D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\marketmaking2.0\\test\\%s\\%s' % (date,id)+'svm.csv'
        print(csvplace)
        svm = pd.read_csv(csvplace)
        svm = svm[['start_time', 'end_time', 'predictForUp']]
        svm.reset_index(drop=True, inplace=True)
        MM = Marketmaking(tikedata=data, transaction_engine=exchange.transaction_engine,svm=svm)
        traded_csv = MM.makeorder()
        n = 0
        m = 0
        for index in range(0, len(traded_csv)):
            if traded_csv.loc[index, 'side'] == 'buy':
                traded_csv.loc[index, 'money'] = 0 - traded_csv.loc[index, 'price']
                n = n + 1
            elif traded_csv.loc[index, 'side'] == 'sell':
                traded_csv.loc[index, 'money'] = traded_csv.loc[index, 'price']
                m = m + 1
        k = traded_csv['money'].sum()
        print(k)
        if n == m:
            traded_csv.loc[0, 'sum'] = traded_csv['money'].sum()
        elif n > m:
            traded_csv.loc[0, 'sum'] = 'wrong' + traded_csv['money'].sum() + traded_csv.loc[len(traded_csv) - 1, 'price']
        elif n < m:
            traded_csv.loc[0, 'sum'] = 'wrong' + traded_csv['money'].sum() - traded_csv.loc[len(traded_csv) - 1, 'price']
        traded_csv.to_csv(os.path.join(tempdir,str(id)+'.csv'), index=False)
        i2=i2+1
    i = i + 1