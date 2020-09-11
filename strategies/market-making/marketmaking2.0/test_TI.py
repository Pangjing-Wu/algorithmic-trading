import numpy as np
import pandas as pd
import json
import sys
import os
#sys.path.append()
sys.path.append('D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master\\datasource')
sys.path.append('D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master')
sys.path.append('D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\exchange')
'''用前五天的数据训练当天的up和down'''
from source.h2 import H2Connection
from datatype.tickdata import TickData
from TI import TI

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

list_date=[20140605,20140606,20140609,20140610,20140611,20140612,20140613,20140616,20140617,20140618,20140619,20140620,20140623,20140624,20140625,20140626,20140627]
list_id=[600000]#,600637,600690,600703,600832,600837,600887,600999,601006,601088,601118,601166,601169,601288,601299,601318,601328,601398,601601,601628,601668,601688,601766,601818,601857,601901,601989,601998]
while i<len(list_date):
    date=list_date[i]
    print(date)
    your_db_dir='F:\\融创课题\\数据百度云\\201406(2)\\%s'%(date)
    tempdir = 'D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\marketmaking2.0\\test\\' + str(date)
    os.makedirs(tempdir, exist_ok=True)
    i2 = 0
    while i2<len(list_id):
        id=str(list_id[i2])
        quote,trade = load(id, your_db_dir, user, password)
        quote = quote.apply(pd.to_numeric, errors='ignore')
        trade = trade.apply(pd.to_numeric, errors='ignore')
        key = quote[quote.time.duplicated(False)]
        k = []
        for ii in key.index:
            if (ii % 2) == 0:
                k.append(ii)
        print(k)
        quote.drop(quote.index[k], inplace=True)
        quote.reset_index(drop=True, inplace=True)
        data = TickData(quote, trade)
        TI2=TI(quote=quote,trade=trade,data=data)
        TI_csv = TI2.BUILDTI()
        TI_csv.to_csv(os.path.join(tempdir, str(id) + 'TI.csv'), index=False)
        i2=i2+1
    i = i + 1


#td.to_csv(r'D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\quote.csv')