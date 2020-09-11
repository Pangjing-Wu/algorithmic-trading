import numpy as np
import pandas as pd
import json
import sys
import os
#sys.path.append()
sys.path.append('D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master\\datasource')
sys.path.append('D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master')
sys.path.append('D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\exchange')
'''用前3天的数据训练当天的up和down'''
from source.h2 import H2Connection
from datatype.tickdata import TickData
from TI import TI
from svm2 import SVM
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



i=3

list_date=[20140603,20140604,20140605,20140606,20140609,20140610,20140611,20140612,20140613,20140616,20140617,20140618,20140619,20140620,20140623,20140624,20140625,20140626,20140627]
list_id=600000#,600637,600690,600703,600832,600837,600887,600999,601006,601088,601118,601166,601169,601288,601299,601318,601328,601398,601601,601628,601668,601688,601766,601818,601857,601901,601989,601998]
while i<len(list_date):
    date1=list_date[i-3]
    date2 = list_date[i - 2]
    date3 = list_date[i - 1]
    date4=list_date[i]
    tempdir1 = 'D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\marketmaking2.0\\test\\' + str(date1)
    tempdir2 = 'D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\marketmaking2.0\\test\\' + str(
        date2)
    tempdir3 = 'D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\marketmaking2.0\\test\\' + str(
        date3)
    tempdir4 = 'D:\\python\\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\marketmaking2.0\\test\\' + str(
        date4)
    csv1=pd.read_csv(os.path.join(tempdir1, str(list_id) + 'TI.csv'))
    csv2 = pd.read_csv(os.path.join(tempdir2, str(list_id) + 'TI.csv'))
    csv3 = pd.read_csv(os.path.join(tempdir3, str(list_id) + 'TI.csv'))
    test_csv = pd.read_csv(os.path.join(tempdir4, str(list_id) + 'TI.csv'))
    svm=SVM(csv1=csv1,csv2=csv2,csv3=csv3)
    (svmresult,rate)=svm.svm3days(test_csv=test_csv)
    svmresult['rate']=rate
    svmresult.to_csv(os.path.join(tempdir4, str(list_id) + 'svm2.csv'), index=False)
    i = i + 1


#td.to_csv(r'D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\quote.csv')