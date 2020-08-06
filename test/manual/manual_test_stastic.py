import datetime
import sys
import traceback
sys.path.append('./')
sys.path.append('./test')

import pandas as pd
import matplotlib.pyplot as plt
from utils.dataloader import load_tickdata, load_case
from datasource.datatype import TickData
from utils.statastic import *

quote, trade = load_tickdata(stock='000001', date='20140704')
data = TickData(quote, trade)
trade = data.get_trade()
time = [34200000, 41400000, 46800000, 54000000]
volume_per_30min = group_trade_volume_by_time(trade, time, 1800000)
print(volume_per_30min)
'''
volume_total = sum(volume_per_5sec['volume'])
ratio = [v / volume_total for v in volume_per_5sec['volume']]
plt.plot(list(range(len(ratio))), ratio)
plt.show()
'''