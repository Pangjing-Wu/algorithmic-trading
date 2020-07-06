import sys
sys.path.append('./')

import pandas as pd

from tickdata import TickData
from env import AlgorithmicTrading

quote = pd.read_csv('test/data/000001-quote-20140704.csv')
trade = pd.read_csv('test/data/000001-trade-20140704.csv')

td = TickData(quote, trade)
trading = AlgorithmicTrading(td, 10000, reward_function='vwap', max_level=10)
s0 = trading.reset()
a = (1, 0, 100)
trading.step(a)
trading.step(a)
next_s, r, signal, info = trading.step(a)
print(next_s)
print(r)
print(signal)
print(info)