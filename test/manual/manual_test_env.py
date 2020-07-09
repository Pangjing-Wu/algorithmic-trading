import sys
sys.path.append('./')

from src.datasource.datatype import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.env import AlgorithmicTradingEnv
from test.utils.dataloader import load_csv, load_case


quote, trade = load_csv(stock='000001', time='20140704')
data = TickData(quote, trade)

exchange = GeneralExchange(data)

env = AlgorithmicTradingEnv(
    tickdata=data,
    transaction_engine=exchange.transaction_engine,
    total_volume=100000,
    reward_function='vwap',
    max_level=10
    )

s0 = env.reset()
a = (1, 0, 100)
env.step(a)
env.step(a)
next_s, r, signal, info = env.step(a)
print(next_s)
print(r)
print(signal)
print(info)