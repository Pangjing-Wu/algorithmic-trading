import sys
sys.path.append('./src')

from datasource.datatype import TickData
from datasource.source.h2 import load
from exchange.stock import GeneralExchange
from strategies.env import AlgorithmicTradingEnv


data = load('000001', 'dbdir', 'cra001', 'cra001')

exchange = GeneralExchange(data)

env = AlgorithmicTradingEnv(
    tickdata=data,
    transaction_engine=exchange.transaction_engine,
    total_volume=100000,
    reward_function='vwap',
    max_level=10
    )

s0 = env.reset()
