import sys
sys.path.append('./')
sys.path.append('./test')

from utils.dataloader import load_tickdata, load_case
from src.datasource.datatype import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.vwap.env import VWAPEnv
from src.strategies.vwap.agent import Baseline

quote, trade = load_tickdata(stock='000001', time='20140704')
data = TickData(quote, trade)
trade = data.get_trade()
time = [34200000, 41400000, 46800000, 53700000]
level_space = ['bid1', 'ask1']
exchange = GeneralExchange(data, 3)

env = VWAPEnv(data, 200000, time, 1200000, trade, exchange.transaction_engine, level_space)

agent = Baseline(side='sell', threshold=0.1)

s = env.reset()
final = env.is_final()


with open('test/results/manual_test_vwap.txt', 'w') as f:
    
    f.write('tasks:\n%s\n' % env.subtasks)

    while not final:

        state_dict = {}
        keys = ['time', 'start', 'end', 'goal', 'filled']
        for k, i in zip(keys, s):
            state_dict[k] = i
        f.write('state: %s\n' % state_dict)
        f.write('vwap: %.3f, market vwap: %.3f.\n' % (env.vwap, env.market_vwap))
        a = agent.action(s)
        s, final = env.step(a)
        
    f.write('tasks:\n%s\n' % env.subtasks)
    f.write('vwap: %.3f\n' % env.vwap)