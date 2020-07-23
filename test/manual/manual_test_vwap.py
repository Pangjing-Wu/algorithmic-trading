import sys
sys.path.append('./')
sys.path.append('./test')

from utils.dataloader import load_tickdata, load_case
from src.datasource.datatype import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.classic.vwap import VWAPAgent, VWAPEnv

quote, trade = load_tickdata(stock='000001', time='20140704')
data = TickData(quote, trade)
trade = data.get_trade()
time = [34200000, 41400000, 46800000, 53700000]

exchange = GeneralExchange(data, 3)

env = VWAPEnv(data, 2000, time, 1200000, trade, exchange.transaction_engine)

agent = VWAPAgent(side='sell', threshold=0.1)

s = env.reset()
final = env.is_final()


with open('test/results/manual_test_vwap.txt', 'w') as f:
    while not final:
        f.write('schedual ratio: %.3f\n' % s)
        a = agent.action(s)
        s, final, info = env.step(a)
        f.write(info)

    f.write('tasks:\n%s\n' % env.subtasks)
    f.write('vwap: %.3f\n' % env.vwap)