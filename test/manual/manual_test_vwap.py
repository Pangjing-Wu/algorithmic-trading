import sys
sys.path.append('./')
sys.path.append('./test')

from utils.dataloader import load_tickdata, load_case
from src.datasource.datatype.tickdata import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.vwap.env import GenerateTranches
from src.strategies.vwap.agent import Baseline
from src.utils.statistic import group_trade_volume_by_time


def state2dict(state) -> dict:
    keys = ['time', 'start', 'end', 'goal', 'filled']
    state_dict = {}
    for k, i in zip(keys, state):
        state_dict[k] = i
    return state_dict


quote, trade = load_tickdata(stock='000001', time='20140704')
data = TickData(quote, trade)
trade = data.get_trade()
time = [34200000, 41400000, 46800000, 53700000]
params = dict(
    tickdata = data,
    level_space = ['bid1', 'ask1'],
    transaction_engine = GeneralExchange(data, 3).transaction_engine,
)

volume_profile = group_trade_volume_by_time(trade, time, 1800000)

envs  = GenerateTranches(200000, volume_profile, **params)
agent = Baseline(side='sell', threshold=0.1)

with open('test/results/manual_test_vwap.txt', 'w') as f:
    
    for env in envs:

        s = env.reset()
        final = env.is_final()

        f.write('state: %s\n' % state2dict(s))

        while not final:
            a = agent.action(s)
            s, r, final = env.step(a)
            f.write('state: %s\n' % state2dict(s))
            f.write('action: %s\n' % a)
            f.write('reward: %s\n' % r)
            f.write('vwap: %.3f, market vwap: %.3f.\n' % (env.vwap, env.market_vwap))