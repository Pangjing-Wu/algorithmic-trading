import datetime
import sys
import traceback
sys.path.append('./')
sys.path.append('./test')

from src.datasource.datatype import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.env import AlgorithmicTradingEnv
from utils.dataloader import load_tickdata, load_case


def test_env_step(env, params, reportdir):
    with open(reportdir, 'a') as f:
        f.write('==========================\n')
        f.write('%s\n' % datetime.datetime.now())
        try:
            _ = env.reset()
            for p in params:
                # p[0] is params, p[1] is excepted.
                action = p[0]['action']
                f.write('param : %s\n' %  action)
                next_s, reward, final, info = env.step(action)
                f.write('-- OUTPUT --\n')
                f.write('next state: %s\n' % next_s)
                f.write('reward: %s\n' % reward)
                f.write('environment terminated signal: %s\n' % final)
                f.write('transaction infomation:\n%s\n' % info)
        except Exception:
            f.write('[ERRO]: an exception occurs:\n%s\n' % traceback.format_exc())


if __name__ == '__main__':
    quote, trade = load_tickdata(stock='000001', time='20140704')
    data = TickData(quote, trade)
    exchange = GeneralExchange(data, 3)
    env = AlgorithmicTradingEnv(
        tickdata=data,
        transaction_engine=exchange.transaction_engine,
        total_volume=100000,
        reward_function='vwap',
        max_level=10
        )

    _, params = load_case('actions.txt')

    reportdir = 'test/results/manual_test_env.txt'
    test_env_step(env, params, reportdir)