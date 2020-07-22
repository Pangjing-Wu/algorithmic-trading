import json
import sys
sys.path.append('./')
sys.path.append('./test')

import pytest

from src.datasource.datatype import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.env.buyer_seller import AlgorithmicTradingEnv
from utils.dataloader import load_tickdata, load_case


cases, params = load_case('actions.txt')


@pytest.fixture(scope='class')
def env():
    data_config   = json.load(open('test/config/data.json', 'r'))
    agent_config  = json.load(open('test/config/agent.json', 'r'))['agent']
    quote, trade = load_tickdata(data_config['stock'], data_config['time'])
    data = TickData(quote, trade)
    exchange = GeneralExchange(data, 3)
    trader = AlgorithmicTradingEnv(
        tickdata=data,
        transaction_engine=exchange.transaction_engine,
        total_volume=agent_config['volume'],
        reward_function=agent_config['reward'],
        max_level=agent_config['max_level']
        )
    _ = trader.reset()
    return trader


class TestAlgorithmicTradingEnv(object):

    @pytest.mark.parametrize('params,excepted', params, ids=cases)
    def test_step(self, env, params, excepted):
        action = params['action']
        next_s, r, final, _ = env.step(action)
        assert all(next_s == excepted['next_s'])
        assert r == excepted['r']
        assert final == excepted['final']