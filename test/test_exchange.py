import json
import sys
sys.path.append('./')
sys.path.append('./test')

import pytest

from src.datasource.datatype import TickData
from src.exchange.stock import GeneralExchange
from utils.dataloader import load_tickdata, load_case


cases, params = load_case('orders.txt')


@pytest.fixture(scope='class')
def exchange():
    data_config = json.load(open('test/config/data.json', 'r'))
    quote, trade = load_tickdata(data_config['stock'], data_config['date'])
    data = TickData(quote, trade)
    return GeneralExchange(data, 3)


class TestGeneralExchange(object):

    @pytest.mark.parametrize('params,excepted', params, ids=cases)
    def test_transaction_engine(self, exchange, params, excepted):
        order = params['order']
        order, trade = exchange.transaction_engine(order)
        assert order == excepted['order']
        assert trade == excepted['trade']