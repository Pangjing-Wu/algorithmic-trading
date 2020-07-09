import datetime
import sys
sys.path.append('.')
sys.path.append('./test')

from src.datasource.datatype import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.env import AlgorithmicTradingEnv
from utils.dataloader import load_tickdata, load_case

def test_transaction(data, params, transaction_engine, reportdir):
    with open(reportdir, 'a') as f:
        f.write('==========================\n')
        f.write('%s\n' % datetime.datetime.now())
        t = iter(data.quote_timeseries)
        for p in params:
            try:
                param = p['order']
                f.write('param : %s\n' %  param)
                quote = data.get_quote(next(t))
                trade = data.get_trade_between(quote)
                trade = data.trade_sum(trade)
                quote_board = data.quote_board(quote)
                f.write('current quote board:\n%s\n' % quote_board)
                f.write('current order:\n%s\n' % param)
                order, traded = transaction_engine(param)
                f.write('-- EXECUTING TRANSACTION MATCHING --\n')
                f.write('order:\n%s\ntraded:\n%s\n\n' % (order, traded))
            except Exception as e:
                f.write('[ERRO]: an exception occurs:%s\n\n' % e)


if __name__ == '__main__':
    quote, trade = load_tickdata(stock='000001', time='20140704')
    data = TickData(quote, trade)
    exchange = GeneralExchange(data)

    case, params = load_case('orders.txt')
    
    reportdir = 'test/results/test_transaction.txt'
    test_transaction(data, params, exchange.transaction_engine, reportdir)
