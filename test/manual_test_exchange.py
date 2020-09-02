import datetime
import sys
import traceback
sys.path.append('./')
sys.path.append('./test')

from datasource.datatype import TickData
from exchange.stock import GeneralExchange
from utils.dataloader import load_tickdata, load_case

def test_transaction(transaction_engine, params, reportdir):
    with open(reportdir, 'w') as f:
        f.write('==========================\n')
        f.write('%s\n' % datetime.datetime.now())
        for p in params:
            try:
                # p[0] is params, p[1] is excepted.
                order = p[0]['order']
                f.write('param : %s\n' %  order)
                quote_board = data.quote_board(quote)
                f.write('current quote board:\n%s\n' % quote_board)
                f.write('current order:\n%s\n' % order)
                order, traded = transaction_engine(order)
                f.write('-- OUTPUT --\n')
                f.write('order:\n%s\ntraded:\n%s\n\n' % (order, traded))
            except Exception:
                f.write('[ERRO]: an exception occurs:\n%s\n' % traceback.format_exc())


if __name__ == '__main__':
    quote, trade = load_tickdata(stock='000001', date='20140704')
    data = TickData(quote, trade)
    exchange = GeneralExchange(data, 3)

    _, params = load_case('orders.txt')
    
    reportdir = 'test/results/manual_test_exchange.txt'
    test_transaction(exchange.transaction_engine, params, reportdir)
