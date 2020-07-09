import datetime
import sys
import traceback
sys.path.append('.')
sys.path.append('./test')

from src.datasource.datatype import TickData
from src.exchange.stock import GeneralExchange
from utils.dataloader import load_tickdata, load_case

def test_transaction(transaction_engine, params, reportdir):
    with open(reportdir, 'a') as f:
        f.write('==========================\n')
        f.write('%s\n' % datetime.datetime.now())
        for p in params:
            try:
                param = p['order']
                f.write('param : %s\n' %  param)
                quote_board = data.quote_board(quote)
                f.write('current quote board:\n%s\n' % quote_board)
                f.write('current order:\n%s\n' % param)
                order, traded = transaction_engine(param)
                f.write('-- OUTPUT --\n')
                f.write('order:\n%s\ntraded:\n%s\n\n' % (order, traded))
            except Exception:
                f.write('[ERRO]: an exception occurs:\n%s\n' % traceback.format_exc())


if __name__ == '__main__':
    quote, trade = load_tickdata(stock='000001', time='20140704')
    data = TickData(quote, trade)
    exchange = GeneralExchange(data)

    _, params = load_case('orders.txt')
    
    reportdir = 'test/results/manual_test_transaction.txt'
    test_transaction(exchange.transaction_engine, params, reportdir)
