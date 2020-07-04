import datetime
import getpass
import sys
sys.path.append('./')

import pandas as pd

from tickdata import TickData
from utils.transaction import transaction_matching


def test_transaction(td, cases, reportdir):
    with open(reportdir, 'a') as f:
        f.write('==========================\n')
        f.write('%s\n' % datetime.datetime.now())
        f.write('user: %s\n' % getpass.getuser())
        t = iter(td.quote_timeseries)
        for i, case in enumerate(cases):
            try:
                f.write('case %d: %s' % (i, case))
                quote = td.get_quote(next(t))
                trade = td.get_trade_between(quote)
                quote_board = td.quote_board(quote)
                f.write('current quote board:\n%s\n' % quote_board)
                if eval(case) != None:
                    order = eval(case)
                f.write('current order:\n%s\n' % order)
                order, traded = transaction_matching(quote_board, trade, order)
                f.write('-- EXECUTING TRANSACTION MATCHING --\n')
                f.write('order:\n%s\ntraded:\n%s\n\n' % (order, traded))
            except Exception as e:
                f.write('[ERRO]: an exception occurs:%s\n\n' % e)


if __name__ == '__main__':
    stock = '000001'
    dbdir = '~/OneDrive/python-programs/reinforcement-learning/data/20140704'
    user = 'cra001'
    password = 'cra001'
    reportdir = 'test/results/test_transaction.txt'

    quote = pd.read_csv('test/data/000001-quote-20140704.csv')
    trade = pd.read_csv('test/data/000001-trade-20140704.csv')
    td = TickData(quote, trade)

    with open('test/case/orders.txt', 'r') as f:
        cases = f.readlines()
    
    test_transaction(td, cases, reportdir)