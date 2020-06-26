import sys
sys.path.append('./')
import time

from tickdata import dataset
from transaction import trasaction_matching

import pandas as pd

stock = '000001'
dbdir = '~/OneDrive/python-programs/reinforcement-learning/data/20140704'
user = 'cra001'
password = 'cra001'

td = dataset(stock, dbdir, user, password)

# black box case test.
with open('test/case/orders.txt', 'r') as f:
    cases = f.readlines()

for i, case in enumerate(cases):
    print('case %d: %s' % (i, case))
    t = iter(td.quote_timeseries)
    quote = td.get_quote(next(t))
    trade = td.get_trade_between(quote)
    quote_board = td.quote_board(quote)
    print('current quote board:\n%s' % quote_board)
    if eval(case) != None:
        order = eval(case)
    print('current order:\n%s' % order)
    order, traded = trasaction_matching(quote=quote_board, trade=trade, simulated_order=order)
    print('\nEXECUTING TRANSACTION MATCHING\n')
    print('order:\n%s\ntraded:\n%s\n' % (order, traded))

