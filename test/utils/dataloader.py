import json

import pandas as pd


def load_tickdata(stock, date):
    quote = pd.read_csv('test/data/tickdata/%s-quote-%s.csv' % (stock, date))
    trade = pd.read_csv('test/data/tickdata/%s-trade-%s.csv' % (stock, date))
    return quote, trade

def load_case(casename):
    filedir = 'test/data/case/' + casename
    with open(filedir, 'r') as f:
        cases = f.readlines()
    cases  = [json.loads(c) for c in cases]
    case   = [c['case'] for c in cases]
    params = [[c['params'], c['excepted']] for c in cases]
    return case, params