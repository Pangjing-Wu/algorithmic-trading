import json
import os
import subprocess
import time

import pandas as pd
import psycopg2

from dataloader import H2Connection


def sql_pattern(stock:str or int, quote_cols:str or list or tuple,
                trade_cols:str or list or tuple, *times:list or tuple)->str:
    if type(quote_cols) in (list, tuple):
        quote_cols = ', '.join(quote_cols)
    if type(trade_cols) in (list, tuple):
        trade_cols = ', '.join(trade_cols)
    sql = \
    'select %s from quote_%s where time between %s and %s or time between %s and %s\n' \
    'union\n' \
    'select %s from trade_%s where time between %s and %s or time between %s and %s\n' \
    'order by time' % (quote_cols, stock, *times, trade_cols, stock, *times)
    return sql


def tick_data_preprocessing(data: pd.DataFrame, config:dict)->pd.DataFrame:
    data.columns = config['quote_cols'][:-2] + config['trade_cols'][-2:]
    data['type'] = None
    for i in data.index:
        data.loc[i, 'type'] = 'trade' if data.loc[i, 'bid1'] == None else 'quote'
    return data


config = json.load(open('config/data.json', 'r'))
h2db = H2Connection('~/OneDrive/python-programs/reinforcement-learning/data/20140704', 'cra001', 'cra001')

if h2db.status:
    sql = sql_pattern('000001', config['quote_cols'], config['trade_cols'], *config['trading_time'])
    data = h2db.query(sql)
    data = tick_data_preprocessing(data, config)
    print(data)