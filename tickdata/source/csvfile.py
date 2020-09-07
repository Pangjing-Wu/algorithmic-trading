import glob
import os
import sys

import pandas as pd
import psycopg2

sys.path.append('./')
from data.source.h2 import load_from_h2


def h2_to_csv(stocklistdir, h2dir, savedir, user, psw, **h2_kwargs):
    with open(stocklistdir, 'r') as f:
        stocks = [s.rstrip('\n') for s in f.readlines()]
    h2filedirs = glob.glob(os.path.join(h2dir, '*.h2.db'))
    h2fliedirs = [h2dir.rstrip('.h2.db') for h2dir in h2filedirs]
    for h2filedir in h2fliedirs:
        date = os.path.basename(h2filedir)
        for stock in stocks:
            quotedir = os.path.join(savedir, stock, 'quote')
            tradedir = os.path.join(savedir, stock, 'trade')
            os.makedirs(quotedir, exist_ok=True)
            os.makedirs(tradedir, exist_ok=True)
            try:
                quote, trade = load_from_h2(stock, h2filedir, user, psw, **h2_kwargs)
            except psycopg2.ProgrammingError as e:
                print(e)
                continue
            else:
                quote.to_csv(os.path.join(quotedir, date+'.csv'), index=False)
                trade.to_csv(os.path.join(tradedir, date+'.csv'), index=False)


if __name__ == "__main__":
    params = dict(
        stocklistdir='./stocklist',
        h2dir='/data/al2',
        savedir='/data/al2/csv',
        user='cra001',
        psw='cra001',
        port='8082'
    )
    h2_to_csv(**params)