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
    h2files = glob.glob(os.path.join(h2dir, '*.h2.db'))
    h2flies = [h2dir.rstrip('.h2.db') for h2dir in h2files]
    for h2file in h2flies:
        date = os.path.basename(h2file)
        for stock in stocks:
            quotedir = os.path.join(savedir, stock, 'quote')
            tradedir = os.path.join(savedir, stock, 'trade')
            os.makedirs(quotedir, exist_ok=True)
            os.makedirs(tradedir, exist_ok=True)
            try:
                quote, trade = load_from_h2(stock, h2file, user, psw, **h2_kwargs)
            except psycopg2.ProgrammingError as e:
                print(e)
                continue
            except Exception as e:
                print(e)
                continue
            else:
                quote.to_csv(os.path.join(quotedir, date+'.csv'), index=False)
                trade.to_csv(os.path.join(tradedir, date+'.csv'), index=False)


if __name__ == "__main__":
    h2_to_csv('./stocklist', '/Volumes/data/201408', '/Volumes/data/processed', 'cra001', 'cra001')