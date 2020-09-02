import glob
import os
import sys

import pandas as pd

sys.path.append('./')
from data.source.h2 import load_from_h2


def h2_to_csv(stockdir, filedir, savedir, user, psw, **h2_kwargs):
    with open(stockdir, 'r') as f:
        stocks = f.readlines()
    h2dirs = glob.glob(os.path.join(filedir, '*.h2.db'))
    h2dirs = [h2dir[:-6] for h2dir in h2dirs]
    for h2dir in h2dirs:
        date = os.path.basename(h2dir)
        tempdir = os.path.join(savedir, date)
        os.makedirs(tempdir, exist_ok=True)
        for stock in stocks:
            quote, trade = load_from_h2(stock, h2dir, user, psw, **h2_kwargs)
            quote.to_csv(os.path.join(tempdir, stock + '_quote.csv'), index=False)
            trade.to_csv(os.path.join(tempdir, stock + '_trade.csv'), index=False)