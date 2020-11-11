import glob
import os
from typing import List

import pandas as pd

from ..type.tickdata import TickData


class CSVDataset(object):
    """
    NOTE: csv dataset should be structed as:
        [stockcode] -> quote, trade -> [date].csv
        e.g.: 600000/quote/20150301.csv
              600000/trade/20150301.csv
    """

    def __init__(self, path:str, stock:str):
        self.__rootpath = os.path.join(path, stock)
        self.__detect_data()
        self.__check_data()
        self.__i = None
        self.__n = len(self.__quotefiles)
        self.__date = [os.path.basename(s).rstrip('.csv') for s in self.__quotefiles]

    def __len__(self):
        return self.__n

    @property
    def date(self):
        return self.__date

    def load(self, date:str):
        self.__check_date_in_range(date)
        self.__i = self.__date.index(date)
        quote = pd.read_csv(self.__quotefiles[self.__i])
        trade = pd.read_csv(self.__tradefiles[self.__i])
        return TickData(quote, trade)

    def loads(self, dates:List[str]):
        for date in dates:
            self.__check_date_in_range(date)
            self.__i = self.__date.index(date)
            quote = pd.read_csv(self.__quotefiles[self.__i])
            trade = pd.read_csv(self.__tradefiles[self.__i])
            yield TickData(quote, trade)

    def load_pre(self, n=1, date=None)->iter:
        if date is None:
            i = self.__i
        else:
            self.__check_date_in_range(date)
            i = self.__date.index(date)
        if i - n <  0:
            raise IndexError('number of previous data out of range.')
        quotes = [pd.read_csv(self.__quotefiles[j]) for j in range(i-n, i)]
        trades = [pd.read_csv(self.__tradefiles[j]) for j in range(i-n, i)]
        for q, t in zip(quotes, trades):
            yield TickData(q, t)

    def load_next(self, n=1, date=None)->iter:
        if date is None:
            i = self.__i
        else:
            self.__check_date_in_range(date)
            i = self.__date.index(date)
        if i + n > self.__n:
            raise IndexError('number of previous data out of range.')
        quotes = [pd.read_csv(self.__quotefiles[j]) for j in range(i, i+n)]
        trades = [pd.read_csv(self.__tradefiles[j]) for j in range(i, i+n)]
        for q, t in zip(quotes, trades):
            yield TickData(q, t)

    def __detect_data(self):
        quotepath = os.path.join(self.__rootpath, 'quote')
        tradepath = os.path.join(self.__rootpath, 'trade')
        exists = [os.path.exists(p) for p in list(quotepath, tradepath)]
        if all(exists):
            self.__quotepath = quotepath
            self.__tradepath = tradepath
        else:
            raise FileNotFoundError('quote or trade floder dose not exist.')

    def __check_data(self):
        quotefiles = sorted(glob.glob(os.path.join(self.__quotepath, '*.csv')))
        tradefiles = sorted(glob.glob(os.path.join(self.__tradepath, '*.csv')))
        if len(quotefiles) == 0 or len(tradefiles) == 0:
            raise FileNotFoundError('quote or trade file dose not exist.')
        if len(quotefiles) != len(tradefiles):
            raise RuntimeError('quote and trade files does not match.')
        for quotefile, tradefile in zip(quotefiles, tradefiles):
            if quotefile != tradefile:
                raise RuntimeError('quote and trade files does not match.')
        self.__quotefiles = quotefiles
        self.__tradefiles = tradefiles

    def __check_date_in_range(self, date:str):
        if date not in self.__date:
            raise KeyError('date %s out of range.' % date)