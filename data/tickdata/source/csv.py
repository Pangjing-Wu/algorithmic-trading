import glob
import os
from typing import List

import pandas as pd

from ..tickdata import TickData


class CSVDataset(object):
    """
    NOTE: csv dataset should be structed as:
        [stockcode] -> quote, trade -> [date].csv
        e.g.: 600000/quote/20150301.csv
              600000/trade/20150301.csv
    """

    def __init__(self, path:str, stock:str):
        self.__rootpath = os.path.join(path, stock)
        self.__quotefiles, self.__tradefiles = self.__detect_data()
        self.__n = len(self.__quotefiles)
        self.__dates = [os.path.basename(s).rstrip('.csv') for s in self.__quotefiles]

    def __len__(self):
        return self.__n

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.__load(key)
        elif isinstance(key, int):
            return self.__load(self.__dates[key])
        elif isinstance(key, list):
            return self.__loads(key)
        elif isinstance(key, slice):
            return self.__loads(self.__dates[key])
        else:
            raise TypeError("invalid key, only support index or date(s).")

    def __iter__(self):
        self.__i = -1
        return self

    def __next__(self):
        self.__i += 1
        if self.__i < self.__n:
            return self.__load(self.__dates[self.__i])
        else:
            raise StopIteration
        
    @property
    def dates(self):
        return self.__dates

    def __load(self, date:str):
        self.__check_date_in_range(date)
        i = self.__dates.index(date)
        quote = pd.read_csv(self.__quotefiles[i])
        trade = pd.read_csv(self.__tradefiles[i])
        quote = quote[~quote['time'].duplicated()]
        return TickData(quote=quote, trade=trade, date=date)

    def __loads(self, dates:List[str])->List[TickData]:
        return [self.__load(date) for date in dates]

    def __detect_data(self):
        quotepath = os.path.join(self.__rootpath, 'quote')
        tradepath = os.path.join(self.__rootpath, 'trade')
        if os.path.exists(quotepath) and os.path.exists(tradepath):
            quotefiles = sorted(glob.glob(os.path.join(quotepath, '*.csv')))
            tradefiles = sorted(glob.glob(os.path.join(tradepath, '*.csv')))
        else:
            raise FileNotFoundError('quote or trade floder dose not exist.')
        if len(quotefiles) == 0 or len(tradefiles) == 0:
            raise FileNotFoundError('quote or trade file dose not exist.')
        if len(quotefiles) != len(tradefiles):
            raise RuntimeError('quote and trade files does not match.')
        for quotefile, tradefile in zip(quotefiles, tradefiles):
            if os.path.basename(quotefile) != os.path.basename(tradefile):
                raise RuntimeError('quote and trade files does not match.')
        return quotefiles, tradefiles

    def __check_date_in_range(self, *args:List[str]):
        for date in args:
            if date not in self.__dates:
                raise KeyError('date %s out of range.' % date)

    def __check_index_in_range(self, *args:List[int]):
        for i in args:
            if i < 0 or i >= self.__len__():
                raise IndexError('index out of data lenth.')