import abc
from typing import Union, List

import numpy as np
import pandas as pd


class BaseData(abc.ABC):

    def __init__(self, data:pd.DataFrame, date:str):
        self.__data = data
        self.__date = date
        self.__timeseries = self.__data['time'].values.tolist()

    def __len__(self):
        return self.__data.shape[0]

    @property
    def date(self)->str:
        return self.__date

    @property
    def timeseries(self)->List[int]:
        return self.__timeseries

    @abc.abstractmethod
    def get(self)->pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_pre(self)->pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_next(self)->pd.DataFrame:
        pass
    

class Quote(BaseData):

    def __init__(self, data:pd.DataFrame, date:str):
        super().__init__(data, date)
        asks = self.__data.filter(like='ask').columns.values.tolist()
        bids = self.__data.filter(like='bid').columns.values.tolist()
        self.__levels = bids[::-1] + asks   # [bid10, ..., ask10]

    @property
    def levels(self):
        return self.__levels

    def get(self)->pd.DataFrame:
        pass

    def get_pre(self)->pd.DataFrame:
        pass

    def get_next(self)->pd.DataFrame:
        pass
    
    def quote_board(self, t:Union[int, pd.DataFrame])->pd.DataFrame:
        if type(t) == int:
            quote = self.__data[self.__data['time'] == t]
        elif type(t) == pd.DataFrame:
            quote = t
        else:
            raise TypeError("argument t must be int or pd.DataFrame.")
        size_tags = [self.__level2size(l) for l in self.__levels]
        tick = np.c_[quote[levels].values[0], quote[size_tags].values[0]]
        tick = pd.DataFrame(data=tick, index=levels, columns=['price', 'size'])
        tick['size'] = tick['size'].astype(int)
        return tick

    def __level2size(self, level:str)->str:
        """ map level to size label.
        e.g.:
            >>> self.__level2size('ask10')
            'asize10'
        """
        sizelabel = level[0] + 'size' + level[3:]
        return sizelabel


class Trade(BaseData):

    def __init__(self):
        pass


class TickData(object):

    def __init__(self, quote: pd.DataFrame, trade: pd.DataFrame):
        self._quote = quote
        self._trade = trade

    @property
    def quote_timeseries(self):
        return self._quote['time'].values.tolist()

    @property
    def trade_timeseries(self):
        return self._trade['time'].values.tolist()

    def quote_board(self, t:int or pd.DataFrame)->pd.DataFrame:
        level2size = lambda l: l[0] + 'size' + l[3:]
        if type(t) == int:
            quote = self._quote[self._quote['time'] == t]
        elif type(t) == pd.DataFrame:
            quote = t
        else:
            raise TypeError("argument 't' must be int or pd.DataFrame.")
        asks  = quote.filter(like='ask').columns.values[::-1]
        bids  = quote.filter(like='bid').columns.values
        levels = np.r_[asks, bids]
        size_tags = [level2size(l) for l in levels]
        tick = np.c_[quote[levels].values[0], quote[size_tags].values[0]]
        tick = pd.DataFrame(data=tick, index=levels, columns=['price', 'size'])
        tick['size'] = tick['size'].astype(int)
        return tick

    def get_quote(self, t:Union[None, int, list]=None)->pd.DataFrame:
        if t == None:
            quote = self.__quote
        elif type(t) == int:
            quote = self.__quote[self.__quote['time'] == t]
        elif type(t) == list:
            quote = self.__quote[self.__quote['time'].isin(t)]
        else:
            raise TypeError('argument t must be None, int, or list.')
        return quote

    def get_trade(self, t:None or int or list = None)->pd.DataFrame:
        if t == None:
            trade = self._trade
        elif type(t) == int:
            trade = self._trade[self._trade['time'] == t]
        else:
            trade = self._trade[self._trade['time'].isin(t)]
        return trade

    def pre_quote(self, t:int or pd.DataFrame)->pd.DataFrame:
        if type(t) == int:
            pass
        elif type(t) == pd.DataFrame:
            t = t['time'].iloc[0]
        else:
            raise TypeError("argument 't' munst be int or pd.DataFrame.")
        quote = self._quote[self._quote['time'] < t]
        return None if quote.empty else quote.iloc[-1:]

    def next_quote(self, t:int or pd.DataFrame)->pd.DataFrame:
        if type(t) == int:
            pass
        elif type(t) == pd.DataFrame:
            t = t['time'].iloc[0]
        else:
            raise TypeError("argument 't' munst be int or pd.DataFrame.")
        quote = self._quote[self._quote['time'] > t]
        return None if quote.empty else quote.iloc[0:1]
    
    def get_trade_between(self, pre_quote:int or pd.DataFrame,
                          post_quote:None or int or pd.DataFrame = None)->pd.DataFrame:
        if type(pre_quote) == int:
            pass
        elif type(pre_quote) == pd.DataFrame:
            pre_quote = int(pre_quote['time'].iloc[0])
        else:
            raise TypeError("pre_quote must be int, or pd.DataFrame")
        # use next quote if post_quote is not specified.
        if post_quote == None:
            post_quote = self.next_quote(pre_quote)['time'].iloc[0]
            if post_quote == None:
                raise KeyError('There is no quote data after pre_quote.')
        elif type(post_quote) == int:
            pass
        elif type(pre_quote) == pd.DataFrame:
            post_quote = post_quote['time'].iloc[0]
        else:
            raise TypeError("post_quote must be 'None', int, or pd.Series")
        trade = self._trade[(self._trade['time'] > pre_quote) & (self._trade['time'] < post_quote)]
        return trade