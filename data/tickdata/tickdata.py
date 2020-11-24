import abc
from typing import Union, List

import numpy as np
import pandas as pd


class MetaQuote(pd.DataFrame):

    def __init__(self, data:pd.DataFrame, date:str, level=List[str]):
        self.__check_lenth(data)
        super().__init__(data)
        self.__date  = date
        self.__level = None
        self.__level = level
    
    @property
    def date(self):
        return self.__date

    @property
    def time(self):
        return self['time'].iloc[0]

    @property
    def size(self):
        index = [self.__level2size(l) for l in self.__level]
        return self[index].iloc[0]

    @property
    def price(self):
        return self[self.__level].iloc[0]

    @property
    def level(self):
        return self.__level

    def to_board(self):
        level = self.__level[::-1]
        size_tags = [self.__level2size(l) for l in level]
        board = np.c_[self[level].values[0], self[size_tags].values[0]]
        board = pd.DataFrame(data=board, index=level, columns=['price', 'size'])
        board['size'] = board['size'].astype(int)
        return board

    def __check_lenth(self, data:pd.DataFrame):
        if data.shape[0] not in [0, 1]:
            raise ValueError("meta quote data can at most contain one record.")

    def __level2size(self, level:str)->str:
        """ map level to size label.
        e.g.:
            >>> self.__level2size('ask10')
            'asize10'
        """
        sizelabel = level[0] + 'size' + level[3:]
        return sizelabel


class BasicSeries(pd.DataFrame, abc.ABC):

    def __init__(self, data:pd.DataFrame, date:str):
        super().__init__(data)
        self.__date = date

    @abc.abstractmethod
    def _get_meta(self, df:pd.DataFrame):
        pass

    @abc.abstractmethod
    def _get_series(self, df:pd.DataFrame):
        pass

    @property
    def date(self):
        return self.__date

    @property
    def timeseries(self)->List[int]:
        return self['time'].drop_duplicates().values.tolist()

    def pre_time_of(self, t:int)->int:
        self.__check_time_in_range(t)
        i = self.timeseries.index(t)
        self.__check_index_in_range(i-1)
        return self.timeseries[i-1]

    def next_time_of(self, t:int)->int:
        self.__check_time_in_range(t)
        i = self.timeseries.index(t)
        self.__check_index_in_range(i+1)
        return self.timeseries[i+1]

    def get(self, t:int)->pd.DataFrame:
        self.__check_time_in_range(t)
        df = self[self['time'] == t]
        return self._get_meta(df)

    def pre_of(self, t:int)->pd.DataFrame:
        t = self.pre_time_of(t)
        return self.get(t)

    def next_of(self, t:int)->pd.DataFrame:
        t = self.next_time_of(t)
        return self.get(t)
        
    def between(self, t1:int=None, t2:int=None)->pd.DataFrame:
        if t1 is None and t2 is None:
            return self
        elif t1 is None:
            df = self[self['time'] < t2]
        elif t2 is None:
            df = self[self['time'] >= t1]
        else:
            df = self[(self['time'] >= t1) & (self['time'] < t2)]
        if df.shape[0] > 1:
            return self._get_series(df)
        else:
            return self._get_meta(df)

    def __check_time_in_range(self, *args):
        for t in args:
            if t not in self.timeseries:
                raise KeyError('argument out of time range.')
    
    def __check_index_in_range(self, *args):
        for i in args:
            if i < 0 or i >= self.__len__():
                raise IndexError('index out of data lenth.')


class Quote(BasicSeries):

    def __init__(self, data:pd.DataFrame, date:str, level:List[str]):
        super().__init__(data, date)
        self.__level = None
        self.__level = level

    @property
    def level(self):
        return self.__level

    def _get_meta(self, df:pd.DataFrame)->MetaQuote:
        return MetaQuote(df, self.date, self.__level)

    def _get_series(self, df:pd.DataFrame)->BasicSeries:
        return Quote(df, self.date, self.__level)


class Trade(BasicSeries):

    def __init__(self, data:pd.DataFrame, date:str):
        super().__init__(data, date)

    def _get_meta(self, df:pd.DataFrame)->BasicSeries:
        return Trade(df, self.date)

    def _get_series(self, df:pd.DataFrame)->BasicSeries:
        return Trade(df, self.date)

    def group_by_price(self)->pd.DataFrame:
        df = self[['price', 'size']].groupby('price')
        df = df.sum().reset_index()
        return df


class TickData(object):

    def __init__(self, quote, trade, date):
        self.__date  = date
        self.__level = self.__get_level(quote)
        self.__quote = Quote(quote, date, self.__level)
        self.__trade = Trade(trade, date)

    @property
    def date(self):
        return self.__date

    @property
    def quote(self):
        return self.__quote

    @property
    def trade(self):
        return self.__trade

    def between(self, t1, t2):
        return TickData(
            quote=self.__quote.between(t1, t2),
            trade=self.__trade.between(t1, t2),
            date=self.__date
            )

    def __get_level(self, quote:pd.DataFrame)->list:
        asks = quote.filter(like='ask').columns.values.tolist()
        bids = quote.filter(like='bid').columns.values.tolist()
        if len(asks) != len(bids):
            raise ValueError('the length of ask and bid level is unequal.')
        else:
            return bids[::-1] + asks