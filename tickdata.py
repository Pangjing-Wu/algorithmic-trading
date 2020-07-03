import json

import numpy as np
import pandas as pd

from h2db import H2Connection


class TickData(object):

    def __init__(self, quote: pd.DataFrame, trade: pd.DataFrame):
        # divide quote and trade.
        self._quote = quote
        self._trade = trade
        # set data type.
        int_type_cols = self._quote.filter(like='size').columns.tolist()
        float_type_cols = self._quote.filter(like='ask').columns.tolist()
        float_type_cols += self._quote.filter(like='bid').columns.tolist()
        self._quote[int_type_cols]  = self._quote[int_type_cols].astype(int)
        self._quote[float_type_cols] = self._quote[float_type_cols].astype(float)
        self._trade['price'] = self._trade['price'].astype(float)
        self._trade['size']  = self._trade['size'].astype(int)
    
    def __len__(self):
        return (self._quote.shape[0] + self._trade.shape[0])

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

    def get_quote(self, t:None or int or list = None)->pd.DataFrame:
        if t == None:
            quote = self._quote
        elif type(t) == int:
            quote = self._quote[self._quote['time'] == t]
        else:
            quote = self._quote[self._quote['time'].isin(t)]
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
        return None if trade.empty else trade

    def trade_sum(self, trade:pd.DataFrame)->pd.DataFrame:
        if trade is None:
            return None
        elif trade.empty:
            return None
        else:
            return trade[['price', 'size']].groupby('price').sum().reset_index()


def load_from_h2(stock, dbdir, user, psw, **kwargs)->TickData:
    h2 = H2Connection(dbdir, user, psw, **kwargs)
    QUOTE_COLS = ["time", "bid1", "bsize1", "ask1", "asize1", "bid2", "bsize2", "ask2",
        "asize2", "bid3", "bsize3", "ask3", "asize3", "bid4", "bsize4", "ask4", "asize4",
        "bid5", "bsize5", "ask5", "asize5", "bid6", "bsize6", "ask6", "asize6", "bid7",
        "bsize7", "ask7", "asize7", "bid8", "bsize8", "ask8", "asize8", "bid9", "bsize9",
        "ask9", "asize9", "bid10", "bsize10", "ask10", "asize10"]
    TRADE_COLS = ["time", "price", "size"]
    TIMES = [34200000, 41400000, 46800000, 54000000]
    if h2.status:
        sql = "select %s from %s where time between %s and %s or time between %s and %s"
        quote = h2.query(sql % (','.join(QUOTE_COLS), 'quote_' + stock, *TIMES))
        trade = h2.query(sql % (','.join(TRADE_COLS), 'trade_' + stock, *TIMES))
        quote.columns = QUOTE_COLS
        trade.columns = TRADE_COLS
    else:
        raise ConnectionError("cannot connect to H2 service, please strat H2 service first.")
    return TickData(quote, trade)