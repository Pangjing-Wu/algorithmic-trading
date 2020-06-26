import json

import numpy as np
import pandas as pd

from h2db import H2Connection


class TickData(object):

    def __init__(self, data: pd.DataFrame):
        self._data = data.copy()
        # divide quote and trade.
        self._quote = data[data['type'] == 'quote'].drop('type', axis=1)
        self._quote = self._quote.dropna(axis=1).reset_index(drop=True)
        self._trade = data[data['type'] == 'trade'].drop('type', axis=1)
        self._trade = self._trade.dropna(axis=1).reset_index(drop=True)
        # set data type.
        int_type_cols   = self._quote.filter(like='size').columns.tolist()
        float_type_cols  = self._quote.filter(like='ask').columns.tolist()
        float_type_cols += self._quote.filter(like='bid').columns.tolist()
        self._quote[int_type_cols]  = self._quote[int_type_cols].astype(int)
        self._quote[float_type_cols] = self._quote[float_type_cols].astype(float)
        self._trade['price'] = self._trade['price'].astype(float)
        self._trade['size']  = self._trade['size'].astype(int)
    
    def __len__(self):
        return self._data.shape[0]

    @property
    def get_data(self):
        return self._data

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


def dataset(stock, dbdir, user, psw, config=None)->TickData:
    if config is None:
        config = json.load(open('config/data.json', 'r'))
    else:
        config = json.load(open(config, 'r'))
    # connect h2db and query.
    h2 = H2Connection(dbdir, user, psw, **config['h2setting'])
    QUOTE_COLS = config['sql']['QUOTE_COLS']
    TRADE_COLS = config['sql']['TRADE_COLS']
    TIMES = config['sql']['TIMES']
    if h2.status:
        sql = config['sql']['str'] % eval(config['sql']['pattern'])
        data = h2.query(sql)
        data.columns = config['tickdata']['TICK_COLS']
        data['type'] = None
        for i in data.index:
            data.loc[i, 'type'] = 'trade' if data.loc[i, 'bid1'] == None else 'quote'
    else:
        raise ConnectionError("cannot connect to H2 service, please strat H2 service first.")
    return TickData(data)
