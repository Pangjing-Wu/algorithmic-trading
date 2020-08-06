import numpy as np
import pandas as pd


class TickData(object):

    def __init__(self, quote: pd.DataFrame, trade: pd.DataFrame):
        self._quote = quote
        self._trade = trade
    
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
        return trade