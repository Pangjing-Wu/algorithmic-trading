import psycopg
import pandas as pd

class QuoteData(object):

    def __init__():
        pass


class TradeData(object):
    
    def __init__():
        pass

class TickData(object):

    def __init__(self, data:h2.database, stock_code:str, range: str or int)->pd.DataFrame:
        # TODO preprocessing
        pass

    def get_quote(self, index:timestamp or order)->quote:
        pass

    def get_trade(self, index:timestamp or order)->quote:
        pass

    def pre_quote(self, current:quote or trade)->quote:
        pass

    def next_quote (self, current:quote or trade)->quote:
        pass
    
    def get_trade_between(self, pre_quote, post_quote)->trade:
        pass

    def get_quote_series(self):
        pass

    def get_trade_series(self):
        pass

    def statastic(self, data:quote or trade):
        pass