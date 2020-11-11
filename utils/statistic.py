from typing import List, Union

import pandas as pd


argmax = lambda a: [i for i, val in enumerate(a) if (val == max(a))][0]


def group_trade_by_price(trade:pd.DataFrame)->pd.DataFrame:

    if trade is None:
        return None
    elif trade.empty:
        return None
    else:
        return trade[['price', 'size']].groupby('price').sum().reset_index()


def tranche_num(time_range:List[int], interval):
    if len(time_range) < 2 and len(time_range) % 2 != 0:
        raise KeyError("argument time should have 2 or multiples of 2 elements.")
    if interval == 0:
        num = 1
    elif interval > 0:
        num = 0
        for i in range(0, len(time_range), 2):
            timelist = list(range(time_range[i], time_range[i+1], interval))
            num += len(timelist)
    else:
        raise KeyError('interval must not be negative.')
    return num