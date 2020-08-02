from typing import List, Union

import pandas as pd


def group_trade_by_price(trade:pd.DataFrame)->pd.DataFrame:

    if trade is None:
        return None
    elif trade.empty:
        return None
    else:
        return trade[['price', 'size']].groupby('price').sum().reset_index()


def group_trade_volume_by_time(trades:Union[pd.DataFrame, List[pd.DataFrame]],
                               time:List[int], interval:int=0) -> pd.DataFrame:
    
    volumes = {'start':[], 'end':[], 'volume':[]}

    if len(time) < 2 and len(time) % 2 != 0:
        raise KeyError("argument time should have 2 or multiples of 2 elements.")

    trades = [trades] if type(trades) == pd.DataFrame else trades
    
    for i in range(0, len(time), 2):
        
        if interval > 0:
            time_slices = list(range(time[i], time[i+1], interval))
        elif interval == 0:
            time_slices = [trades[0]['time'].iloc[0]]
        else:
            raise KeyError('interval must not be negative.')

        if time_slices[-1] != time[i+1]:
            time_slices.append(time[i+1])

        for j in range(len(time_slices) - 1):
            t0 = time_slices[j]
            t1 = time_slices[j+1]
            volume = 0

            for trade in trades:
                index = (trade['time'] >= t0) & (trade['time'] < t1)
                volume += trade[index]['size'].sum()

            volumes['start'].append(t0)
            volumes['end'].append(t1)
            volumes['volume'].append(int(volume / len(trades)))

    return pd.DataFrame(volumes)


def tranche_num(time:List[int], interval):

    if len(time) < 2 and len(time) % 2 != 0:
        raise KeyError("argument time should have 2 or multiples of 2 elements.")

    if interval == 0:
        num = 1
    elif interval > 0:
        num = 0
        for i in range(0, len(time), 2):
            timelist = list(range(time[i], time[i+1], interval))
            num += len(timelist)
    else:
        raise KeyError('interval must not be negative.')
    
    return num