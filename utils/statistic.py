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


def volume_profile(trades:Union[pd.DataFrame, List[pd.DataFrame]],
                  time_range:List[int], interval=0) -> pd.DataFrame:
    
    volumes = dict(start=list(), end=list(), volume=list())

    if len(time_range) < 2 and len(time_range) % 2 != 0:
        raise KeyError("argument time should have 2 or multiples of 2 elements.")

    trades = [trades] if type(trades) == pd.DataFrame else trades
    
    for i in range(0, len(time_range), 2):
        
        if interval > 0:
            time_slices = list(range(time_range[i], time_range[i+1], interval))
        elif interval == 0:
            time_slices = [trades[0]['time'].iloc[0]]
        else:
            raise KeyError('interval must not be negative.')

        if time_slices[-1] != time_range[i+1]:
            time_slices.append(time_range[i+1])

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


def distribute_task(goal:int, profile:pd.DataFrame):
    ratio = [v / (profile['volume'].sum() + 1e-8) for v in profile['volume']]
    subgoals = [int(goal * r // 100 * 100) for r in ratio]
    subgoals[argmax(subgoals)] += goal - sum(subgoals)
    tasks = pd.DataFrame(dict(start=profile['start'], end=profile['end'], goal=subgoals))
    return tasks