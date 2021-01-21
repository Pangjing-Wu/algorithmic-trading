import sys
from typing import List, Union, Tuple

import numpy as np
import pandas as pd

from data.tickdata.source.csv import CSVDataset


def get_tranche_time(time_range:List[int], interval:int)->Tuple[int]:
    _check_pair_in_list(time_range)
    times = list()
    for i in range(0, len(time_range), 2):
        if interval > 0:
            series = list(range(time_range[i], time_range[i+1], interval))
        elif interval == 0:
            series = [time_range[0]]
        else:
            raise ValueError('interval must not be negative.')
        if series[-1] != time_range[i+1]:
            series += [time_range[i+1]]
        for i, j in zip(series[:-1], series[1:]):
            times.append((i,j))
    return tuple(times)


def volume_profile(trade, time_range:List[int],
                  interval:int, ratio=True) -> pd.DataFrame:
    profile = dict(start=list(), end=list(), volume=list())
    tranche_time = get_tranche_time(time_range, interval)
    total_volume = trade['size'].sum()
    for time in tranche_time:
        profile['start'].append(time[0])
        profile['end'].append(time[1])
        volume = trade.between(time[0], time[1])['size'].sum()
        volume = volume / total_volume if ratio else volume
        profile['volume'].append(volume)
    return pd.DataFrame(profile)


def _check_pair_in_list(x):
    if len(x) < 2 and len(x) % 2 != 0:
        raise ValueError("argument should contain 2 or multiples of 2 elements.")