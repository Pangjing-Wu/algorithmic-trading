import pandas as pd

def group_trade_by_price(trade:pd.DataFrame)->pd.DataFrame:

    if trade is None:
        return None
    elif trade.empty:
        return None
    else:
        return trade[['price', 'size']].groupby('price').sum().reset_index()


def group_trade_volume_by_time(trade:pd.DataFrame, time:list, interval:int=0) -> pd.DataFrame:
    volumes = {'start':[], 'end':[], 'volume':[]}

    if len(time) < 2 and len(time) % 2 != 0:
        raise KeyError("argument time should have 2 or multiples of 2 elements.")
    
    for i in range(0, len(time), 2):

        if interval > 0:
            time_slices = list(range(time[i], time[i+1], interval))
        elif interval == 0:
            time_slices = [trade['time'].iloc[0]]
        else:
            raise KeyError('interval must not be negative.')

        for j in range(len(time_slices)):
            t0 = time_slices[j]
            if j < len(time_slices) - 1:
                t1 = time_slices[j+1]
                index = (trade['time'] >= t0) & (trade['time'] < t1)
                volumes['start'].append(t0)
                volumes['end'].append(t1)
                volumes['volume'].append(int(trade[index]['size'].sum()))
            else:
                index = trade['time'] >= t0
                volumes['start'].append(t0)
                volumes['end'].append(time[i+1])
                volumes['volume'].append(int(trade[index]['size'].sum()))

    return pd.DataFrame(volumes)