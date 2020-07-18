import numpy as np

class GeneralExchange(object):

    def __init__(self, tickdata, wait_t):
        '''
        arguments:
        ----------
            tickdata: TickData, tick-level data.
            wait_t: int, waiting number of trade records before
                transaction.
        '''

        if wait_t < 0:
            raise KeyError("wait_t must greater than or equal to 0.")

        self._data = tickdata
        self._wait_t = wait_t
        self._next_level = lambda level: level[:3] + str(int(level[3:]) + 1)
    
    def transaction_engine(self, order)->tuple:
        '''
        arguments:
        ----------
        order: dict, simulated order issued by agent,
            keys are ('time', 'side', 'price', 'size', 'pos').

        returns:
        --------
        order: dict, remaining orders，
            keys are ('time', 'side', 'price', 'size', 'pos').
        filled: dict, filled orders
            keys are ('price', 'size').
        '''

        self._check_order(order)

        quote, trade = self._query_data(order['time'])

        filled = {'price': [], 'size': []}

        if order['pos'] == -1:
            order['pos'] = self._wait_t

        order_level = quote[quote['price'] == order['price']]
        
        # if order price is not in quote.
        if order_level.empty:
            return (order, filled)
        else:
            order_level = order_level.index[0]

        # case 1, transact directly.
        if order['side'] == 'buy' and order_level[:3] == 'ask':
            l = 'ask1'
            order['pos'] = 0

        # case 2, wait in trading queue.    
        elif order['side'] == 'buy' and order_level[:3] == 'bid':
            l = order_level
            order['pos'] = self._update_pos_by_trade(order, trade)

        # case 3, transact directly.        
        elif order['side'] == 'sell' and order_level[:3] == 'bid':
            l = 'bid1'
            order['pos'] = 0

        # case 4, wait in trading queue.    
        elif order['side'] == 'sell' and order_level[:3] == 'ask':
            l = order_level
            order['pos'] = self._update_pos_by_trade(order, trade)

        else:
            raise Exception("Unexcepted error occoured in calculating order position.")
        
        # execute orders.
        if order['pos'] == 0:
            while l <= order_level:
                if quote.loc[l, 'size'] <= 0:
                    l = self._next_level(l)
                    continue
                elif quote.loc[l, 'size'] < order['size']:
                    filled['price'].append(quote.loc[l, 'price'])
                    filled['size'].append(quote.loc[l, 'size'])
                    order['size'] -= quote.loc[l, 'size']
                    l = self._next_level(l)
                else:
                    filled['price'].append(quote.loc[l, 'price'])
                    filled['size'].append(quote.loc[l, 'size'])
                    order['size'] = 0
                    break
        return (order, filled)

    def _query_data(self, time):
        if time not in self._data.quote_timeseries:
            raise KeyError("order's time not in range, "\
                           "cannot find corresponding data.")
        quote = self._data.quote_board(time)
        trade = self._data.get_trade_between(time)
        return quote, trade

    def _update_pos_by_trade(self, order, trade) -> int:
        pos = order['pos']
        for _ in range(len(trade[trade['price']==order['price']])):
                if pos == 0:
                    break
                else:
                    pos -= 1
        return pos

    def _check_order(self, order):
        if type(order) != dict:
            raise TypeError("argument type of order must be dict.")
        if order['side'] not in ['buy', 'sell']:
            raise KeyError("order['side'] value must be 'buy' or 'sell'.")
        if type(order['time']) not in [int, np.int32, np.int64]:
            raise TypeError("argument type of order['time'] must be int.")
        if type(order['price']) not in [int, np.int32, np.int64, 
                                        float, np.float32, np.float64]:
            raise TypeError("argument type of order['price'] "\
                            "must be int or float.")
        if order['size'] < 0:
            raise KeyError("order['size'] value must be positive integer or 0.")                    
        if type(order['size']) not in [int, np.int32, np.int64]:
            raise TypeError("argument type of order['size'] must be int.")
        if type(order['pos']) not in [int, np.int32, np.int64]:
            raise TypeError("argument type of order['pos'] must be int.")