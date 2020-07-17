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
        self._data = tickdata
        self._wait_t = wait_t
    
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
        
        next_level = lambda level: level[:3] + str(int(level[3:]) + 1)

        self._check_order(order)
        
        # query data from datasource.
        quote, trade = self._query_data(order['time'])

        # initial variable.
        filled = {'price': [], 'size': []}
        
        # return blank filled if there is no order issued.
        if order['size'] <= 0:
            return (order, filled)

        # map price to level.
        order_level = quote[quote['price'] == order['price']]
        # return blank filled if the price is not in quote.
        if order_level.empty:
            return (order, filled)
        else:
            order_level = order_level.index[0]

        # main matching process
        # ---------------------
        # case 1, side is 'buy' and level is 'ask', transact directly.
        if order['side'] == 'buy' and order_level[:3] == 'ask':
            l = 'ask1' # iterative level.
            order['pos'] = 0 # transact directly.
            # keep buying until reach order’s level.
            while l <= order_level:
                if quote.loc[l, 'size'] <= 0:
                    l = next_level(l)
                    continue
                if quote.loc[l, 'size'] < order['size']:
                    filled['price'].append(quote.loc[l, 'price'])
                    filled['size'].append(quote.loc[l, 'size'])
                    order['size'] -= quote.loc[l, 'size']
                    l = next_level(l)
                else:
                    filled['price'].append(quote.loc[l, 'price'])
                    filled['size'].append(order['size'])
                    order['size'] = 0
                    break
            return (order, filled)

        # case 2, side is 'buy' and level is 'bid', wait in trading queue.
        if order['side'] == 'buy' and order_level[:3] == 'bid':
            # init order position if pos is -1.
            if order['pos'] == -1:
                order['pos'] = self._wait_t
            # return blank filled if trade is empty.
            if trade.empty:
                return (order, filled)
            # update order position.
            for _ in trade[trade['price']==order['price']]:
                if order['pos'] == 0:
                    break
                order['pos'] -= 1
            # transaction.
            if order['pos'] == 0:
                size = min(order['size'], quote.loc[order_level,'size'])
                filled['price'].append(quote.loc[order_level, 'price'])
                filled['size'].append(quote.loc[order_level, 'size'])
                order['size'] -= quote.loc[order_level, 'size']
            return (order, filled)

        # case 3, side is 'sell' and level is 'bid', transact directly.                            
        if order['side'] == 'sell' and order_level[:3] == 'bid':
            l = 'bid1'    # iterative level.
            order['pos'] = 0  # transact directly.
            # keep buying until reach the issued order’s level.
            while l <= order_level:
                # continue if quote size is 0.
                if quote.loc[l, 'size'] <= 0:
                    l = next_level(l)
                    continue
                if quote.loc[l, 'size'] <= order['size']:
                    filled['price'].append(quote.loc[l, 'price'])
                    filled['size'].append(quote.loc[l, 'size'])
                    order['size'] -= quote.loc[l, 'size']
                    l = next_level(l)
                else:
                    filled['price'].append(quote.loc[l, 'price'])
                    filled['size'].append(order['size'])
                    order['size'] = 0
                    break
            return(order, filled)

        # case 4, side is 'sell' and level is 'ask', wait in trading queue.
        if order['side'] == 'sell' and order_level[:3] == 'ask':
            # init order position if pos is -1.
            if order['pos'] == -1:
                order['pos'] = self._wait_t
            # return blank filled if trade is empty.
            if trade.empty:
                return (order, filled)
            # update order position.
            for _ in trade[trade['price']==order['price']]:
                if order['pos'] == 0:
                    break
                order['pos'] -= 1
            # transaction.
            if order['pos'] == 0:
                size = min(order['size'], quote.loc[order_level,'size'])
                filled['price'].append(quote.loc[order_level, 'price'])
                filled['size'].append(quote.loc[order_level, 'size'])
                order['size'] -= quote.loc[order_level, 'size']
            return (order, filled)

        # raise exception if order is not in previous 4 conditions.
        raise Exception("An unknown error occured, " \
                        "exchange cannot handle this order.")

    def _query_data(self, time):
        if time not in self._data.quote_timeseries:
            raise KeyError("order's time not in range, "\
                           "cannot find corresponding data.")
        quote = self._data.quote_board(time)
        trade = self._data.get_trade_between(time)
        return quote, trade

    def _check_order(self, order):
        if type(order) != dict:
            raise TypeError("argument type of order must be dict.")
        if order['side'] not in ['buy', 'sell']:
            raise KeyError("argument value of order['side'] "\
                           "must be 'buy' or 'sell'.")
        if type(order['time']) not in [int, np.int32, np.int64]:
            raise TypeError("argument type of order['time'] must be int.")
        if type(order['price']) not in [int, np.int32, np.int64, 
                                        float, np.float32, np.float64]:
            raise TypeError("argument type of order['price'] "\
                            "must be int or float.")
        if type(order['size']) not in [int, np.int32, np.int64]:
            raise TypeError("argument type of order['size'] must be int.")
        if type(order['pos']) not in [int, np.int32, np.int64]:
            raise TypeError("argument type of order['pos'] must be int.")