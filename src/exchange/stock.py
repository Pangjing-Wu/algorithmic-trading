class GeneralExchange(object):

    def __init__(self, tickdata):
        self._data = tickdata
    
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
        
        # arguments checking
        self._order_check(order)
        
        # shortcut function.
        next_level = lambda level: level[:3] + str(int(level[3:]) + 1)
        
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
                # skip if quote volume is 0.
                if quote.loc[l, 'size'] <= 0:
                    continue
                # if actual quote size is less than our order need.
                if quote.loc[l, 'size'] < order['size']:
                    filled['price'].append(quote.loc[l, 'price'])
                    filled['size'].append(quote.loc[l, 'size'])
                    order['size'] -= quote.loc[l, 'size']
                    l = next_level(l)
                # if actual quote size is more than our order need.
                else:
                    filled['price'].append(quote.loc[l, 'price'])
                    filled['size'].append(order['size'])
                    order['size'] = 0
                    break
            return (order, filled)

        # case 2, side is 'buy' and level is 'bid', wait in trading queue.
        if order['side'] == 'buy' and order_level[:3] == 'bid':
            # return if no order is traded at this moment.
            if trade is None:
                return (order, filled)
            # init order position if pos is -1.
            if order['pos'] == -1:
                order['pos'] = quote.loc[order_level]['size']
            # if an order whose price is lower or equal to ours is filled.
            if trade['price'][0] <= order['price']:
                # keep buying...
                for price, size in zip(trade['price'], trade['size']):
                    # until reach order’s price.
                    if price > order['price']:
                        break
                    # calculate available quantity.
                    available_size = max(0, size - order['pos'])
                    # refresh order position.
                    order['pos'] = max(0, order['pos'] - size)
                    # execute order if it is on the front.
                    if order['pos'] == 0:
                        # if actual trade is less than our order need.
                        if available_size < order['size']:
                            filled['price'].append(order['price'])
                            filled['size'].append(available_size)
                            order['size'] -= available_size
                        # if actual trade is more than our order need.
                        else:
                            filled['price'].append(order['price'])
                            filled['size'].append(order['size'])
                            order['size'] = 0
                            break
            return (order, filled)

        # case 3, side is 'sell' and level is 'bid', transact directly.                            
        if order['side'] == 'sell' and order_level[:3] == 'bid':
            l = 'bid1'    # iterative level.
            order['pos'] = 0  # transact directly.
            # keep buying until reach the issued order’s level.
            while l <= order_level:
                # continue if quote size is 0.
                if quote.loc[l, 'size'] <= 0:
                    continue
                # if actual quote size is less than our order need.
                if quote.loc[l, 'size'] <= order['size']:
                    filled['price'].append(quote.loc[l, 'price'])
                    filled['size'].append(quote.loc[l, 'size'])
                    order['size'] -= quote.loc[l, 'size']
                    l = next_level(l)
                # if actual quote size is more than our order need.
                else:
                    filled['price'].append(quote.loc[l, 'price'])
                    filled['size'].append(order['size'])
                    order['size'] = 0
                    break
            return(order, filled)

        # case 4, side is 'sell' and level is 'ask', wait in trading queue.
        if order['side'] == 'sell' and order_level[:3] == 'ask':
            # return if no order is traded at this moment.
            if trade is None:
                return (order, filled)
            # init order position.
            if order['pos'] == -1:
                order['pos'] = quote.loc[order_level]['size']
            trade = trade[::-1] # reverse price order of trade.    
            # if an order whose price is higher or equal to ours is filled.
            if trade['price'][len(trade)-1] >= order['price']:
                # keep selling...
                for price, size in zip(trade['price'], trade['size']):
                    # until reach order’s price.
                    if price < order['price']:
                        break
                    # calculate available quantity.
                    available_size = max(0, size - order['pos'])
                    # refresh order position.
                    order['pos'] = max(0, order['pos'] - size)
                    # execute order if it is on the front.
                    if order['pos'] == 0:
                        # if actual trade is less than our order need.
                        if available_size < order['size']:
                            filled['price'].append(order['price'])
                            filled['size'].append(available_size)
                            order['size'] -= available_size
                        # if actual trade is more than our order need.
                        else:
                            filled['price'].append(order['price'])
                            filled['size'].append(order['size'])
                            order['size'] = 0
                            break
            return(order, filled)

        # raise exception if order is not in previous 4 conditions.
        raise Exception('An unknown error occured, exchange cannot handle this order.')

    def _query_data(self, time):
        quote = self._data.get_quote(time)
        trade = self._data.get_trade_between(time)
        trade = self._data.trade_sum(trade)
        return quote, trade

    def _order_check(self, order):
        if type(order) != dict:
            raise TypeError("argument type of order must be dict.")
        if order['side'] not in ['buy', 'sell']:
            raise KeyError("argument value of order['side'] must be 'buy' or 'sell'.")
        if type(order['time']) != int:
            raise TypeError("argument type of order['time'] must be int.")
        if type(order['price']) not in [int, float]:
            raise TypeError("argument type of order['price'] must be int or float.")
        if type(order['size']) != int:
            raise TypeError("argument type of order['size'] must be int.")
        if type(order['pos']) != int:
            raise TypeError("argument type of order['pos'] must be int.")