def trasaction_matching(quote, trade, simulated_order)->tuple:
    '''
    arguments:
    ----------
    quote: pd.DataFrame, quote_board likes.
    trade: pd.Dataframe, trade series.
    simulated_order: dict, keys are formed by
                     ('direction', 'price', 'size', 'pos').

    returns:
    --------
    simulated_trade: (dict of remaining order, dict of traded)
                     keys are formed by ('price', 'size').
    '''

    # TODO arguments' input form checking.

    # shortcut function
    next_level = lambda level: level[:-1] + str(int(level[-1]) + 1)

    # initial variable
    simulated_trade = {'price': [], 'size': []}

    # return blank simulated_trade if there is no order issued.
    if simulated_order['size'] <= 0:
        return (simulated_order, simulated_trade)

    # map price to level
    simulated_order_level = quote[quote['price'] == simulated_order['price']]
    # return blank simulated_trade if the price is not in quote.
    if simulated_order_level.empty:
        return (simulated_order, simulated_trade)
    else:
        simulated_order_level = simulated_order_level.index[0]
    
    # main matching process
    # ---------------------
    # case 1, direction is 'buy' and level is 'ask', transact directly.
    if simulated_order['direction'] == 'buy':
        # if simulated_order level is 'ask'
        if simulated_order_level[:-1] == 'ask':
            l = 'ask1'    # iterative level.
            simulated_order['pos'] = 0  # transact directly.
            # keep buying until reach simulated_order’s level.
            while l <= simulated_order_level:
                # skip if quote volume is 0.
                if quote.loc[l, 'size'] <= 0:
                    continue
                # if actual quote size is less than our simulated_order need.
                if quote.loc[l, 'size'] < simulated_order['size']:
                    simulated_trade['price'].append(quote.loc[l, 'price'])
                    simulated_trade['size'].append(quote.loc[l, 'size'])
                    simulated_order['size'] -= quote.loc[l, 'size']
                    l = next_level(l)
                # if actual quote size is more than our simulated_order need.
                else:
                    simulated_trade['price'].append(quote.loc[l, 'price'])
                    simulated_trade['size'].append(simulated_order['size'])
                    simulated_order['size'] = 0
                    break
    return(simulated_order, simulated_trade)

    # case 2, direction is 'buy' and level is 'bid', wait in trading queue.
    if simulated_order['direction'] == 'buy':
        if simulated_order_level[:-1] == 'bid':
            # init order position if pos is -1.
            if simulated_order['pos'] == -1:
                simulated_order['pos'] = quote[simulated_order_level]['size']
            trade = trade[::-1] # reverse price order of trade.    
            # if there is a trade whose price is lower or equal to ours.
            if trade['price'][0] <= simulated_order['price']:
                # keep buying...
                for price, size in zip(trade['price'], trade['size']):
                    # until reach simulated_order’s price.
                    if price < simulated_order['price']:
                        break
                    # refresh order position.
                    simulated_order['pos'] = max(0, simulated_order['pos'] - size)
                    # execute order if it is on the front.
                    if simulated_order['pos'] == 0:
                        # if actual trade is less than our simulated_order need.
                        if size < simulated_order['size']:
                            simulated_trade['price'].append(simulated_order['price'])
                            simulated_trade['size'].append(size)
                            simulated_order['size'] -= size
                        # if actual trade is more than our simulated_order need.
                        else:
                            simulated_trade['price'].append(simulated_order['price'])
                            simulated_trade['size'].append(simulated_order['size'])
                            simulated_order['size'] = 0
                            break
    return (simulated_order, simulated_trade)

    # case 3, direction is 'sell' and level is 'bid', transact directly.                            
    if simulated_order['direction'] == 'sell':
        if simulated_order_level[:-1] == 'bid':
            l = 'bid1'    # iterative level.
            simulated_order['pos'] = 0  # transact directly.
            # keep buying until reach the issued simulated_order’s level.
            while l <= simulated_order_level:
                # continue if quote size is 0.
                if quote.loc[l, 'size'] <= 0:
                    continue
                # if actual quote size is less than our simulated_order need.
                if quote.loc[l, 'size'] <= simulated_order['size']:
                    simulated_trade['price'].append(quote.loc[l, 'price'])
                    simulated_trade['size'].append(quote.loc[l, 'size'])
                    simulated_order['size'] -= quote.loc[l, 'size']
                    i_level = next_level(i_level)
                # if actual quote size is more than our simulated_order need.
                else:
                    simulated_trade['price'].append(quote.loc[l, 'price'])
                    simulated_trade['size'].append(simulated_order['size'])
                    simulated_order['size'] = 0
                    break
    return(simulated_order, simulated_trade)

    # case 4, direction is 'sell' and level is 'ask', wait in trading queue.
    if simulated_order['direction'] == 'sell':
        if simulated_order_level[:-1] == 'ask':
            # init order position.
            if simulated_order['pos'] == -1:
                simulated_order['pos'] = quote[simulated_order_level]['size']
            trade = trade[::-1] # reverse price order of trade.    
            # if there is a trade whose price is higher or equal to ours.
            if trade['price'][0] >= simulated_order['price']:
                # keep selling...
                for price, size in zip(trade['price'], trade['size']):
                    # until reach simulated_order’s price.
                    if price < simulated_order['price']:
                        break
                    # refresh order position.
                    simulated_order['pos'] = max(0, simulated_order['pos'] - size)
                    # execute order if it is on the front.
                    if simulated_order['pos'] == 0:
                        # if actual trade is less than our simulated_order need.
                        if size < simulated_order['size']:
                            simulated_trade['price'].append(simulated_order['price'])
                            simulated_trade['size'].append(size)
                            simulated_order['size'] -= size
                        # if actual trade is more than our simulated_order need.
                        else:
                            simulated_trade['price'].append(simulated_order['price'])
                            simulated_trade['size'].append(simulated_order['size'])
                            simulated_order['size'] = 0
                            break
    return(simulated_order, simulated_trade)