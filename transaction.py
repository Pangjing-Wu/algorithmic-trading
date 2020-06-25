def trasaction_matching(quote, trade, simulated_order)->tuple(dict, dict):
    '''
    arguments:
    ----------
    quote: pd.DataFrame 
    trade: pd.Dataframe
    simulated_order: dict, keys are formed by
                     ('direction', 'level', 'size', 'pos')
    # TODO actual order is given by price, not by level.
    
    returns:
    --------
    simulated_trade: (dict of traded, dict of res)
                     keys are formed by ('price', 'volume')
    '''

    # TODO arguments' input form checking.

    # shortcut function
    level2size = lambda level: level[0] +'size' + level[-1]
    next_level = lambda level: level[:-1] + str(int(level[-1]) + 1)
    
    # initial variable
    simulated_trade = {'price': [], 'size': []}

    # return blank simulated_trade if there is no order issued.
    if simulated_order['size'] <= 0:
        return (simulated_order, simulated_trade)

    # main matching process
    # ---------------------
    # case 1, direction is 'buy' and level is 'ask', transact directly.
    if simulated_order['direction'] == 'buy':
        # if simulated_order level is 'ask'
        if simulated_order['level'][:-1] == 'ask':
            simulated_order['pos'] = 0
            i_level = 'ask1'    # iterative level.
            # keep buying until reach simulated_order’s level.
            while i_level <= simulated_order['level']:
                size_tag = level2size(i_level)
                # continue if quote size is 0.
                if quote[size_tag].iloc[0] <= 0:
                    continue
                # if actual quote size is less than our simulated_order need.
                if quote[size_tag].iloc[0] < simulated_order['size']:
                    simulated_trade['price'].append(quote[i_level].iloc[0])
                    simulated_trade['size'].append(quote[size_tag].iloc[0])
                    simulated_order['size'] -= quote[size_tag].iloc[0]
                    i_level = next_level(i_level)
                # if actual quote size is more than our simulated_order need.
                else:
                    simulated_trade['price'].append(quote[i_level].iloc[0])
                    simulated_trade['size'].append(simulated_order['size'])
                    simulated_order['size'] = 0
                    break
    return(simulated_order, simulated_trade)

    # case 2, direction is 'buy' and level is 'bid', wait in trading queue.
    if simulated_order['direction'] == 'buy':
        if simulated_order['level'][:-1] == 'bid':
            size_tag = level2size(simulated_order['level'])
            # init order position.
            if simulated_order['pos'] == -1:
                simulated_order['pos'] = size_tag
            trade = trade[::-1] # reverse price order of trade.    
            order_price = quote[simulated_order['level']]
            # if there is a trade whose price is lower or equal to ours.
            if trade['price'][0] <= order_price:
                # keep buying...
                for price, size in zip(trade['price'], trade['size']):
                    # until reach simulated_order’s price.
                    if price < order_price:
                        break
                    simulated_order['pos'] = max(0, simulated_order['pos'] - size)
                    # execute order if it is on the front.
                    if simulated_order['pos'] == 0:
                        # if actual trade is less than our simulated_order need.
                        if size < simulated_order['size']:
                            simulated_trade['price'].append(order_price)
                            simulated_trade['size'].append(size)
                            simulated_order['size'] -= size
                        # if actual trade is more than our simulated_order need.
                        else:
                            simulated_trade['price'].append(order_price)
                            simulated_trade['size'].append(simulated_order['size'])
                            simulated_order['size'] = 0
                            break
    return (simulated_order, simulated_trade)

    # case 3, direction is 'sell' and level is 'bid', transact directly.                            
    if simulated_order['direction'] == 'sell':
        if simulated_order['level'][:-1] == 'bid':
            simulated_order['pos'] = 0
            i_level = 'bid1'    # iterative level.
            # keep buying until reach the issued simulated_order’s level.
            while i_level <= simulated_order['level']:
                size_tag = level2size(i_level)
                # continue if quote size is 0.
                if quote[size_tag].iloc[0] <= 0:
                    continue
                # if actual quote size is less than our simulated_order need.
                if quote[size_tag].iloc[0] <= simulated_order['size']:
                    simulated_trade['price'].append(quote[i_level].iloc[0])
                    simulated_trade['size'].append(quote[size_tag].iloc[0])
                    simulated_order['size'] -= quote[size_tag].iloc[0]
                    i_level = next_level(i_level)
                # if actual quote size is more than our simulated_order need.
                else:
                    simulated_trade['price'].append(quote[i_level].iloc[0])
                    simulated_trade['size'].append(simulated_order['size'])
                    simulated_order['size'] = 0
                    break
    return(simulated_order, simulated_trade)

    # case 4, direction is 'sell' and level is 'ask', wait in trading queue.
    if simulated_order['direction'] == 'sell':
        if simulated_order['level'][:-1] == 'ask':
            size_tag = level2size(simulated_order['level'])
            # init order position.
            if simulated_order['pos'] == -1:
                simulated_order['pos'] = size_tag
            trade = trade[::-1] # reverse price order of trade.    
            order_price = quote[simulated_order['level']]
            # if there is a trade whose price is higher or equal to ours.
            if trade['price'][0] >= order_price:
                # keep selling...
                for price, size in zip(trade['price'], trade['size']):
                    # until reach simulated_order’s price.
                    if price < order_price:
                        break
                    simulated_order['pos'] = max(0, simulated_order['pos'] - size)
                    # execute order if it is on the front.
                    if simulated_order['pos'] == 0:
                        # if actual trade is less than our simulated_order need.
                        if size < simulated_order['size']:
                            simulated_trade['price'].append(order_price)
                            simulated_trade['size'].append(size)
                            simulated_order['size'] -= size
                        # if actual trade is more than our simulated_order need.
                        else:
                            simulated_trade['price'].append(order_price)
                            simulated_trade['size'].append(simulated_order['size'])
                            simulated_order['size'] = 0
                            break
    return(simulated_order, simulated_trade)