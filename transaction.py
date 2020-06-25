def action2simulated_order():
    pass



def trasaction_matching(self, quote, trade, simulated_order) -> 'simulated_trade':
    '''
    arguments:
    ----------
    quote: pd.DataFrame 
    trade: pd.Dataframe
    simulated_order: dict, keys are formed by ('direction', 'level', 'size')

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
        return simulated_trade

    # main matching process
    # ---------------------
    # case 1, direction is 'buy' and level is 'ask', transact directly.
    if simulated_order['direction'] == 'buy':
        # if simulated_order level is 'ask'
        if simulated_order['level'][:-1] == 'ask':
            i_level = 'ask1'    # iterative level
            # keep buying until reach simulated_order’s level.
            while i_level <= simulated_order['level']:
                size_tag = level2size(i_level)
                # continue if quote size is 0.
                if quote[size_tag].iloc[0] <= 0:
                    continue
                # if actual quote is less than our simulated_order need.
                if quote[size_tag].iloc[0] < simulated_order['size']:
                    simulated_trade['price'].append(quote[i_level].iloc[0])
                    simulated_trade['size'].append(quote[size_tag].iloc[0])
                    simulated_order['size'] -= quote[size_tag].iloc[0]
                # if actual quote is more than our simulated_order need.
                else:
                    simulated_trade['price'].append(quote[i_level].iloc[0])
                    simulated_trade['size'].append(simulated_order['size'])
                    simulated_order['size'] = 0
                    break

    # case 2, direction is 'buy' and level is 'bid', wait in trading queue.
    if simulated_order['direction'] == 'buy':
        if simulated_order['level'][:-1] == 'bid':
            # TODO need rewrite
            wait_t=self.wait_t
            if action['size'] != 0:
                self._wait_signal = 0
            else :
                self._wait_signal= self._wait_signal +1
            if self._wait_signal >= wait_t:
                # if there is a trade whose price is lower or equal to ours.
                if trade['price'][0] <= self.simulated_quote['price']:
                    # keep buying...
                    for price, size in zip(trade['price'], trade['size']):
                        # until reach simulated_order’s price.
                        if price > quote[simulated_order['level']]:
                            break
                        # if actual trade is less than our simulated_order need.
                        if size < simulated_order['size']:
                            simulated_trade['price'].append(price)
                            simulated_trade['size'].append(size)
                            simulated_order['size'] -= size
                        # if actual trade is more than our simulated_order need.
                        else:
                            simulated_trade['price'].append(price)
                            simulated_trade['size'].append(simulated_order['size'])
                            simulated_order['size'] = 0
                            break

    # case 3, direction is 'sell' and level is 'bid', transact directly.                            
    if simulated_order['direction'] == 'sell':
        if simulated_order['level'][:-1] == 'bid':
            i_level = 'bid1'    # iterative level
            # keep buying until reach the issued simulated_order’s level.
            while i_level <= simulated_order['level']:
                size_tag = level2size(i_level)
                # continue if quote size is 0.
                if quote[size_tag].iloc[0] <= 0:
                    continue
                # if actual quote is less than our simulated_order need.
                if quote[size_tag].iloc[0] <= simulated_order['size']:
                    simulated_trade['price'].append(quote[i_level].iloc[0])
                    simulated_trade['size'].append(quote[size_tag].iloc[0])
                    simulated_order['size'] -= quote[size_tag].iloc[0]
                # if actual quote is more than our simulated_order need.
                else:
                    simulated_trade['price'].append(quote[i_level].iloc[0])
                    simulated_trade['size'].append(simulated_order['size'])
                    simulated_order['size'] = 0
                    break

    # case 4, direction is 'sell' and level is 'ask', wait in trading queue.
    if simulated_order['direction'] == 'sell':
        if simulated_order['level'][:-1] == 'ask':
            # TODO need rewrite
            wait_t = self.wait_t
            if action['size'] != 0:
                self._wait_signal = 0
            else:
                self._wait_signal = self._wait_signal + 1
            if self._wait_signal >= wait_t:
                # if there is a trade whose price is higher or equal to ours.
                if trade['price'][0] >= self.simulated_quote['price']:
                    # keep selling...
                    for price, size in zip(trade['price'], trade['size']):
                        # until reach simulated_order’s price.
                        if price < quote[simulated_order['level']]:
                            break
                        # if actual trade is less than our simulated_order need.
                        if size < simulated_order['size']:
                            simulated_trade['price'].append(price)
                            simulated_trade['size'].append(size)
                            simulated_order['size'] -= size
                        # if actual trade is more than our simulated_order need.
                        else:
                            simulated_trade['price'].append(price)
                            simulated_trade['size'].append(simulated_order['size'])
                            simulated_order['size'] = 0
                            break

    return(simulated_order, simulated_trade)