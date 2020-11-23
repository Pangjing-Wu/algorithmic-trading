import numpy as np

from .order import ClientOrder, ExchangeOrder


class AShareExchange(object):

    def __init__(self, tickdata, wait_trade=0):
        '''
        arguments:
        ----------
            tickdata: TickData, tick-level data.
            wait_trade: int, waiting trade number before transaction.
        '''
        if wait_trade < 0:
            raise ValueError("wait_trade must be non-negative.")
        self.__data  = tickdata
        self.__wait  = wait_trade
        self.__order = None
        self.reset()

    def __str__(self):
        if self.__order is None:
            return self.__order
        else:
            return self.__order.__str__()

    def reset(self):
        self.__time = [-1]

    def issue(self, code=0, order:ClientOrder=None):
        if code == 0:
            pass
        elif code == 1:
            self.__issue_order(order)
        elif code == 2:
            self.__cancel_order()
        else:
            raise ValueError('unknown operation code.')
    
    def step(self, time):
        self.__check_time(time)
        self.__t = time
        self.__time.append(time)
        return self.__transaction()
    
    def __issue_order(self, order):
        self.__check_time(order.time)
        if self.__order is None:
            self.__order = ExchangeOrder(order, self.__wait)
        else:
            raise RuntimeError("exchange can only contain 1 order, "
                               "cancel previous order first.")

    def __cancel_order(self):
        self.__order = None

    def __transaction(self):
        if self.__order is None:
            return None
        quote = self.__data.quote.get(self.__t).to_board()
        trade = self.__data.trade.between(
            self.__t,
            self.__data.quote.next_time_of(self.__t)
            )
        order_level = quote[quote['price'] == self.__order.price]
        if order_level.empty:
            return self.__order # order price is not in quote
        else:
            order_level = order_level.index[0]
        # case 1, transact directly.
        if self.__order.side == 'buy' and order_level[:3] == 'ask':
            level = 'ask1'
            self.__order.update_pos(0)
        # case 2, wait in trading queue.    
        elif self.__order.side == 'buy' and order_level[:3] == 'bid':
            level = order_level
            self.__order.update_pos(self.__update_pos(self.__order, trade))
        # case 3, transact directly.        
        elif self.__order.side == 'sell' and order_level[:3] == 'bid':
            level = 'bid1'
            self.__order.update_pos(0)
        # case 4, wait in trading queue.    
        elif self.__order.side == 'sell' and order_level[:3] == 'ask':
            level = order_level
            self.__order.update_pos(self.__update_pos(self.__order, trade))
        else:
            raise RuntimeError("unknown error occured during transaction.") 
        # execute orders.
        while self.__order.pos == 0 and level <= order_level:
            if quote.loc[level, 'size'] <= 0:
                level = self.__next_level(level)
            elif quote.loc[level, 'size'] < self.__order.remain:
                self.__order.update_filled(
                    time=self.__t,
                    price=quote.loc[level, 'price'],
                    size=quote.loc[level, 'size']
                    )
                level = self.__next_level(level)
            else:
                self.__order.update_filled(
                    time=self.__t,
                    price=quote.loc[level, 'price'],
                    size=self.__order.remain
                    )
                break
        ret = self.__order
        if self.__order.remain == 0:
            self.__order = None
        return ret

    def __next_level(self, level:str)->str:
        level = level[:3] + str(int(level[3:]) + 1)
        return level

    def __check_time(self, time):
        if time not in self.__data.quote.timeseries:
            raise ValueError('illegal time, cannot find in quote timeseries.')
        if time <= max(self.__time):
            raise RuntimeError('time reverses, current time must be not happend.')

    def __update_pos(self, order, trade):
        pos = order.pos
        for _ in trade[trade['price'] == order.price].index:
            if pos == 0:
                break
            else:
                pos -= 1
        return pos