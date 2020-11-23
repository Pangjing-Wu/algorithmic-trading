import numpy as np

class ClientOrder(object):

    def __init__(self, time, side, price, size):
        self.__check_order(time, side, price, size)
        self.__order = dict(time=time, side=side,
                            price=price, size=size)

    def __str__(self):
        return self.__order.__str__()

    def __eq__(self, other):
        return other == self.__hash__()

    def __hash__(self):
        return hash(str(self.__order))

    @property
    def time(self):
        return self.__order['time']

    @property
    def side(self):
        return self.__order['side']

    @property
    def price(self):
        return self.__order['price']

    @property
    def size(self):
        return self.__order['size']

    def __check_order(self, time, side, price, size):
        if type(time) not in [int, np.int32, np.int64]:
            raise TypeError("order time must be int.")
        if time <= 0:
            raise ValueError("order time must be positive.")
        if side not in ['buy', 'sell']:
            raise ValueError("order side must be 'buy' or 'sell'.")
        if type(price) not in [float, np.float32, np.float64]:
            raise TypeError("order price must be float.")         
        if type(size) not in [int, np.int32, np.int64]:
            raise TypeError("order size must be int.")
        if size <= 0:
            raise ValueError("order size must be positive.")


class ExchangeOrder(object):

    def __init__(self, order:ClientOrder, init_pos:int):
        self.__order  = order
        self.__pos    = init_pos
        self.__remain = order.size
        self.__filled  = dict(time=[], price=[], size=[])

    def __str__(self):
        ret = "order: %s\nqueue position: %s\nfilled: %s\nremain: %d" % (
               self.__order, self.__pos, self.__filled, self.__remain
               )
        return ret

    def __eq__(self, other):
        return other == self.__hash__()

    def __hash__(self):
        return hash(self.__order)
            
    @property
    def time(self):
        return self.__order.time

    @property
    def side(self):
        return self.__order.side

    @property
    def price(self):
        return self.__order.price

    @property
    def size(self):
        return self.__order.size
    
    @property
    def pos(self):
        return self.__pos

    @property
    def remain(self):
        return self.__remain

    @property
    def filled(self):
        return self.__filled

    def update_pos(self, i):
        self.__pos = max(0, i)

    def update_filled(self, time, price, size):
        self.__check_waiting()
        self.__check_filled_range(time, price, size)
        self.__filled['time'].append(time)
        self.__filled['price'].append(price)
        self.__filled['size'].append(size)
        self.__remain -= size

    def __check_waiting(self):
        if self.__pos != 0:
            raise RuntimeError('order is waiting in queue (pos=%d), '
                               'cannot execute.' % self.__pos)

    def __check_filled_range(self, time, price, size):
        if time < self.__order.time:
            raise ValueError('illegal filled time')
        if size > self.__remain:
            raise ValueError('illegal filled size')