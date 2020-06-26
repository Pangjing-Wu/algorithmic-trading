import sys
from tickdata import dataset


def main(stock, dbdir, user, psw):
        td = dataset(stock, dbdir, user, psw)
        print(td.get_quote(td.quote_timeseries[0]))
        print(td.get_trade(td.trade_timeseries[0]))
        print(td.pre_quote(td.quote_timeseries[0]))
        print(td.next_quote(td.quote_timeseries[0]))
        print(td.pre_quote(td.quote_timeseries[2]))
        trade = td.get_trade_between(td.quote_timeseries[0])
        print(trade)
        print(td.trade_sum(trade))


if __name__ == "__main__":
    stock = '000001'
    dbdir = '~/OneDrive/python-programs/reinforcement-learning/data/20140704'
    user = 'cra001'
    password = 'cra001'
    main(stock, dbdir, user, password)
    # main(*sys.argv[1:])
