import json
import os
import subprocess
import time

import pandas as pd
import psycopg2

# TODO complete logger.


class H2Connection(object):

    def __init__(self, dbdir, user, password, host='localhost', port='5435', h2_start_wait=3):
        self.new_connect(dbdir, user, password, host, port, h2_start_wait)

    @property
    def status(self):
        # for windows
        if os.name == 'nt':
            try:
                return self.conn
            except NameError:
                return False
        # for Linux/MacOS
        else:
            if self._is_h2_online():
                return self.conn
            else:
                return False

    def new_connect(self, dbdir, user, password, host='localhost', port='5435', h2_start_wait=3):
        try:
            self.conn = psycopg2.connect(dbname=dbdir, user=user, password=password, host=host, port=port)
        except psycopg2.OperationalError as e:
            if os.name == 'nt':
                raise ConnectionError("H2 service is not running." \
                    " Since windows doesn't support H2 automatic start, please start h2 service manually.")
            if self._is_h2_online():
                raise ConnectionError("H2 service is running, but connection is refused." \
                    " Please double check username and password or restart h2 service manually.")
            else:
                self._start_h2_service(h2_start_wait)
                self.conn = psycopg2.connect(dbname=dbdir, user=user, password=password, host=host, port=port)
        finally:
            self.cur = self.conn.cursor()

    def query(self, sql: str, *args)->pd.DataFrame:
        self.cur.execute(sql, args)
        data = self.cur.fetchall()
        data = pd.DataFrame(data)
        return data

    def _is_h2_online(self):
        output = subprocess.getoutput('ps -la|grep h2|grep -v grep')
        status = True if len(output) else False
        return status

    # NOTE @wupj _start_h2_service() is only available when running code in terminal.
    # NOTE @wupj H2 start process is paralleled, sleep 3 secs to wait it completely start.
    def _start_h2_service(self, h2_start_wait):
        print("[INFO] try to start H2 service automatically by execute 'h2' in shell.")
        # NOTE @wupj use subprocess.getstatusoutput('h2 &') will suspend this program, since it need complete output. 
        ret = subprocess.check_call('h2 &', shell=True)
        time.sleep(h2_start_wait)
        # start success.
        if ret == 0:
            print('[INFO] start H2 service sucessfully.')
        # start failure, try to start manually.
        else:
            # allow five times to try to start mannully.
            for i in range(5):
                # specify H2 direction or quit.
                h2dir = input("Fail to start H2, please specify h2*.jar direction or input 'q!' to quit:")
                if h2dir != 'q!':
                    ret = subprocess.check_call('java -cp %s org.h2.tools.Server' % h2dir, shell=True)
                    time.sleep(h2_start_wait)
                else:
                    raise KeyboardInterrupt("exit by user input 'q!'.")
                # mannual start successfully, break for loop.
                if ret == 0:
                    subprocess.call('echo "java -cp %s org.h2.tools.Server" > /usr/local/bin/h2' % h2dir, shell=True)
                    print('[INFO] start H2 service sucessfully.')
                    break
                else:
                    print("[WARN] Fail to start H2, you still have %d times to try." % (4-i))
            # mannual start failed.
            raise ConnectionError('Cannot start H2 service, exit program.')


class TickData(object):

    def __init__(self, data: pd.DataFrame):
        self._data = data
        self._quote = self._data[self._data['type'] == 'quote']
        self._quote = self._quote.drop('type', axis=1).dropna(axis=1).reset_index(drop=True)
        self._trade = self._data[self._data['type'] == 'trade']
        self._trade = self._trade.drop('type', axis=1).dropna(axis=1).reset_index(drop=True)

    def __len__(self):
        return self._data.shape[0]

    @property
    def data(self):
        return self._data

    @property
    def quote_timeseries(self):
        return self._quote['time'].values.tolist()

    @property
    def trade_timeseries(self):
        return self._trade['time'].values.tolist()

    def get_quote(self, t:None or int or list = None)->pd.DataFrame:
        if t == None:
            quote = self._quote
        elif type(t) == int:
            quote = self._quote[self._quote['time'] == t]
        else:
            quote = self._quote[self._quote['time'].isin(t)]
        return quote

    def get_trade(self, t:None or int or list = None)->pd.DataFrame:
        if t == None:
            trade = self._trade
        elif type(t) == int:
            trade = self._trade[self._trade['time'] == t]
        else:
            trade = self._trade[self._trade['time'].isin(t)]
        return trade

    def pre_quote(self, t:int or pd.DataFrame)->pd.DataFrame:
        if type(t) == int:
            pass
        elif type(t) == pd.DataFrame:
            t = t['time'].iloc[0]
        else:
            raise TypeError("argument 't' munst be 'int' or 'pd.DataFrame'.")
        quote = self._quote[self._quote['time'] < t]
        return None if quote.empty else quote.iloc[-1:]

    def next_quote(self, t:int or pd.DataFrame)->pd.DataFrame:
        if type(t) == int:
            pass
        elif type(t) == pd.DataFrame:
            t = t['time'].iloc[0]
        else:
            raise TypeError("argument 't' munst be int or pd.DataFrame.")
        quote = self._quote[self._quote['time'] > t]
        return None if quote.empty else quote.iloc[0:1]
    
    def get_trade_between(self, pre_quote:int or pd.DataFrame,
                          post_quote:None or int or pd.DataFrame = None)->pd.DataFrame:
        if type(pre_quote) == int:
            pass
        elif type(pre_quote) == pd.DataFrame:
            pre_quote = pre_quote['time'].iloc[0]
        else:
            raise TypeError("pre_quote must be int, or pd.DataFrame")
        # use next quote if post_quote is not specified.
        if post_quote == None:
            post_quote = self.next_quote(pre_quote)['time'].iloc[0]
            if post_quote == None:
                raise KeyError('There is no quote data after pre_quote.')
        elif type(post_quote) == int:
            pass
        elif type(pre_quote) == pd.DataFrame:
            post_quote = post_quote['time'].iloc[0]
        else:
            raise TypeError("post_quote must be 'None', int, or pd.Series")
        trade = self._trade[(self._trade['time'] > pre_quote) & (self._trade['time'] < post_quote)]
        return None if trade.empty else trade

    def trade_sum(self, trade:pd.DataFrame)->pd.DataFrame:
        if trade is None:
            return None
        elif trade.empty:
            return None
        else:
            return trade[['price', 'size']].groupby('price').sum().reset_index()


def dataset(stock, dbdir, user, psw)->TickData:
    config = json.load(open('config/data.json', 'r'))
    h2db = H2Connection(dbdir, user, psw, **config['h2setting'])
    QUOTE_COLS = config['sql']['QUOTE_COLS']
    TRADE_COLS = config['sql']['TRADE_COLS']
    TIMES = config['sql']['TIMES']
    if h2db.status:
        sql = config['sql']['str'] % eval(config['sql']['pattern'])
        data = h2db.query(sql)
        data.columns = config['tickdata']['TICK_COLS']
        data['type'] = None
        for i in data.index:
            data.loc[i, 'type'] = 'trade' if data.loc[i, 'bid1'] == None else 'quote'
    else:
        raise ConnectionError("cannot connect to H2 service, please strat H2 service first.")
    return TickData(data)