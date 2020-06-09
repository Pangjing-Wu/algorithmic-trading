import os
import time
import subprocess
import psycopg2
import pandas as pd

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
        output = subprocess.getoutput('ps -l|grep h2|grep -v grep')
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
        self._quote = self._data[self._data['type'] == 'quote'].dropna(axis=1).reset_indix()
        self._trade = self._data[self._data['type'] == 'trade'].dropna(axis=1).reset_indix()

    def __len__(self):
        return self._data.shape[0]

    def __str__(self):
        return self._data

    @property
    def data(self):
        return self._data

    def get_quote(self, t:None or int or list = None)->pd.DataFrame:
        quote = self._quote if t is None else self._quote.loc[t, :]
        quote = quote.drop(['index', 'type'])
        return quote

    def get_trade(self, t:None or int or list = None)->pd.DataFrame:
        trade = self._trade if t is None else self._trade.loc[t, :]
        trade = trade.drop(['index', 'type'])
        return trade

    def pre_quote(self, t:int or pd.Series)->pd.Series:
        if type(t) == int:
            pass
        elif type(t) == pd.DataFrame:
            t = t['time'].iloc[0]
        elif type(t) == pd.Series:
            t = t['time']
        else:
            raise TypeError("argument 't' munst be 'int', 'pd.Series', or 'pd.DataFrame'.")
        quote = self._quote[self._quote['time'] < t]
        return None if quote.empty else quote.iloc[-1]

    def next_quote(self, t:int or pd.Series)->pd.Series:
        if type(t) == int:
            pass
        elif type(t) == pd.DataFrame:
            t = t['time'].iloc[0]
        elif type(t) == pd.Series:
            t = t['time']
        else:
            raise TypeError("argument 't' munst be 'int', 'pd.Series', or 'pd.DataFrame'.")
        quote = self._quote[self._quote['time'] > t]
        return None if quote.empty else quote.iloc[0]
    
    def get_trade_between(self, pre_quote:pd.Series, post_quote:None or pd.Series = None)->pd.DataFrame:
        # use next quote if post_quote is not specified.
        if post_quote == None:
            post_quote = self.next_quote(pre_quote)
        time1, time2 = pre_quote['time'], post_quote['time']
        trade = self._trade[(self._trade['time'] > time1) & (self._trade['time'] < time2)]
        return None if trade.empty else trade

    def trade_statistic(self, trade:pd.DataFrame)->pd.DataFrame:
        return trade.groupby('price').sum()