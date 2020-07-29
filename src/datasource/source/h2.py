import os
import subprocess
import time

import pandas as pd
import psycopg2

# TODO improve logger module


QUOTE_COLS = ["time", "bid1", "bsize1", "ask1", "asize1", "bid2", "bsize2", "ask2",
        "asize2", "bid3", "bsize3", "ask3", "asize3", "bid4", "bsize4", "ask4", "asize4",
        "bid5", "bsize5", "ask5", "asize5", "bid6", "bsize6", "ask6", "asize6", "bid7",
        "bsize7", "ask7", "asize7", "bid8", "bsize8", "ask8", "asize8", "bid9", "bsize9",
        "ask9", "asize9", "bid10", "bsize10", "ask10", "asize10"]
TRADE_COLS = ["time", "price", "size"]
TIMES = [34200000, 41400000, 46800000, 54000000]


class H2Connection(object):

    def __init__(self, dbdir, user, password, host='localhost', port='5435', h2_start_wait=3):
        self.new_connect(dbdir, user, password, host, port, h2_start_wait)

    @property
    def status(self):
        # for windows
        if os.name == 'nt':
            try:
                return self._conn
            except NameError:
                return False
        # for Linux/MacOS
        else:
            if self._is_h2_online():
                return self._conn
            else:
                return False

    def new_connect(self, dbdir, user, password, host='localhost', port='5435', h2_start_wait=3):
        try:
            self._conn = psycopg2.connect(dbname=dbdir, user=user, password=password, host=host, port=port)
        except psycopg2.OperationalError as e:
            if os.name == 'nt':
                raise ConnectionError("H2 service is not running." \
                    " Since windows doesn't support H2 automatic start, please start h2 service manually.")
            else:
                if self._is_h2_online():
                    raise ConnectionError("H2 service is running, but connection is refused." \
                        " Please double check username and password or restart h2 service manually.")
                else:
                    self._start_h2_service(h2_start_wait)
        finally:
            self._conn = psycopg2.connect(dbname=dbdir, user=user, password=password, host=host, port=port)
            self._cur = self._conn.cursor()

    def query(self, sql: str, *args)->pd.DataFrame:
        self._cur.execute(sql, *args)
        data = self._cur.fetchall()
        data = pd.DataFrame(data)
        return data

    def _is_h2_online(self):
        output = subprocess.getoutput('ps -la|grep h2|grep -v grep')
        status = True if len(output) else False
        return status

    
    def _start_h2_service(self, h2_start_wait):

        # NOTE H2 start process is paralleled, sleep a few seconds to wait for starting.
        
        print("[INFO] try to start H2 service automatically by execute 'h2' in shell.")
        # NOTE use subprocess.getstatusoutput('h2 &') will suspend this program,
        # since it need wait the command terminated. 
        ret = subprocess.check_call('h2 &', shell=True)
        time.sleep(h2_start_wait)

        # start successfully.
        if ret == 0:
            print('[INFO] start H2 service sucessfully.')
        else:
            # try to start h2 mannully.
            h2dir = input("Fail to start H2, please specify h2*.jar direction:")
            ret = subprocess.check_call('java -cp %s org.h2.tools.Server' % h2dir, shell=True)
            time.sleep(h2_start_wait)

            # mannual start h2 service successfully.
            if ret == 0:
                subprocess.call('echo "java -cp %s org.h2.tools.Server" > /usr/local/bin/h2' % h2dir, shell=True)
                print('[INFO] start H2 service sucessfully.')
            else:
                raise ConnectionError('Cannot start H2 service.')


def load(stock, dbdir, user, psw, **kwargs):
    '''
    arguments:
    ----------
    dbdir: str, data base direction.
    user: str, user name of data base.
    paw: str, password of data base.

    returns:
    --------
    quote: pandas.DataFrame, quote data.
    trade: pandas.DataFrame, trade data.
    '''
    
    h2 = H2Connection(dbdir, user, psw, **kwargs)

    if h2.status:
        sql = "select %s from %s where time between %s and %s or time between %s and %s"
        quote = h2.query(sql % (','.join(QUOTE_COLS), 'quote_' + stock, *TIMES))
        trade = h2.query(sql % (','.join(TRADE_COLS), 'trade_' + stock, *TIMES))
        quote.columns = QUOTE_COLS
        trade.columns = TRADE_COLS
        # form data type.
        int_cols = quote.filter(like='size').columns.tolist()
        float_cols = quote.filter(like='ask').columns.tolist()
        float_cols += quote.filter(like='bid').columns.tolist()
        quote[int_cols]  = quote[int_cols].astype(int)
        quote[float_cols] = quote[float_cols].astype(float)
        trade['price'] = trade['price'].astype(float)
        trade['size']  = trade['size'].astype(int)
    else:
        raise ConnectionError("cannot connect to H2 service, please strat H2 service first.")
    return quote, trade