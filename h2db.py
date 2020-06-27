import os
import subprocess
import time

import pandas as pd
import psycopg2


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