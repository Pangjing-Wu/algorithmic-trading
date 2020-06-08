import psycopg2
from dataloader import H2Connection
import subprocess
import os
import time
import pandas as pd

h2db = H2Connection('~/OneDrive/python-programs/reinforcement-learning/data/20140704', 'cra001', 'cra001')
print(h2db.status)
quote_cols = ['time', 'bid1', 'bsize1', 'ask1', 'asize1', 'bid2', 'bsize2', 'ask2', 'asize2', 'bid3', 'bsize3',
              'ask3', 'asize3', 'bid4', 'bsize4', 'ask4', 'asize4', 'bid5', 'bsize5','ask5', 'asize5', 'bid6',
              'bsize6', 'ask6', 'asize6', 'bid7', 'bsize7', 'ask7', 'asize7', 'bid8', 'bsize8', 'ask8', 'asize8',
              'bid9', 'bsize9', 'ask9', 'asize9', 'bid10', 'bsize10', 'ask10', 'asize10', 'null', 'null']
trade_cols = ['time', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null',
              'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null',
              'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null',
              'null', 'null', 'null', 'null', 'null', 'price', 'size']

sql = 'select %s from quote_%s where time between %s and %s or time between %s and %s\n' \
      'union\n' \
      'select %s from trade_%s where time between %s and %s or time between %s and %s\n' \
      'order by time' % (', '.join(quote_cols), '000001', '34200000', '41400000', '46800000', '54000000',
                         ', '.join(trade_cols), '000001', '34200000', '41400000', '46800000', '54000000')


if h2db.status:
    data = h2db.query(sql)
    data = pd.DataFrame(data, columns=quote_cols[:-2]+trade_cols[-2:])
    data = data.reset_index(drop=True)
    print(data)