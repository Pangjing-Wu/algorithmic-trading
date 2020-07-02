import sys
sys.path.append('./')

from h2db import H2Connection
from tickdata import TickData

def load(stock, dbdir, user, psw)->TickData:
    h2 = H2Connection(dbdir, user, psw)
    QUOTE_COLS = ["time", "bid1", "bsize1", "ask1", "asize1", "bid2", "bsize2", "ask2",
        "asize2", "bid3", "bsize3", "ask3", "asize3", "bid4", "bsize4", "ask4", "asize4",
        "bid5", "bsize5","ask5", "asize5", "bid6", "bsize6", "ask6", "asize6", "bid7",
        "bsize7", "ask7", "asize7", "bid8", "bsize8", "ask8", "asize8", "bid9", "bsize9",
        "ask9", "asize9", "bid10", "bsize10", "ask10", "asize10"]
    TRADE_COLS = ["time", "price", "size"]
    TIMES = [34200000, 41400000, 46800000, 54000000]
    if h2.status:
        sql = "select %s from %s where time between %s and %s or time between %s and %s"
        quote = h2.query(sql % (','.join(QUOTE_COLS), 'quote_' + stock, *TIMES))
        trade = h2.query(sql % (','.join(TRADE_COLS), 'trade_' + stock, *TIMES))
        quote.columns = QUOTE_COLS
        trade.columns = TRADE_COLS
    else:
        raise ConnectionError("cannot connect to H2 service, please strat H2 service first.")
    return TickData(quote, trade)