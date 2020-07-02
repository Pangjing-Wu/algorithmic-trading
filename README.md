# Algorithm-trading-environment

**Algorithm-trading-environment** is a simulating trading environment designed for reinforcement learning agents. It is running on **tick-level** quote and trade data to reproduce trading process incorprating with new simulated orders. 
 
Like general reinforcement learning environment, **algorithm-trading-environment** can be represented by a tuple of *(s, a, p, r)*, where:
* environment state *s* is a numerical vector that consists of current actual quote (*i.e.* market state) and simulated order status (*i.e.* agent state);
* action *a* is a tuple whose shape likes (trading direction, level, size);
* transition probability *p* describes state transition in general reinforcement learning environment, but the transition is constant and sequential in this environment;
* reward *r* denotes transaction cost of trading strategies.


## Installation

Download zip file or use `git clone`.

```bash
$ git clone https://github.com/Pangjing-Wu/algorithm-trading-strategy-environment.git
```

*TODO*: we will release it as a python module in the future.


## Usage

Here is a quick start example for loading **algorithm-trading-environment**.

> Connection H2 data base and query quote and trade data, and create `TickData` to store raw tick data.
```python
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

td = load(quote, trade)
```
> Build transaction environment by `TickData`.
```python
from env import AlgorithmTrader
trader = AlgorithmTrader(td=td, total_volume=20000, reward_function='vwap', wait_t=0, max_level=5)
trader.reset()
action = {'direction': 'buy', 'price': 10.0, 'size': 500}
(s_next, reward, signal, info) = trader.step(action)
```

## Contents

* [env.py](env.py) is main file of **algorithm-trading-environment**, provides reacting trading environment interface for agent.
* [transaction.py](transaction.py) is transaction matching module to executing simulated order by matching it with actual quote and trade data.
* [tickdata.py](tickdata.py) creates a class for tick-level data, which provides abundant function for query and processing quote or trade records.
* [h2db.py](h2db.py) is h2 database connection and query module.

Besides, [tutorial](./tutorial) contains some examples of guiding to use the envronment, [test](./test) contains test configuration and test cases, [config](./config) defines some constant and variable that not changes often.


## Module design

The core of this program is to match simulated order with actual quote and trade data to evaluate transaction cost. To address this problem, we need to

1. read data from H2 database and convert to some type that is easy to handle in Python, such as `pd.DataFrame`.
2. preprocess actual quote and trade data;
3. issue simulated order and match it with actual quote and trade data;
4. sequencially execute transaction matching process.

> Read data from H2 database and convert to some type that is easy to handle in Python

Since raw data is stored in H2 database, we firstly design a module [h2bd.py](h2db.py) to read the raw tick data. [h2bd.py](h2db.py) provides `H2Connection` class to get connection with H2 database under the protocol provided by `psycopg2` library. It support automatically detect H2 service status and start H2 service for MacOS/Linux. Data query is conducted by executing `H2Connection.query`.

> Preprocess actual quote and trade data

These raw data are preprocessed and converted to `TickData` class which defined in [tickdata.py](tickdata.py). It provides abundant function for processing quote and trade data, such as `get_quote` or `get_trade` record(s) by timestamp, get `pre_quote` or `next_quote` by timestamp or quote record, and `get_trade_between` two quotes, even you can tranform quote from record form to quote board form by `quote_board`.

> Issue simulated order and match it with actual quote and trade data

We design `transaction_matching` function in [transaction.py](transaction.py) to match simulated order with actual quote and trade data. It considers 4 transaction matching situations, which are
1. when simulated order's direction is 'buy' and price level is 'ask', the order deals directly.
2. when simulated order's direction is 'buy' and price level is 'bid', the order waits in queue until previous quote orders are traded.
3. when simulated order's direction is 'sell' and price level is 'bid', the order deals directly.
4. when simulated order's direction is 'sell' and price level is 'ask', the order waits in queue until previous quote orders are traded.

> Sequencially execute transaction matching process

We design `AlgorithmTrader` in [env.py](evn.py) whose interface is like the famous reinforcement learning library `gym`. It provides `reset` to initiate or reset environment and `step` to execute agent's action in environment.

## Interface document
The detailed interface document are presented in [interface](doc/interface.md).

## Contributer
* [Pangjing-Wu](https://github.com/Pangjing-Wu)
* [cdanni9120](https://github.com/cdanni9120)

## License
No license, private program, **DO NOT** distribute.
