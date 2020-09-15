# Algorithm-trading-environment

**Algorithm-trading-environment** is a simulating trading environment designed for reinforcement learning agents. It is running on **tick-level** quote and trade data to reproduce trading process incorprating with new simulated orders. 
 
Like general reinforcement learning environment, **algorithm-trading-environment** can be represented by a tuple of *(s, a, p, r)*, where:
* environment state *s* is a numerical vector that consists of current real quote (*i.e.* market state) and simulated order status (*i.e.* agent state);
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

> Train agent based on vwap trading strategies
```bash
nohup python -u vwap.py --mode train --env histrical_hard_constrain --agent linear --stock 600000 --side sell --level 1 --tranche_id 0 2>&1 >./logs/600000-histical-hard-linear-0-8.log &
```
> Test agent based on vwap trading strategies
```bash
python -u vwap.py --mode test --env histrical_hard_constrain --agent linear --stock 600000 --side sell --level 1 --tranche_id 0
```

## Contents

* [env.py](env.py) is main file of **algorithm-trading-environment**, provides algorithmic trading environment interface for agent.
* [tickdata.py](tickdata.py) creates a class for tick-level data, which provides abundant function for query and processing quote or trade records.
* [utils/](./utils) contains some functions to support algorithmic trading.
    * [transaction.py](transaction.py) is transaction matching module to executing simulated order by matching it with real quote and trade data.
    * [h2db.py](h2db.py) is h2 database connection and query module.

Besides, [tutorial/](./tutorial) contains some examples of guiding to use the envronment, [test](./test) contains test configuration and test cases, [doc/](./doc) contains documents of this program.


## Module design

The core of this program is to match simulated order with real quote and trade data to evaluate transaction cost. To address this problem, we need to

1. read tick data from H2 database and convert to some type that is easy to handle in Python, such as `pd.DataFrame`.
2. preprocess real quote and trade data;
3. issue simulated order and match it with real quote and trade data;
4. sequencially execute transaction matching process.

> Read data from H2 database and convert to some type that is easy to handle in Python

Since raw data is stored in H2 database, we firstly design a module [h2bd.py](h2db.py) to read the raw tick data. [h2bd.py](h2db.py) provides `H2Connection` class to get connection with H2 database under the protocol provided by `psycopg2` library. It support automatically detect H2 service status and start H2 service for MacOS/Linux. Data query is conducted by executing `H2Connection.query`.

> Preprocess real quote and trade data

These raw data are preprocessed and converted to `TickData` class which defined in [tickdata.py](tickdata.py). It provides abundant function for processing quote and trade data, such as `get_quote` or `get_trade` record(s) by timestamp, get `pre_quote` or `next_quote` by timestamp or quote record, and `get_trade_between` two quotes, even you can tranform quote from record form to quote board form by `quote_board`.

> Issue simulated order and match it with real quote and trade data

We design `transaction_matching` function in [transaction.py](transaction.py) to match simulated order with real quote and trade data. It considers 4 transaction matching situations, which are
1. when simulated order's direction is 'buy' and price level is 'ask', the order is closed directly.
2. when simulated order's direction is 'buy' and price level is 'bid', the order waits in queue until previous quote orders are closed.
3. when simulated order's direction is 'sell' and price level is 'bid', the order is closed directly.
4. when simulated order's direction is 'sell' and price level is 'ask', the order waits in queue until previous quote orders are closed.

> Sequencially execute transaction matching process

We design `AlgorithmicTrading` in [env.py](evn.py) whose interface is like the famous reinforcement learning library `gym`. It provides `reset` to initiate or reset environment and `step` to execute agent's action in environment.

## Interface document
The detailed interface document are presented in [interface](doc/interface.md).

## Update
#### 2020/7/x
Release first version.

## Contributors
* [Pangjing-Wu](https://github.com/Pangjing-Wu)
* [cdanni9120](https://github.com/cdanni9120)

## License
No license, private program, **DO NOT** distribute.
