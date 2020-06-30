# Algorithm-trading-environment

**Algorithm-trading-environment** is a simulating trading environment designed for reinforcement learning agents. It is running on **tick-level** quote and trade data to reproduce trading process incorprating with new simulated orders. 
 
Like general reinforcement learning environment, **algorithm-trading-environment** can be represented by a tuple of $(s, a, p, r)$, where:
* environment state $s$ is a numerical vector that consists of current actual quote (i.e. market state) and simulated order status (i.e. agent state);
* action $a$ is a tuple likes ("trading direction", "level", "size") given by agent;
* transition probability $p$ describes state transition in general reinforcement learning environment, but the transition is constant and sequential in this environment;
* reward $r$ denotes transaction cost of trading strategies.


## Installation

Download zip file or use `git clone`.

```bash
$ git clone https://github.com/Pangjing-Wu/algorithm-trading-strategy-environment.git
```

*TODO*: we will release it as a python module in the future.


## Usage

Here is a quick start example for loading **algorithm-trading-environment**.

> Connection H2 data base and query quote and trade data.
```python
from h2db import H2Connextion
h2 = H2Connection(dbdir, user, psw, config)
quote = h2.query('QUOTE_000001')
trade = h2.query('TRADE_000001')
```
> Create `TickData` to store and preprocess raw tick data.
```python
from tickdata import TickData
td = TickData(quote, trade)
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

1. preprocess actual quote and trade data;
2. issue simulated order and match it with actual quote and trade data;
3. sequencially execute transaction matching process.

> Preprocess actual quote and trade data

For the first point, since raw data is stored in H2 database, we firstly design a module [h2bd.py](h2db.py) to read the raw tick data. [h2bd.py](h2db.py) provides `H2Connection` class to get connection with H2 database by `psycopg2` library. It support automatically detect H2 service status and start H2 service for MacOS/Linux. Data query is conducted by executing `H2Connection.query` with normal SQL language.

Then, these raw data are preprocessed and transformed to `TickData` class which defined in [tickdata.py](tickdata.py). It provides abundant function for processing quote and trade data, such as `get_quote` or `get_trade` record(s) by timestamp, get `pre_quote` or `next_quote` by timestamp or quote record, and `get_trade_between` two quotes, even you can tranform quote from record form to quote board form by `quote_board`.

> Issue simulated order and match it with actual quote and trade data

For the second point, we design `transaction_matching` function in [transaction.py](transaction.py) to match simulated order with actual quote and trade data. It considers 4 transaction matching situations, which are
1. when simulated order's direction is 'buy' and price level is 'ask', the order deals directly.
2. when simulated order's direction is 'buy' and price level is 'bid', the order waits in queue until previous quote orders are traded.
3. when simulated order's direction is 'sell' and price level is 'bid', the order deals directly.
4. when simulated order's direction is 'sell' and price level is 'ask', the order waits in queue until previous quote orders are traded.

> Sequencially execute transaction matching process

For the third point, we design `AlgorithmTrader` in [env.py](evn.py) whose interface is like the famous reinforcement learning library `gym`. It provides `reset` to initiate or reset environment and `step` to execute agent's action in environment.

## Interfaces description
#### [env.py](env.py)


* AlgorithmTrader()
    > Main class of algorithm trading environment, simulate order stransaction and provide reacting interface for agent.

    ##### input:
    |variable|type|description|
    |:---|:---|:---|
    |td|`TickData`|tick data.|
    |total_volume|`int`|total issued orders' volume.|
    |reward_function|`str` or `callable`|environment reward funciton.|
    |wait_t|`int`|waiting time befor executing order.|
    |max_level|`int`|max level of trading environment.|

    #### attributes:
    |variable|type|description|
    |:---|:---|:---|
    |_i|`int`|iteration index|
    |_td|`TickData`|tick data.|
    |_total_volume|`int`|total volume.|
    |_wait_t|`int`|waiting time befor executing order.|
    |_level_space|`list`|all levels in action space.|
    |_level_space_n|`int`|number of levels in action space.|
    |_reward_function|`str` or `callable`|environment reward funciton.|
    |_time|`list`|timestamp series of quote.|
    |_init|`bool`|is the environment has been initiated.|
    |_final|`bool`|is the environment terminated.|
    |_res_volume|`int`|residual orders' volume|
    |_simulated_all_trade|`dict`|all traded simulated order |records, keys are ('price', 'size')|
    |level_space|`@property`|public interface of `_level_space`|
    |level_space_n|`@property`|public interface of `_level_space_n`|
    |current_time|`@property`|public interface of `_time`|
    |trade_results|`@property`|public interface of `_simulated_all_trade`|


    #### methods:
    * reset()
    > A function to intiate or reset environment, including set `self._init = True`, `self._final = False`, `self._i = 0`, `self._res_volume = self._total_volume` and clear `simulated_all_trade = {'price': [], 'size': []}`. It returns the initial state of environment and agent.
    ##### output:
    |variable|type|description|
    |:---|:---|:---|
    |s_0|`np.array`|initial state of environment and agent.|

    * step()
    > A function to step-by-step simulate transaction of orders issued by reinforcement learning agent. It firstly calls `transaction_matching()` to match simulated orders with real quote and trade data. Then according trading results, it calculates transaction cost and determines whether the environment is terminated.

    ##### input:
    variable|type|description|
    |:---|:---|:---|
    |action|`tuple`, `list`, array like|agent's action, like (direction, price, size).|

    ##### output:
    variable|type|description|
    |:---|:---|:---|
    |next_s|`np.array`|next state of environment and agent|
    |reward|`float`|environment rewards|
    |signal|`bool`|final signal|
    |info|`str`|detailed transaction information of simulated orders|
#### [transaction.py](transaction.py)
* transaction_matching()
    > A function for matching simulated order with real quote and trade data. It considers 4 transaction matching situations, which are 1) when simulated order's direction is 'buy' and price level is 'ask', the order deals directly, 2) when simulated order's direction is 'buy' and price level is 'bid', the order waits in queue until previous quote orders are traded, 3) when simulated order's direction is 'sell' and price level is 'bid', the order deals directly, 4) when simulated order's direction is 'sell' and price level is 'ask', the order waits in queue until previous quote orders are traded.
    ##### input:

    |name|type|mean|
    |:---|:---|:---|
    |quote|`pd.DataFrame`|quote board data of time t.|
    |trade|`pd.DataFrame`|trade data of time t.|
    |simulated_order|`dict`|issued simulated order, keys are (direction, price, size, pos), where `pos` is order's position in waiting queue, `pos=-1` denote a new order.|

    ##### output:

    |name|type|mean|
    |:---|:---|:---|
    |simulated_order|`dict`|residual simulated order, keys are ('direction', 'price', 'size', 'pos'), where `pos` is order's position in waiting queue, `pos=-1` denote a new order.|
    |simulated_trade|`dict`|traded records, keys are ('price', 'size').

#### [tickdata.py](tickdata.py)
#### [h2db.py](h2db.py)
