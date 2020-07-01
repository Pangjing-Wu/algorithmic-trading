# Interfaces description
#### [env.py](env.py)
* Class AlgorithmTrader
    > Main class of algorithm trading environment, simulate order stransaction and provide reacting interface for agent.

    ##### input:
    |argument|type|description|
    |:---|:---|:---|
    |td|`TickData`|tick data.|
    |total_volume|`int`|total issued orders' volume.|
    |reward_function|`str` or `callable`|environment reward funciton.|
    |wait_t|`int`|waiting time befor executing order.|
    |max_level|`int`|max level of trading environment.|

    ##### attributes:
    |variable|type|description|
    |:---|:---|:---|
    |_i|`int`|iteration index.|
    |_td|`TickData`|tick data.|
    |_total_volume|`int`|total volume.|
    |_wait_t|`int`|waiting time befor executing order.|
    |_level_space|`list`|all levels in action space.|
    |_level_space_n|`int`|number of levels in action space.|
    |_reward_function|`str` or `callable`|environment reward funciton.|
    |_time|`list`|timestamp series of quote.|
    |_init|`bool`|is the environment has been initiated.|
    |_final|`bool`|is the environment terminated.|
    |_res_volume|`int`|residual orders' volume.|
    |_simulated_all_trade|`dict`|all traded simulated order |records, keys are ('price', 'size').|
    |level_space|`@property`|public interface of `_level_space`.|
    |level_space_n|`@property`|public interface of `_level_space_n`.|
    |current_time|`@property`|public interface of `_time`.|
    |trade_results|`@property`|public interface of `_simulated_all_trade`.|


    ##### methods:
    * reset()
        > A function to intiate or reset environment, including set `self._init = True`, `self._final = False`, `self._i = 0`, `self._res_volume = self._total_volume` and clear `simulated_all_trade = {'price': [], 'size': []}`. It inputs none and returns the initial state of environment and agent.
    ##### output:
    |variable|type|description|
    |:---|:---|:---|
    |s_0|`np.array`|initial state of environment and agent.|

    * step()
        > A function to step-by-step simulate transaction of orders issued by reinforcement learning agent. It firstly calls `transaction_matching()` to match simulated orders with real quote and trade data. Then according trading results, it calculates transaction cost and determines whether the environment is terminated.

    ##### input:
    |argument|type|description|
    |:---|:---|:---|
    |action|`tuple`, `list`, array like|agent's action, shape likes (direction, price, size).|

    ##### output:
    |variable|type|description|
    |:---|:---|:---|
    |next_s|`np.array`|next state of environment and agent|
    |reward|`float`|environment rewards|
    |signal|`bool`|final signal|
    |info|`str`|detailed transaction information of simulated orders|

#### [transaction.py](transaction.py)
* transaction_matching()
    > A function for matching simulated order with real quote and trade data. It considers 4 transaction matching situations, which are 1) when simulated order's direction is 'buy' and price level is 'ask', the order deals directly, 2) when simulated order's direction is 'buy' and price level is 'bid', the order waits in queue until previous quote orders are traded, 3) when simulated order's direction is 'sell' and price level is 'bid', the order deals directly, 4) when simulated order's direction is 'sell' and price level is 'ask', the order waits in queue until previous quote orders are traded.
    ##### input:
    |argument|type|mean|
    |:---|:---|:---|
    |quote|`pd.DataFrame`|quote board data of time t.|
    |trade|`pd.DataFrame`|trade data of time t.|
    |simulated_order|`dict`|issued simulated order, keys are (direction, price, size, pos), where `pos` is order's position in waiting queue, `pos=-1` denote a new order.|

    ##### output:
    |variable|type|mean|
    |:---|:---|:---|
    |simulated_order|`dict`|residual simulated order, keys are ('direction', 'price', 'size', 'pos'), where `pos` is order's position in waiting queue, `pos=-1` denote a new order.|
    |simulated_trade|`dict`|traded records, keys are ('price', 'size').|

#### [tickdata.py](tickdata.py)
* Class TickData
    > Create a class for tick-level data, which provides abundant function for query and processing quote or trade records.

    ##### input:
    |argument|type|description|
    |:---|:---|:---|
    |quote|`pd.DataFrame`|quote data.|
    |trade|`pd.DataFrame`|trade data.|

    ##### attributes:
    |variable|type|description|
    |:---|:---|:---|
    |_quote|`pd.DataFrame`|quote data.|
    |_trade|`pd.DataFrame`|trade data.|
    |quote_timeseries|`@property`|timestamp series of quote.|
    |trade_timeseries|`@property`|timestamp series of trade.|

    ##### method:
    * quote_board()
        > Load a quote record in quote board shape, *i.e.*

        |level|price|size|
        |:-:|:-:|:-:|
        |ask10|10.10|3250|
        |...|...|...|
        |ask1|10.00|2333|
        |bid1|9.99|1515|
        |...|...|...|
        |bid10|9.90|1390|
        ##### input:
        |argument|type|description|
        |:---|:---|:---|
        |t|`int` or `pd.DataFrame`|timestame of a quote record.|
        ##### output:
        |variable|type|description|
        |:---|:---|:---|
        |tick|`pd.DataFrame`|a quote record in quote board shape.|
    * get_quote()
        > get quote record(s) by timestamp(s), *i.e.*

        |time|ask1|asize1|bid1|bsize1|...|ask10|asize10|bid10|bsize10|
        |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
        |34200000|10.00|2333|9.99|1515|...|10.10|3250|9.90|1390|
        |...|...|...|...|...|...|...|...|...|...|
        ##### input:
        |argument|type|description|
        |:---|:---|:---|
        |t|`None`, `int` or `pd.DataFrame`|timestames of quote records, load all quote records if it is `None`.|
        ##### output:
        |variable|type|description|
        |:---|:---|:---|
        |quote|`pd.DataFrame`|quote records.|

    * get_trade()
        > get quote record(s) by timestamp(s), *i.e.*

        |time|price|size|
        |:-:|:-:|:-:|
        |34200000|10.00|233|
        |34200000|10.01|200|
        |...|...|...|
        ##### input:
        |argument|type|description|
        |:---|:---|:---|
        |t|`None`, `int` or `pd.DataFrame`|timestames of trade records, load all trade records if `t=None`.|
        ##### output:
        |variable|type|description|
        |:---|:---|:---|
        |trade|`pd.DataFrame`|quote records.|

    * pre_quote()
        > get previous one quote record by timestamp
        ##### input:
        |argument|type|description|
        |:---|:---|:---|
        |t|`int` or `pd.DataFrame`|timestame of a quote record.|
        ##### output:
        |variable|type|description|
        |:---|:---|:---|
        |quote|`pd.DataFrame`|previous one quote record, return `None` if there is no previous record|
    * next_quote()
        > get next one quote record by timestamp
        ##### input:
        |argument|type|description|
        |:---|:---|:---|
        |t|`int` or `pd.DataFrame`|timestame of a quote record.|
        ##### output:
        |variable|type|description|
        |:---|:---|:---|
        |quote|`pd.DataFrame`|next one quote record, return `None` if there is no next record|
    * get_trade_between()
        > get all trade record(s) between two quote records.
        ##### input:
        |argument|type|description|
        |:---|:---|:---|
        |pre_quote|`int` or `pd.DataFrame`|timestame of quote.|
        |post_quote|`None`, `int` or `pd.DataFrame`|timestame of quote, use next quote timestamp of `pre_quote` if it is `None`|
        ##### output:
        |variable|type|description|
        |:---|:---|:---|
        |trade|`pd.DataFrame`|trade records, return `None` if there is no next record|
    * trade_sum()
        > combine trade records with the same price.
        ##### input:
        |argument|type|description|
        |:---|:---|:---|
        |trade|`pd.DataFrame`|trade records.|
        ##### output:
        |variable|type|description|
        |:---|:---|:---|
        |trade|`pd.DataFrame`|trade records' sum grouped by thier price.|

#### [h2db.py](h2db.py)