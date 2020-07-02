# Interfaces document
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
    ```python
    class AlgorithmTrader(object):

    def __init__(
            self,
            td: TickData,
            total_volume: int,
            reward_function: callable or str,
            wait_t=0,
            max_level=5
    ):
        self._td = td
        self._total_volume = total_volume
        self._wait_t = wait_t
        self._level_space = list(range(max_level * 2))
        self._level_space_n = len(self._level_space)
        self._reward_function = reward_function
        self._time = self._td.quote_timeseries
        self._init = False
        self._final = False
    ```
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
        
        ```python
        def reset(self)->np.array:
            self._init = True
            self._final = False
            self._i = 0
            self._res_volume = self._total_volume
            self._simulated_all_trade = {'price': [], 'size': []}
            env_s = self._td.quote(self._time[0]).drop('time', axis=1)
            env_s = env_s.values.reshape(-1)
            agt_s = [self._res_volume, 0, 0]
            s_0   = np.append(env_s, agt_s, axis=0)
            return s_0
        ```

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

        ```python
        def step(self, action):
            # raise exception if not initiate or reach final.
            if self._init == False:
                raise NotInitiateError
            if self._final == True:
                raise EnvTerminatedError
            # get current timestamp.
            t = self._td.quote_timeseries[self._i]
            info = 'At %s ms, ' % t
            # load quote and trade.
            quote = self._td.quote_board(t)
            trade = self._td.get_trade_between(t)
            # issue an order if the size of action great than 0.
            if action[-1] > 0:
                order = self._action2order(action) 
                info += 'issue an order %s; ' % order
            else:
                info += 'execute remaining order; '
            # transaction matching
            order, traded = transaction_matching(quote, trade, order)
            self._simulated_all_trade['price'] += traded['price']
            self._simulated_all_trade['size']  += traded['size']
            self._res_volume -= sum(traded['size'])
            info += 'after matching, %s hand(s) were traded at %s and' \
                    '%s hand(s) waited to trade at %s; total.' % (
                        sum(traded['size']), sum(traded['price']),
                        order['size'], order['price']
                    )
            # give a final signal
            if t == self._td.quote_timeseries[-2]:
                self._final = True
            elif self._res_volume == 0:
                self._final = True
            elif self._res_volume < 0:
                # NOTE verify if there is self._res_volume < 0
                print('[WARN] residual volume less than 0.')
                self._final = True
            else:
                self._final = False
            # calculate trasaction cost as reward.
            if self._final == True:
                # if order completed.
                if self._res_volume == 0:
                    if self._reward_function == 'vwap':
                        reward = self._vwap(self._simulated_all_trade)
                    elif self._reward_function == 'twap':
                        reward = self._twap(self._simulated_all_trade)
                    else:
                        reward = self._reward_function(self._simulated_all_trade)
                # if order not completed.
                else:
                    reward = -999.
            else:
                reward = 0.
            # go to next step.
            env_s = self._td.next_quote(t).drop('time', axis=1).values.reshape(-1)
            agt_s = [self._res_volume] + traded['price'] + traded['size']
            next_s = np.append(env_s, agt_s, axis=0)
            return (next_s, reward, self._final, info)
        ```

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
    
    ```python
    def transaction_matching(quote, trade, simulated_order)->tuple:

        # shortcut function
        next_level = lambda level: level[:-1] + str(int(level[-1]) + 1)

        # initial variable
        simulated_trade = {'price': [], 'size': []}

        # return blank simulated_trade if there is no order issued.
        if simulated_order['size'] <= 0:
            return (simulated_order, simulated_trade)

        # map price to level
        simulated_order_level = quote[quote['price'] == simulated_order['price']]
        # return blank simulated_trade if the price is not in quote.
        if simulated_order_level.empty:
            return (simulated_order, simulated_trade)
        else:
            simulated_order_level = simulated_order_level.index[0]
        
        # main matching process
        # ---------------------
        # case 1, direction is 'buy' and level is 'ask', transact directly.
        if simulated_order['direction'] == 'buy':
            # if simulated_order level is 'ask'
            if simulated_order_level[:-1] == 'ask':
                l = 'ask1'    # iterative level.
                simulated_order['pos'] = 0  # transact directly.
                # keep buying until reach simulated_order’s level.
                while l <= simulated_order_level:
                    # skip if quote volume is 0.
                    if quote.loc[l, 'size'] <= 0:
                        continue
                    # if actual quote size is less than our simulated_order need.
                    if quote.loc[l, 'size'] < simulated_order['size']:
                        simulated_trade['price'].append(quote.loc[l, 'price'])
                        simulated_trade['size'].append(quote.loc[l, 'size'])
                        simulated_order['size'] -= quote.loc[l, 'size']
                        l = next_level(l)
                    # if actual quote size is more than our simulated_order need.
                    else:
                        simulated_trade['price'].append(quote.loc[l, 'price'])
                        simulated_trade['size'].append(simulated_order['size'])
                        simulated_order['size'] = 0
                        break
        return(simulated_order, simulated_trade)

        # case 2, direction is 'buy' and level is 'bid', wait in trading queue.
        if simulated_order['direction'] == 'buy':
            if simulated_order_level[:-1] == 'bid':
                # init order position if pos is -1.
                if simulated_order['pos'] == -1:
                    simulated_order['pos'] = quote[simulated_order_level]['size']
                trade = trade[::-1] # reverse price order of trade.    
                # if there is a trade whose price is lower or equal to ours.
                if trade['price'][0] <= simulated_order['price']:
                    # keep buying...
                    for price, size in zip(trade['price'], trade['size']):
                        # until reach simulated_order’s price.
                        if price < simulated_order['price']:
                            break
                        # refresh order position.
                        simulated_order['pos'] = max(0, simulated_order['pos'] - size)
                        # execute order if it is on the front.
                        if simulated_order['pos'] == 0:
                            # if actual trade is less than our simulated_order need.
                            if size < simulated_order['size']:
                                simulated_trade['price'].append(simulated_order['price'])
                                simulated_trade['size'].append(size)
                                simulated_order['size'] -= size
                            # if actual trade is more than our simulated_order need.
                            else:
                                simulated_trade['price'].append(simulated_order['price'])
                                simulated_trade['size'].append(simulated_order['size'])
                                simulated_order['size'] = 0
                                break
        return (simulated_order, simulated_trade)

        # case 3, direction is 'sell' and level is 'bid', transact directly.                            
        if simulated_order['direction'] == 'sell':
            if simulated_order_level[:-1] == 'bid':
                l = 'bid1'    # iterative level.
                simulated_order['pos'] = 0  # transact directly.
                # keep buying until reach the issued simulated_order’s level.
                while l <= simulated_order_level:
                    # continue if quote size is 0.
                    if quote.loc[l, 'size'] <= 0:
                        continue
                    # if actual quote size is less than our simulated_order need.
                    if quote.loc[l, 'size'] <= simulated_order['size']:
                        simulated_trade['price'].append(quote.loc[l, 'price'])
                        simulated_trade['size'].append(quote.loc[l, 'size'])
                        simulated_order['size'] -= quote.loc[l, 'size']
                        i_level = next_level(i_level)
                    # if actual quote size is more than our simulated_order need.
                    else:
                        simulated_trade['price'].append(quote.loc[l, 'price'])
                        simulated_trade['size'].append(simulated_order['size'])
                        simulated_order['size'] = 0
                        break
        return(simulated_order, simulated_trade)

        # case 4, direction is 'sell' and level is 'ask', wait in trading queue.
        if simulated_order['direction'] == 'sell':
            if simulated_order_level[:-1] == 'ask':
                # init order position.
                if simulated_order['pos'] == -1:
                    simulated_order['pos'] = quote[simulated_order_level]['size']
                trade = trade[::-1] # reverse price order of trade.    
                # if there is a trade whose price is higher or equal to ours.
                if trade['price'][0] >= simulated_order['price']:
                    # keep selling...
                    for price, size in zip(trade['price'], trade['size']):
                        # until reach simulated_order’s price.
                        if price < simulated_order['price']:
                            break
                        # refresh order position.
                        simulated_order['pos'] = max(0, simulated_order['pos'] - size)
                        # execute order if it is on the front.
                        if simulated_order['pos'] == 0:
                            # if actual trade is less than our simulated_order need.
                            if size < simulated_order['size']:
                                simulated_trade['price'].append(simulated_order['price'])
                                simulated_trade['size'].append(size)
                                simulated_order['size'] -= size
                            # if actual trade is more than our simulated_order need.
                            else:
                                simulated_trade['price'].append(simulated_order['price'])
                                simulated_trade['size'].append(simulated_order['size'])
                                simulated_order['size'] = 0
                                break
        return(simulated_order, simulated_trade)
    ```

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

    ```python
    class TickData(object):

        def __init__(self, quote: pd.DataFrame, trade: pd.DataFrame):
            # divide quote and trade.
            self._quote = quote
            self._trade = trade
            # set data type.
            int_type_cols = self._quote.filter(like='size').columns.tolist()
            float_type_cols = self._quote.filter(like='ask').columns.tolist()
            float_type_cols += self._quote.filter(like='bid').columns.tolist()
            self._quote[int_type_cols]  = self._quote[int_type_cols].astype(int)
            self._quote[float_type_cols] = self._quote[float_type_cols].astype(float)
            self._trade['price'] = self._trade['price'].astype(float)
            self._trade['size']  = self._trade['size'].astype(int)
    ```

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

        ```python
        def quote_board(self, t:int or pd.DataFrame)->pd.DataFrame:
            level2size = lambda l: l[0] + 'size' + l[3:]
            if type(t) == int:
                quote = self._quote[self._quote['time'] == t]
            elif type(t) == pd.DataFrame:
                quote = t
            else:
                raise TypeError("argument 't' must be int or pd.DataFrame.")
            asks  = quote.filter(like='ask').columns.values[::-1]
            bids  = quote.filter(like='bid').columns.values
            levels = np.r_[asks, bids]
            size_tags = [level2size(l) for l in levels]
            tick = np.c_[quote[levels].values[0], quote[size_tags].values[0]]
            tick = pd.DataFrame(data=tick, index=levels, columns=['price', 'size'])
            tick['size'] = tick['size'].astype(int)
            return tick
        ```

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

        ```python
        def get_quote(self, t:None or int or list = None)->pd.DataFrame:
            if t == None:
                quote = self._quote
            elif type(t) == int:
                quote = self._quote[self._quote['time'] == t]
            else:
                quote = self._quote[self._quote['time'].isin(t)]
            return quote
        ```

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

        ```python
        def get_trade(self, t:None or int or list = None)->pd.DataFrame:
            if t == None:
                trade = self._trade
            elif type(t) == int:
                trade = self._trade[self._trade['time'] == t]
            else:
                trade = self._trade[self._trade['time'].isin(t)]
            return trade
        ```

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

        ```python
        def pre_quote(self, t:int or pd.DataFrame)->pd.DataFrame:
            if type(t) == int:
                pass
            elif type(t) == pd.DataFrame:
                t = t['time'].iloc[0]
            else:
                raise TypeError("argument 't' munst be int or pd.DataFrame.")
            quote = self._quote[self._quote['time'] < t]
            return None if quote.empty else quote.iloc[-1:]
        ```

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

        ```python
        def next_quote(self, t:int or pd.DataFrame)->pd.DataFrame:
            if type(t) == int:
                pass
            elif type(t) == pd.DataFrame:
                t = t['time'].iloc[0]
            else:
                raise TypeError("argument 't' munst be int or pd.DataFrame.")
            quote = self._quote[self._quote['time'] > t]
            return None if quote.empty else quote.iloc[0:1]
        ```

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

        ```python
        def get_trade_between(self, pre_quote:int or pd.DataFrame,
                          post_quote:None or int or pd.DataFrame = None)->pd.DataFrame:
            if type(pre_quote) == int:
                pass
            elif type(pre_quote) == pd.DataFrame:
                pre_quote = int(pre_quote['time'].iloc[0])
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
        ```

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

        ```python
        def trade_sum(self, trade:pd.DataFrame)->pd.DataFrame:
            if trade is None:
                return None
            elif trade.empty:
                return None
            else:
                return trade[['price', 'size']].groupby('price').sum().reset_index()
        ```

        ##### output:
        |variable|type|description|
        |:---|:---|:---|
        |trade|`pd.DataFrame`|trade records' sum grouped by thier price.|

#### [h2db.py](h2db.py)
* Class H2Connection
    > connect H2 database and query data by SQL.
    ##### input:
    |argument|type|description|
    |:---|:---|:---|
    |dbdir|`str`|quote data.|
    |user|`str`|trade data.|
    |password|`str`|database password.|
    |host|`str`|database host, defult is `'localhost'`|
    |port|`str`|database port, defult is `'5435'`|
    |h2_start_wait|`int`|wait time to start h2 service|

    ```python
    class H2Connection(object):

    def __init__(self, dbdir, user, password, host='localhost', port='5435', h2_start_wait=3):
        self.new_connect(dbdir, user, password, host, port, h2_start_wait)
    ```

    ##### attributes:
    |variable|type|description|
    |:---|:---|:---|
    |status|`@property`|H2 connection status, return `None` if it is not connecting.|

    ##### method:
    * new_connect()
        > create new H2 connection, it can automatically detect and start H2 service for MacOS/Linux.
        ##### input:
        |argument|type|description|
        |:---|:---|:---|
        |dbdir|`str`|quote data.|
        |user|`str`|trade data.|
        |password|`str`|database password.|
        |host|`str`|database host, defult is `'localhost'`|
        |port|`str`|database port, defult is `'5435'`|
        |h2_start_wait|`int`|wait time to start h2 service|

        ```python
        def new_connect(self, dbdir, user, password, host='localhost', port='5435', h2_start_wait=3):
            try:
                self._conn = psycopg2.connect(dbname=dbdir, user=user, password=password, host=host, port=port)
            except psycopg2.OperationalError as e:
                if os.name == 'nt':
                    raise ConnectionError("H2 service is not running." \
                        " Since windows doesn't support H2 automatic start, please start h2 service manually.")
                if self._is_h2_online():
                    raise ConnectionError("H2 service is running, but connection is refused." \
                        " Please double check username and password or restart h2 service manually.")
                else:
                    self._start_h2_service(h2_start_wait)
                    self._conn = psycopg2.connect(dbname=dbdir, user=user, password=password, host=host, port=port)
            finally:
                self._cur = self._conn.cursor()
        ```

        * query()
        > query data in H2 database by SQL.
        ##### input:
        |argument|type|description|
        |:---|:---|:---|
        |sql|`str`|SQL.|
        |\*args||argments for execute SQL.|

        ```python
        def query(self, sql: str, *args)->pd.DataFrame:
            self._cur.execute(sql, *args)
            data = self._cur.fetchall()
            data = pd.DataFrame(data)
            return data
        ```

        ##### output:
        |variable|type|description|
        |:---|:---|:---|
        |data|`pd.DataFrame`|SQL execution results.|
