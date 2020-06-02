# trading-strategy-environment
 A reinforcement learning environment for simulating trading strategy
## Configuration
### dataloader.py
#### class TickData()
* Attributes:
    * Some attributes to describe data characteristics.
    * Such as **quote** and **trade** data length, start and end timestamps, etc.
* Methods:
    * \_\_init__(data:_H2.database_)->_pandas.DataFrame_:
        1. Use python library "psycopg" to link h2 database and transform to pandas.DataFrame.
        2. Select a specific stock.
        3. Drop noise.
        4. Meger (or concatenate) **quote** and **trade** data and sort by time.
        5. Set tags to specify which type (quote or trade) each line is.
        6. Fill null data. What value to use to fill N/A caused by limit up/down?
    * Get previous/next quote data of current quote/trade.
    * Get trade data between two quote (support one or two arguments).
    * Sum trade data between two quote (support one or two arguments).
    * Get quote series
    * Generate quote series (which is generator type).
    * Get time series of all/quote/trade data.

### env.py
#### class Trader()
* Attributes:
    * Some attributes to describe _<s,a,r,p>_.
    * Simulated order book.
* Methods:
    * \_\_init__(TickData, strategy_direction, reward: _callable_ or _str_):
    * reset(): rest environment.
    * next_s, reward, signal, info = step(action):
        1. issue an order;
        2. matching transactions;
        3. calculate reward;
        4. give a final signal;
        5. refresh the real/simulated order book and go to next step.

