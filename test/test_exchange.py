import sys

sys.path.append('./')
from data.tickdata import CSVDataset
from exchange.stock import AShareExchange, ClientOrder

path = '/data/al2/'
stock = '600000'

dataset = CSVDataset(path, stock)
tickdata = dataset[0]

exchange = AShareExchange(tickdata, wait_trade=3)

'''
order = ClientOrder(tickdata.quote.timeseries[1], 'sell', 9.58, 100)

exchange.issue(1, order)
exchange.issue(2, order)
exchange.issue(1, order)

print(tickdata.quote.get(tickdata.quote.timeseries[1]))
order = exchange.step(tickdata.quote.timeseries[1])
print(order)
'''


order = ClientOrder(tickdata.quote.timeseries[2], 'sell', 9.6, 2000000)
exchange.issue(1, order)

for i in range(2, 10):
    print(tickdata.quote.get(tickdata.quote.timeseries[i]))
    order = exchange.step(tickdata.quote.timeseries[i])
    print(order.filled[order.filled['time'] == tickdata.quote.timeseries[i]]['size'].sum())
