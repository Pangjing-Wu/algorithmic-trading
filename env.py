from dataloader import TickData

class AlgorithmTrader(object):

    def __init__(self, data:TickData, strategy_direction,
        volume:int, reward_function:callable or str,
        *args:'[arguments to describe environment]'):
        pass

    def reset(self):
        # TODO reset invironment.
        pass

    def step(self, action)->'(next_s, reward, signal, info)':
        # TODO environment step:
        # Issue an order
        # Matching transactions;
        # Calculate reward;
        # Give a final signal;
        # Refresh the real/simulated order book and go to next step.
        pass

    def _transaction_matching(self, args):
        pass

    def _vwap(self, price, volumn):
        pass

    def _twap(self, price, volumn):
        pass
