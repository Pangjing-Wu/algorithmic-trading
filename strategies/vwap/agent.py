import torch
import torch.nn as nn


INF = 0x7FFFFFF

class Baseline(object):
    
    def __init__(self, side: str, threshold: int=0.1):
        self._side = 0 if side == 'buy' else 1
        self._threshold = threshold
        self._level = [0, 1] if side == 'buy' else [1, 0]   # level[0] is benifit, level[1] is cost.

    def __call__(self, state):
        '''
        arugment:
        ---------
        state: list,
            elements consist of ['time', 'start', 'end', 'goal', 'filled'].
        return:
        -------
        action: int.
        '''
        if state[4] >= state[3]:
            return 2
        if self._schedule_ratio(state) < 1.05:
            return 0
        elif self._schedule_ratio(state) > 1.05 + self._threshold:
            return 2
        else:
            return 1

    def _schedule_ratio(self, state):
        if state[3] == 0:
            return INF
        filled_ratio = state[4] / state[3]
        time_ratio = (state[0] - state[1]) / (state[2] - state[1])
        if filled_ratio == 1:
            return INF
        if time_ratio == 0:
            return 1
        else:
            return filled_ratio / time_ratio

    
class Linear(nn.Module):
    
    def __init__(self, input_size, output_size, criterion=None, optimizer=None):
        super().__init__()
        self.criterion = nn.MSELoss if criterion is None else criterion
        self.optimizer = torch.optim.Adam if optimizer is None else optimizer
        self.l1 = nn.Linear(input_size, output_size, bias=True)
        nn.init.uniform_(self.l1.weight, 0, 0.001)
    
    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x)
        x = self.l1(x)
        return x

    
class LSTM(nn.Module):
    
    def __init__(self, input_size:tuple, output_size:int, hidden_size=20,
                 num_layers=1, dropout=0, criterion=None, optimizer=None):
        super().__init__()
        self.criterion = nn.MSELoss if criterion is None else criterion
        self.optimizer = torch.optim.Adam if optimizer is None else optimizer
        self.l1   = nn.Linear(input_size[0]+hidden_size, output_size, bias=True)
        self.lstm = nn.LSTM(input_size[1], hidden_size, num_layers, dropout=dropout, batch_first=True)
        nn.init.uniform_(self.l1.weight, 0, 0.001)
        
    def forward(self, x):
        if x.ndim == 1:
            x0 = torch.tensor(x[0])
            x1 = torch.tensor(x[1])
        elif x.ndim == 2:
            x0 = torch.tensor(x[:,0])
            x1 = torch.tensor(x[:,1])
        else:
            raise KeyError('unexcepted dimension of x.')
        x1, _ = self.lstm(x1)
        x = [x0, x1[:,-1,:]]
        x = torch.cat(x, dim=1)
        x = self.l1(x)
        return x
