import torch
import torch.nn as nn


class Baseline(object):
    
    def __init__(self, side: str, threshold: int=0.1):
        self._side = 0 if side == 'buy' else 1
        self._threshold = threshold
        self._level = [0, 1] if side == 'buy' else [1, 0]   # level[0] is benifit, level[1] is cost.

    def __call__(self, state):
        '''
        Arugment:
        ---------
        state: list,
            elements consist of ['time', 'start', 'end', 'goal', 'filled'].
        Return:
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


class BasicDNN(nn.Module):

    def __init__(self, criterion, optimizer):
        super().__init__()
        self._criterion = criterion
        self._optimizer = optimizer

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    
class Linear(BasicDNN):
    
    def __init__(self, input_size, output_size, criterion, optimizer):
        super().__init__(criterion, optimizer)
        self.l1 = nn.Linear(input_size, output_size, bias=False)
        nn.init.uniform_(self.l1.weight, 0, 0.001)
    
    def forward(self, x):
        x = torch.Tensor(x)
        x = self.l1(x)
        return x

    
class Lstm(BasicDNN):
    pass