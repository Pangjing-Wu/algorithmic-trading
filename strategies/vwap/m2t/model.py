import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroBaseline(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x) 
        return x.mean(axis=1).reshape(-1,1)


class MicroBaseline(object):
    
    def __init__(self, side:str, level=1, lb=1.01, ub=1.1):
        if not lb < ub:
            raise ValueError('lb must be less than ub.')
        self._lb = lb
        self._ub = ub
        self._action = [
            torch.zeros(2*level+1),
            torch.zeros(2*level+1),
            torch.zeros(2*level+1)
            ]
        # action[0] is benifit, action[1] is cost.
        if side == 'buy':
            self._action[0][level-1] = 1
            self._action[1][level]   = 1
            self._action[2][-1]      = 1
        elif side == 'sell':
            self._action[0][level]   = 1
            self._action[1][level-1] = 1
            self._action[2][-1]      = 1
        else:
            raise ValueError('unknown transaction side.')

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
        time_ratio, filled_ratio = state
        time_ratio = 1e-10 if time_ratio == 0 else time_ratio
        if filled_ratio == 1:
            return self._action[2]
        if filled_ratio / time_ratio < self._lb:
            return self._action[1]
        elif filled_ratio / time_ratio > self._ub:
            return self._action[2]
        else:
            return self._action[0]


class Linear(nn.Module):

    def __init__(self, input_size:int, output_size:int, device='cpu'):
        super().__init__()
        self.l1 = nn.Linear(input_size, output_size)
        self.__device = device
    
    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x)
        x = x.to(self.__device)
        return self.l1(x)
         

class MLP(nn.Module):

    def __init__(self, input_size:int, hidden_size:tuple,
                 output_size:int, device='cpu'):
        super().__init__()
        layers = list()
        layer_size = [input_size] + list(hidden_size) + [output_size]
        for i, j in zip(layer_size[:-1], layer_size[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(nn.Dropout(p=0.5))
        self.layer = nn.Sequential(*layers)
        self.__device = device
    
    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x)
        x = x.to(self.__device)
        return self.layer(x)


class LSTM(nn.Module):
    
    def __init__(self, input_size:int, output_size:int, hidden_size=20,
                 num_layers=1, dropout=0, device='cpu'):
        super().__init__()
        self.l1   = nn.Linear(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.__device = device
          
    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x)
        x = x.to(self.__device)
        x = x.reshape(*x.shape, 1)
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = F.relu(x)
        return self.l1(x)


class HybridLSTM(nn.Module):
    
    def __init__(self, input_size:tuple, output_size:int, hidden_size=20,
                 num_layers=1, dropout=0, device='cpu'):
        super().__init__()
        self.l1   = nn.Linear(input_size[0]+ 2 * hidden_size, output_size, bias=True)
        self.lstm11 = nn.LSTM(input_size[1] // 2, hidden_size, num_layers,
                              dropout=dropout, batch_first=True)
        self.lstm12 = nn.LSTM(input_size[1] // 2, hidden_size, num_layers,
                              dropout=dropout, batch_first=True)
        self.__device = device
        
    def forward(self, x):
        '''
        x shape = (in_state, ex_state) or (N, (in_state, ex_state))
        ex_state shape = ([price, volume], seq, level)
        '''
        x = np.array(x, dtype=object)
        if x.ndim == 2:
            x0  = torch.tensor(np.array(x[0], dtype=np.float32), device=self.__device).unsqueeze(0)
            x11 = torch.tensor(np.array(x[1, 0], dtype=np.float32), device=self.__device).unsqueeze(0)
            x12 = torch.tensor(np.array(x[1, 1], dtype=np.float32), device=self.__device).unsqueeze(0)
        elif x.ndim == 3:
            x0  = torch.tensor(np.array(x[:,0], dtype=np.float32), device=self.__device, dtype=torch.float32)
            x11 = torch.tensor(np.array([*x[:,1, 0]], dtype=np.float32), device=self.__device, dtype=torch.float32)
            x12 = torch.tensor(np.array([*x[:,1, 1]], dtype=np.float32), device=self.__device, dtype=torch.float32)
        else:
            raise KeyError('unexcepted dimension of x.')
        x11, _ = self.lstm11(x11)
        x12, _ = self.lstm11(x12)
        x = torch.cat([x0, x11[:,-1,:], x12[:,-1,:]], dim=1)
        x = self.l1(x)
        return x