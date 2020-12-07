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
        # action[0] is benifit, action[1] is cost.
        if side == 'buy':
            self._action = [level-1, level, 2 * level]
        elif side == 'sell':
            self._action = [level, level-1, 2 * level]
        else:
            raise ValueError('unknown transaction side.')

    def __call__(self, time_ratio, filled_ratio):
        '''
        arugment:
        ---------
        state: list,
            elements consist of ['time', 'start', 'end', 'goal', 'filled'].
        return:
        -------
        action: int.
        '''
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
        self.l1   = nn.Linear(input_size[0]+hidden_size, output_size, bias=True)
        self.lstm = nn.LSTM(input_size[1], hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.__device = device
        
    def forward(self, x):
        x = np.array(x, dtype=object)
        if x.ndim == 1:
            x0 = torch.tensor(x[0], device=self.__device).unsqueeze(0)
            x1 = torch.tensor(x[1], device=self.__device).unsqueeze(0)
        elif x.ndim == 2:
            x0 = torch.tensor(np.vstack(x[:,0]), device=self.__device, dtype=torch.float32)
            x1 = torch.tensor(np.array([*x[:,1]]), device=self.__device, dtype=torch.float32)
        else:
            raise KeyError('unexcepted dimension of x.')
        x1, _ = self.lstm(x1)
        x = torch.cat([x0, x1[:,-1,:]], dim=1)
        x = self.l1(x)
        return x


class HybridAttenBiLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,
                 num_layers, dropout, attention_size, device='cpu'):
        super().__init__()

        self.__device = device
        self.__att_w  = torch.zeros(hidden_size * num_layers * 2, attention_size).to(self.__device)
        self.__att_u = torch.zeros(attention_size).to(self.__device)
        self.l1 = nn.Linear(hidden_size * num_layers * 2, output_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, bidirectional=True,
                            batch_first=True)

    def __attention_net(self, x):
        ''' M = tanh(wH); a = softmax(uM); r = Ha
        '''
        sequen_size = x.shape[1]
        hidden_size = x.shape[2]
        attention = torch.tanh(torch.mm(x.reshape(-1, hidden_size), self.__att_w))
        attention = torch.mm(attention, self.__att_u.reshape(-1, 1))
        alphas = F.softmax(attention, dim=0).reshape(-1, sequen_size, 1)
        x = torch.sum(x * alphas, 1)
        return x

    def forward(self, x, batch_size=None):
        x, _ = self.lstm(x)
        x = self.__attention_net(x)
        x = self.l1(x)
        return x