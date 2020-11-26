import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMacro(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x) 
        return x.mean(axis=1).reshape(-1,1)


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
        x = np.array(x)
        if x.ndim == 1:
            x0 = torch.tensor(x[0], device=self.__device)
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
