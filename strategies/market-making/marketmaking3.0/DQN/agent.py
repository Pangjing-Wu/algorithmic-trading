import numpy as np
import torch
import torch.nn as nn
INF = 0x7FFFFFF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Linear(nn.Module):#线性层
    
    def __init__(self, input_size, output_size, criterion=None, optimizer=None):
        super().__init__()
        self.criterion = nn.MSELoss if criterion is None else criterion
        self.optimizer = torch.optim.Adam if optimizer is None else optimizer
        self.l1 = nn.Linear(input_size, output_size, bias=True)
        nn.init.uniform_(self.l1.weight, 0, 0.001)
    
    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x)
        x = x.to(device)
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
        x = np.array(x)
        if x.ndim == 1:
            x0 = torch.tensor(x[0], device=device)
            x1 = torch.tensor(x[1], device=device).unsqueeze(0)
        elif x.ndim == 2:
            x0 = torch.tensor(np.vstack(x[:,0]), device=device, dtype=torch.float32)
            x1 = torch.tensor(np.array([*x[:,1]]), device=device, dtype=torch.float32)
        else:
            raise KeyError('unexcepted dimension of x.')
        x1, _ = self.lstm(x1)
        x = [x0, x1[:,-1,:]]
        x = torch.cat(x, dim=1)
        x = self.l1(x)
        return x
