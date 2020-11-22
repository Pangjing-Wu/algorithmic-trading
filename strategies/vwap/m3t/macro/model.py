import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):

    def __init__(self, input_size:int, output_size:int):
        super().__init__()
        self.criterion = nn.MSELoss
        self.optimizer = torch.optim.Adam
        self.l1 = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x)
        x = self.l1(x)
        return F.relu(x)


class MLP(nn.Module):

    def __init__(self, input_size:int, hidden_size:tuple, output_size):
        super().__init__()
        self.criterion = nn.MSELoss
        self.optimizer = torch.optim.Adam
        layers = list()
        layer_size = [input_size] + list(hidden_size) + [output_size]
        for i, j in zip(layer_size[:-1], layer_size[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(nn.ReLU(True))
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x)
        return self.layer(x)


class LSTM(nn.Module):
    
    def __init__(self, input_size:int, output_size:int, hidden_size=20,
                 num_layers=1, dropout=0):
        super().__init__()
        self.criterion = nn.MSELoss
        self.optimizer = torch.optim.Adam
        self.l1   = nn.Linear(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        
    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x)
        x = x.reshape(*x.shape, 1)
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = F.relu(x)
        return self.l1(x)