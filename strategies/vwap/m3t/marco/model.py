import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = x if torch.is_tensor(x) else torch.tensor(x)
        x = self.lstm(x)
        return F.relu(x)