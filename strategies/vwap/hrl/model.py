import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    
class HybridLSTM(nn.Module):
    
    def __init__(self, input_size:tuple, output_size:int, hidden_size:int,
                 num_goals:int, num_layers:int, dropout:float, device='cpu'):
        super().__init__()
        self.lstm = nn.LSTM(input_size[1], hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.l1 = nn.Linear(input_size[0]+hidden_size, num_goals, bias=True)
        self.l2 = nn.Linear(num_goals * 2, output_size, bias=True)
        self.__device = device
        self.__num_goals = num_goals
        
    def forward(self, x, goal):
        x = np.array(x, dtype=object)
        if x.ndim == 1:
            x0 = torch.tensor(x[0], device=self.__device).unsqueeze(0)
            x1 = torch.tensor(x[1], device=self.__device).unsqueeze(0)
        elif x.ndim == 2:
            x0 = torch.tensor(np.vstack(x[:,0]), device=self.__device, dtype=torch.float32)
            x1 = torch.tensor(np.array([*x[:,1]]), device=self.__device, dtype=torch.float32)
        else:
            raise ValueError('unexcepted dimension of x.')
        goal = self.__goal_encode(x0.shape[0], goal)
        x1, _ = self.lstm(x1)
        x = torch.cat([x0, x1[:,-1,:]], dim=1)
        x = self.l1(x)
        x = torch.cat([x, goal], dim=1)
        return self.l2(x)

    def __goal_encode(self, batch_size, goal):
        if isinstance(goal, int):
            goal = [goal]
        encode = torch.zeros(batch_size, self.__num_goals)
        for i, g in enumerate(goal):
            encode[i,g] = 1
        return encode.to(self.__device)

'''
model = HybridLSTM((2,5), 2, 20, 5, 1, 0)
batch = 5
x = [[torch.randn(2).numpy(), torch.randn(3,5).numpy()] for _ in range(batch)] 
x = np.array(x, dtype=object)
goal = [1,2,3,0,2]
print(model(x,goal))
'''