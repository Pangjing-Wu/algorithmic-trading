import numpy as np
import torch
import torch.nn as nn
INF = 0x7FFFFFF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''class linear1(nn.Module):#线性层
    
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
        return x'''


class lstm1(nn.Module):
    def __init__(self,input_size, output_size:int, hidden_size=20,
                 num_layer=1, dropout=0, criterion=None, optimizer=None):
        super().__init__()
        self.criterion = nn.MSELoss if criterion is None else criterion
        self.optimizer = torch.optim.Adam if optimizer is None else optimizer
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True)
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, dropout=dropout, batch_first=True)
        nn.init.uniform_(self.layer2.weight, 0, 0.001)
        
    def forward(self, x):
        dataX = np.array(x)
        if dataX.ndim==1:
            dataX=[[dataX]]
            x=torch.tensor(dataX, dtype=torch.float32,device=device)
            out,_= self.layer1(x)
            out = out[-1,:,:]
            out = self.layer2(out)
            #print(out)
            out=nn.functional.softmax(out,dim=1) 
        else:
            dataX=[dataX]
            x=torch.tensor(dataX, dtype=torch.float32,device=device)
            out,_= self.layer1(x)
            out = out[-1,:,:]
            out = self.layer2(out)
            out=nn.functional.softmax(out,dim=1) 

            
        return out

