import torch


weight_file = './results/vwap/600000/hard_constrain/linear/8-tranches/task0_best.pth'
param = torch.load(weight_file)
print(param)