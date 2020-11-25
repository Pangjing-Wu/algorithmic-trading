import glob
import json
import os
import sys

sys.path.append('/home/chendy/code/algorithmic-trading-master/datasource')
sys.path.append('/home/chendy/code/algorithmic-trading-master/exchange')
import pandas as pd
import torch
import torch.nn as nn

from datatype import TickData
#from options import parse_args
from stock import GeneralExchange
from agentt import lstm1
from envs import Env
from train import QLearning


arg_agent='lstm'
arg_mode='train'#或者'test'
#选择日期
date=['20140603','20140604','20140605','20140606','20140609','20140610']
list_id=[600000]
where='/data/al2/%s/quote'%(list_id[0])
where2='/data/al2/%s/trade'%(list_id[0])

modeldir  = os.path.join('/home/chendy/code/algorithmic-trading-master/strategies/market-making/marketmaking3.0/DQN/result/%s_best.pth'%date[0])
print(modeldir)


criterion = nn.MSELoss
optimizer = torch.optim.Adam
agent = lstm1(166,6, criterion=criterion, optimizer=optimizer)  

name='20140611'
quote1=pd.read_csv(where+'/'+name+'.csv')
trade1=pd.read_csv(where2+'/'+name+'.csv')
data=TickData(quote1, trade1)
env_params=list()
exchange=GeneralExchange(tickdata=data,wait_t=3)

for num in range(4,len(quote1)-1):
    param = dict(tickdata=data,number=num,transaction_engine=exchange.transaction_engine)
    env_params.append(param)
val_envs = [Env(**param) for param in env_params]

if arg_mode == 'train':
    if os.path.exists(modeldir):
        agent.load_state_dict(torch.load(modeldir))
    trainer = QLearning(agent)
    trainer.train(date=date,where=where,where2=where2,TickData=TickData, val_split=0.2,val_envs=val_envs,savedir=modeldir)#episodes不对要改！！！！！
elif arg_mode == 'test':
    if os.path.exists(modeldir):
        agent.load_state_dict(torch.load(modeldir, map_location='cpu'))
    else:
        raise FileNotFoundError('cannot find model file in %s' % modeldir)
    trainer = QLearning(agent)
    for env in val_envs:
        reward  = trainer.test(env)
        print('test reward = %.5f' % reward)
        #print('test metric = %s' % env.metrics())
else:
    raise KeyError('argument mode must be train or test.')  
