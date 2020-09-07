import glob
import os
import sys

import pandas as pd
import torch
import torch.nn as nn

from exchange.stock import GeneralExchange
from strategies.vwap.agent import Baseline, Linear
from strategies.vwap.env import GenerateTranches, HardConstrainTranche
from strategies.vwap.train import BaselineTraining, EpisodicTraining
from tickdata.datatype import TickData
from utils.statistic import group_trade_volume_by_time, tranche_num


action_encoder = lambda a: [1, 0, 0] if a == 2 else [1, a, 100]


mode        = 'train'
stock       = '600000'
goals       = 20000   # share, goals per days. 
pre_days    = 20
time_range  = [34200000, 41400000, 46800000, 53700000]
interval    = 1800000
tranche_id  = 0
method      = 'baseline'
side        = 'sell'
exchange    = 'general'
level_spcae = ['bid1', 'ask1']
SAVEDIR     = os.path.join('./results', stock)
overwrite   = False


n_tranche = tranche_num(time_range, interval)
if tranche_id not in list(range(n_tranche)):
    raise KeyError('tranche_id should in range(0, %d), but got %d' % (n_tranche, tranche_id))

if mode == 'train':
    quotefiles = sorted(glob.glob('./data/%s/quote/*.csv' % stock))
    tradefiles = sorted(glob.glob('./data/%s/trade/*.csv' % stock))

    if len(quotefiles) != len(tradefiles):
        raise Exception('quote files do not match trade files.')

    quotes = [pd.read_csv(file) for file in quotefiles]
    trades = [pd.read_csv(file) for file in tradefiles]
    datas  = [TickData(quote, trade) for quote, trade in zip(quotes, trades)]
    volume_profiles = [group_trade_volume_by_time(trades[i-pre_days:i], time_range, interval) for i in range(pre_days, len(trades))]

    env_params = list()
    for i in range(pre_days, len(trades)):
        param = dict(tickdata=datas[i], level_space=level_spcae)
        if exchange == 'general':
            param['transaction_engine'] = GeneralExchange(datas[i], 3) 
        else:
            raise KeyError('unkonwn exchange.')
        env_params.append(param)

    envs = [GenerateTranches(HardConstrainTranche, goals, profile, **param)[tranche_id] for profile, param in zip(volume_profiles, env_params)]
            
    if method == 'linear':
        criterion = nn.MSELoss
        optimizer = torch.optim.Adam
        agent = Linear(envs[0].observation_space_n, envs[0].action_space_n, criterion=nn.MSELoss, optimizer=torch.optim.Adam)
        modeldir = os.path.join(SAVEDIR, method, '%dtranches' % n_tranche, 'task%d_best.pth' %i)
        if not overwrite and os.path.exists(modeldir):
            agent.load_state_dict(torch.load(modeldir))
        trainer = EpisodicTraining(agent, action_map=action_encoder)
        trainer.sequence_train(envs, 1000, modeldir)
    else:
        raise KeyError

elif mode == 'test':
    pass

else:
    raise KeyError('argument mode must be train or test.')