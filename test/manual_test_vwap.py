import glob
import os
import sys

import pandas as pd
import torch
import torch.nn as nn

sys.path.append('./')
from exchange.stock import GeneralExchange
from strategies.vwap.agent import Baseline, Linear
from strategies.vwap.env import GenerateTranches, HardConstrainTranche
from strategies.vwap.train import BaselineTraining, EpisodicTraining
from tickdata.datatype import TickData
from utils.statistic import group_trade_volume_by_time


action_encoder = lambda a: [1, 0, 0] if a == 2 else [1, a, 100]


def state2dict(state) -> dict:
    keys = ['time', 'start', 'end', 'goal', 'filled']
    state_dict = {}
    for k, i in zip(keys, state):
        state_dict[k] = i
    return state_dict


stock         = '600000'
goals         = 20000   # share, goals per days. 
pre_days      = 20
time_range    = [34200000, 41400000, 46800000, 53700000]
time_interval = 1800000
tranche_id    = 0

quotefiles = sorted(glob.glob('./data/%s/quote/*.csv' % stock))
tradefiles = sorted(glob.glob('./data/%s/trade/*.csv' % stock))

if len(quotefiles) != len(tradefiles):
    raise Exception('quote files do not match trade files.')

quotes = [pd.read_csv(file) for file in quotefiles]
trades = [pd.read_csv(file) for file in tradefiles]

datas = [TickData(quote, trade) for quote, trade in zip(quotes, trades)]

env_params     = list()
volume_profiles = list()
for i in range(pre_days, len(trades)):
    param  = dict(tickdata = datas[i], level_space = ['bid1', 'ask1'], transaction_engine = GeneralExchange(datas[i], 3).transaction_engine)
    profile = group_trade_volume_by_time(trades[i-pre_days:i], time_range, time_interval)
    env_params.append(param)
    volume_profiles.append(profile)

envs = [GenerateTranches(HardConstrainTranche, goals, profile, **param)[tranche_id] for profile, param in zip(volume_profiles, env_params)]

agent = Linear(envs[0].observation_space_n, envs[0].action_space_n, criterion = nn.MSELoss, optimizer = torch.optim.Adam)
trainer = EpisodicTraining(agent, action_map=action_encoder)
trainer.sequence_train(envs, 1000, 'test/results/temp/temp.pth')

# nohup python -u ./test/manual_test_vwap.py 2>&1 > ./test/1000e.log &