import glob
import os

import pandas as pd
import torch
import torch.nn as nn

from exchange.stock import GeneralExchange
from options import parse_args
from strategies.vwap.agent import *
from strategies.vwap.env import *
from strategies.vwap.train import BaselineTraining, QLearning
from tickdata.datatype import TickData
from utils.statistic import group_trade_volume_by_time, tranche_num


arg = parse_args(strategy='vwap')

n_tranche = tranche_num(arg.time_range, arg.interval)
modeldir  = os.path.join(arg.save_dir, arg.stock, arg.env, arg.agent, '%d-tranches'% n_tranche, 'task%d_best.pth' %arg.tranche_id)

quotefiles = sorted(glob.glob('./data/%s/quote/*.csv' % arg.stock))
tradefiles = sorted(glob.glob('./data/%s/trade/*.csv' % arg.stock))
quotes = [pd.read_csv(file) for file in quotefiles]
trades = [pd.read_csv(file) for file in tradefiles]
datas  = [TickData(quote, trade) for quote, trade in zip(quotes, trades)]

env_params = list()
for i in range(arg.pre_days, len(trades)):
    param = dict(
        tickdata=datas[i], 
        level=arg.level, 
        side=arg.side,
        volume_profile = group_trade_volume_by_time(trades[i-arg.pre_days:i], arg.time_range, arg.interval)
        )
    if arg.exchange == 'general':
        param['transaction_engine'] = GeneralExchange(datas[i], arg.wait_t).transaction_engine
    else:
        raise KeyError('unkonwn exchange.')
    env_params.append(param)

if arg.env == 'hard_constrain':
    env = HardConstrainTranche
elif arg.env == 'histrical_hard_constrain':
    env = HistoricalHardConstrainTranche
    for param in env_params:
        param['historical_quote_num'] = arg.hist_quote
elif arg.env == 'recurrent_hard_constrain':
    env = RecurrentHardConstrainTranche
    for param in env_params:
        param['historical_quote_num'] = arg.hist_quote
else:
    raise KeyError('unknown environment.')

envs = [GenerateTranches(env, arg.goal, **param)[arg.tranche_id] for param in env_params]

if arg.agent == 'baseline':
    agent = Baseline(arg.side)
    if arg.mode == 'train':
        raise KeyError('Baseline only support test mode.')
    elif arg.mode == 'test':
        env_test = envs[-1]
        trainer = BaselineTraining(agent)
        reward  = trainer.test(env_test)
        print('test reward = %.5f' % reward)
        print('test metric = %s' % env_test.metrics())
    else:
        raise KeyError('argument mode must be test for running baseline.')

elif arg.agent == 'linear':
    criterion = nn.MSELoss
    optimizer = torch.optim.Adam
    agent = Linear(envs[0].observation_space_n, envs[0].action_space_n, criterion=criterion, optimizer=optimizer)
    
    if arg.mode == 'train':
        if os.path.exists(modeldir) and not arg.overwrite:
            agent.load_state_dict(torch.load(modeldir))
        trainer = QLearning(agent)
        trainer.train(envs[:-1], arg.episodes, val_split=.2, savedir=modeldir)

    elif arg.mode == 'test':
        env_test = envs[-1]
        if os.path.exists(modeldir):
            agent.load_state_dict(torch.load(modeldir))
        else:
            raise FileNotFoundError('cannot find model file in %s' % modeldir)
        trainer = QLearning(agent)
        reward  = trainer.test(env_test)
        print('test reward = %.5f' % reward)
        print('test metric = %s' % env_test.metrics())
    else:
        raise KeyError('argument mode must be train or test.')

elif arg.agent == 'lstm':
    criterion = nn.MSELoss
    optimizer = torch.optim.Adam
    agent = LSTM(envs[0].observation_space_n, envs[0].action_space_n, criterion=criterion, optimizer=optimizer)
    
    if arg.mode == 'train':
        if os.path.exists(modeldir) and not arg.overwrite:
            agent.load_state_dict(torch.load(modeldir))
        trainer = QLearning(agent)
        trainer.train(envs[:-1], arg.episodes, val_split=.2, savedir=modeldir)

    elif arg.mode == 'test':
        env_test = envs[-1]
        if os.path.exists(modeldir):
            agent.load_state_dict(torch.load(modeldir))
        else:
            raise FileNotFoundError('cannot find model file in %s' % modeldir)
        trainer = QLearning(agent)
        reward  = trainer.test(env_test)
        print('test reward = %.5f' % reward)
        print('test metric = %s' % env_test.metrics())
    else:
        raise KeyError('argument mode must be train or test.')

else:
    raise KeyError('unknown agent')