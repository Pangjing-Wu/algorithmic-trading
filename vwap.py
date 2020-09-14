import glob
import os

import pandas as pd
import torch
import torch.nn as nn

from exchange.stock import GeneralExchange
from options import parse_args
from strategies.vwap.agent import Baseline, Linear
from strategies.vwap.env import GenerateTranches, HardConstrainTranche, HistoricalHardConstrainTranche
from strategies.vwap.train import BaselineTraining, EpisodicTraining
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
volume_profiles = [group_trade_volume_by_time(trades[i-arg.pre_days:i], arg.time_range, arg.interval) for i in range(arg.pre_days, len(trades))]

env_params = list()
for i in range(arg.pre_days, len(trades)):
    param = dict(tickdata=datas[i], level=arg.level, side=arg.side)
    if arg.exchange == 'general':
        param['transaction_engine'] = GeneralExchange(datas[i], 3).transaction_engine
    else:
        raise KeyError('unkonwn exchange.')
    env_params.append(param)

if arg.env == 'hardconstrain':
    env = HardConstrainTranche
elif arg.env == 'histricalhardconstrain':
    env = HistoricalHardConstrainTranche
else:
    raise KeyError('unknown environment.')
envs = [GenerateTranches(env, arg.goal, profile, **param)[arg.tranche_id] for profile, param in zip(volume_profiles, env_params)]

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
    agent = Linear(envs[0].observation_space_n, envs[0].action_space_n, criterion=nn.MSELoss, optimizer=torch.optim.Adam)
    
    if arg.mode == 'train':
        if os.path.exists(modeldir) and not arg.overwrite:
            agent.load_state_dict(torch.load(modeldir))
        trainer = EpisodicTraining(agent)
        trainer.train(envs, arg.episodes, savedir=modeldir)

    elif arg.mode == 'test':
        env_test = envs[-1]
        if os.path.exists(modeldir):
            agent.load_state_dict(torch.load(modeldir))
        else:
            raise FileNotFoundError('cannot find model file in %s' % modeldir)
        trainer = EpisodicTraining(agent)
        reward  = trainer.test(env_test)
        print('test reward = %.5f' % reward)
        print('test metric = %s' % env_test.metrics())
    else:
        raise KeyError('argument mode must be train or test.')
else:
    raise KeyError('unknown agent')
# nohup python -u vwap.py --mode train --env histricalhardconstrain --agent linear --stock 600000 --side sell --tranche_id 0 > "./results/600000/historicalhardconstrain/linear/8-tranches/0-tranche.log"
# python -u vwap.py --mode test --env histricalhardconstrain --agent linear --stock 600000 --side sell --tranche_id 0