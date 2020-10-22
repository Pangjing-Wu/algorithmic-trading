import glob
import json
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
from utils.cache import VolumeProfileCache
from utils.statistic import distribute_task, tranche_num, volume_profile

arg   = parse_args(strategy='vwap', mode='train/test')
config = json.load(open('./config/vwap.json', 'r'))

n_tranche = tranche_num(config['env']['time_range'], config['env']['interval'])
modeldir  = os.path.join(config['path']['savedir'], arg.stock, arg.env, arg.agent,
                         '%d-tranches'% n_tranche, 'task%d_best.pth' %arg.tranche_id)

quotefiles = sorted(glob.glob(os.path.join(config['path']['datadir'], arg.stock, 'quote/*.csv')))
tradefiles = sorted(glob.glob(os.path.join(config['path']['datadir'], arg.stock, 'trade/*.csv')))
quotes = [pd.read_csv(file) for file in quotefiles]
trades = [pd.read_csv(file) for file in tradefiles]
dates  = [os.path.basename(file).rstrip('.csv') for file in quotefiles]
datas  = [TickData(quote, trade) for quote, trade in zip(quotes, trades)]

env_params   = list()
volume_cache = VolumeProfileCache(config['path']['cachedir'], arg.stock,
                                 n_tranche, config['env']['volume_pre_day'])
for i in range(config['env']['volume_pre_day'], len(trades)):
    profile = volume_cache.load(dates[i])
    if profile is None:
        profile = volume_profile(
            trades=trades[i-config['env']['volume_pre_day']:i],
            time_range=config['env']['time_range'],
            interval=config['env']['interval']
            )
        volume_cache.push(dates[i], profile)
    tasks = distribute_task(arg.goal, profile)
    param = dict(
        tickdata=datas[i], 
        level=arg.level, 
        side=arg.side,
        task=tasks.loc[arg.tranche_id]
        )
    if config['env']['exchange'] == 'general':
        param['transaction_engine'] = GeneralExchange(datas[i], config['env']['wait_t']).transaction_engine
    else:
        raise KeyError('unkonwn exchange.')
    env_params.append(param)
    
if arg.env == 'hard_constrain':
    environment = HardConstrainTranche
elif arg.env == 'historical_hard_constrain':
    environment = HistoricalHardConstrainTranche
    for param in env_params:
        param['historical_quote_num'] = config['env']['hist_quote']
elif arg.env == 'recurrent_hard_constrain':
    environment = RecurrentHardConstrainTranche
    for param in env_params:
        param['historical_quote_num'] = config['env']['hist_quote']
else:
    raise KeyError('unknown environment.')

envs  = [environment(**param) for param in env_params]
split = int(len(envs) * config['train']['train_test_split'])

if arg.agent == 'baseline':
    agent = Baseline(arg.side)
    if arg.mode == 'train':
        raise KeyError('Baseline only support test mode.')
    elif arg.mode == 'test':
        env_test = envs[split:]
        trainer = BaselineTraining(agent)
        for env in env_test:
            reward  = trainer.test(env)
            print('test reward = %.5f' % reward)
            print('test metric = %s' % env.metrics())
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
        trainer.train(envs=envs[:split], episodes=arg.episodes,
                      val_split=config['train']['val_split'], savedir=modeldir)
    elif arg.mode == 'test':
        env_test = envs[split:]
        if os.path.exists(modeldir):
            agent.load_state_dict(torch.load(modeldir, map_location='cpu'))
        else:
            raise FileNotFoundError('cannot find model file in %s' % modeldir)
        trainer = QLearning(agent)
        for env in env_test:
            reward  = trainer.test(env)
            print('test reward = %.5f' % reward)
            print('test metric = %s' % env.metrics())
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
        trainer.train(envs=envs[:split], episodes=arg.episodes,
                      val_split=config['train']['val_split'], savedir=modeldir)
    elif arg.mode == 'test':
        env_test = envs[split:]
        if os.path.exists(modeldir):
            agent.load_state_dict(torch.load(modeldir, map_location='cpu'))
        else:
            raise FileNotFoundError('cannot find model file in %s' % modeldir)
        trainer = QLearning(agent)
        for env in env_test:
            reward  = trainer.test(env)
            print('test reward = %.5f' % reward)
            print('test metric = %s' % env.metrics())
    else:
        raise KeyError('argument mode must be train or test.')

else:
    raise KeyError('unknown agent')