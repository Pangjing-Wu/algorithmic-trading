'''
给定数据源和数据范围 -> 载入历史tickdata数据 -> 分别基于历史数据计算day t 的volume profile
-> list of tickdata[day] -> 建立list of envs [day, tranche]
-> 对每天，每个tranche训练agent[tranche]

input: filename, stock
load historical tickdata
output: list of tickdata[day]

input: list of tickdata[day], volume profile args, envs args
calculate volume profile of each day by historical tickdata
build envs
output: envs[day, tranche]

input: envs[day, tranche], agents[tranche]
train agents
'''
import os

import torch
import torch.nn as nn

from data.source.h2 import load_from_h2
from data.datatype import TickData
from utils.statistic import group_trade_volume_by_time, tranche_num
from exchange.stock import GeneralExchange
from strategies.vwap.agent import Baseline, Linear
from strategies.vwap.env import GenerateTranches
from strategies.vwap.train import BaselineTraining, EpisodicTraining


stock = '000001'

dbdir = './data/'
user  = 'cra001'
psw   = 'cra001'
times = [34200000, 41400000, 46800000, 54000000]

pre_day = 20
interval = 1800000

method = 'baseline'

side = 'sell'
goal = 200000
exchange = 'general'
level_spcae = ['bid1', 'ask1']
SAVEDIR = './results'
SAVEDIR = os.path.join(SAVEDIR, stock)

action_map = lambda a: [1, 0, 0] if a == len(level_spcae) else [1, a, 100]

dates = [f[:8] for f in os.listdir(dbdir) if f[-5:] == 'h2.db']


tickdatas = []
for date in dates[:1]:
    dbname = os.path.join(dbdir, date)
    quote, trade = load_from_h2(stock, dbname, user, psw, times, port='8082')
    tickdatas.append(TickData(quote, trade))


volume_profiles = []
for i in range(pre_day, len(tickdatas)):
    trades = [tickdata.get_trade() for tickdata in tickdatas[i-pre_day, i]]
    volume_profile = group_trade_volume_by_time(trades, times, interval)
    volume_profiles.append(volume_profile)

dates = dates[pre_day:]
tickdatas = tickdatas[pre_day:]


for date, tickdata, volume_profile in zip(dates, tickdatas, volume_profiles):

    if exchange == 'general':
        engine = GeneralExchange(tickdata).transaction_engine

    envs = GenerateTranches(goal, volume_profile, tickdata=tickdata, transaction_engine=engine)

    if method == 'baseline': 
        agents = [Baseline(side) for _ in range(len(envs))]
        for i, agent, env in enumerate(zip(agents, envs)):
            trainer = BaselineTraining(agent)
            trainer.train(env)

            savedir = os.path.join(SAVEDIR, method, str(date), '%dtranches' % n, 'task%d' %i)
            with open(os.path.join(savedir, 'results.txt'), 'a') as f:
                f.write('vwap=%.6f, market vwap=%.6f\n' %(env.vwap, env.market_vwap))
            with open(os.path.join(savedir, 'actions.txt'), 'a') as f:
                f.writelines(trainer.action_track)
            with open(os.path.join(savedir, 'reward.txt'), 'a') as f:
                f.writelines(trainer.reward)

    elif method == 'linear':
        criterion = nn.MSELoss
        optimizer = torch.optim.Adam
        for i, env in enumerate(envs):
            agent = Linear(env.observation_spcae_n, env.action_space_n, criterion, optimizer)
            trainer = EpisodicTraining(agent, 1000)
            modeldir = os.path.join(SAVEDIR, method, '%dtranches' % n, 'task%d' %i, 'best.pth')
            if os.path.exists(modeldir):
                trainer.load(modeldir)
            trainer.train(env, modeldir)

            resultdir = os.path.join(SAVEDIR, method, str(date), '%dtranches' % n, 'task%d' %i)
            with open(os.path.join(savedir, 'results.txt'), 'a') as f:
                f.write('vwap=%.6f, market vwap=%.6f\n' %(env.vwap, env.market_vwap))

    else:
        raise KeyError