import sys
sys.path.append('./')
sys.path.append('./test')

import torch
import torch.nn as nn

from utils.dataloader import load_tickdata, load_case
from src.datasource.datatype.tickdata import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.vwap.env import GenerateTranches
from src.strategies.vwap.agent import Baseline, Linear
from src.strategies.vwap.train import BaselineTraining, EpisodicTraining
from src.utils.statistic import group_trade_volume_by_time


def state2dict(state) -> dict:
    keys = ['time', 'start', 'end', 'goal', 'filled']
    state_dict = {}
    for k, i in zip(keys, state):
        state_dict[k] = i
    return state_dict


quote, trade = load_tickdata(stock='000001', date='20140704')
data = TickData(quote, trade)
trade = data.get_trade()
time = [34200000, 41400000, 46800000, 53700000]
env_params = dict(
    tickdata = data,
    level_space = ['bid1', 'ask1'],
    transaction_engine = GeneralExchange(data, 3).transaction_engine,
)

volume_profile = group_trade_volume_by_time(trade, time, 1800000)

action_encoder = lambda a: [1, 0, 0] if a == 2 else [1, a, 100]

env     = GenerateTranches(200000, volume_profile, **env_params)[0]
agent   = Baseline(side='sell', threshold=0.1)
trainer = BaselineTraining(agent, action_map=action_encoder)
trainer.train(env)
guides  = trainer.action_track

agent = Linear(env.observation_space_n, env.action_space_n,
               criterion = nn.MSELoss, optimizer = torch.optim.Adam)
trainer = EpisodicTraining(agent, 10, action_map=action_encoder)
trainer.pre_train(env, guides, 10)
print(env.task)
print(trainer.parameters)

trainer.pre_train(env, guides, 10)
print(env.task)
print(trainer.parameters)

r = trainer.test(env)
print(env.task)