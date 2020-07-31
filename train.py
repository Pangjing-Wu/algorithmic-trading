import random

import numpy as np
import torch
import torch.nn as nn

from src.datasource.datatype.tickdata import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.vwap.agent import Linear
from src.strategies.vwap.env import VwapEnv


from time import sleep
import sys
sys.path.append('./test')
from utils.dataloader import load_tickdata, load_case


def episodic_training(env, agent, epsilon, gamma):
    pass


quote, trade = load_tickdata(stock='000001', time='20140704')
data = TickData(quote, trade)
trade = data.get_trade()
time = [34200000, 41400000, 46800000, 53700000]
level_space = ['bid1', 'ask1']
exchange = GeneralExchange(data, 3)

env = VwapEnv(data, 200000, time, 1200000, trade, exchange.transaction_engine, level_space)
agent = Linear(env.observation_space_n, env.action_space_n)


epsilon = 0.1
delta_eps = 0.998
episodes = 1
gamma = 0.99
max_step = 100
reward_list = []
Q_list = []


torch.manual_seed(1)
agent = Linear(env.observation_space_n, env.action_space_n)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(agent.parameters())

for _ in range(episodes):

    s = env.reset()
    final = False
    R = 0

    while not final:
    
        Q = agent(s)
        # choose action by epsilon greedy strategy.
        if np.random.rand() < epsilon:
            a = random.sample(env.action_space, 1)[0]
        else: 
            a = torch.argmax(Q).item()
        order = [1, 0, 0] if a == env.action_space[-1] else [1, a, 100]
        # execute action a.
        s1, r, final = env.step(order)
        R += r
        # calculate next state's Q-values.
        Q1max = agent(s1).max()
        with torch.no_grad():
            Q_target = Q.clone()
            Q_target[a] = r + gamma * Q1max
        # network training.
        loss = criterion(Q, Q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # State transition.
        s = s1
        if final is True:
            epsilon *= delta_eps
            break
    # record results pre episode.
    reward_list.append(R)

print(R)