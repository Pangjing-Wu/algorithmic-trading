import random
import sys
sys.path.append('./')


import numpy as np
import torch
import torch.nn as nn

from src.datasource.datatype.tickdata import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.vwap.agent import Linear
from src.strategies.vwap.env import GenerateTranches
from src.utils.statistic import group_trade_volume_by_time

# remove in future
sys.path.append('./test')
from utils.dataloader import load_tickdata, load_case


def episodic_training(env, agent, episodes, epsilon=0.1, gamma=0.99, delta_eps = 0.998):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(agent.parameters())

    reward_list = []

    for _ in range(episodes):
        s = env.reset()
        final = False
        R = 0

        while not final:
            Q = agent(s)
            # choose action by epsilon greedy.
            if np.random.rand() < epsilon:
                a = random.sample(env.action_space, 1)[0]
            else: 
                a = torch.argmax(Q).item()
            order = [1, 0, 0] if a == env.action_space[-1] else [1, a, 100]

            s1, r, final = env.step(order)
            R += r
            
            # calculate next state's Q-values.
            Q1max = agent(s1).max()
            with torch.no_grad():
                Q_target = Q.clone()
                Q_target[a] = r + gamma * Q1max

            loss = criterion(Q, Q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            s = s1
            if final is True:
                epsilon *= delta_eps
                break

        # record results after pre episode.
        reward_list.append(R)

    print(reward_list)


if __name__ == '__main__':

    quote, trade = load_tickdata(stock='000001', time='20140704')
    data = TickData(quote, trade)
    trade = data.get_trade()
    time = [34200000, 41400000, 46800000, 53700000]
    params = dict(
        tickdata = data,
        level_space = ['bid1', 'ask1'],
        transaction_engine = GeneralExchange(data, 3).transaction_engine,
    )

    volume_profile = group_trade_volume_by_time(trade, time, 1800000)

    envs = GenerateTranches(20000, volume_profile, **params)
    agent = Linear(envs.observation_space_n, envs.action_space_n)

    episodic_training(envs[0], agent, 10)