import functools
import os

import torch
import torch.nn as nn

from src.datasource.datatype.tickdata import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.vwap.agent import Linear
from src.strategies.vwap.env import GenerateTranches
from src.utils.statistic import group_trade_volume_by_time
from src.utils.train import episodic_training

# remove in future
sys.path.append('./test')
from utils.dataloader import load_case, load_tickdata


def baseline():
    pass

'''
给定数据源和数据范围 -> 载入历史tickdata数据 -> list of tickdata[day]
-> 分别基于历史数据计算day t 的volume profile -> 建立list of envs [day, tranche]
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


def train(stock:str, dates:list, data_source:callable,
          goal:int, time:list, interval:int,
          agent_fn:callable, savedir:str, episodes:int,
          epsilon:float, gamma:float, delta_eps:float,
          level_space:list, exchange:callable,
          load_model=False):

    envlist = []

    for date in dates:
        quote, trade = data_source(stock, date)
        data = TickData(quote, trade)
        trade = data.get_trade()
        volume_profile = group_trade_volume_by_time(trade, time, interval)
        
        envs = GenerateTranches(
            goal,
            volume_profile,
            tickdata=data,
            level_space=level_space,
            transaction_engine=exchange(data).transaction_engine
            )
        envlist.append([env for env in envs])

    agents = [agent_fn(envs.observation_space_n, envs.action_space_n) for _ in range(len(envlist[0]))]
    
    model_loaded = False

    for envs in envlist:
        for i, env in enumerate(envs):

            if load_model ==True and model_loaded == False:
                model = torch.load(od.path.join(savedir, stock, 'task%d' % i, 'best.pth'))
                agents[i].load_state_dict(model)

            episodic_training(env, agents[i], os.path.join(savedir, stock, 'task%d' % i), episodes)

        model_loaded = True


time = [34200000, 41400000, 46800000, 53700000]
interval = 1800000
agent = functools.partial(Linear, criterion=nn.MSELoss, optimizer=torch.optim.Adam)
exchange = functools.partial(GeneralExchange, wait_t=3)
torch.manual_seed(1)


train(stock='000001', dates=[20140704], data_source=load_tickdata,
      goal=200000, time=time, interval=interval,
      agent_fn=agent, savedir='results/', episodes=1,
      epsilon=0.1, gamma=0.9, delta_eps=0.998,
      level_space=['bid1', 'ask1'], exchange=exchange,
      load_model=False)

