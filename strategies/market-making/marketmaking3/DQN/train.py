import os
import copy
import random
import sys
from collections import namedtuple
sys.path.append('/home/chendy/code/algorithmic-trading-master/datasource')
sys.path.append('/home/chendy/code/algorithmic-trading-master/exchange')

from datatype import TickData
import torch
import numpy as np
import pandas as pd
from envs import Env
from stock import GeneralExchange



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object): #经验重放--

    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = list()
        self._position = 0
        self._mdp = namedtuple('mdp', ('state', 'action', 'next_state', 'reward'))

    def __len__(self):
        return len(self._memory)

    def push(self, *args):#存储
        if len(self._memory) < self._capacity:
            self._memory.append(None)
        self._memory[self._position] = self._mdp(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):#采样
        sample = random.sample(self._memory, batch_size)
        sample = self._mdp(*zip(*sample))
        return sample


class QLearning(object):

    def __init__(self, agent, epsilon=0.7, gamma=0.99, delta_eps=0.95, lr=0.1, batch=128, memory=10000):
        self._policy_net = agent.to(device)
        self._target_net = copy.deepcopy(agent).to(device)
        self._epsilon    = epsilon
        self._gamma      = gamma 
        self._delta_eps  = delta_eps
        self._criterion  = agent.criterion()
        self._optimizer  = agent.optimizer(agent.parameters(), lr=lr)
        self._batch      = min(batch, memory)
        self._memory     = ReplayMemory(max(1, memory))
        self._target_net.eval()

    @property
    def parameters(self):  #envs
        return self._policy_net.state_dict()
    
    def train(self,date:list,where,where2,TickData,val_split:int,val_envs,savedir:str): 
        #val_split为数据比例，可以决定将envs中的百分之多少分配为训练集，多少是测试集
        best_reward = None
        epsilon     = self._epsilon #??????贪心策略的参数!

        ii=0
        while ii<len(date)-1:
            #for episode in range(episodes): #episodes需要是什么??????  
            name=date[ii]
            quote1=pd.read_csv(where+'/'+name+'.csv')
            trade1=pd.read_csv(where2+'/'+name+'.csv')
            data=TickData(quote1, trade1)
            modeldir  = os.path.join('/home/chendy/code/algorithmic-trading-master/strategies/market-making/marketmaking3.0/DQN/result/%s_best.pth'%date[ii])
            env_params=list()
            exchange=GeneralExchange(tickdata=data,wait_t=3)
            for num in range(4,len(quote1)-2):
                param = dict(tickdata=data,number=num,transaction_engine=exchange.transaction_engine)
                env_params.append(param)
            envs  = [Env(**param) for param in env_params]
            
            rewards = list()
            num_episodes=10
            for i_episode in range(num_episodes):
                s,money,position,resorder= envs[0].reset()#看看你的state到底包括了什么，money、position、resorder都应该包括在state里的把？
                #print(resorder,len(resorder))
                self.sellnum=0
                self.buynum=0
                waitsum=0
                for i in range(len(envs)-1):
                    env=envs[i]
                    final = False
                    reward = 0
                    time=int(env._data._quote.loc[env.number,'time'])
                    while not final:
                    # select action by epsilon greedy
                    # 做动作1的时候应该包含了！撤销目前市面上所有单！！
                        with torch.no_grad(): #action[0,1,2,3,4,5] 其中0代表不下单，1代表卖ask1买bid1，2代表卖ask1,3代表买bid1,4代表卖bid1(卖ask1三个quote都不成交的时候进行跨河交易),5代表买ask1
                            #print(resorder)
                            actionspace,waitsum=self.chooseaction(resorder=resorder,money=money,position=position,time=time,waitsum=waitsum)
                            if i==0 :
                                a=1
                            elif random.random()< epsilon:#用resorder判断一下
                                #print(random.sample(actionspace,1))
                                a=random.sample(actionspace,1)[0]
                            else:
                                k=self._policy_net(s)
                                mouyigezhi=0
                                for actions in actionspace:
                                    if  k[0,actions]>mouyigezhi:
                                        mouyigezhi=k[0,actions]
                                        a=actions                   
                        s1, r, final,money,position,resorder,sell_num,buy_num= env.step(a,money,position,resorder)  #  这里得到的r应该不是真实的reward
                        
                        self.sellnum=self.sellnum+sell_num
                        self.buynum=self.buynum+buy_num
                        #print(sell_num,buy_num,self.sellnum,self.buynum)
                        if position!=0 and position!=100 and self.sellnum==self.buynum:
                            print('!!!!')
                        reward += r
                        self._memory.push(s, a, s1, r)
                        if len(self._memory) >= self._batch:
                            batch = self._memory.sample(self._batch)
                            action_batch = torch.tensor(batch.action, device=device).view(-1,1)
                            reward_batch = torch.tensor(batch.reward, device=device).view(-1,1)
                            non_final_mask   = torch.tensor([s is not None for s in batch.next_state], 
                                                           device=device, dtype=torch.bool)          
                            non_final_next_s = [s for s in batch.next_state if s is not None]
                            #print(batch.state)
                            Q  = self._policy_net(batch.state).gather(1, action_batch)#targetnet是policynet的历史版本
                            Q1 = torch.zeros(self._batch, device=device)
                            Q1[non_final_mask] = self._target_net(non_final_next_s).max(1)[0].detach()#q1是训练之后的
                            Q_target = self._gamma * Q1.view(-1,1) + reward_batch
                            loss = self._criterion(Q, Q_target)
                            self._optimizer.zero_grad()
                            loss.backward()
                            self._optimizer.step()
                        s=s1
                        final=True
                    rewards.append(reward)
                    i=i+1 
                   
                #print('rewards=',rewards)
                if  i_episode % 2 == 0:
                    self._target_net.load_state_dict(self._policy_net.state_dict())
                train_reward = sum(rewards)
                print('In the end, our reward is %5.f, money is %.5f, position is %.5f, sell-num is %.5f,buy-num is %.5f'%(train_reward,money,position,self.sellnum,self.buynum))
                
                epsilon     *= self._delta_eps
                val_reward,val_money= self.validation(val_envs)#神经网络中用已知网络算出的reward
                # / len(rewards)#平均值
                print('Episode %d/%d: train reward = %.5f, validation reward = %.5f., train money =%.5f, validation money=%.5f' % (i_episode+1, num_episodes, train_reward, val_reward, money, val_money))
                if best_reward == None or best_reward < val_reward:
                    best_reward = val_reward
                    self.save(savedir)
                    print('Get best model with reward %.5f! saved.\n' % best_reward)
                else:
                    print('GG! current reward is %.5f, best reward is %.5f.\n' % (val_reward, best_reward))

    def chooseaction(self,resorder,money,position,waitsum,time):
        #print(resorder)
        if time>=53000000:
            if position>100:
                actionspace=[2]
            elif position<100:
                actionspace=[3]
            else:
                actionspace=[0]               
        elif len(resorder)==0:
            actionspace=[1]
            waitsum=0
        else:
            if resorder.loc[0,'time']!=time:
                waitsum=waitsum+1
            if len(resorder)==2:
                actionspace=[0,1]
                waitsum=0
            elif len(resorder)==1:
                if waitsum>=4:
                    if resorder.loc[0,'side']=='sell':
                        actionspace=[4]
                        waitsum=0
                    else:
                        actionspace=[5]
                        waitsum=0
                else:
                    if resorder.loc[0,'side']=='sell':
                        if abs(self.sellnum-self.buynum)>0:
                            actionspace=[0]
                        else:    
                            actionspace=[0,3]
                    elif resorder.loc[0,'side']=='buy':
                        if abs(self.sellnum-self.buynum)>0:
                            actionspace=[0]
                        else:    
                            actionspace=[0,2] 
                    else:
                        print("warning!some errors!") 
            else:
                print("warning2!some errors!")                       
        return (actionspace,waitsum)           

    def validation(self, envs):#验证集？？神经网络中的reward
        rewards = list()
        s,money,position,resorder= envs[0].reset()
        waitsum=0
        for i in range(len(envs)-3):
            env=envs[i+1]
            final = False
            reward = 0
            time=int(env._data._quote.loc[env.number,'time'])
            while not final:
                with torch.no_grad():
                    actionspace,waitsum=self.chooseaction(resorder=resorder,money=money,position=position,time=time,waitsum=waitsum)
                    if i==0 :
                        a=1
                    else:
                        k=self._policy_net(s)
                        mouyigezhi=0
                        for actions in actionspace:
                            if  k[0,actions]>mouyigezhi:
                                mouyigezhi=k[0,actions]
                                a=actions
                s1, r, final,money,position,resorder,sell_num,buy_num= env.step(a,money,position,resorder) 
                reward += r
                s=s1
                final=True
            rewards.append(reward)
        rewards = sum(rewards) #/ len(rewards)
        return (rewards,money)

    def test(self, env):
        s = env.reset()
        final = False
        reward = 0
        while not final:
            with torch.no_grad():
                Q = self._policy_net(s)
                a = torch.argmax(Q).item()
            s1, r, final = env.step(a)
            reward += r
            s = s1
        return reward

    def save(self, savedir):
        os.makedirs(os.path.dirname(savedir), exist_ok=True)
        torch.save(self._policy_net.state_dict(), savedir)