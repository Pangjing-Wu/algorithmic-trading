import os
import copy
import random
from collections import namedtuple

import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):

    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = list()
        self._position = 0
        self._mdp = namedtuple('mdp', ('state', 'action', 'next_state', 'reward'))

    def __len__(self):
        return len(self._memory)

    def push(self, *args):
        if len(self._memory) < self._capacity:
            self._memory.append(None)
        self._memory[self._position] = self._mdp(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):
        sample = random.sample(self._memory, batch_size)
        sample = self._mdp(*zip(*sample))
        return sample


class BaselineTraining(object):

    def __init__(self, agent):
        self._agent = agent
        self._actions = list()

    @property
    def action_track(self):
        return self._actions

    def test(self, env):
        self._actions = list()
        s = env.reset()
        final = env.is_final()
        reward = 0
        while not final:
            a = self._agent(s)
            s, r, final = env.step(a)
            self._actions.append(a)
            reward += r
        return reward


class QLearning(object):

    def __init__(self, agent, epsilon=0.1, gamma=0.99, delta_eps=0.95, lr=0.1, batch=128, memory=10000):
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
    def parameters(self):
        return self._policy_net.state_dict()

    def train(self, envs:list, episodes:int, val_split:int, savedir:str):
        if len(envs) < 2:
            raise KeyError('Too less environments to train, at least need 2.') 
        random.shuffle(envs)
        i_spilt     = max(1, int(len(envs) * val_split))
        val_envs    = envs[:i_spilt]
        train_envs  = envs[i_spilt:]
        best_reward = None
        epsilon     = self._epsilon
        for episode in range(episodes):
            rewards = list()
            for env in train_envs:
                s = env.reset()
                final = False
                reward = 0
                while not final:
                    # select action by epsilon greedy
                    with torch.no_grad():
                        if random.random() < epsilon:
                            a = random.sample(env.action_space, 1)[0]
                        else:
                            a = torch.argmax(self._policy_net(s)).item()
                    s1, r, final = env.step(a)
                    reward += r
                    self._memory.push(s, a, s1, r)
                    if len(self._memory) >= self._batch:
                        batch = self._memory.sample(self._batch)
                        action_batch = torch.tensor(batch.action, device=device).view(-1,1)
                        reward_batch = torch.tensor(batch.reward, device=device).view(-1,1)
                        Q  = self._policy_net(batch.state).gather(1, action_batch)
                        Q1 = self._target_net(batch.next_state).argmax().detach()
                        Q_target = self._gamma * Q1 + reward_batch
                        loss = self._criterion(Q, Q_target)
                        self._optimizer.zero_grad()
                        loss.backward()
                        self._optimizer.step()
                    s = s1
                rewards.append(reward)
            if episode % 5 == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())
            epsilon     *= self._delta_eps
            val_reward   = self.validation(val_envs)
            train_reward = sum(rewards) / len(rewards)
            print('Episode %d/%d: train reward = %.5f, validation reward = %.5f.' % (episode+1, episodes, train_reward, val_reward))
            if best_reward == None or best_reward < val_reward:
                best_reward = val_reward
                self.save(savedir)
                print('Get best model with reward %.5f! saved.\n' % best_reward)
            else:
                print('GG! current reward is %.5f, best reward is %.5f.\n' % (val_reward, best_reward))

    def pre_train(self, envs:list, actions:list, episodes:int):
        for episode in range(episodes):
            for env, action in zip(envs, actions):
                if len(action) != len(env)-1:
                    raise KeyError("action length is not matching state length.")
                guides = iter(action)
                s = env.reset()
                final = False
                while not final:
                    Q = self._policy_net(s)
                    a = next(guides)
                    s1, r, final = env.step(a)
                    Q1max = self._target_net(s1).max()
                    with torch.no_grad():
                        Q_target = Q.clone()
                        Q_target[a] = r +  self._gamma * Q1max
                    loss = self._criterion(Q, Q_target)
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    s = s1
            if episode % 5 == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())

    def validation(self, envs):
        rewards = list()
        for env in envs:
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
            rewards.append(reward)
        rewards = sum(rewards) / len(rewards)
        return rewards

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