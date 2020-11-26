import abc
import copy
import os
import random

import numpy as np
import torch

from .utils import ReplayMemory


INF = 0x7FFFFFF


class Baseline(object):
    
    def __init__(self, side:str, level=1, lb=1.01, ub=1.1):
        if not lb < ub:
            raise ValueError('lb must be less than ub.')
        self.__lb = lb
        self.__ub = ub
        # action[0] is benifit, action[1] is cost.
        self.__action = [level, level+1] if side == 'buy' else [level+1, level]

    def __call__(self, time_ratio, filled_ratio):
        '''
        arugment:
        ---------
        state: list,
            elements consist of ['time', 'start', 'end', 'goal', 'filled'].
        return:
        -------
        action: int.
        '''
        if filled_ratio == 1:
            return 2
        if filled_ratio / time_ratio < self.__lb:
            return 1
        elif filled_ratio / time_ratio > self.__ub:
            return 2
        else:
            return 0


class BasicTD(abc.ABC):

    def __init__(self, epsilon, gamma, delta_eps, 
                 batch, memory, device):
        self._epsilon   = epsilon
        self._gamma     = gamma
        self._delta_eps = delta_eps
        self._batch     = min(batch, memory)
        self._memory    = ReplayMemory(max(1, memory))
        self._device    = device

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def _validation(self):
        pass
    
    @abc.abstractmethod
    def _load_weight(self):
        pass
    
    @abc.abstractmethod
    def _save_weight(self):
        pass


class QLearning(BasicTD):

    def __init__(self, epsilon=0.1, gamma=0.99, delta_eps=0.95,
                 batch=128, memory=10000, device='cpu'):
        super().__init__(epsilon=epsilon, gamma=gamma,
                         delta_eps=delta_eps, batch=batch,
                         memory=memory, device=device)

    def train(self, train_envs:list, val_envs:list,
              model, model_dir:str, criterion,
              optimizer, episode:int, checkpoint=25,
              start_episode=0):
        self.__policy_net = model.to(self._device)
        self.__target_net = copy.deepcopy(model).to(self._device)
        self.__target_net.eval()
        start_episode = 'best' if start_episode == -1 else start_episode
        if start_episode != 0:
            self._load_weight(model_dir, start_episode)
        best_reward = None
        epsilon = self._epsilon
        for e in range(start_episode + 1, start_episode + episode + 1):
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
                            a = torch.argmax(self.__policy_net(s)).item()
                    s1, r, final = env.step(a)
                    reward += r
                    self._memory.push(s, a, s1, r)
                    if len(self._memory) >= self._batch:
                        batch = self._memory.sample(self._batch)
                        action_batch  = torch.tensor(batch.action, device=self._device).view(-1,1)
                        reward_batch  = torch.tensor(batch.reward, device=self._device).view(-1,1)
                        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                                      device=self._device, dtype=torch.bool)     
                        non_final_next_s = [s for s in batch.next_state if s is not None]
                        Q  = self.__policy_net(batch.state).gather(1, action_batch)
                        Q1 = torch.zeros(self._batch, device=self._device)
                        Q1[non_final_mask] = self.__target_net(non_final_next_s).max(1)[0].detach()
                        Q_target = self._gamma * Q1.view(-1,1) + reward_batch
                        loss = criterion(Q, Q_target)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    s = s1
                rewards.append(reward)
            if e % 5 == 0:
                self.__target_net.load_state_dict(self._policy_net.state_dict())
            epsilon     *= self._delta_eps
            val_reward   = self._validation(val_envs)
            train_reward = sum(rewards) / len(rewards)
            print('Episode %d/%d: train reward = %.5f, validation reward = %.5f.' % (
                  e+1, start_episode + episode, train_reward, val_reward))
            if checkpoint and e % checkpoint == 0:
                self._save_weight(model_dir, e)
            if best_reward == None or best_reward < val_reward:
                best_reward = val_reward
                self._save_weight(model_dir, -1)
                print('Get best model with reward %.5f! saved.\n' % best_reward)
            else:
                print('GG! current reward is %.5f, best reward is %.5f.\n' % (
                      val_reward, best_reward))

    def _validation(self, envs):
        rewards = list()
        for env in envs:
            s = env.reset()
            final = False
            reward = 0
            while not final:
                with torch.no_grad():
                    Q = self.__policy_net(s)
                    a = torch.argmax(Q).item()
                s1, r, final = env.step(a)
                reward += r
                s = s1
            rewards.append(reward)
        rewards = sum(rewards) / len(rewards)
        return rewards

    def _load_weight(self, model_dir, episodes=-1):
        episodes = 'best' if episodes == -1 else episodes
        load_dir = os.path.join(model_dir, "%s.pt" % episodes)
        weight = torch.load(load_dir, map_location=self._device)
        self.__policy_net.load_state_dict(weight)
        self.__target_net.load_state_dict(weight)

    def _save_weight(self, model_dir, episodes=-1):
        os.makedirs(model_dir, exist_ok=True)
        episodes = 'best' if episodes == -1 else episodes
        save_dir = os.path.join(model_dir, "%s.pt" % episodes)
        self.__policy_net.to('cpu')
        torch.save(self.__policy_net.state_dict(), save_dir)
        self.__policy_net.to(self._device)