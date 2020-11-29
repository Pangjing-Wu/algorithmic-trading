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
        self._lb = lb
        self._ub = ub
        # action[0] is benifit, action[1] is cost.
        self._action = [level, level+1] if side == 'buy' else [level+1, level]

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
        if filled_ratio / time_ratio < self._lb:
            return 1
        elif filled_ratio / time_ratio > self._ub:
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

    def train(self, envs:list, model, model_dir:str,
              criterion, optimizer, episode:int, 
              checkpoint=0, start_episode=0):
        self._policy_net = model.to(self._device)
        self._target_net = copy.deepcopy(model).to(self._device)
        self._target_net.eval()
        start_episode = 'best' if start_episode == -1 else start_episode
        if start_episode != 0:
            self._load_weight(model_dir, start_episode)
            self._target_net.load_state_dict(self._policy_net.state_dict())
        epsilon = self._epsilon
        e = start_episode
        while e < episode:
            e += 1
            env = random.sample(envs, k=1)[0]
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
                    action_batch  = torch.tensor(batch.action, device=self._device).view(-1,1)
                    reward_batch  = torch.tensor(batch.reward, device=self._device).view(-1,1)
                    non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                                device=self._device, dtype=torch.bool)     
                    non_final_next_s = [s for s in batch.next_state if s is not None]
                    Q  = self._policy_net(batch.state).gather(1, action_batch)
                    Q1 = torch.zeros(self._batch, device=self._device)
                    Q1[non_final_mask] = self._target_net(non_final_next_s).max(1)[0].detach()
                    Q_target = self._gamma * Q1.view(-1,1) + reward_batch
                    loss = criterion(Q, Q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                s = s1
            print('Episode %d/%d: train reward = %.5f' % (e, start_episode + episode, reward))
            if e % 5 == 0:
                epsilon *= self._delta_eps
                self._target_net.load_state_dict(self._policy_net.state_dict())
            if checkpoint and e % checkpoint == 0:
                self._save_weight(model_dir, e)
        self._save_weight(model_dir, e)

    def _load_weight(self, model_dir, episodes=-1):
        episodes = 'best' if episodes == -1 else episodes
        load_dir = os.path.join(model_dir, "%s.pt" % episodes)
        weight = torch.load(load_dir, map_location=self._device)
        self._policy_net.load_state_dict(weight)
        self._target_net.load_state_dict(weight)

    def _save_weight(self, model_dir, episodes=-1):
        os.makedirs(model_dir, exist_ok=True)
        episodes = 'best' if episodes == -1 else episodes
        save_dir = os.path.join(model_dir, "%s.pt" % episodes)
        self._policy_net.to('cpu')
        torch.save(self._policy_net.state_dict(), save_dir)
        self._policy_net.to(self._device)