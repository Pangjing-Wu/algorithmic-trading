import copy
import os
import random

import numpy as np
import torch

from .base import BasicQ


class QLearning(BasicQ):

    def __init__(self, criterion, optimizer, epsilon=0.5,
                 gamma=0.99, delta_eps=0.95, batch=128,
                 memory=10000, device='cpu'):
        super().__init__(criterion=criterion,
                         optimizer=optimizer,
                         epsilon=epsilon, gamma=gamma,
                         delta_eps=delta_eps, batch=batch,
                         memory=memory, device=device)

    def train(self, envs:list, model, model_dir:str,
              episode:int, checkpoint=0, start_episode=0):
        self.__policy_net = model.to(self._device)
        self.__target_net = copy.deepcopy(model).to(self._device)
        self.__target_net.eval()
        if start_episode != 0:
            self._load_weight(model_dir, start_episode)
        epsilon = self._epsilon
        e = start_episode
        while e < episode:
            e += 1
            env = random.sample(envs, k=1)[0]
            s = env.reset()
            reward = 0
            while not env.final:
            # select action by epsilon greedy
                with torch.no_grad():
                    if random.random() < epsilon:
                        a = random.sample(env.action_space, 1)[0]
                    else:
                        a = torch.argmax(self.__policy_net(s)).item()
                s1, r = env.step(a)
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
                    loss = self._criterion(Q, Q_target)
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                s = s1
            print('Episode %d/%d: train reward = %.5f, metrics = %s' % (e, episode, reward, env.metrics()))
            if e % 5 == 0:
                epsilon *= self._delta_eps
                self.__target_net.load_state_dict(self.__policy_net.state_dict())
            if checkpoint and e % checkpoint == 0:
                self._save_weight(model_dir, e)
        self._save_weight(model_dir, e)

    def _load_weight(self, model_dir, episode):
        load_dir = os.path.join(model_dir, '%s.pt' % episode)
        weight = torch.load(load_dir, map_location=self._device)
        self.__policy_net.load_state_dict(weight)
        self.__target_net.load_state_dict(weight)

    def _save_weight(self, model_dir, episode):
        os.makedirs(model_dir, exist_ok=True)
        save_dir = os.path.join(model_dir, '%s.pt' % episode)
        self.__policy_net.to('cpu')
        torch.save(self.__policy_net.state_dict(), save_dir)
        self.__policy_net.to(self._device)