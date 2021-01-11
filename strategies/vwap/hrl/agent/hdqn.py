import abc
import copy
import os
import random

import torch

from .base import BasicHRL


average = lambda x: sum(x) / float(len(x))


class HierarchicalQ(BasicHRL):

    def __init__(self, criterion, optimizer, epsilon=0.5,
                 gamma=0.99, delta_eps=0.95, batch=128,
                 memory=10000, device='cpu', warmup_coef=10):
        super().__init__(criterion=criterion,
                         optimizer=optimizer,
                         epsilon=epsilon, gamma=gamma,
                         delta_eps=delta_eps, batch=batch,
                         memory=memory, device=device)
        self.__warmup_coef = warmup_coef

    def train(self, envs, macro_model, micro_model, model_dir,
              episode:int, checkpoint=0, start_episode=0):
        self.__macro_policy = macro_model.to(self._device)
        self.__micro_policy = micro_model.to(self._device)
        self.__macro_target = copy.deepcopy(macro_model).to(self._device)
        self.__micro_target = copy.deepcopy(micro_model).to(self._device)
        if start_episode != 0:
            self._load_weight(model_dir, start_episode)
        macro_epsilon = self._epsilon
        micro_epsilon = self._epsilon
        e = start_episode
        while e < episode:
            e += 1
            env = random.sample(envs, k=1)[0]
            ex_reward = list()
            in_reward = list()
            ex_s = env.reset()
            while not env.final:
                with torch.no_grad():
                    if random.random() < macro_epsilon:
                        goal = random.sample(env.extrinsic_action_space, 1)[0]
                    else:
                        goal = torch.argmax(self.__macro_policy(ex_s)).item()
                in_s = env.update_subgoal(goal)
                while not env.subfinal:
                    with torch.no_grad():
                        if random.random() < micro_epsilon:
                            a = random.sample(env.intrinsic_action_space, 1)[0]
                        else:
                            a = torch.argmax(self.__micro_policy(in_s, goal)).item()
                    in_s1, in_r = env.step(a)
                    in_reward.append(in_r)
                    self._micro_memory.push(in_s, a, in_s1, in_r, goal)
                    if len(self._micro_memory) >= self._batch * self.__warmup_coef:
                        self.__update_micro()
                    if len(self._macro_memory) >= self._batch * self.__warmup_coef:
                        self.__update_macro()
                    in_s = in_s1
                ex_s1, ex_r = env.extrinsic_state, env.extrinsic_reward
                ex_reward.append(ex_r)
                self._macro_memory.push(ex_s, goal, ex_s1, ex_r)
                ex_s = ex_s1
            print('Episode %d/%d: ' % (e, episode), end='')
            print('ave. intrinsic reward = %.5f, ' % average(in_reward), end='')
            print('ave. extrinsic reward = %.5f, ' % average(ex_reward), end='')
            print('metrics = %s, ' % env.metrics())
            if e % 5 == 0:
                macro_epsilon *= self._delta_eps
                micro_epsilon *= self._delta_eps
                self.__macro_target.load_state_dict(self.__macro_policy.state_dict())
                self.__micro_target.load_state_dict(self.__micro_policy.state_dict())
            if checkpoint and e % checkpoint == 0:
                self._save_weight(model_dir, e)
        self._save_weight(model_dir, e)

    def _load_weight(self, model_dir, episode):
        macro_dir = os.path.join(model_dir, 'macro-%s.pt' % episode)
        micro_dir = os.path.join(model_dir, 'micro-%s.pt' % episode)
        macro_weight = torch.load(macro_dir, map_location=self._device)
        micro_weight = torch.load(micro_dir, map_location=self._device)
        self.__macro_policy.load_state_dict(macro_weight)
        self.__micro_policy.load_state_dict(micro_weight)
        self.__macro_target.load_state_dict(macro_weight)
        self.__micro_target.load_state_dict(micro_weight)
    
    def _save_weight(self, model_dir, episode):
        os.makedirs(model_dir, exist_ok=True)
        macro_dir = os.path.join(model_dir, 'macro-%s.pt' % episode)
        micro_dir = os.path.join(model_dir, 'micro-%s.pt' % episode)
        self.__macro_policy.to('cpu')
        self.__micro_policy.to('cpu')
        torch.save(self.__macro_policy.state_dict(), macro_dir)
        torch.save(self.__micro_policy.state_dict(), micro_dir)
        self.__macro_policy.to(self._device)
        self.__micro_policy.to(self._device)

    def __update_macro(self):
        batch = self._macro_memory.sample(self._batch)
        action_batch  = torch.tensor(batch.action, device=self._device).view(-1,1)
        reward_batch  = torch.tensor(batch.reward, device=self._device).view(-1,1)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                     device=self._device, dtype=torch.bool)     
        non_final_next_s = [s for s in batch.next_state if s is not None]
        Q  = self.__macro_policy(batch.state).gather(1, action_batch)
        Q1 = torch.zeros(self._batch, device=self._device)
        Q1[non_final_mask] = self.__macro_target(non_final_next_s).max(1)[0].detach()
        Q_target = self._gamma * Q1.view(-1,1) + reward_batch
        loss = self._macro_criterion(Q.float(), Q_target.float())
        self._macro_optimizer.zero_grad()
        loss.backward()
        self._macro_optimizer.step()

    def __update_micro(self):
        batch = self._micro_memory.sample(self._batch)
        goal_batch    = batch.goal
        action_batch  = torch.tensor(batch.action, device=self._device).view(-1,1)
        reward_batch  = torch.tensor(batch.reward, device=self._device).view(-1,1)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                     device=self._device, dtype=torch.bool)     
        non_final_next_s = [s for s in batch.next_state if s is not None]
        non_final_goal   = [g for g, s in zip(goal_batch, batch.next_state) if s is not None]
        Q  = self.__micro_policy(batch.state, goal_batch).gather(1, action_batch)
        Q1 = torch.zeros(self._batch, device=self._device)
        Q1[non_final_mask] = self.__micro_target(non_final_next_s, non_final_goal).max(1)[0].detach()
        Q_target = self._gamma * Q1.view(-1,1) + reward_batch
        loss = self._micro_criterion(Q.float(), Q_target.float())
        self._micro_optimizer.zero_grad()
        loss.backward()
        self._micro_optimizer.step()
