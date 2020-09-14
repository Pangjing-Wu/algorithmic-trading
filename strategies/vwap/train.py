import os
import random

import torch

random.seed(0)

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


class EpisodicTraining(object):

    def __init__(self, agent, epsilon=0.1, gamma=0.99, delta_eps=0.998):
        self._agent      = agent
        self._epsilon    = epsilon
        self._gamma      = gamma
        self._delta_eps  = delta_eps
        self._criterion  = self._agent.criterion()
        self._optimizer  = self._agent.optimizer(self._agent.parameters(), lr=0.1)
    
    @property
    def parameters(self):
        return self._agent.state_dict()

    def train(self, envs:list, episodes:int, val_split:int, savedir:str):
        if len(envs) < 2:
            raise KeyError('Too less environments to train, at least need 2.') 
        random.shuffle(envs)
        i_spilt     = int(len(envs) * val_split)
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
                    Q = self._agent(s)
                    # select action by epsilon greedy
                    if random.random() < epsilon:
                        a = random.sample(env.action_space, 1)[0]
                    else: 
                        a = torch.argmax(Q).item()
                    s1, r, final = env.step(a)
                    reward += r
                    Q1max = self._agent(s1).max()
                    with torch.no_grad():
                        Q_target = Q.clone()
                        Q_target[a] = r + self._gamma * Q1max
                    loss = self._criterion(Q, Q_target)
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    s = s1
                rewards.append(reward)
                epsilon *= self._delta_eps
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
        for _ in range(episodes):
            for env, action in zip(envs, actions):
                if len(action) != len(env)-1:
                    raise KeyError("action length is not matching state length.")
                guides = iter(action)
                s = env.reset()
                final = False
                while not final:
                    Q = self._agent(s)
                    a = next(guides)
                    s1, r, final = env.step(a)
                    Q1max = self._agent(s1).max()
                    with torch.no_grad():
                        Q_target = Q.clone()
                        Q_target[a] = r +  self._gamma * Q1max
                    loss = self._criterion(Q, Q_target)
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    s = s1

    def validation(self, envs):
        rewards = list()
        for env in envs:
            s = env.reset()
            final = False
            reward = 0
            while not final:
                with torch.no_grad():
                    Q = self._agent(s)
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
                Q = self._agent(s)
                a = torch.argmax(Q).item()
            s1, r, final = env.step(a)
            reward += r
            s = s1
        return reward

    def save(self, savedir):
        os.makedirs(os.path.dirname(savedir), exist_ok=True)
        torch.save(self._agent.state_dict(), savedir)