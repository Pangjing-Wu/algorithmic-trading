import os
import random

import torch

class BaselineTraining(object):

    def __init__(self, agent, action_map:callable):
        self._agent = agent
        self._actions = list()
        self._action_map = action_map
        self._metric = dict(vwap=None, market_vwap=None)

    @property
    def action_track(self):
        return self._actions

    def evaluate(self):
        return self._metric

    def train(self, env):
        self._actions = list()
        s = env.reset()
        final = env.is_final()
        while not final:
            a = self._agent(s)
            action = a if self._action_map == None else self._action_map(a)
            s, r, final = env.step(action)
            self._actions.append(a)
        self._metric['vwap'] = env.vwap
        self._metric['market_vwap'] = env.market_vwap


class EpisodicTraining(object):

    def __init__(self, agent, epsilon=0.1, gamma=0.99,
                 delta_eps=0.998, action_map=None):
        self._agent      = agent
        self._epsilon    = epsilon
        self._gamma      = gamma
        self._delta_eps  = delta_eps
        self._criterion  = self._agent.criterion()
        self._optimizer  = self._agent.optimizer(self._agent.parameters(), lr=0.1)
        self._action_map = action_map
        self._metric     = dict(vwap=None, market_vwap=None)
    
    @property
    def parameters(self):
        return self._agent.state_dict()

    def evaluate(self):
        return self._metric

    def train(self, env, episodes, savedir):
        best_reward = None
        epsilon = self._epsilon

        for episode in range(episodes):
            s = env.reset()
            final = False
            reward = 0

            while not final:
                Q = self._agent(s)
                # epsilon greedy
                if random.random() < epsilon:
                    a = random.sample(env.action_space, 1)[0]
                else: 
                    a = torch.argmax(Q).item()
                action = a if self._action_map == None else self._action_map(a)
                s1, r, final = env.step(action)
                reward += r
                # calculate next state's Q-values.
                Q1max = self._agent(s1).max()
                with torch.no_grad():
                    Q_target = Q.clone()
                    Q_target[a] = r + self._gamma * Q1max
                loss = self._criterion(Q, Q_target)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                s = s1
                if final is True:
                    epsilon *= self._delta_eps
                    break

            if best_reward == None or best_reward < reward:
                best_reward = reward
                self.save(savedir)
                print('get best model at %d episode with reward %.5f, saved.' % (episode, best_reward))
                self._metric['vwap'] = env.vwap
                self._metric['market_vwap'] = env.market_vwap

    def pre_train(self, env, actions:list, episodes:int):
        if len(actions) != len(env)-1:
            raise KeyError("the length of action and environment are not matching.")
        
        for _ in range(episodes):
            guides = iter(actions)
            s = env.reset()
            final = False
            reward = 0
            while not final:
                Q = self._agent(s)
                a = next(guides)
                action = a if self._action_map == None else self._action_map(a)
                s1, r, final = env.step(action)
                reward += r
                Q1max = self._agent(s1).max()
                with torch.no_grad():
                    Q_target = Q.clone()
                    Q_target[a] = r +  self._gamma * Q1max
                loss = self._criterion(Q, Q_target)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                s = s1
    
    def test(self, env):
        s = env.reset()
        final = False
        reward = 0
        while not final:
            with torch.no_grad():
                Q = self._agent(s)
                a = torch.argmax(Q).item()
            action = a if self._action_map == None else self._action_map(a)
            s1, r, final = env.step(action)
            reward += r
            s = s1
        return reward

    def load(self, modeldir):
        self._agent.load_state_dict(torch.load(modeldir))

    def save(self, savedir):
        os.makedirs(os.path.dirname(savedir), exist_ok=True)
        torch.save(self._agent.state_dict(), savedir)