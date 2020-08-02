import os

import torch

class BaselineTraining(object):

    def __init__(self, agent):
        self._agent = agent
        self._actions = []
        self._reward = 0

    @property
    def action_track(self):
        return self._actions

    @property
    def reward(self):
        return self._reward

    def train(self, env):
        self._actions = []
        s = env.reset()
        final = env.is_final()
        while not final:
            a = agent(s)
            s, r, final = env.step(a)
            self._actions.append(a)
            self._reward.append(r)


class EpisodicTraining(BaselineTraining):

    def __init__(self, agent, episodes, epsilon=0.1, gamma=0.99,
                 delta_eps=0.998, action_map=None):
        super().__init__(agent)
        self._episodes   = episodes
        self._epsilon    = epsilon
        self._gamma      = gamma
        self._delta_eps  = delta_eps
        self._criterion  = self._agent.criterion
        self._optimizer  = self._agent.optimizer(self._agent.parameters())
        self._action_map = action_map

    def train(self, env, savedir):

        best_reward = None

        for episode in range(self._episodes):
            s = env.reset()
            final = False
            reward = 0

            while not final:
                Q = self._agent(s)

                # epsilon greedy
                if random.random() < self._episodes:
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
                    Q_target[a] = r + gamma * Q1max

                loss = self._criterion(Q, Q_target)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                s = s1
                if final is True:
                    epsilon *= delta_eps
                    break
            
            if best_reward == None or best_reward < reward:
                best_reward = reward
                os.makedirs(os.getcwd(savedir), exist_ok=True)
                torch.save(self._agent.state_dict(), savedir)
                print('get best model at %d episode with reward %.5f, saved.' % (episode, best_reward))

    def load(self, modeldir):
        self._agent.load_state_dict(torch.load(modeldir))