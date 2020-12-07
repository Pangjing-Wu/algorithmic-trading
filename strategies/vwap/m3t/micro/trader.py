import torch


class MicroTrader(object):

    def __init__(self, model):
        self._model = model

    def __call__(self, env):
        self._actions = list()
        s = env.reset()
        step   = 0
        reward = 0
        while not env.final:
            a = self._model(s)
            a = torch.argmax(a).item()
            s, r = env.step(a)
            self._actions.append(a)
            step += 1
            reward += r
        ret = dict(sum_reward=reward, step=step, metrics=env.metrics())
        return ret