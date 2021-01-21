import random
from collections import namedtuple


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