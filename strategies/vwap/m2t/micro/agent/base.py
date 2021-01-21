import abc

from .utils import ReplayMemory

class BasicQ(abc.ABC):

    def __init__(self, epsilon, gamma, delta_eps, 
                 batch, memory, criterion, optimizer,
                 device):
        self._epsilon   = epsilon
        self._gamma     = gamma
        self._delta_eps = delta_eps
        self._batch     = min(batch, memory)
        self._device    = device
        self._criterion = criterion
        self._optimizer = optimizer
        self._memory    = ReplayMemory(max(1, memory))

    @abc.abstractmethod
    def train(self):
        pass
    
    @abc.abstractmethod
    def _load_weight(self):
        pass
    
    @abc.abstractmethod
    def _save_weight(self):
        pass