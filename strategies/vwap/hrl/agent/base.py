import abc

from .utils import MacroReplayMemory, MicroReplayMemory


class BasicHRL(abc.ABC):

    def __init__(self, epsilon, gamma, delta_eps, 
                 batch, memory, criterion, optimizer,
                 device):
        self._epsilon   = epsilon
        self._gamma     = gamma
        self._delta_eps = delta_eps
        self._batch     = min(batch, memory)
        self._device    = device
        self._macro_memory = MacroReplayMemory(max(1, memory))
        self._micro_memory = MicroReplayMemory(max(1, memory))
        self._macro_criterion = criterion['macro']
        self._macro_optimizer = optimizer['macro']
        self._micro_criterion = criterion['micro']
        self._micro_optimizer = optimizer['micro']

    @abc.abstractmethod
    def train(self):
        pass
    
    @abc.abstractmethod
    def _load_weight(self):
        pass
    
    @abc.abstractmethod
    def _save_weight(self):
        pass