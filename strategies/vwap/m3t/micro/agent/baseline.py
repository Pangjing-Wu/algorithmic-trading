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