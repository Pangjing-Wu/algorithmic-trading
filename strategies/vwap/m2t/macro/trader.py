import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .profile import get_tranche_time
from ..model import MacroBaseline


class MacroTrader(object):

    def __init__(self, model, model_file, time_range, interval):
        self.__times = get_tranche_time(time_range, interval)
        self.__model = model
        if model_file is not None:
            self.__model.load_state_dict(torch.load(model_file))

    def __call__(self, x, goal):
        self.__model.eval()
        x = self.__model(x)
        tasks = list()
        for i, in range(x.shape[0]):
            subgoals = [int(goal * r // 100 * 100) for r in x[i,:]]
            subgoals[np.argmax(subgoals)] += goal - sum(subgoals)
            task = dict(
                start=[t[0] for t in self.__times],
                end=[t[1] for t in self.__times],
                goal=subgoals
                )
            tasks.append(pd.DataFrame(task))
        return tasks