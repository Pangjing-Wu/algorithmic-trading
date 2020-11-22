import abc
import os

import torch
import torch.nn as nn


class BaselineMacroTrader(object):

    def __init__(self):
        pass
 
    def predict(self, x):
        x = x if torch.is_tensor(x) else torch.tensor(x) 
        return x.mean().item()


class DeepMacroTrader(object):

    def __init__(self, model, model_file):
        self.__model = model.to('cpu')
        self.__model.load_state_dict(torch.load(model_file, map_location='cpu'))

    def predict(self, x):
        x = x.to('cpu')
        self.__model.eval()
        return self.__model(x)