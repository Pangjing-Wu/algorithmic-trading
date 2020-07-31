import random
import sys
sys.path.append('./')


import numpy as np
import torch
import torch.nn as nn

from src.datasource.datatype.tickdata import TickData
from src.exchange.stock import GeneralExchange
from src.strategies.vwap.agent import Linear
from src.strategies.vwap.env import VwapEnv

# remove in future
sys.path.append('./test')
from utils.dataloader import load_tickdata, load_case

torch.manual_seed(1)