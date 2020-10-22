import glob
import json
import os

import pandas as pd


from exchange.stock import GeneralExchange
from options import parse_args
from strategies.vwap.agent import *
from strategies.vwap.env import *
from strategies.vwap.train import BaselineTraining, QLearning
from tickdata.datatype import TickData
from utils.statistic import distribute_task, tranche_num, volume_profile

arg   = parse_args(strategy='vwap', mode='simulate')
config = json.load(open('./config/vwap.json', 'r'))


n_tranche = tranche_num(config['env']['time_range'], config['env']['interval'])

tradefiles = sorted(glob.glob(os.path.join(config['path']['datadir'], arg.stock, 'trade/*.csv')))
i = tradefiles.index(os.path.join(config['path']['datadir'], arg.stock, 'trade/%s.csv' %arg.date))
if i < config['env']['volume_pre_day']:
    raise RuntimeError('there is not enough trade data to calculate volume profile')
tradefiles = tradefiles[i - config['env']['volume_pre_day']:i]
trades    = [pd.read_csv(file) for file in tradefiles]
profile    = volume_profile(trades, config['env']['time_range'], config['env']['interval'])

quote  = pd.read_csv(os.path.join(config['path']['datadir'], arg.stock, 'quote/%s.csv' %arg.date))
trade  = pd.read_csv(os.path.join(config['path']['datadir'], arg.stock, 'trade/%s.csv' %arg.date))
data   = TickData(quote, trade)
tasks  = distribute_task(arg.goal, profile)
print('intraday volume profile:')
print(tasks)


if config['env']['exchange'] == 'general':
    engine = GeneralExchange(data, config['env']['wait_t']).transaction_engine
else:
    raise KeyError('unkonwn exchange.')

if arg.env == 'hard_constrain':
    environment = HardConstrainTranche
elif arg.env == 'historical_hard_constrain':
    environment = HistoricalHardConstrainTranche
elif arg.env == 'recurrent_hard_constrain':
    environment = RecurrentHardConstrainTranche
else:
    raise KeyError('unknown environment.')

for i in range(n_tranche):
    print('execute tranche %d:' % i)
    env = environment(data, tasks.loc[i], engine, arg.level, arg.side)
    modeldir  = os.path.join(config['path']['savedir'], arg.stock, arg.env, arg.agent,
                             '%d-tranches'% n_tranche, 'task%d_best.pth' %i)
    if arg.agent == 'baseline':
        agent = Baseline(arg.side)
        trainer = BaselineTraining(agent)
        
    elif arg.agent == 'linear':
        criterion = nn.MSELoss
        optimizer = torch.optim.Adam
        agent = Linear(env.observation_space_n, env.action_space_n, criterion=criterion, optimizer=optimizer)
        if os.path.exists(modeldir):
            agent.load_state_dict(torch.load(modeldir, map_location='cpu'))
        else:
            raise FileNotFoundError('cannot find model file in %s' % modeldir)
        trainer = QLearning(agent)
    elif arg.agent == 'lstm':
        criterion = nn.MSELoss
        optimizer = torch.optim.Adam
        agent = LSTM(env.observation_space_n, env.action_space_n, criterion=criterion, optimizer=optimizer)
        if os.path.exists(modeldir):
            agent.load_state_dict(torch.load(modeldir, map_location='cpu'))
        else:
            raise FileNotFoundError('cannot find model file in %s' % modeldir)
        trainer = QLearning(agent)
    else:
        raise KeyError('unknown agent')
    reward  = trainer.test(env)
    print('transaction detail:\n%s' % env.filled)
    print('test reward = %.5f' % reward)
    print('test metric = %s' % env.metrics())