import argparse
import json
import os
import sys
import random

import pandas as pd
import torch
import torch.nn as nn

sys.path.append('./')
from data.tickdata import CSVDataset
from exchange.stock import AShareExchange
from strategies.vwap.m3t.macro.profile import get_tranche_time
from strategies.vwap.m3t.micro.datamgr import TrancheDataset
from strategies.vwap.m3t.micro.agent import QLearning
from strategies.vwap.m3t.micro.env import HistoricalTranche, RecurrentTranche
from strategies.vwap.m3t.model import LSTM, MLP, HybridLSTM, Linear


def parse_args():
    parser = argparse.ArgumentParser('train reinforcement micro tader')
    parser.add_argument('--env', type=str, help='RL environment {Historical/Recurrent}')
    parser.add_argument('--cuda', action='store_true', help='use cuda in training')
    parser.add_argument('--stock', type=str, help='stock code')
    parser.add_argument('--model', type=str, help='RL agent based model {Linear/HybridLSTM}')
    parser.add_argument('--agent', type=str, help='RL agent {QLearning/}')
    parser.add_argument('--episode', default=200, type=int, help='episode for training')
    parser.add_argument('--checkpoint', default=0, type=int, help='save model per checkpoint')
    parser.add_argument('--quote_length', default=1, type=int, help='length of quote data for training')
    parser.add_argument('--start_episode', default=0, type=int, help='start episode')
    
    return parser.parse_args()


def generate_tranche_envs(dataset, env, time, args, config):
    envs = list()
    for data in dataset:
        goal = random.sample(config['m3t']['micro']['goal_pool'], 1)[0]
        task = pd.Series(dict(start=time[0], end=time[1], goal=goal))
        exchange = AShareExchange(data, wait_trade=config['exchange']['wait_trade'])
        envs.append(
            env(
                tickdata=data,
                task=task,
                exchange=exchange,
                level=1,
                side=config['m3t']['side'],
                quote_length=args.quote_length
                )
            )
    return envs


def main(args, config):

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    times = get_tranche_time(
        time_range=config['data']['times'], 
        interval=config['m3t']['interval']
        )
    n_tranches = len(times)
    dataset = CSVDataset(config['data']['path'], args.stock)

    for n, time in enumerate(times):
        tranches = TrancheDataset(
            dataset=dataset,
            split=config['m3t']['split'],
            time=time,
            history_length=config['m3t']['macro']['n_history']
            )
        if args.env == 'Historical':
            env = HistoricalTranche
        elif args.env == 'Recurrent':
            env = RecurrentTranche
        else:
            raise ValueError('unknown environment')
        train_envs = generate_tranche_envs(tranches.train_set, env, time, args, config)
        valid_envs = generate_tranche_envs(tranches.valid_set, env, time, args, config)

        model_config = config['m3t']['micro']['model']
        if args.model == 'Linear':
            model = Linear(
                input_size=train_envs[0].observation_space_n, 
                output_size=train_envs[0].action_space_n,
                device=device
                )
        elif args.model == 'lstm':
            model = HybridLSTM(
                input_size=train_envs[0].observation_space_n,
                output_size=train_envs[0].action_space_n,
                hidden_size=model_config[args.model]['hidden_size'],
                num_layers=model_config[args.model]['num_layers'],
                dropout=model_config[args.model]['dropout'],
                device=device
                )

        if args.agent == 'QLearning':
            agent = QLearning(device=device, **config['m3t']['micro']['agent']['QLearning'])
        else:
            raise ValueError('unkonwn agent')

        model_dir = os.path.join(config['model_dir'], 'm3t', 'micro', args.stock,
                                 args.agent, "%s-%d" % (args.model, args.quote_length),
                                 '%d-%d' % (n+1, n_tranches))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
        agent.train(
            train_envs=train_envs,
            val_envs=valid_envs,
            model=model,
            model_dir=model_dir,
            criterion=criterion,
            optimizer=optimizer,
            episode=args.episode,
            start_episode=args.start_episode
            )
        

if __name__ == '__main__':
    args  = parse_args()
    config = json.load(open('./config/vwap.json', 'r'), encoding='utf-8')
    main(args, config)
