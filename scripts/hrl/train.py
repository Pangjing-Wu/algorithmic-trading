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
from strategies.vwap.hrl.datamgr import TrancheDataset
from strategies.vwap.hrl.agent import HierarchicalQ
from strategies.vwap.hrl.env import HistoricalTranche, RecurrentTranche
from strategies.vwap.hrl.model import HybridLSTM, MLP


def parse_args():
    parser = argparse.ArgumentParser('train hierarchical reinforcement trader')
    parser.add_argument('--env', type=str, help='RL environment {Historical/Recurrent}')
    parser.add_argument('--eps', type=float, help='epsilon greedy rate')
    parser.add_argument('--cuda', action='store_true', help='use cuda in training')
    parser.add_argument('--stock', type=str, help='stock code')
    parser.add_argument('--model', type=str, help='RL agent based model {Linear/HybridLSTM}')
    parser.add_argument('--agent', type=str, help='RL agent {HierarchicalQ/}')
    parser.add_argument('--reward', type=str, help='environment reward type {sparse/dense}')
    parser.add_argument('--episode', type=int, help='episode for training')
    parser.add_argument('--checkpoint', default=0, type=int, help='save model per checkpoint')
    parser.add_argument('--quote_length', default=1, type=int, help='length of quote data for training')
    parser.add_argument('--start_episode', default=0, type=int, help='start episode')
    return parser.parse_args()


def generate_tranche_envs(dataset, env, args, config):
    envs = list()
    for data in dataset:
        goal = random.sample(config['hrl']['goal_pool'], 1)[0]
        exchange = AShareExchange(data, wait_trade=config['exchange']['wait_trade'])
        envs.append(
            env(
                tickdata=data,
                goal=goal,
                exchange=exchange,
                level=config['hrl']['level'],
                side=config['hrl']['side'],
                quote_length=args.quote_length,
                reward=args.reward,
                unit_size=config['exchange']['unit_size']
                )
            )
    return envs


def main(args, config):

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = CSVDataset(config['data']['path'], args.stock)

    tranches = TrancheDataset(
        dataset=dataset,
        split=config['hrl']['split'],
        time_range=config['data']['times'],
        interval=config['hrl']['interval'],
        drop_length=config['hrl']['n_history']
        )

    if args.env == 'Recurrent':
        env = RecurrentTranche
    else:
        raise ValueError('unknown environment')

    envs = generate_tranche_envs(tranches.train_set, env, args, config)

    model_config = config['hrl']['model']

    macro_model = MLP(
        input_size=envs[0].extrinsic_observation_space_n, 
        hidden_size=model_config['MLP']['hidden_size'],
        output_size=envs[0].extrinsic_action_space_n,
        device=device
        )

    if args.model == 'HybridLSTM':
        micro_model = HybridLSTM(
            input_size=envs[0].intrinsic_observation_space_n, 
            output_size=envs[0].intrinsic_action_space_n,
            hidden_size=model_config[args.model]['hidden_size'],
            num_goals=envs[0].extrinsic_action_space_n,
            num_layers=model_config[args.model]['num_layers'],
            dropout=model_config[args.model]['dropout'],
            device=device
            )
    else:
        raise ValueError('unknown model.')

    micro_criterion = nn.MSELoss()
    macro_criterion = nn.MSELoss()
    macro_optimizer = torch.optim.Adam(macro_model.parameters(), lr=model_config['lr'])
    micro_optimizer = torch.optim.Adam(micro_model.parameters(), lr=model_config['lr'])

    if args.agent == 'HierarchicalQ':
        agent = HierarchicalQ(
            epsilon=args.eps,
            criterion=dict(macro=macro_criterion, micro=micro_criterion),
            optimizer=dict(macro=macro_optimizer, micro=micro_optimizer),
            device=device,
            **config['hrl']['agent']['HierarchicalQ']
            )
    else:
        raise ValueError('unkonwn agent')

    model_dir = os.path.join(config['model_dir'], 'hrl', args.stock, args.agent, 
                             "%s-%d" % (args.model, args.quote_length))
    
    agent.train(envs=envs, macro_model=macro_model, micro_model=micro_model,
                model_dir=model_dir, episode=args.episode, 
                start_episode=args.start_episode, checkpoint=args.checkpoint)
        

if __name__ == '__main__':
    args  = parse_args()
    config = json.load(open('./config/vwap.json', 'r'), encoding='utf-8')
    main(args, config)
