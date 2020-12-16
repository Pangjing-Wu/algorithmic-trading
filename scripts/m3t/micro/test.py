import argparse
import json
import os
import sys
import random

import pandas as pd
import torch

sys.path.append('./')
from data.tickdata import CSVDataset
from exchange.stock import AShareExchange
from strategies.vwap.m3t.micro.datamgr import TrancheDataset
from strategies.vwap.m3t.micro.env import BaselineTranche, HistoricalTranche, RecurrentTranche
from strategies.vwap.m3t.model import HybridLSTM, Linear, MicroBaseline
from strategies.vwap.m3t.micro.trader import MicroTrader


def parse_args():
    parser = argparse.ArgumentParser('train reinforcement micro tader')
    parser.add_argument('--stock', type=str, help='stock code')
    parser.add_argument('--agent', type=str, help='RL agent {QLearning/}')
    parser.add_argument('--model', type=str, help='RL agent based model {/Baseline/Linear/HybridLSTM}')
    parser.add_argument('--reward', type=str, help='environment reward type {sparse/dense}')
    parser.add_argument('--i_tranche', type=int, help='train model for i tranche, start from 1')
    parser.add_argument('--quote_length', type=int, help='length of quote data for training')
    parser.add_argument('--model_episode', type=int, help='model episode')
    return parser.parse_args()


def generate_tranche_envs(dataset, env, time, args, config):
    envs = list()
    for data in dataset:
        goal = random.sample(config['m3t']['micro']['goal_pool'], 1)[0]
        # remove environment with bad liquidity
        trade = data.trade.between(time[0],time[1])
        if trade['size'].sum() < goal * 100:
            continue
        task = pd.Series(dict(start=time[0], end=time[1], goal=goal))
        exchange = AShareExchange(data, wait_trade=config['exchange']['wait_trade'])
        envs.append(
            env(
                tickdata=data,
                task=task,
                exchange=exchange,
                level=config['m3t']['micro']['level'],
                side=config['m3t']['side'],
                quote_length=args.quote_length,
                reward=args.reward
                )
            )
    return envs


def test(model, envs, verbose=False):
    trader = MicroTrader(model)
    n = len(envs)
    sum_rewards = list()
    ave_rewards = list()
    steps = list()
    slippages = list()
    for i, env in enumerate(envs):
        ret = trader(env)
        sum_rewards.append(ret['sum_reward'])
        ave_rewards.append(ret['sum_reward'] / float(ret['step']))
        steps.append(ret['step'])
        slippages.append(ret['metrics']['vwap'] - ret['metrics']['market_vwap'])
        if verbose:
            print('sample %d/%d: ' % (i+1, n), end='')
            print('sum reward = %.5f, ' % sum_rewards[-1], end='')
            print('ave reward = %.5f, ' % ave_rewards[-1], end='')
            print('step = %d, slippages = %s' % (steps[-1], slippages[-1]))
    report = dict(sum_rewards=sum_rewards, ave_rewards=ave_rewards,
                  steps=steps, slippages=slippages)
    report = pd.DataFrame(report)
    print(report.describe())


def main(args, config):

    device = torch.device('cpu')
    dataset = CSVDataset(config['data']['path'], args.stock)
    tranches = TrancheDataset(
        dataset=dataset,
        split=config['m3t']['micro']['split'],
        i_tranche=args.i_tranche,
        time_range=config['data']['times'],
        interval=config['m3t']['interval'],
        drop_length=config['m3t']['macro']['n_history']
        )
    
    if args.model == 'Linear':
        env = HistoricalTranche
    elif args.model == 'HybridLSTM':
        env = RecurrentTranche
    elif args.model == 'Baseline':
        env = BaselineTranche
    else:
        raise ValueError('unknown environment')

    train_envs = generate_tranche_envs(tranches.train_set, env,
                                       tranches.time, args, config)
    test_envs  = generate_tranche_envs(tranches.test_set, env,
                                       tranches.time, args, config)

    model_config = config['m3t']['micro']['model']
    if args.model == 'Linear':
        model = Linear(
            input_size=train_envs[0].observation_space_n, 
            output_size=train_envs[0].action_space_n,
            device=device
            )
    elif args.model == 'HybridLSTM':
        model = HybridLSTM(
            input_size=train_envs[0].observation_space_n,
            output_size=train_envs[0].action_space_n,
            hidden_size=model_config[args.model]['hidden_size'],
            num_layers=model_config[args.model]['num_layers'],
            dropout=model_config[args.model]['dropout'],
            device=device
            )
    elif args.model == 'Baseline':
        model = MicroBaseline(
            side=config['m3t']['side'],
            level=config['m3t']['micro']['level'],
            lb=1.01,
            ub=1.1
            )
    else:
        raise ValueError('unknown model.')

    model_file = os.path.join(config['model_dir'], 'm3t', 'micro', args.stock,
                             args.agent, "%s-%d" % (args.model, args.quote_length),
                             '%d-%d' % (args.i_tranche, tranches.n),
                             '%d.pt' % args.model_episode)
    if not isinstance(model, MicroBaseline):
        weight = torch.load(model_file, map_location='cpu')
        model.load_state_dict(weight)
        model.eval()
    
    print('agent: %s, model: %s, ' % (args.agent, args.model), end='')
    print('model episode: %d, ' % args.model_episode, end='')
    print('reward type: %s, ' % args.reward, end='')
    print('quote length: %d, ' % args.quote_length, end='')
    print('%d/%d tranche.' % (args.i_tranche, tranches.n))

    print('test on training samples')
    test(model, train_envs)
    print('test on testing samples')
    test(model, test_envs)
    print('\n')


if __name__ == '__main__':
    args  = parse_args()
    config = json.load(open('./config/vwap.json', 'r'), encoding='utf-8')
    main(args, config)