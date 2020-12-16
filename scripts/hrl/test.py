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
from strategies.vwap.hrl.datamgr import TrancheDataset
from strategies.vwap.hrl.env import RecurrentTranche
from strategies.vwap.hrl.model import HybridAttenBiLSTM, HybridLSTM, MLP
from strategies.vwap.hrl.trader import HRLTrader


def parse_args():
    parser = argparse.ArgumentParser('train reinforcement micro tader')
    parser.add_argument('--stock', type=str, help='stock code')
    parser.add_argument('--agent', type=str, help='RL agent {HierarchicalQ/}')
    parser.add_argument('--model', type=str, help='RL agent based model {HybridLSTM/HybridAttenBiLSTM}')
    parser.add_argument('--reward', type=str, help='environment reward type {sparse/dense}')
    parser.add_argument('--i_tranche', type=int, help='train model for i tranche, start from 1')
    parser.add_argument('--quote_length', type=int, help='length of quote data for training')
    parser.add_argument('--model_episode', type=int, help='model episode')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    return parser.parse_args()


def generate_tranche_envs(dataset, env, time, args, config):
    envs = list()
    for data in dataset:
        goal = random.sample(config['hrl']['goal_pool'], 1)[0]
        # remove environment with bad liquidity
        trade = data.trade.between(time[0],time[1])
        if trade['size'].sum() < goal * 100:
            continue
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


def test(macro_model, micro_model, envs, verbose=False):
    trader = HRLTrader(macro_model, micro_model)
    n = len(envs)
    steps       = list()
    subgoals    = list()
    ex_rewards  = list()
    in_rewards  = list()
    slippages   = list()
    for i, env in enumerate(envs):
        ret = trader(env)
        steps.append(ret['step'])
        subgoals += ret['subgoals']
        ex_rewards.append(ret['ex_reward'])
        in_rewards.append(ret['in_reward'])
        slippages.append(ret['metrics']['vwap'] - ret['metrics']['market_vwap'])
        if verbose:
            print('sample %d/%d: ' % (i+1, n), end='')
            print('sobgoals: %s' % ret['subgoals'])
            print('ave. ex. reward = %.5f, ' % ex_rewards[-1], end='')
            print('ave. in. reward = %.5f, ' % in_rewards[-1], end='')
            print('step = %d, slippages = %.5f' % (steps[-1], slippages[-1]))
    report = dict(ex_reward=ex_rewards, in_reward=in_rewards,
                  step=steps, metrics=env.metrics())
    report = pd.DataFrame(report)
    print(report.describe())
    print(pd.Series(subgoals).value_counts())


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
    
    if args.model == 'HybridLSTM':
        env = RecurrentTranche
    elif args.model == 'HybridAttenBiLSTM':
        env = RecurrentTranche
    else:
        raise ValueError('unknown environment')

    train_envs = generate_tranche_envs(tranches.train_set, env,
                                       tranches.time, args, config)
    test_envs  = generate_tranche_envs(tranches.test_set, env,
                                       tranches.time, args, config)

    model_config = config['hrl']['model']

    macro_model = MLP(
        input_size=train_envs[0].extrinsic_observation_space_n, 
        hidden_size=model_config['MLP']['hidden_size'],
        output_size=train_envs[0].extrinsic_action_space_n,
        device=device
        )

    if args.model == 'HybridLSTM':
        micro_model = HybridLSTM(
            input_size=train_envs[0].intrinsic_observation_space_n, 
            output_size=train_envs[0].intrinsic_action_space_n,
            hidden_size=model_config[args.model]['hidden_size'],
            num_goals=train_envs[0].extrinsic_action_space_n,
            num_layers=model_config[args.model]['num_layers'],
            dropout=model_config[args.model]['dropout'],
            device=device
            )
    elif args.model == 'HybridAttenBiLSTM':
        micro_model = HybridAttenBiLSTM(
            input_size=train_envs[0].intrinsic_observation_space_n, 
            output_size=train_envs[0].intrinsic_action_space_n,
            hidden_size=model_config[args.model]['hidden_size'],
            num_goals=train_envs[0].extrinsic_action_space_n,
            num_layers=model_config[args.model]['num_layers'],
            dropout=model_config[args.model]['dropout'],
            attention_size=model_config[args.model]['attention_size'],
            device=device
        )
    else:
        raise ValueError('unknown model.')

    model_path = os.path.join(config['model_dir'], 'hrl', args.stock, args.agent,
                              "%s-len%d" % (args.model, args.quote_length),
                              '%s-eps10' % args.reward)
    macro_file  = os.path.join(model_path, 'macro-%d.pt' % args.model_episode)
    mairo_file  = os.path.join(model_path, 'micro-%d.pt' % args.model_episode)
    macro_weight = torch.load(macro_file, map_location='cpu')
    micro_weight = torch.load(mairo_file, map_location='cpu')
    macro_model.load_state_dict(macro_weight)
    micro_model.load_state_dict(micro_weight)
    macro_model.eval()
    micro_model.eval()
    
    print('agent: %s, model: %s, ' % (args.agent, args.model), end='')
    print('model episode: %d, ' % args.model_episode, end='')
    print('reward type: %s, ' % args.reward, end='')
    print('quote length: %d, ' % args.quote_length, end='')
    print('%d/%d tranche.' % (args.i_tranche, tranches.n))

    print('test on training samples')
    test(macro_model, micro_model, train_envs, args.verbose)
    print('test on testing samples')
    test(macro_model, micro_model, test_envs, args.verbose)
    print('\n')


if __name__ == '__main__':
    args  = parse_args()
    config = json.load(open('./config/vwap.json', 'r'), encoding='utf-8')
    main(args, config)