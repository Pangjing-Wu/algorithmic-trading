import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append('./')
from data.tickdata import CSVDataset
from exchange.stock import AShareExchange
from strategies.vwap.m3t.datamgr import VolumeProfileDataset, TrancheDataset 
from strategies.vwap.m3t.env import RecurrentTranche
from strategies.vwap.m3t.model import HybridAttenBiLSTM, HybridLSTM, MLP
from strategies.vwap.m3t.trader import HRLTrader
from strategies.vwap.m2t.model import MacroBaseline, LSTM


dotmut = lambda x, y: sum([a * b for a, b in zip(x, y)])


def parse_args():
    parser = argparse.ArgumentParser('test macro trader')
    parser.add_argument('--goal', type=int, help='total order volume')
    parser.add_argument('--stock', type=str, help='stock code')
    parser.add_argument('--macro', type=str, help='macro model {Baseline/LSTM}')
    parser.add_argument('--micro', type=str, help='RL agent based model {HybridLSTM/HybridAttenBiLSTM}')
    parser.add_argument('--agent', type=str, help='RL agent {QLearning/}')
    parser.add_argument('--reward', type=str, help='environment reward type {sparse/dense}')
    parser.add_argument('--macro_epoch', type=int, help='macro model epoch')
    parser.add_argument('--micro_episode', type=int, help='micro model episode')
    parser.add_argument('--quote_length', type=int, help='length of quote data for training')
    return parser.parse_args()


def generate_tranche_envs(dataset, env, volume_profiles, time, args, config)->list:
    envs = list()
    for data, vp in zip(dataset, volume_profiles):
        subgoal = int(args.goal * vp // 100 * 100)
        trade = data.trade.between(time[0],time[1])
        # remove environment with bad liquidity
        if trade['size'].sum() < subgoal * 100:
            envs.append(None)
        else:
            exchange = AShareExchange(data, wait_trade=config['exchange']['wait_trade'])
            envs.append(
                env(
                    tickdata=data,
                    goal=subgoal,
                    exchange=exchange,
                    level=config['m3t']['level'],
                    side=config['m3t']['side'],
                    quote_length=args.quote_length,
                    reward=args.reward,
                    unit_size=config['exchange']['unit_size']
                    )
                )
    return envs


def main(args, config):
    device = torch.device('cpu')
    dataset = CSVDataset(config['data']['path'], args.stock)

    # macro dataset
    vp_data = VolumeProfileDataset(
        dataset=dataset,
        split=config['m2t']['micro']['split'], 
        time_range=config['data']['times'],
        interval=config['m2t']['interval'],
        history_length=config['m2t']['macro']['n_history']
        )

    macro_config = config['m2t']['macro']['model']
    micro_config = config['m3t']['model']

    # macro model
    if args.macro == 'Baseline':
        macro = MacroBaseline()
    elif args.macro == 'LSTM':
        macro = LSTM(
            input_size=1,
            output_size=1,
            hidden_size=macro_config[args.macro]['hidden_size'],
            num_layers=macro_config[args.macro]['num_layers'],
            dropout=macro_config[args.macro]['dropout']
            )
    else:
        raise ValueError('unknown macro model')

    # load macro model
    epoch = args.macro_epoch if args.macro_epoch not in [-1, None] else 'best'
    macro_file = os.path.join(config['model_dir'], 'm2t', 'macro',
                                args.stock, args.macro, '%s.pt' % epoch)
    macro_file = None if isinstance(macro, MacroBaseline) else macro_file
    if macro_file is not None:
        macro.load_state_dict(torch.load(macro_file))
    macro.eval()

    # micro environment
    if args.micro == 'HybridLSTM':
        env = RecurrentTranche
    elif args.micro == 'HybridAttenBiLSTM':
        env = RecurrentTranche
    else:
        raise ValueError('unknown environment')

    # generate environments
    train_envs = list()
    test_envs  = list()
    for i in range(vp_data.n_tranche):
        train_vp = macro(vp_data.train_set[i].X).reshape(-1).detach().numpy()
        test_vp  = macro(vp_data.test_set[i].X).reshape(-1).detach().numpy()
        # micro dataset
        tranches = TrancheDataset(
            dataset=dataset,
            split=config['m3t']['split'],
            i_tranche=i+1,
            time_range=config['data']['times'],
            interval=config['m3t']['interval'],
            drop_length=config['m3t']['n_history']
        )
        train_env = generate_tranche_envs(tranches.train_set, env, train_vp,
                                          tranches.time, args, config)
        test_env  = generate_tranche_envs(tranches.test_set, env, test_vp,
                                          tranches.time, args, config)
        train_envs += train_env
        test_envs  += test_env
    train_envs = np.array(train_envs, dtype=object).reshape(8, -1).T
    test_envs  = np.array(test_envs, dtype=object).reshape(8, -1).T

    for e in train_envs[:,0]:
        if e is not None:
            demo_env = e
            break

    # set agents
    medium_model = MLP(
        input_size=demo_env.extrinsic_observation_space_n, 
        hidden_size=micro_config['MLP']['hidden_size'],
        output_size=demo_env.extrinsic_action_space_n,
        device=device
        )

    if args.micro == 'HybridLSTM':
        micro_model = HybridLSTM(
            input_size=demo_env.intrinsic_observation_space_n, 
            output_size=demo_env.intrinsic_action_space_n,
            hidden_size=micro_config[args.micro]['hidden_size'],
            num_goals=demo_env.extrinsic_action_space_n,
            num_layers=micro_config[args.micro]['num_layers'],
            dropout=micro_config[args.micro]['dropout'],
            device=device
            )
    elif args.micro == 'HybridAttenBiLSTM':
        micro_model = HybridAttenBiLSTM(
            input_size=demo_env.intrinsic_observation_space_n, 
            output_size=demo_env.intrinsic_action_space_n,
            hidden_size=micro_config[args.micro]['hidden_size'],
            num_goals=demo_env.extrinsic_action_space_n,
            num_layers=micro_config[args.micro]['num_layers'],
            dropout=micro_config[args.micro]['dropout'],
            attention_size=micro_config[args.micro]['attention_size'],
            device=device
            )
    else:
        raise ValueError('unknown model.')

    # load agents
    model_path = os.path.join(config['model_dir'], 'm3t', args.stock, args.agent,
                              "%s-len%d" % (args.micro, args.quote_length),
                              '%s-eps10' % args.reward)
    medium_file  = os.path.join(model_path, 'macro-%d.pt' % args.micro_episode)
    micro_file   = os.path.join(model_path, 'micro-%d.pt' % args.micro_episode)
    medium_weight = torch.load(medium_file, map_location='cpu')
    micro_weight  = torch.load(micro_file, map_location='cpu')
    medium_model.load_state_dict(medium_weight)
    micro_model.load_state_dict(micro_weight)
    medium_model.eval()
    micro_model.eval()
    agent = HRLTrader(medium_model, micro_model)

    del demo_env

    #print args
    print('goal: %d, ' % args.goal, end='')
    print('macro: %s, epoch: %s, ' % (args.macro, epoch), end='')
    print('agent: %s, med: MLP, micro: %s, ' % (args.agent, args.micro), end='')
    print('micro episode: %d, ' % args.micro_episode, end='')
    print('eps:1.0, reward type: %s, ' % args.reward, end='')
    print('quote length: %d, ' % args.quote_length)

    # test on train set
    for envs in [train_envs, test_envs]:
        subgoals  = [[] for _ in range(envs.shape[1])]
        slippages = []
        for t in range(envs.shape[0]):
            demo_env = None
            for e in envs[t,:]:
                if e is not None:
                    demo_env = e
                    break
            if demo_env is None:
                continue
            trade = dataset[demo_env._data.date].trade
            del demo_env
            market_vwap = dotmut(trade['price'], trade['size']) / trade['size'].sum()
            filled = dict(price=[], size=[])
            for i in range(envs.shape[1]):
                subgoal = list()
                env = envs[t,i]
                if env is None:
                    continue
                else:
                    ret = agent(env)
                    subgoals[i] += ret['subgoals']
                    filled['price'] += env.filled['price']
                    filled['size']  += env.filled['size']
            vwap = dotmut(filled['price'], filled['size']) / sum(filled['size'])
            slippages.append((vwap - market_vwap) * 10000)
        if envs is train_envs:
            print('test results of training set')
        else:
            print('test results of test set')
        print('slippage:')
        print(pd.Series(slippages).describe())
        print('subgoals:')
        for i, subgoal in enumerate(subgoals):
            print('tranche %d:' % (i+1))
            print(pd.Series(subgoals[i]).value_counts()) 
if __name__ == '__main__':
    args  = parse_args()
    config = json.load(open('./config/vwap.json', 'r'), encoding='utf-8')
    main(args, config)

# python ./scripts/m3t/test.py --goal 200000 --stock 600000 --macro Baseline --micro HybridLSTM --agent HierarchicalQ --reward dense --micro_episode 10000 --quote_length 5
