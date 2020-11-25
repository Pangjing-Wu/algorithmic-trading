import argparse
import json
import sys

sys.path.append('./')
from data.tickdata import CSVDataset
from exchange.stock import AShareExchange
from strategies.vwap.m3t.macro.datamgr import VolumeProfileDataset
from strategies.vwap.m3t.macro.model import LSTM, MLP, Linear
from strategies.vwap.m3t.macro.trader import (BaselineMacroTrader,
                                              DeepMacroTrader)
from strategies.vwap.m3t.micro.agent import QLearning
from strategies.vwap.m3t.micro.env import HistoricalTranche, RecurrentTranche
from strategies.vwap.m3t.micro.model import HybridLSTM, Linear


def parse_args():
    parser = argparse.ArgumentParser('train reinforcement micro tader')
    parser.add_argument('--cuda', action='store_true', help='use cuda in training')
    parser.add_argument('--stock', type=str, help='stock code')
    
    parser.add_argument('--model', type=str, help='RL agent based model {Linear/LSTM}')
    parser.add_argument('--macro', type=str, help='macro model {Baseline/Linear/MLP/LSTM}')
    parser.add_argument('--method', type=str, help='RL agent {QLearning/}')
    parser.add_argument('--episode', default=200, type=int, help='episode for training')
    parser.add_argument('--checkpoint', default=0, type=int, help='save model per checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    return parser.parse_args()


def load_macro(args, config):
    dataset = CSVDataset(config['data']['path'], args.stock)
    data = VolumeProfileDataset(
        dataset = dataset,
        split=config['m3t']['split'], 
        time_range=config['data']['times'],
        interval=config['m3t']['interval'],
        history_length=config['m3t']['macro']['n_history']
        )
    if args.macro == 'baseline':
        trader = BaselineMacroTrader()
    elif args.macro in ['Linear', 'MLP', 'LSTM']:
        model_config = config['m3t']['macro']['model']
        if args.model == 'Linear':
            model = Linear(
                input_size=data.X_len, 
                output_size=1
                )
        elif args.model == 'MLP':
            model = MLP(
                input_size=data.X_len,
                hidden_size=model_config[args.model]['hidden_size'],
                output_size=1
                )
        elif args.model == 'LSTM':
            model = LSTM(
                input_size=1,
                output_size=1,
                hidden_size=model_config[args.model]['hidden_size'],
                num_layers=model_config[args.model]['num_layers'],
                dropout=model_config[args.model]['dropout']
                )
        trader = DeepMacroTrader()

def main(args, config):
    pass
        
    

if __name__ == '__main__':
    args  = parse_args()
    config = json.load(open('./config/vwap.json', 'r'), encoding='utf-8')
    main()
