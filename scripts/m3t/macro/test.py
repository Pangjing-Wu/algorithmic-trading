import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.append('./')
from data.tickdata import CSVDataset
from strategies.vwap.m3t.macro.datamgr import VolumeProfileDataset
from strategies.vwap.m3t.model import LSTM, MLP, Linear, BaselineMacro


def parse_args():
    parser = argparse.ArgumentParser('test macro trader')
    parser.add_argument('--stock', type=str, help='stock code')
    parser.add_argument('--model', type=str, help='macro model {Baseline/Linear/MLP/LSTM}')
    return parser.parse_args()


def test(model, model_file, dataset, criterion):
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    model.eval()
    pred = model(dataset.train_set.X)
    loss = criterion(dataset.train_set.y, pred)
    print('training set MSE loss = %.5f' %  loss)
    pred = model(dataset.valid_set.X)
    loss = criterion(dataset.valid_set.y, pred)
    print('validation set MSE loss = %.5f' %  loss)
    pred = model(dataset.test_set.X)
    loss = criterion(dataset.test_set.y, pred)
    print('test set MSE loss = %.5f' %  loss)

def main(args, config):
    dataset = CSVDataset(config['data']['path'], args.stock)
    data = VolumeProfileDataset(
        dataset = dataset,
        split=config['m3t']['split'], 
        time_range=config['data']['times'],
        interval=config['m3t']['interval'],
        history_length=config['m3t']['macro']['n_history']
        )

    model_config = config['m3t']['macro']['model']
    if args.model == 'Baseline':
        model = BaselineMacro()
    elif args.model == 'Linear':
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
    else:
        raise ValueError('unknown macro model')
    model_file = os.path.join(config['model_dir'], 'm3t', 'macro',
                                args.stock, args.model, 'best.pt')
    model_file = None if isinstance(model, BaselineMacro) else model_file
        
    criterion = nn.MSELoss()
    test(model=model, model_file=model_file, dataset=data, criterion=criterion)


if __name__ == '__main__':
    args  = parse_args()
    config = json.load(open('./config/vwap.json', 'r'), encoding='utf-8')
    main(args, config)
