import argparse
import json
import os
import sys
import random

import numpy as np

sys.path.append('./')
from data.tickdata import CSVDataset
from strategies.dnn.datamgr import TrancheDataset, dataloader


def parse_args():
    parser = argparse.ArgumentParser('train hierarchical reinforcement trader')
    parser.add_argument('--cuda', action='store_true', help='use cuda in training')
    parser.add_argument('--stock', type=str, help='stock code')
    parser.add_argument('--i', type=int, help='i tranche')
    return parser.parse_args()
        

def main(args, config):
    window = config['data']['windows']

    dataset = CSVDataset(config['data']['path'], args.stock)
    tranche = TrancheDataset(
        dataset=dataset,
        split=config['data']['split'],
        time_range=config['data']['times'],
        interval=config['data']['interval'],
        drop_length=config['data']['nday_history'],
        i_tranche=args.i
        )
    data = dataloader(tranche, window)
    np.save(f'./scripts/dnn/dataset/{args.stock}-{args.i}.train.X', data['train']['X'])
    np.save(f'./scripts/dnn/dataset/{args.stock}-{args.i}.train.y', data['train']['y'])
    np.save(f'./scripts/dnn/dataset/{args.stock}-{args.i}.test.X', data['test']['X'])
    np.save(f'./scripts/dnn/dataset/{args.stock}-{args.i}.test.y', data['test']['y'])
    

if __name__ == '__main__':
    args  = parse_args()
    config = json.load(open('./config/dnn.json', 'r'), encoding='utf-8')
    main(args, config)