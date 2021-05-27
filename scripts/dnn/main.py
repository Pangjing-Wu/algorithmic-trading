import argparse
import json
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn

sys.path.append('./')
from data.tickdata import CSVDataset
from strategies.dnn.datamgr import TrancheDataset, dataloader


def parse_args():
    parser = argparse.ArgumentParser('train hierarchical reinforcement trader')
    parser.add_argument('--cuda', action='store_true', help='use cuda in training')
    parser.add_argument('--stock', type=str, help='stock code')
    return parser.parse_args()

        

def main(args, config):
    i = 1
    window = 20

    if args.cuda and torch.cuda.is_available():
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = CSVDataset(config['data']['path'], args.stock)
    tranche = TrancheDataset(
        dataset=dataset,
        split=config['data']['split'],
        time_range=config['data']['times'],
        interval=config['data']['interval'],
        drop_length=config['data']['n_history'],
        i_tranche=i
        )
    data = dataloader(tranche, window)
    np.save('./dataset/600000.train.X', data['train']['X'])
    np.save('./dataset/600000.train.y', data['train']['y'])
    np.save('./dataset/600000.test.X', data['test']['X'])
    np.save('./dataset/600000.test.y', data['test']['y'])
    model = None
    



if __name__ == '__main__':
    args  = parse_args()
    config = json.load(open('./config/dnn.json', 'r'), encoding='utf-8')
    main(args, config)