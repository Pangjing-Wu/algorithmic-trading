import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.append('./')
from data.tickdata import CSVDataset
from strategies.vwap.m2t.macro.datamgr import VolumeProfileDataset
from strategies.vwap.m2t.model import MacroBaseline, LSTM, MLP, Linear


def parse_args():
    parser = argparse.ArgumentParser('train deep macro trader')
    parser.add_argument('--cuda', action='store_true', help='use cuda in training')
    parser.add_argument('--stock', type=str, help='stock code')
    parser.add_argument('--model', type=str, help='macro model {Baseline/Linear/MLP/LSTM}')
    parser.add_argument('--epoch', default=200, type=int, help='epoch for training')
    parser.add_argument('--checkpoint', default=0, type=int, help='save model per checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    return parser.parse_args()


# training dnn
def train(model, model_dir, train_data, test_data,
          epoch, criterion, optimizer,
          start_epoch=0, checkpoint=0,
          device='cpu'):

    model.to(device)

    def load_weight(epoch):
        epoch    = 'best' if epoch == -1 else epoch
        load_dir = os.path.join(model_dir, "%s.pt" % epoch)
        model.load_state_dict(torch.load(load_dir, map_location=device))

    def save_weight(epoch):
        os.makedirs(model_dir, exist_ok=True)
        epoch    = 'best' if epoch == -1 else epoch
        save_dir = os.path.join(model_dir, "%s.pt" % epoch)
        model.to('cpu')
        torch.save(model.state_dict(), save_dir)
        model.to(device)

    if start_epoch != 0:
        load_weight(epoch=start_epoch)
    
    best_test_loss = 1e8
    for e in range(start_epoch + 1, start_epoch + epoch + 1):
        model.train()
        X, y = train_data.X.to(device), train_data.y.to(device)
        pred = model(X)
        loss = criterion(y, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        X, y = test_data.X.to(device), test_data.y.to(device)
        pred = model(X)
        test_loss = criterion(y, pred)
        print('Epoch: %d/%d, ' % (e, start_epoch + epoch), end='')
        print('train MSE = %.5f, test MSE = %.5f.' % (loss, test_loss))
        if checkpoint > 0 and e % checkpoint == 0:
            save_weight(epoch=e)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_weight(epoch=-1)
            print('Get best model with MSE loss %.5f! saved.' % best_test_loss)
        else:
            print('GG, best MSE loss is %.5f.' % best_test_loss)


def main(args, config):
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # set dataset
    dataset = CSVDataset(config['data']['path'], args.stock)
    data = VolumeProfileDataset(
        dataset = dataset,
        split=config['m2t']['macro']['split'], 
        time_range=config['data']['times'],
        interval=config['m2t']['interval'],
        history_length=config['m2t']['macro']['n_history']
        )

    # set model
    model_config = config['m2t']['macro']['model']
    if args.model == 'Baseline':
        model = MacroBaseline()
    elif args.model == 'Linear':
        model = Linear(
            input_size=data.X_len, 
            output_size=1,
            device=device
            )
    elif args.model == 'MLP':
        model = MLP(
            input_size=data.X_len,
            hidden_size=model_config[args.model]['hidden_size'],
            output_size=1,
            device=device
            )
    elif args.model == 'LSTM':
        model = LSTM(
            input_size=1,
            output_size=1,
            hidden_size=model_config[args.model]['hidden_size'],
            num_layers=model_config[args.model]['num_layers'],
            dropout=model_config[args.model]['dropout'],
            device=device
            )
    else:
        raise ValueError('unknown model.')

    # set training
    model_dir = os.path.join(config['model_dir'], 'm2t', 'macro', args.stock, args.model)
    
    if args.model == 'Baseline':
        X, y = data.test_set.X.to(device), data.test_set.y.to(device)
        pred  = model(X)
        criterion = nn.MSELoss()
        loss  = criterion(y, pred)
        print('Baseline test MSE loss %.5f!' % loss)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
        criterion = nn.MSELoss()
        train(
            model=model,
            model_dir=model_dir,
            train_data=data.train_set,
            test_data=data.test_set,
            epoch=args.epoch,
            optimizer=optimizer,
            criterion=criterion,
            start_epoch=args.start_epoch,
            checkpoint=args.checkpoint,
            device=device
            )


# python -u ./scripts/m2t/macro/train.py --stock 600000 --epoch 1 --model Baseline --checkpoint 0

if __name__ == '__main__':
    args  = parse_args()
    config = json.load(open('./config/vwap.json', 'r'), encoding='utf-8')
    main(args, config)
