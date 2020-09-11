import argparse

def parse_args(strategy):
	parser = argparse.ArgumentParser('algorithmic trading')
	if strategy == 'vwap':
		parser.add_argument('--env', type=str, help='uncompleted, specify environment type')
		parser.add_argument('--mode', type=str, help='train/test, specify running mode')
		parser.add_argument('--stock', type=str, help='stock code')
		parser.add_argument('--side', default='buy', type=str, help='buy/sell, transaction side')
		parser.add_argument('--goal', default=20000, type=int, help='total sahres of trading goal per day')
		parser.add_argument('--agent', default='linear', help='baseline/linear/LSTM, backbone of agent')
		parser.add_argument('--level', default=1, type=int, help='[1,10], available bid/ask level in action space')
		parser.add_argument('--episodes', default=400, type=int, help='episode number of training')
		parser.add_argument('--pre_days', default=20, type=int, help='data of pre n days for calculating volume profile')
		parser.add_argument('--exchange' , default='general', type=str, help='specify exchange')
		parser.add_argument('--time_range', default=[34200000, 41400000, 46800000, 53700000], type=int, nargs='+', help='legal time range of transaction')
		parser.add_argument('--interval'  , default=1800000, type=int, help='time length of each tranche')
		parser.add_argument('--tranche_id', default=0, type=int, help='specify tranche of each day')
		parser.add_argument('--overwrite', action='store_true', help='train a new model and overwrite existed')
		parser.add_argument('--hist_quote', default=5, type=int, help='specify number of historical quote in state')
		parser.add_argument('--save_dir', '--model_dir', default='./results', type=str, help='direction of load/save models')
	else:
		raise ValueError('Unknown strategy.')

	return parser.parse_args()