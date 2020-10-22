import argparse

def parse_args(strategy, mode):
	parser = argparse.ArgumentParser('algorithmic trading')
	if strategy == 'vwap':
		parser.add_argument('--env', type=str, help='hard_constrain/historical_hard_constrain, specify environment type')
		parser.add_argument('--agent', help='baseline/linear/LSTM, backbone of agent')
		parser.add_argument('--stock', type=str, help='stock code')
		parser.add_argument('--side', default='buy', type=str, help='buy/sell, transaction side')
		parser.add_argument('--goal', default=20000, type=int, help='total sahres of trading goal per day')
		parser.add_argument('--level', default=1, type=int, help='[1,10], available bid/ask level in action space')
		if mode == 'train/test':
			parser.add_argument('--mode', type=str, help='train/test, specify running mode')
			parser.add_argument('--episodes', default=400, type=int, help='episode number of training')
			parser.add_argument('--overwrite', action='store_true', help='train a new model and overwrite existed')
			parser.add_argument('--tranche_id', default=0, type=int, help='specify tranche of each day')
		elif mode == 'simulate':
			parser.add_argument('--date', type=str, required=True, help='date of data for simulate')
		else:
			raise ValueError('Unknown mode.')
	else:
		raise ValueError('Unknown strategy.')

	return parser.parse_args()