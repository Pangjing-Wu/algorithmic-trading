import os

import pandas as pd


class VolumeProfileCache(object):

    def __init__(self, cache_dir:str, stock:str, tranche_num:int, predays:int):
        os.makedirs(cache_dir, exist_ok=True)
        cachename = '%s-%d-pre%d.cache' % (stock, tranche_num, predays)
        self._file = os.path.join(cache_dir, cachename)
        if not os.path.exists(self._file):
            self._init_cache()
        self._cache = pd.read_csv(self._file, index_col=['date', 'i_tranche'])
        self._cache = self._cache[~self._cache.index.duplicated(keep='first')]

    def __del__(self):
        cache = pd.read_csv(self._file, index_col=['date', 'i_tranche'])
        index = cache.index.duplicated(keep='first')
        if any(index):
            cache = cache[~index]
            cache.to_csv(self._file, mode='w')

    def load(self, date:str):
        try:
            cache = self._cache.loc[int(date)]
        except KeyError:
            cache = None
        finally:
            return cache

    def push(self, date:str, df:pd.DataFrame):
        cache = pd.DataFrame(df.values, pd.MultiIndex.from_product([[date], df.index]))
        cache.to_csv(self._file, mode='a', header=False)
        self._cache = pd.read_csv(self._file, index_col=['date', 'i_tranche'])
        self._cache = self._cache[~self._cache.index.duplicated(keep='first')]

    def _init_cache(self):
        cols = ['date', 'i_tranche', 'start', 'end', 'volume']
        pd.DataFrame(columns=cols).to_csv(self._file, index=False)