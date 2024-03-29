import os
import re
import time
from pathlib import Path
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
import argparse
import inspect

import numpy as np
import pandas as pd
from tqdm import tqdm

# os.chdir('src')
from paths import *
import features.denoising as deno
import util.s3_functions as s3


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[name] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''

    def __init__(self, slide_size, series_type='normal', denoising=False, ojisan=False):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.slide_size = slide_size
        self.series_type = series_type
        self.denoising = denoising
        self.ojisan = ojisan

        if ojisan:
            self.save_dir = FEATURES_DIR / f'add_ojisan36'
        else:
            self.save_dir = FEATURES_DIR / f'{slide_size}_slides'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.test_files = []
        submission = pd.read_csv(DATA_DIR / 'input/sample_submission.csv')
        for seg_id in submission.seg_id.values:
            self.test_files.append((seg_id, DATA_DIR / f'input/test/{seg_id}.csv'))

    def run(self, **kwargs):
        with timer(self.name):
            self.create_features(**kwargs)
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def calc_features(self, x, record):
        raise NotImplementedError

    def save(self):
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        name = prefix + self.name + suffix
        self.train.to_feather(str(self.save_dir / f'{name}_train.ftr'))
        self.test.to_feather(str(self.save_dir / f'{name}_test.ftr'))

    # For lanl
    def create_features(self, train):
        if self.ojisan:
            index_tuple_list = self.create_index_tuple_ojisan36()
        else:
            index_tuple_list = self.create_index_tuple(self.slide_size, train.shape[0])

        # 訓練データ用
        records = []
        for seg_id, index_tuple in tqdm(enumerate(index_tuple_list)):
            x, y, seg_id = self.slice_train_data(train, seg_id, index_tuple)

            if self.denoising:
                self.suffix = 'denoised'
                x = deno.denoise_signal(deno.high_pass_filter(x, low_cutoff=10000, SAMPLE_RATE=4000000), wavelet='haar', level=1)

            # Convert x
            x = self.convert(x)

            # Initialize
            record = dict()
            record['target'] = y
            record['seg_id'] = seg_id

            # Create features
            record = self.calc_features(x, record)

            records.append(record)
        self.train = pd.DataFrame(records)

        # テストデータ用
        records = []
        for seg_id, f in tqdm(self.test_files):
            # seg_id, f = test_files[0]; slide_size = 50000
            df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
            # x = df.acoustic_data.values[-self.slide_size:]
            x = df.acoustic_data.values

            if self.denoising:
                self.suffix = 'denoised'
                x = deno.denoise_signal(deno.high_pass_filter(x, low_cutoff=10000, SAMPLE_RATE=4000000), wavelet='haar', level=1)

            # Convert x
            x = self.convert(x)

            # Initialize
            record = dict()
            record['target'] = np.nan
            record['seg_id'] = seg_id

            # Create features
            record = self.calc_features(x, record)

            records.append(record)
        self.test = pd.DataFrame(records)

    def convert(self, x):
        x = pd.Series(x)
        zc = np.fft.fft(x)
        if self.series_type == 'fftr':
            x = pd.Series(np.real(zc))
            self.prefix = 'fftr'
        elif self.series_type == 'ffti':
            x = pd.Series(np.imag(zc))
            self.prefix = 'ffti'
        return x

    def slice_train_data(self, train, seg_id, index_tuple, denoising=False):
        start, end = index_tuple
        sub_train = train.iloc[start:end, :]

        x = list(sub_train.acoustic_data.values)
        if denoising:
            x = deno.denoise_signal(deno.high_pass_filter(x, low_cutoff=10000, SAMPLE_RATE=4000000), wavelet='haar', level=1)

        y = sub_train.time_to_failure.values[-1]
        seg_id = f'train_{seg_id}'

        return x, y, seg_id

    def create_index_tuple(self, slide_size, data_length):
        start = 0
        end = 150000
        index_tuple_list = []

        while end <= data_length:
            index_tuple_list.append((start, end))
            start += slide_size
            end += slide_size
        index_tuple_list.append((start, data_length-1))

        return index_tuple_list

    def create_index_tuple_ojisan36(self):
        start_indices = s3.read_csv_in_s3('s3://kaggle-nowcast/kaggle_lanl/data/input/featured/ojisan36/start_indices_4k.csv', header=None)
        index_tuple_list = []
        for i in range(6):
            # i = 1
            for start_index in (start_indices[i] + i*104_832_580).values:
                index_tuple_list.append((start_index, start_index+150000))
        return index_tuple_list


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()
