""" 공용 함수

    * File I/O
    * Logger
    * Model Load / Save
    * Seed
    * System
"""

import os
import math
import json
import pickle
import yaml
import random
import logging
import numpy as np
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
import torch


"""
File I/O
"""
def load_csv(path: str):
    return pd.read_csv(path)

def save_csv(path: str, obj: dict, index=False):
    try:
        obj.to_csv(path, index=index)
        message = f'csv saved {path}'
    except Exception as e:
        message = f'Failed to save : {e}'
    return message

def load_feather(path):
    df = pd.read_feather(path)
    print(f'File Loaded from {path}')
    return df

def save_feather(df_input, path):
    df = df_input.copy()

    files_present = glob.glob(path)
    if not files_present:
        df.to_feather(path)
        print(f'File Saved to {path}')
    else:
        print('WARNING: This feather file already exists!')

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json():
    try:
        with open(path, 'w') as f:
            json.dump(obj, f, indent=4, sort_keys=False)
        message = f'Json saved {path}'
    except Exception as e:
        message = f'Failed to save : {e}'
    print(message)
    return message

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def save_yaml(path, obj):
    try:
        with open(path, 'w') as f:
            yaml.dump(obj, f, sort_keys=False)
        message = f'Json saved {path}'
    except Exception as e:
        message = f'Failed to save : {e}'
    print(message)
    return message


"""
Logger
"""
def get_logger(name: str, file_path: str, stream=False)-> logging.RootLogger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


"""
System
"""
def make_directory(directory: str)-> str:
    """경로가 없으면 생성
    Args:
        directory (str): 새로 만들 경로

    Returns:
        str: 상태 메시지
    """

    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            msg = f"Create directory {directory}"
        
        else:
            msg = f"{directory} already exists"

    except OSError as e:
        msg = f"Fail to create directory {directory} {e}"

    return msg

def count_csv_row(path):
    """
    CSV 열 수 세기
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        n_row = sum(1 for row in reader)
        
"""
scheduler
"""
class CosineAnnealingWarmUpRestart(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                    1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


if __name__ == '__main__':
    pass
