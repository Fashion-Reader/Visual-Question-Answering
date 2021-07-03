""" 공용 함수

    * File I/O
    * Logger
    * Model Load / Save
    * Seed
    * System
"""

import os
import json
import pickle
import yaml
import random
import logging
import numpy as np
import pandas as pd

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


if __name__ == '__main__':
    pass
