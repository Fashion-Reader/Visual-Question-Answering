import os
import random
import argparse
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.get_model import get_model
from modules.dataset import CustomDatasetV1
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder
from modules.trainer import CustomTrainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory, CosineAnnealingWarmUpRestart
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
import warnings
import os

warnings.filterwarnings("ignore")

# DEBUG
DEBUG = False

# CONFIG
PROJECT_DIR = "./"
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = "/DATA/Final_DATA/"
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config_v3.yaml')
config = load_yaml(TRAIN_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# DATALOADER
NUM_WORKERS = 4
PIN_MEMORY = config['DATALOADER']['pin_memory']

# DATA
MAX_TOKEN = config['DATA']['max_token']

# MODEL
MODEL = config['MODEL']['model_str']
NUM_TARGETS = config['MODEL']['num_targets']
DIM_I = config['MODEL']['dim_i']
DIM_Q = config['MODEL']['dim_q']
DIM_H = config['MODEL']['dim_h']

# TRAIN
EPOCHS = config['TRAIN']['num_epochs']
BATCH_SIZE = config['TRAIN']['batch_size']
LEARNING_RATE = config['TRAIN']['learning_rate']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_function']
METRIC_FN = config['TRAIN']['metric_function']

# TRAIN SERIAL
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TRAIN_SERIAL = f'{MODEL}_{TRAIN_TIMESTAMP}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
        
if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set train result directory
    make_directory(PERFORMANCE_RECORD_DIR)

    # Set system logger
    system_logger = get_logger(name='train',
                               file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))

    # Load dataset & dataloader
    train_dataset = CustomDatasetV1(root_dir=DATA_DIR, result_dir=PERFORMANCE_RECORD_DIR, config=config, mode='train')
    validation_dataset = CustomDatasetV1(root_dir=DATA_DIR, result_dir=PERFORMANCE_RECORD_DIR, config=config, mode='val')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False, drop_last=True)
    
    # Load Model
    Model = get_model(model_str=MODEL)
    model = Model(num_targets=NUM_TARGETS, dim_i=DIM_I, dim_q=DIM_Q, dim_h=DIM_H, large=False, res152=True).to(device)
#     model.load_state_dict(torch.load("/fashion_reader/code/results/train/vqa_model_20210704124859/model.pt")["model"])
    
    # Set optimizer, scheduler, loss function, metric function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#     scheduler = CosineAnnealingWarmUpRestart(optimizer, T_0=3, T_mult=1, eta_max=2e-5,  T_up=1, gamma=0.6)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                   num_warmup_steps=len(train_dataset)//BATCH_SIZE,
                                                                   num_training_steps=len(train_dataset),
                                                                   num_cycles=EPOCHS)
    loss_fn = FocalLoss()
    metric_fn = accuracy_score

    # Set trainer
    trainer = CustomTrainer(model, device, loss_fn, metric_fn, optimizer, scheduler, logger=system_logger)

    # Set earlystopper
    early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE, verbose=False, logger=system_logger)
    
    # Set performance recorder
    key_column_value_list = [
        TRAIN_SERIAL,
        TRAIN_TIMESTAMP,
        MODEL,
        OPTIMIZER,
        LOSS_FN,
        METRIC_FN,
        EARLY_STOPPING_PATIENCE,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        RANDOM_SEED]

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=None)

    # Save config yaml file
    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'train_config_v3.yaml'), config)
    
    # Train
    for epoch in range(EPOCHS):
        trainer.train_epoch(train_dataloader, epoch_index=epoch, verbose=True)
        trainer.validate_epoch(validation_dataloader, epoch_index=epoch, verbose=True)

        # Performance record - csv & save elapsed_time
        performance_recorder.add_row(epoch_index=epoch,
                                     train_loss=trainer.train_loss_mean,
                                     validation_loss=trainer.validation_loss_mean,
                                     train_score=trainer.train_score,
                                     validation_score=trainer.validation_score)

        # Performance record - plot
        performance_recorder.save_performance_plot(final_epoch=epoch)
        # early_stopping check
        early_stopper.check_early_stopping(loss=trainer.validation_loss_mean)
        
        if early_stopper.stop:
            break

        trainer.clear_history()

    # last model save