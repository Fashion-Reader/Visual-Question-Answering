import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.get_model import get_model
from modules.utils import load_yaml
import transformers

class TestDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'data/test')
        self.img_dir = "/DATA/Final_DATA/task07_test/images/"
        self.csv_path = "/DATA/Final_DATA/task07_test/test.csv"
        self.data_df = pd.read_csv(self.csv_path)
        self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.max_token = 30
        self.transform = transforms.Compose([
            transforms.Resize((356, 356)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        img_fn = self.data_df['image'][idx]
        question = self.data_df['question'][idx]

        tokenized = self.tokenizer.encode_plus("".join(question),
                                     None,
                                     add_special_tokens=True,
                                     max_length=self.max_token,
                                     truncation=True,
                                    #  padding=True)
                                     pad_to_max_length=True)

        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']

        image_path = os.path.join(self.img_dir, img_fn)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'image': image}
    
# CONFIG
PROJECT_DIR = './'
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config_v1.yaml')
config = load_yaml(TRAIN_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# DATALOADER
NUM_WORKERS = config['DATALOADER']['num_workers']
PIN_MEMORY = config['DATALOADER']['pin_memory']

# DATA
MAX_TOKEN = config['DATA']['max_token']

# MODEL
MODEL = config['MODEL']['model_str']
MODEL_PATH = './results/train/train_v1/model.pt'

NUM_TARGETS = config['MODEL']['num_targets']
DIM_I = config['MODEL']['dim_i']
DIM_Q = config['MODEL']['dim_q']
DIM_H = config['MODEL']['dim_h']
BATCH_SIZE = 1


torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset = TestDataset(root_dir=PROJECT_DIR)
answer_list = pd.read_csv("./results/train/train_v1/answers.csv")['answer'].tolist()
dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE)
Model = get_model(model_str=MODEL)
model = Model(num_targets=NUM_TARGETS, dim_i=DIM_I, dim_q=DIM_Q, dim_h=DIM_H).to(device)
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model'])

output_id = []
output_pred = []

model.to(device).eval()

with torch.no_grad():
    for batch_index, data in enumerate(dataloader):
        if batch_index % 50 == 0:
            print(f"{batch_index}")
        q_bert_ids = data['ids'].to(device)
        q_bert_mask = data['mask'].to(device)
        imgs = data['image'].to(device)
        target_pred = model(q_bert_ids, q_bert_mask, imgs)
        logits = target_pred[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        answer = answer_list[result]
        output_id.append(batch_index)
        output_pred.append(answer)

pd.DataFrame({'ID': output_id, 'answer': output_pred}).to_csv('./results/train/train_v1/submission.csv', index=False)
