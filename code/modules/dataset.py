"""Dataset 클래스 정의

TODO:

NOTES:

UPDATED:
"""

import os
import sys
import pandas as pd
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import transformers
import cv2
import torch
import re
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class TestDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'data')
        self.img_dir = os.path.join(self.data_dir, 'test_images')
        self.csv_path = os.path.join(self.data_dir,'test.csv')
        self.data_df = pd.read_csv(self.csv_path)
        
        self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.max_token = 30
        self.transform = transforms.Compose([
            transforms.Resize((356,356)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        img_fn = self.data_df['image'][idx]
        question = self.data_df['question'][idx]

        # BERT기반의 Tokenizer로 질문을 tokenize한다.
        tokenized = self.tokenizer.encode_plus("".join(question),
                                     None,
                                     add_special_tokens=True,
                                     max_length=self.max_token,
                                     truncation=True,
                                     pad_to_max_length=True)

        # BERT기반의 Tokenize한 질문의 결과를 변수에 저장
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']

        image_path = os.path.join(self.img_dir, img_fn)
        image = Image.open(image_path).convert('RGB')  #이미지 데이터를 RGB형태로 읽음
        image = self.transform(image)  #이미지 데이터의 크기 및 각도등을 변경


        #전처리가 끝난 질문, 이미지 데이터를 반환
        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'image': image}

    
class CustomDatasetBase(Dataset):
    def __init__(self, root_dir, result_dir, config, mode='train'):
        root_dir = os.path.join(root_dir, 'task07_train')
        PREPROCESS_SERIAL = str(config['PREPROCESS']['preprocess_serial'])
        if PREPROCESS_SERIAL=='None':
            self.data_dir = os.path.join(root_dir)
        else:
            self.data_dir = os.path.join(root_dir, PREPROCESS_SERIAL)
        self.result_dir = result_dir

        self.img_dir = os.path.join(self.data_dir, 'images')
        self.csv_path = os.path.join(self.data_dir, 'train.csv')

        print(f'Loading {self.csv_path}...')
        self.data_df = pd.read_csv(self.csv_path)
        print('Loaded')
        
        # Creating a dataframe
        train, valid = train_test_split(self.data_df, test_size=0.2, random_state=config['SEED']['random_seed'])

        if mode == 'train':
            self.data_df = train.reset_index(drop=True)
        elif mode == 'val':
            self.data_df = valid.reset_index(drop=True)
        print(f'{mode} Loaded')
        
        # Save answer list
        answer_list_path = os.path.join(self.result_dir, 'answers.csv')
        # if mode =='train'
        if mode == 'train':
            self.answer_list = self.data_df['answer'].unique().tolist()
            pd.DataFrame({'answer': self.answer_list}).to_csv(answer_list_path, index=False)
        else:
            self.answer_list = pd.read_csv(answer_list_path)['answer'].tolist()

        # Set Tokenizer
        self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.max_token = 30
        # Set transform
        self.transform = A.Compose([
            A.Resize(356,475),
            A.RandomCrop(224,224),
            A.CLAHE(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ])
        
    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        img_fn = self.data_df['image'][idx]
        question = self.data_df['question'][idx]
        answer = self.data_df['answer'][idx]

        # BERT기반의 Tokenizer로 질문을 tokenize한다.
        tokenized = self.tokenizer.encode_plus("".join(question),
                                     None,
                                     add_special_tokens=True,
                                     max_length=self.max_token,
                                     truncation=True,
                                    #  padding=True)
                                     pad_to_max_length=True)

        # BERT기반의 Tokenize한 질문의 결과를 변수에 저장
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']

        image_path = os.path.join(self.img_dir, img_fn)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.transform(image=image)['image']  #이미지 데이터의 크기 및 각도등을 변경
        
        if answer not in self.answer_list:
            print(f"Unexpected Target Token! {answer}")
            sys.exit()
        answer_id = self.answer_list.index(answer)

        #전처리가 끝난 질문, 응답, 이미지 데이터를 반환
        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'answer': torch.tensor(answer_id, dtype=torch.long),
                'image': image}


class CustomDatasetTemplate(Dataset):
    """Train Dataset 클래스 정의
    
    args:
        data_dir (str)
        mode (str) : train, val, test 중 1개
    """

    def __init__(self, data_dir, mode='train'):

        self.mode = mode
        self.data_path = os.path.join(data_dir)
        self.data = ...

        self.features = self.data[...]
        self.targets = self.data['target']

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx:int):
        feature = torch.tensor(self.features[idx, :], dtype=torch.float)
        if self.mode == 'test':
            return (feature, _)
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            return (feature, target)



if __name__ == '__main__':
    pass

