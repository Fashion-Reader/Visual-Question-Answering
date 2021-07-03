"""
"""
import transformers

import torch
from torch import nn
import torchvision.models as models

class VQAModel(nn.Module):
    def __init__(self, num_targets, dim_i, dim_q, dim_h=1024, large=False):
        super(VQAModel, self).__init__()

        #The BERT model: 질문 --> Vector 처리를 위한 XLM-Roberta모델 활용
        model = "xlm-roberta-large" if large else "xlm-roberta-base"
        self.bert = transformers.XLMRobertaModel.from_pretrained(model)

        #Backbone: 이미지 --> Vector 처리를 위해 ResNet50을 활용
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, dim_i)
        self.i_relu = nn.ReLU()
        self.i_drop = nn.Dropout(0.2)

        #classfier: MLP기반의 분류기를 생성
        self.linear1 = nn.Linear(dim_i, dim_h)
        self.q_relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_h, num_targets)
        self.q_drop = nn.Dropout(0.2)


    def forward(self, idx, mask, image):

        output = self.bert(idx, mask)
        q_f = output['pooler_output']

        i_f = self.i_drop(self.resnet(image)) # 이미지를 resnet을 활용해 Vector화
        uni_f = i_f * q_f #이미지와 질문 vector를 point-wise연산을 통해 통합 vector생성

        return self.linear2(self.q_relu(self.linear1(uni_f))) #MLP classfier로 답변 예측
