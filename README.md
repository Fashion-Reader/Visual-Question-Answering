## Visual Question Answering
> - 이미지를 보고 주어진 질문에 답변하는 Visual Question Answering 모델 개발

### Members

| [권태양](https://github.com/sunnight9507) | [류재희](https://github.com/JaeheeRyu) | [박종헌](https://github.com/PJHgh) | [신찬엽](https://github.com/chanyub) | [조원](https://github.com/jo-member) |

### Set up

#### 1. Install Requirements
```
$ pip install -r requirements.txt
```

#### 2. Train
```
$ python3 train_v1.py # version 1
$ python3 train_v2.py # version 2
```

#### 3. Inference
```
$ 
```

### Code File
```
$ fashion_reader
├── config
│   ├── train_config_base.yaml
├── models
│   ├── get_model.py
│   └── vqa_model.py
├── modulus
│   ├── dataset.py
│   ├── earlystoppers.py
│   ├── recorders.py
│   ├── trainer.py
│   └── utils.py
├── results
├── train_v1.py
└── train_v2.py
```

### Output
```
$ fashion_reader
└── results
    ├── train_v1
    │     ├── loss.png
    │     ├── model.pt
    │     ├── answers.csv
    │     ├── score.jpg
    │     ├── train_config_base.yaml
    │     └── train_log.log
    ├── train_v2
    │     └── ...
```

### Description
|Version|Pre-trained Model|Config|
|:---|:---|:---|
| V1 | xlm-roberta-base & resnet50 | [Link](https://github.com/Fashion-Reader/Visual-Question-Answering/blob/main/code/results/train_v1/train_config_v1.yaml) |
| V2 | xlm-roberta-large & resnet50 |  |
| V3 | xlm-roberta-base & resnet152 |  |
| V4 | xlm-roberta-base & resnet152 |  |
| V5 | xlm-roberta-base & resnet152 |  |
