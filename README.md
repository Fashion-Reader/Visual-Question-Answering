# Visual-Question-Answering

### Set up

#### 1. Install Requirements
```
$ pip install -r requirements.txt
```

#### 2. Train
```
$ python3 train_v1.py
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
└── train_v1.py
```

### Output
```
$ fashion_reader
└── results
    ├── loss.png
    ├── model.pt
    ├── answers.csv
    ├── score.jpg
    ├── train_config_base.yaml
    └── train_log.log
```

### Description
|Version|Pre-trained Model|Link|
|:---|:---|:---|
| <pre>V1 | <pre>xlm-roberta-base & resnet50 | <pre> |
| <pre>V2 | <pre>xlm-roberta-large & resnet50 | <pre> |
| <pre>V3 | <pre>xlm-roberta-base & resnet152 | <pre> |
| <pre>V4 | <pre>xlm-roberta-base & resnet152 | <pre> |
| <pre>V5 | <pre>xlm-roberta-base & resnet152 | <pre> |
| <pre>V6 | <pre>koelectra-base-v3-discriminator & resnet50 | <pre> |
| <pre>V7 | <pre>koelectra-base-v3-discriminator & efficientnet b0 | <pre> |
| <pre>V8 | <pre>xlm-roberta-large & resnet50 | <pre> |
