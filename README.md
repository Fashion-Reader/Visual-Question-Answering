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
    └── train_log.log
```
