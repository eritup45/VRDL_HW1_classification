# VRDL_HW1_Fine-Grained_classification

### Requirement
 
python 3.6

PyTorch >= 1.3.1

torchvision >= 0.4.2

* conda create environment from file.
```bash
conda env create -f environment.yml
```

* pip install requirements.
```bash
pip install -r requirements.txt
```

> Use cuda:0 as default

### Data preparation

1. Download datatsets 
https://drive.google.com/drive/folders/1vzr_gRZri9kDt7Rbq8gBt1N9M5J46v_9?usp=sharing

2. organize the structure as follows:
```
data
├── train_valid
│   ├── train
│   │   ├── class_001
│   |   |      ├── 1.jpg
│   |   |      ├── 2.jpg
│   |   |      └── ...
│   │   ├── class_002
│   |   |      ├── 1.jpg
│   |   |      ├── 2.jpg
│   |   |      └── ...
│   │   └── ...
│   └── val
│       ├── class_001
│       |      ├── 1.jpg
│       |      ├── 2.jpg
│       |      └── ...
│       ├── class_002
│       |      ├── 1.jpg
│       |      ├── 2.jpg
│       |      └── ...
│       └── ...
└── test
    └── class_001
        ├── 1.jpg
        ├── 2.jpg
        └── ...
```

> In google drive, data has been categorized as above format.

### Training

Train and validate: 
```bash
python train.py
```

### Testing

```bash
python inference.py
```

### Pretrained model

ep5_vloss1.990_vacc81.8_vac81.7.pth
https://drive.google.com/drive/folders/1vzr_gRZri9kDt7Rbq8gBt1N9M5J46v_9?usp=sharing

### Results
Test score: 0.76624
Rank: 34/99
![](https://i.imgur.com/dbRHjo8.png)


### Report
https://hackmd.io/o-cz5Z65TySKpGFF47usBA?view


### Reference
[Progressive Multi-Granularity Training](https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training)

