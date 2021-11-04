# VRDL_HW1_Fine-Grained_classification

### Objective
Classify 200 bird species. 

### Requirement

1. conda create environment from file.
```bash
conda env create -f environment.yml
conda activate TransFG
```
> May show some error. Just ignore them.

2. install pytorch correspond to your cuda version.

python 3.7

PyTorch >= 1.5.1

torchvision >= 0.6.1

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
Put pretrained model in the path: 
"./ckpt/ep5_vloss1.990_vacc81.8_vac81.7.pth"

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
https://hackmd.io/@Bmj6Z_QbTMy769jUvLGShA/VRDLHW1

### Reference
[Progressive Multi-Granularity Training](https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training)

