# VRDL_HW1_classification

# Progressive Multi-Granularity Training
 
Code release for Fine-Grained Visual Classiﬁcation via Progressive Multi-Granularity Training of Jigsaw Patches (ECCV2020)
 
### Requirement
 
python 3.6

PyTorch >= 1.3.1

torchvision >= 0.4.2

> Use cuda:0 as default

### Training

1. Download datatsets and organize the structure as follows:
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

2. Train from scratch: 
```bash
python train.py
```

### Testing

```bash
python inference.py
```

> The results will be saved in ./answer.txt

### Pretrained model

ep5_vloss1.990_vacc81.8_vac81.7.pth


### Results
![](https://i.imgur.com/dbRHjo8.png)



### Reference

https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training
