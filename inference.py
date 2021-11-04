from __future__ import print_function
import os
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torchvision
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from utils import *
# from utils import cosine_anneal_schedule, load_model, jigsaw_generator, test

from datasets import make_test_loader


def main():
    # submission
    submission = []
    with open('./data/testing_img_order.txt') as f:
        # all the testing images
        test_images = [x.strip() for x in f.readlines()]

    # dictionary of label
    classes, class_to_idx = find_classes('./data/train_valid/train/')

    # Model
    resume = "./ckpt/ep5_vloss1.990_vacc81.8_vac81.7.pth"
    batch_size = 1
    net = torch.load(resume)

    # GPU
    device = torch.device("cuda:0")
    net.to(device)
    # cudnn.benchmark = True

    net.eval()

    testloader = make_test_loader()
    # transform_test = transforms.Compose([
    #     transforms.Resize((500, 500), Image.BILINEAR),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])
    # testset = torchvision.datasets.ImageFolder(
    #     root='./data/test/',
    #     transform=transform_test)
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        epoch_iterator = tqdm(testloader)
        for idx, (inputs, targets) in enumerate(epoch_iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            output_1, output_2, output_3, output_concat = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)

            # Dictionary get key from value
            ans = list(class_to_idx.keys())[list(class_to_idx.values()).index(predicted.item())]
            submission.append([test_images[idx], ans])

            if idx <= 5 or idx >= 3030:
                print(
                    f'idx: {idx}. predicted.item(): {predicted.item()}. ' +
                    f'submit=[test_images[idx]: ' +
                    f'{test_images[idx]}, ans:{ans}]')

    np.savetxt('answer.txt', submission, fmt='%s')


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(
        entry.name for entry in os.scandir(directory) if entry.is_dir())

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    # print('class_to_idx: ', class_to_idx)
    return classes, class_to_idx

if __name__ == '__main__':
    main()
