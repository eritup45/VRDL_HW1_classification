import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image

from model import *
from Resnet import *

class Smooth_CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, class_num, label_smooth=0.1):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        ''' 
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)	# softmax + log
            target = F.one_hot(target, self.class_num)	# one-hot
            
            # label smoothing
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num 	
            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
            loss = -1*torch.sum(target*logprobs, 1)
        else:
            # standard cross entropy loss
            loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))
        return loss.mean()

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 200)

    return net


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def test(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda:0")

    transform_test = transforms.Compose([
        transforms.Resize((550, 550), Image.BILINEAR),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    testset = torchvision.datasets.ImageFolder(root='./data/train_valid/val',
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    is_train = False
    with torch.set_grad_enabled(is_train):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            output_1, output_2, output_3, output_concat= net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('(Test) Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss


