# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import argparse
import time
import csv
import datetime
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.functional as F

from logger import TermLogger, AverageMeter
from path import Path
from itertools import chain
from tensorboardX import SummaryWriter

from utils import tensor2array, save_checkpoint

# from mnist_em import AutoEncoder

parser = argparse.ArgumentParser(description='MNIST and SVHN training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size')

parser.add_argument('--pretrained-autoencoder', dest='pretrained_autoencoder', default=None, metavar='PATH',
                    help='path to pre-trained autoencoder')
parser.add_argument('--pretrained-mod', dest='pretrained_mod', default=None, metavar='PATH',
                    help='path to pre-trained moderator')

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, 3, 1, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1, 1)
        self.fc1 = nn.Linear(40*7*7, 20)
        self.fc2 = nn.Linear(20, 40*7*7)
        self.deconv1 = nn.ConvTranspose2d(40, 40, 2, 2)
        self.deconv2 = nn.ConvTranspose2d(40, 1, 2, 2)
        # self.conv3 = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 40*7*7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 40, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        # x = F.relu(self.conv3(x))
        return x

    def name(self):
        return "AutoEncoder"


class LeNet(nn.Module):
    def __init__(self, nout=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.fc1 = nn.Linear(40*5*5, 40)
        self.fc2 = nn.Linear(40, nout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 40*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"

def main():
    global args
    args = parser.parse_args()

    args.data = Path(args.data)

    print("=> fetching dataset")
    mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    valset_mnist = torchvision.datasets.MNIST(args.data/'mnist', train=False, transform=mnist_transform, target_transform=None, download=True)

    svhn_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(28,28)),
                                                    torchvision.transforms.Grayscale(),
                                                    torchvision.transforms.ToTensor()])
    valset_svhn = torchvision.datasets.SVHN(args.data/'svhn', split='test', transform=svhn_transform, target_transform=None, download=True)
    val_set = torch.utils.data.ConcatDataset([valset_mnist, valset_svhn])


    print('{} Test samples found in MNIST'.format(len(valset_mnist)))
    print('{} Test samples found in SVHN'.format(len(valset_svhn)))

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    val_loader_mnist = torch.utils.data.DataLoader(
        valset_mnist, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    val_loader_svhn = torch.utils.data.DataLoader(
        valset_svhn, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # create model
    print("=> creating model")
    # define 10 autoencoder, 1 for each mnist class
    autoencoder_list = nn.ModuleList([AutoEncoder() for i in range(10)])
    mod_net = LeNet()

    print("=> using pre-trained weights from {}".format(args.pretrained_autoencoder))
    weights = torch.load(args.pretrained_autoencoder)
    autoencoder_list.load_state_dict(weights['state_dict'])

    print("=> using pre-trained weights from {}".format(args.pretrained_mod))
    weights = torch.load(args.pretrained_mod)
    mod_net.load_state_dict(weights['state_dict'])

    cudnn.benchmark = True
    autoencoder_list = autoencoder_list.cuda()
    mod_net = mod_net.cuda()

    # evaluate on validation set
    errors_mnist, error_names_mnist, mod_count_mnist, autoencoder_class_mnist = validate(val_loader_mnist, autoencoder_list, mod_net)
    errors_svhn, error_names_svhn, mod_count_svhn, autoencoder_class_svhn = validate(val_loader_svhn, autoencoder_list, mod_net)
    errors_total, error_names_total, mod_count_total, autoencoder_class_total = validate(val_loader, autoencoder_list, mod_net)

    accuracy_string_mnist = ', '.join('{} : {:.3f}'.format(name, 100*(error)) for name, error in zip(error_names_mnist, errors_mnist))
    accuracy_string_svhn = ', '.join('{} : {:.3f}'.format(name, 100*(error)) for name, error in zip(error_names_svhn, errors_svhn))
    accuracy_string_total = ', '.join('{} : {:.3f}'.format(name, 100*(error)) for name, error in zip(error_names_total, errors_total))

    print("MNIST Error")
    print(accuracy_string_mnist)
    for i in range(10): # hardcoded for 10 classes
        print ("MNIST Picking Percentage - AutoEncoder_{}: {:.5f}, Class: {}".format(i, mod_count_mnist[i]*100, autoencoder_class_mnist[i]))

    print("SVHN Error")
    print(accuracy_string_svhn)
    for i in range(10):
        print ("SVHN Picking Percentage - AutoEncoder_{}: {:.5f}, Class: {}".format(i, mod_count_svhn[i]*100, autoencoder_class_svhn[i]))

    print("TOTAL Error")
    print(accuracy_string_total)

def validate(val_loader, autoencoder_list, mod_net):
    global args
    accuracy = AverageMeter(i=1, precision=4)
    mod_count = AverageMeter(i=10)

    # switch to evaluate mode
    for model in autoencoder_list:
        model.eval()
    mod_net.eval()

    pred_mod_count = [[0]*10 for i in range(10)]
    autoencoder_accuracy = []
    autoencoder_class = []

    for i, (img, target) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            img_var = Variable(img.cuda())
            target_var = Variable(target.cuda())

            pred_autoencoder = []
            for mode in autoencoder_list:
                pred = model(img_var)
                pred_autoencoder.append(pred)
            
            pred_mod = F.softmax(torch.sigmoid(mod_net(img_var)), dim=1)

            _, pred_label = torch.max(pred_mod, 1)
            pred_label = pred_label.squeeze().data

            total_accuracy = (pred_label.cpu() == target).sum().item() / img.size(0)
            accuracy.update([total_accuracy])
        
        for i in range(img.size(0)):
            pred_mod_count[int(pred_label[i])][int(target[i])] += 1

    
    for i in range(10):
        curr_arr = np.array(pred_mod_count[i])
        max_class = np.argmax(curr_arr)
        class_acc = np.max(curr_arr)/np.sum(curr_arr)
        autoencoder_class.append(max_class)
        autoencoder_accuracy.append(class_acc)

    # pred_mod_count = [pred_mod_count[i]/img.size(0) for i in range(10)]
    mod_count.update(autoencoder_accuracy)
    # mod_count.update((pred_mod.cpu().data > 0.5).sum().item() / img.size(0))

    return [1-accuracy.avg[0]], ['Total'], mod_count.avg, autoencoder_class
    # return list(map(lambda x: 1-x, accuracy.avg)), ['Total', 'alice', 'bob'] , mod_count.avg



if __name__ == '__main__':
    # import sys
    # with open("experiment_recorder.md", "a") as f:
    #     f.write('\n python3 ' + ' '.join(sys.argv))
    main()
