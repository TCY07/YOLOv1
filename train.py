from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
from torchvision import models

from resnet import resnet50
from yoloLoss import YoloLoss
from dataset import Yolodata
import time

parser = argparse.ArgumentParser(description='PyTorch Yolov1 Trainer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', default=15, type=int, help='batch size')
parser.add_argument('--num_epoches', default=1, type=int, help='training length')
parser.add_argument('--sgrid', default=7, type=int, help='grid number 7*7 for default')
parser.add_argument('--bbxnumber', default=2, type=int, help='bounding box number')
parser.add_argument('--classnumber', default=20, type=int, help='class number default is 20')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
print('cuda available:', torch.cuda.is_available())
print('use cuda:', torch.cuda.is_available() and args.cuda, '\n')

print('loading dataset ...')
Datasetinstance = Yolodata(train_file_root='./data/VOCdevkit/VOC2007/JPEGImages/',
                           train_listano='voc2007.txt',
                           test_file_root='./data/VOCdevkit/VOC2007/JPEGImages/',
                           test_listano='voc2007.txt', batchsize=args.batch_size, snumber=args.sgrid,
                           bnumber=args.bbxnumber, cnumber=args.classnumber)
train_loader, test_loader = Datasetinstance.getdata()
print('dataloader created\n')

print('batch_size: %d' % args.batch_size)
print('the dataset has %d * %d = %d images for training\n' % (len(train_loader), args.batch_size,
                                                              len(train_loader) * args.batch_size))

print('loading network structure ...')
net = resnet50()
net = net.to(device)

print('loading pre-trined model...\n')
resnet = models.resnet50()

if args.resume:
    print('resume')
    net.load_state_dict(torch.load('result.model'))
else:
    # pre-trained model:
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    resnet.load_state_dict(torch.load('./data/resnet50-0676ba61.pth'))
    new_state_dict = resnet.state_dict()
    op = net.state_dict()
    for k in new_state_dict.keys():
        if k in op.keys() and not k.startswith('fc'):
            op[k] = new_state_dict[k]
    net.load_state_dict(op)

criterion = YoloLoss(args.batch_size, args.bbxnumber, args.classnumber, lambda_coord=0.5, lambda_noobj=0.5)

net.train()
# different learning rate
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr': args.lr * 1}]
    else:
        params += [{'params': [value], 'lr': args.lr}]
optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)


def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        pred = net(inputs)

        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        print('batch %s of total batch %s' % (batch_idx, len(train_loader)),
              'Loss: %.3f ' % (train_loss / (batch_idx + 1)))

    end_time = time.time()
    epoch_time = end_time - start_time
    data = [epoch, train_loss / (batch_idx + 1), epoch_time]
    print('trainloss:{}, time_used:{}'.format(train_loss / (batch_idx + 1), epoch_time))
    return data


log_file = open("log.txt", 'w')

for epoch in range(args.num_epoches):
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epoches))

    nd = train(epoch)
    log_file.write(' '.join('%s' % item for item in nd) + '\n')
    torch.save(net.state_dict(), 'result.model')


log_file.close()
