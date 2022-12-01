import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import math
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import time
import json

import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(2)
device


def sample(mu, var, nb_samples=500):
    """
    sample guassian random variable
    :param mu: torch.Tensor (features)
    :param var: torch.Tensor (features) (note: zero covariance)
    :return: torch.Tensor (nb_samples, features)
    """
    out = []
    for i in range(nb_samples):
        out += [
            torch.normal(mu, var.sqrt())
        ]
    return torch.stack(out, dim=0)


# data_x = torch.cat([cluster3, cluster4])
# data_y = torch.cat([cluster3_labels, cluster4_labels])

# data_x = torch.cat([cluster1, cluster2, cluster3, cluster4])
# data_y = torch.cat([cluster1_labels, cluster2_labels, cluster3_labels, cluster4_labels])


class Data(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index].float(), self.labels[index].long()

    def __len__(self):
        return len(self.data)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, n_channels, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(n_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.softmax = nn.Softmax()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)

        out = self.block1(out)

        out = self.block2(out)

        out = self.block3(out)

        out = self.relu(self.bn1(out))

        out = F.avg_pool2d(out, 8)

        out = out.view(-1, self.nChannels)

        fc = self.fc(out)
        out = self.softmax(fc)
        return out, fc  # return noth softmax output and logits


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def metrics_print(net, num_experts, expert_fns, n_classes, loader):
    '''
    Computes metrics for deferal
    -----
    Arguments:
    net: model
    expert_fn: expert model
    n_classes: number of classes
    loader: data loader
    '''
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    losses = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device).float(), labels.to(device)
            # outputs = net(images)  # MoG
            outputs, _ = model(images)  # CIFAR10
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size

            expert_predictions = []
            collection_Ms = []  # a collection of 3-tuple
            for i, fn in enumerate(expert_fns, 0):
                exp_prediction1 = fn(images, labels)
                m = [0] * batch_size
                m2 = [0] * batch_size
                for j in range(0, batch_size):
                    if exp_prediction1[j] == labels[j][0].item():
                        m[j] = 1
                        m2[j] = 1
                    else:
                        m[j] = 0
                        m2[j] = 1

                m = torch.tensor(m)
                m2 = torch.tensor(m2)
                m = m.to(device)
                m2 = m2.to(device)
                collection_Ms.append((m, m2))
                expert_predictions.append(exp_prediction1)
            # second expert
            # exp_prediction2 = expert_fn2(images, labels)
            # n = [0]*batch_size
            # n2 = [0] * batch_size
            # for j in range(0, batch_size):
            # 	if exp_prediction2[j] == labels[j].item():
            # 		n[j] = 1
            # 		n2[j] = 1
            # 	else:
            # 		n[j] = 0
            # 		n2[j] = 1
            # n = torch.tensor(n)
            # n2 = torch.tensor(n2)
            # n = n.to(device)
            # n2 = n2.to(device)

            loss = reject_CrossEntropyLoss(outputs, labels[:, 0], collection_Ms, n_classes)
            losses.append(loss.item())

            for i in range(0, batch_size):
                # whether the decision is to defer to any of the expert
                boo = []
                for j in range(num_experts):
                    boo.append(predicted[i].item() == n_classes - (j + 1))
                if sum(boo) >= 1:
                    r = 1
                else:
                    r = 0
                prediction = predicted[i]
                # if the expert predicts
                if predicted[i] >= n_classes - num_experts:
                    max_idx = 0
                    # get max from classifier
                    for j in range(0, n_classes - num_experts):
                        if outputs.data[i][j] >= outputs.data[i][max_idx]:
                            max_idx = j
                    prediction = max_idx
                # else the classifier predicts
                else:
                    prediction = predicted[i]
                alone_correct += (prediction == labels[i][0]).item()
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i][0]).item()
                    correct_sys += (predicted[i] == labels[i][0]).item()
                if r == 1:
                    # print(r, boo)
                    for j in range(len(boo)):
                        if boo[j] == True:
                            # print("in if ", boo[j])
                            exp_prediction = expert_predictions[j][i]
                    exp += (exp_prediction == labels[i][0].item())
                    correct_sys += (exp_prediction == labels[i][0].item())
                    exp_total += 1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system_accuracy": 100 * correct_sys / real_total,
                "expert_accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier_accuracy": 100 * correct / (total + 0.0001),
                "alone_classifier": 100 * alone_correct / real_total,
                "validation_loss": np.average(losses)}
    print(to_print, flush=True)
    return to_print


def reject_CrossEntropyLoss(outputs, labels, collection_Ms, n_classes, logits=False):
    '''
    The L_{CE} loss implementation for CIFAR
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    labels: target
    m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    n_classes: number of classes
    '''
    if logits:
        outputs = F.softmax(outputs)
    batch_size = outputs.size()[0]  # batch_size
    m2 = collection_Ms[0][1]
    rcs = []
    for i, _ in enumerate(collection_Ms, 0):
        rcs.append([n_classes - (i + 1)] * batch_size)
    # rc1 = [n_classes-2] * batch_size
    # rc2 = [n_classes-1] * batch_size
    # print(labels)
    temp = - m2 * torch.log2(outputs[range(batch_size), labels])
    for i, (m, _) in enumerate(collection_Ms):
        temp -= m * torch.log2(outputs[range(batch_size), rcs[len(rcs) - 1 - i]])
    # outputs = -m * torch.log2(outputs[range(batch_size), rc1]) - n * torch.log2(outputs[range(batch_size), rc2]) - m2 * torch.log2(
    # outputs[range(batch_size), labels])
    return torch.sum(temp) / batch_size


def my_CrossEntropyLoss(outputs, labels):
    # Regular Cross entropy loss
    batch_size = outputs.size()[0]  # batch_size
    outputs = - torch.log2(outputs[range(batch_size), labels])  # regular CE
    return torch.sum(outputs) / batch_size


def sample_target(batch_size):
    target = []
    for i in range(batch_size):
        target.append(random.randint(0, 9))
    return torch.tensor(target)


def train_reject(iters, warmup_iters, lrate, train_loader, model, optimizer, scheduler, epoch, num_experts, expert_fns,
                 n_classes, alpha):
    """Train for one epoch on the training set with deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    epoch_train_loss = []
    for i, (input, target) in enumerate(train_loader):
        if iters < warmup_iters:
            lr = lrate * float(iters) / warmup_iters
            print(iters, lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        target = target.to(device)
        input = input.to(device).float()

        # compute output
        # output = model(input)  # MoG
        output, _ = model(input)  # CIFAR10

        # get expert  predictions and costs
        batch_size = output.size()[0]  # batch_size

        # first expert
        collection_Ms = []
        # We only support \alpha=1
        for _, fn in enumerate(expert_fns):
            m = fn(input, target)
            m2 = [0] * batch_size
            for j in range(0, batch_size):
                if m[j] == target[j][0].item():
                    m[j] = 1
                    m2[j] = alpha
                else:
                    m[j] = 0
                    m2[j] = 1
            m = torch.tensor(m)
            m2 = torch.tensor(m2)
            m = m.to(device)
            m2 = m2.to(device)
            collection_Ms.append((m, m2))

        rand_target = target[:, 0]
        loss = reject_CrossEntropyLoss(output, rand_target, collection_Ms, n_classes)
        epoch_train_loss.append(loss.item())

        # measure accuracy and record loss
        prec1 = accuracy(output.data, rand_target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not iters < warmup_iters:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        iters += 1

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1), flush=True)
    return iters, np.average(epoch_train_loss)


def run_reject(model, train_dataset, valid_dataset, n_classes, num_experts, expert_fns, epochs, alpha, batch_size, k,
               save_path='./'):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(valid_dataset,
                                              batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])), flush=True)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)

    best_validation_loss = np.inf
    patience = 0
    iters = 0
    warmup_iters = 0 * (len(train_loader))
    lrate = 0.1
    for epoch in range(0, epochs):
        # train for one epoch
        iters, train_loss = train_reject(iters, warmup_iters, lrate, train_loader, model, optimizer, scheduler, epoch,
                                         num_experts, expert_fns, n_classes, alpha)
        metrics = metrics_print(model, num_experts, expert_fns, n_classes, test_loader)
        print(metrics)
        validation_loss = metrics["validation_loss"]
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            print("Saving the model with classifier accuracy {}".format(metrics['classifier_accuracy']), flush=True)
            torch.save(model.state_dict(), save_path + '.pt')
            patience = 0
        else:
            patience += 1
        if patience >= 50:
            print("Early Exiting Training.", flush=True)
            break


# n_dataset = 4  # cifar-10


class synth_expert:
    '''
    simple class to describe our synthetic expert on CIFAR-10
    ----
    k: number of classes expert can predict
    n_classes: number of classes (10+1 for CIFAR-10)
    '''

    def __init__(self, k, n_classes):
        self.k = k
        self.n_classes = n_classes

    def predict(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() <= self.k:
                outs[i] = labels[i][0].item()
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs

    def predict_biasedK(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() <= self.k:
                coin_flip = np.random.binomial(1, 0.7)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs

    def predict_biased(self, input, labels):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            coin_flip = np.random.binomial(1, 0.7)
            if coin_flip == 1:
                outs[i] = labels[i][0].item()
            if coin_flip == 0:
                outs[i] = random.randint(0, self.n_classes - 1)
        return outs

    def predict_random(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            prediction_rand = random.randint(0, self.n_classes - 1)
            outs[i] = prediction_rand
        return outs

    # expert which only knows k labels and even outside of K predicts randomly from K
    def predict_severe(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() <= self.k:
                outs[i] = labels[i][0].item()
            else:
                prediction_rand = random.randint(0, self.k)
                outs[i] = prediction_rand
        return outs

    # when the input is OOD, expert predicts corrects else not
    def oracle(self, input, labels):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][1].item() == 0:
                outs[i] = labels[i][0]
            else:
                if labels[i][0].item() <= self.k:
                    outs[i] = labels[i][0].item()
                else:
                    prediction_rand = random.randint(0, self.n_classes - 1)
                    outs[i] = prediction_rand
        return outs


if __name__ == "__main__":
    from data_utils import *

    train, val = cifar.read(test=False, only_id=True, data_aug=True)
    print(len(train), len(val))

    alpha = 1.0
    bsz = 1024
    k = 5  # classes expert is oracle
    n_dataset = 10
    os.makedirs('./checkpoints', exist_ok=True)
    pth = './checkpoints/'
    model = 'synthetic_mutiple_experts'

    for n in [2, 4, 6, 8]:
        print("Training with {} experts".format(n), flush=True)
        path = pth + model + '_' + str(n) + '_experts'
        num_experts = n
        expert = synth_expert(k, n_dataset)
        model = WideResNet(28, 3, n_dataset + num_experts, 4, dropRate=0)
        # fill experts in the reverse order
        expert_fns = [expert.predict] * n
        run_reject(model, train, val, n_dataset + num_experts, num_experts, expert_fns, 200, alpha, bsz, k,
                   save_path=path)  # train for 200 epochs
