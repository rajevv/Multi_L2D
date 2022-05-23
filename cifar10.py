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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class Net(nn.Module):
    '''
    Linear multiclass classifier with unit init
    '''

    def __init__(self, input_dim=2, dims=[32, 16, 8, 5]):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, dims[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dims[0], dims[1])
        self.fc3 = nn.Linear(dims[1], dims[2])
        self.fc4 = nn.Linear(dims[2], dims[3])
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))
        x = self.softmax(x)
        return x


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
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size

            expert_predictions = []
            collection_Ms = []  # a collection of 3-tuple
            for i, fn in enumerate(expert_fns, 0):
                exp_prediction1 = fn(images, labels)
                m = [0] * batch_size
                m2 = [0] * batch_size
                for j in range(0, batch_size):
                    if exp_prediction1[j] == labels[j].item():
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

            loss = reject_CrossEntropyLoss(outputs, labels, collection_Ms, n_classes)
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
                alone_correct += (prediction == labels[i]).item()
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    # print(r, boo)
                    for j in range(len(boo)):
                        if boo[j] == True:
                            # print("in if ", boo[j])
                            exp_prediction = expert_predictions[j][i]
                    exp += (exp_prediction == labels[i].item())
                    correct_sys += (exp_prediction == labels[i].item())
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
        output = model(input)

        # get expert  predictions and costs
        batch_size = output.size()[0]  # batch_size

        # first expert
        collection_Ms = []
        # We only support \alpha=1
        for _, fn in enumerate(expert_fns):
            m = fn(input, target)
            m2 = [0] * batch_size
            for j in range(0, batch_size):
                if m[j] == target[j].item():
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

        rand_target = target
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


# expert predicts first and second class correctly with 95%confidence
# expert predicts zeroth and third class correctly with 70% confidence
class synth_expert:
    '''
    simple class to describe our synthetic expert on CIFAR-10
    ----
    k: number of classes expert can predict
    n_classes: number of classes (10+1 for CIFAR-10)
    '''

    def __init__(self, n_classes):
        # self.k = k
        self.n_classes = n_classes

    def predict(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i].item() == 1 or labels[i].item() == 2:
                coin_flip = np.random.binomial(1, 0.7)
                if coin_flip == 1:
                    outs[i] = labels[i].item()
                else:
                    outs[i] = random.randint(0, self.n_classes - 1)
            if labels[i].item() == 0 or labels[i].item() == 3:
                coin_flip = np.random.binomial(1, 0.2)
                if coin_flip == 1:
                    outs[i] = labels[i].item()
                else:
                    outs[i] = random.randint(0, self.n_classes - 1)
        return outs


# expert = synth_expert(n_dataset)


if __name__ == "__main__":
    import os

    experimental_data_rej1 = []
    experimental_data_rej5 = []
    experimental_data_rej0 = []
    experimental_data_madras = []
    experimental_data_ora = []
    experimental_data_conf = []
    trials = 1
    TO_PRINT = False
    for exp in tqdm(range(0, trials)):
        d = 2
        total_samples = 20000
        group_proportion = np.random.uniform()
        if group_proportion <= 0.02:
            group_proportion = 0.02
        if group_proportion >= 0.98:
            group_proportion = 0.98
        # group_proportion = 0.4
        cluster1_mean = torch.rand(d) * d - 1
        cluster1_var = torch.rand(d) * d

        print(cluster1_mean, cluster1_var)

        cluster1 = sample(
            cluster1_mean,
            cluster1_var,
            nb_samples=math.floor(total_samples * group_proportion * 0.5)
        )
        cluster1_labels = torch.zeros([math.floor(total_samples * group_proportion * 0.5)], dtype=torch.long)
        cluster2_mean = torch.rand(d) * d + 4
        cluster2_var = torch.rand(d) * d

        print(cluster2_mean, cluster2_var)
        cluster2 = sample(
            cluster2_mean,
            cluster2_var,
            nb_samples=math.floor(total_samples * group_proportion * 0.5)
        )
        cluster2_labels = torch.ones([math.floor(total_samples * group_proportion * 0.5)], dtype=torch.long)
        cluster3_mean = torch.rand(d) * d + 5
        cluster3_var = torch.rand(d) * d

        print(cluster3_mean, cluster3_var)

        cluster3 = sample(
            cluster3_mean,
            cluster3_var,
            nb_samples=math.floor(total_samples * (1 - group_proportion) * 0.5)
        )
        cluster3_labels = torch.ones([math.floor(total_samples * (1 - group_proportion) * 0.5)], dtype=torch.long) + 1.0

        cluster4_mean = torch.rand(d) * d - 5
        cluster4_var = torch.rand(d) * d

        print(cluster4_mean, cluster4_var)

        cluster4 = sample(
            cluster4_mean,
            cluster4_var,
            nb_samples=math.floor(total_samples * (1 - group_proportion) * 0.5)
        )
        cluster4_labels = torch.ones([math.floor(total_samples * (1 - group_proportion) * 0.5)], dtype=torch.long) + 2.0

        # valid data
        cluster1_valid = sample(
            cluster1_mean,
            cluster1_var,
            nb_samples=math.floor(total_samples * group_proportion * 0.5)
        )
        cluster1_labels_valid = torch.zeros([math.floor(total_samples * group_proportion * 0.5)], dtype=torch.long)

        cluster2_valid = sample(
            cluster2_mean,
            cluster2_var,
            nb_samples=math.floor(total_samples * group_proportion * 0.5)
        )
        cluster2_labels_valid = torch.ones([math.floor(total_samples * group_proportion * 0.5)], dtype=torch.long)

        cluster3_valid = sample(
            cluster3_mean,
            cluster3_var,
            nb_samples=math.floor(total_samples * (1 - group_proportion) * 0.5)
        )
        cluster3_labels_valid = torch.ones([math.floor(total_samples * (1 - group_proportion) * 0.5)],
                                           dtype=torch.long) + 1.0

        cluster4_valid = sample(
            cluster4_mean,
            cluster4_var,
            nb_samples=math.floor(total_samples * (1 - group_proportion) * 0.5)
        )
        cluster4_labels_valid = torch.ones([math.floor(total_samples * (1 - group_proportion) * 0.5)],
                                           dtype=torch.long) + 2.0
        data_x_valid = torch.cat([cluster1_valid, cluster2_valid, cluster3_valid, cluster4_valid])
        data_y_valid = torch.cat(
            [cluster1_labels_valid, cluster2_labels_valid, cluster3_labels_valid, cluster4_labels_valid])

        # test data
        cluster1_test = sample(
            cluster1_mean,
            cluster1_var,
            nb_samples=math.floor(total_samples * group_proportion * 0.5)
        )
        cluster1_labels_test = torch.zeros([math.floor(total_samples * group_proportion * 0.5)], dtype=torch.long)

        cluster2_test = sample(
            cluster2_mean,
            cluster2_var,
            nb_samples=math.floor(total_samples * group_proportion * 0.5)
        )
        cluster2_labels_test = torch.ones([math.floor(total_samples * group_proportion * 0.5)], dtype=torch.long)

        cluster3_test = sample(
            cluster3_mean,
            cluster3_var,
            nb_samples=math.floor(total_samples * (1 - group_proportion) * 0.5)
        )
        cluster3_labels_test = torch.ones([math.floor(total_samples * (1 - group_proportion) * 0.5)],
                                          dtype=torch.long) + 1.0

        cluster4_test = sample(
            cluster4_mean,
            cluster4_var,
            nb_samples=math.floor(total_samples * (1 - group_proportion) * 0.5)
        )
        cluster4_labels_test = torch.ones([math.floor(total_samples * (1 - group_proportion) * 0.5)],
                                          dtype=torch.long) + 2.0
        data_x_test = torch.cat([cluster1_test, cluster2_test, cluster3_test, cluster4_test])
        data_y_test = torch.cat(
            [cluster1_labels_test, cluster2_labels_test, cluster3_labels_test, cluster4_labels_test])

    data_x = torch.cat([cluster1, cluster2, cluster3, cluster4])
    data_y = torch.cat([cluster1_labels, cluster2_labels, cluster3_labels, cluster4_labels])

    train_d = Data(data_x, data_y)
    valid_d = Data(data_x_valid, data_y_valid)
    test_d = Data(data_x_test, data_y_test)

    alpha = 1.0
    bsz = 1024
    k = 2
    n_dataset = 4
    os.makedirs('./checkpoints', exists_ok=True)
    pth = './checkpoints/'
    model = 'synthetic_mutiple_experts'
    for n in [2, 4, 6, 8]:
        print("Training with {} experts".format(n), flush=True)
        path = pth + model + '_' + str(n) + '_experts'
        num_experts = n
        expert = synth_expert(n_dataset)
        model = Net(dims=[32, 16, 8, n_dataset + num_experts])
        # fill experts in the reverse order
        expert_fns = [expert.predict] * n
        run_reject(model, train_d, valid_d, n_dataset + num_experts, num_experts, expert_fns, 200, alpha, bsz, k,
                   save_path=path)  # train for 200 epochs