
# To include lib
import sys

sys.path.insert(0, '../')

import argparse
import copy
import json
import math
import os
import random
import shutil
import time

import numpy as np
import pickle5 as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from galaxyzoodataset import GalaxyZooDataset
from models.experts import synth_expert
from models.resnet50 import ResNet50_defer
from torch.autograd import Variable

from lib.losses import Criterion
from lib.utils import AverageMeter, accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,  flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(model,
             expert_fns,
             loss_fn,
             n_classes,
             data_loader,
             config):
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
    #  === Individual Expert Accuracies === #
    expert_correct_dic = {k: 0 for k in range(len(expert_fns))}
    expert_total_dic = {k: 0 for k in range(len(expert_fns))}
    #  === Individual  Expert Accuracies === #
    alpha = config["alpha"]
    losses_log = []
    losses = AverageMeter()
    top1 = AverageMeter()
    with torch.no_grad():
        for data in data_loader:
            images, labels, hpred = data
            images, labels, hpred = images.to(device), labels.to(device), hpred
            outputs = model(images)
            if config["loss_type"] == "softmax":
                outputs = F.log_softmax(outputs, dim=1)
            # if config["loss_type"] == "ova":
            # 	ouputs = F.sigmoid(outputs)

            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size
            loss = loss_fn(outputs, labels)  # , collection_Ms, n_classes)
            losses_log.append(loss.item())

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, labels, topk=(1,))[0]
            losses.update(loss.data.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))

        # if i % 10 == 0:
        print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  loss=losses, top1=top1), flush=True)
        to_print = {'system_accuracy': top1.avg,
                    'validation_loss': np.average(losses_log)}

        return to_print

def train_epoch(iters,
                warmup_iters,
                lrate,
                train_loader,
                model,
                optimizer,
                scheduler,
                epoch,
                expert_fns,
                loss_fn,
                n_classes,
                alpha,
                config):
    """ Train for one epoch """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    epoch_train_loss = []

    for i, (input, target, hpred) in enumerate(train_loader):
        if iters < warmup_iters:
            lr = lrate * float(iters) / warmup_iters
            print(iters, lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        target = target.to(device)
        input = input.to(device)
        hpred = hpred

        # compute output
        output = model(input)

        if config["loss_type"] == "softmax":
            output = F.log_softmax(output, dim=1)

        # get expert  predictions and costs
        batch_size = output.size()[0]  # batch_size

        # compute loss
        loss = loss_fn(output, target)  # , collection_Ms, n_classes)
        epoch_train_loss.append(loss.item())

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
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


def train(model,
          train_dataset,
          validation_dataset,
          expert_fns,
          config,
          seed=""):
    n_classes = config["n_classes"] + len(expert_fns)
    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
    model = model.to(device)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), config["lr"],
                                 weight_decay=config["weight_decay"])
    #criterion = Criterion()
    loss_fn = nn.NLLLoss()  # getattr(criterion, config["loss_type"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader) * config["epochs"])
    best_validation_loss = np.inf
    patience = 0
    iters = 0
    warmup_iters = config["warmup_epochs"] * len(train_loader)
    lrate = config["lr"]

    for epoch in range(0, config["epochs"]):
        iters, train_loss = train_epoch(iters,
                                        warmup_iters,
                                        lrate,
                                        train_loader,
                                        model,
                                        optimizer,
                                        scheduler,
                                        epoch,
                                        expert_fns,
                                        loss_fn,
                                        n_classes,
                                        config["alpha"],
                                        config)
        metrics = evaluate(model,
                           expert_fns,
                           loss_fn,
                           n_classes,
                           valid_loader,
                           config)

        validation_loss = metrics["validation_loss"]

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            print("Saving the model with system accuracy {}".format(
                metrics['system_accuracy']), flush=True)
            save_path = os.path.join(config["ckp_dir"],
                                     config["experiment_name"] + '_' + str(len(expert_fns)) + '_experts' + '_seed_' + str(seed))
            torch.save(model.state_dict(), save_path + '.pt')
            # Additionally save the whole config dict
            with open(save_path + '.json', "w") as f:
                json.dump(config, f)
            patience = 0
        else:
            patience += 1

        if patience >= config["patience"]:
            print("Early Exiting Training.", flush=True)
            break


def OneClassifier(config):
    config["ckp_dir"] = "./" + config["loss_type"] + "_classifier"
    os.makedirs(config["ckp_dir"], exist_ok=True)

    # experiment_experts = [1] #,2,3,4,5,6,7,8] #,9,10]

    expert_fns = []
    # , 948,  625,  436,  791]: #, 1750,  812, 1331, 1617,  650, 1816]:
    for seed in ['', 948, 625, 436, 791]:
        print("run for seed {}".format(seed))
        if seed != '':
            set_seed(seed)
        model = model = ResNet50_defer(int(config["n_classes"]))
        trainD = GalaxyZooDataset()
        valD = GalaxyZooDataset(split='val')
        train(model, trainD, valD, expert_fns, config, seed=seed)

        # pth = os.path.join(config['ckp_dir'], config['experiment_name'] + '_log_' + '_seed_' + str(seed))
        # with open(pth + '.json', 'w') as f:
        # 	json.dump(log, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="scaling parameter for the loss function, default=1.0.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10,
                        help="number of patience steps for early stopping the training.")
    parser.add_argument("--expert_type", type=str, default="predict_prob",
                        help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.")
    parser.add_argument("--n_classes", type=int, default=2,
                        help="K for K class classification.")
    parser.add_argument("--k", type=int, default=0)
    
    parser.add_argument("--n_experts", type=int, default=2)
    
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--loss_type", type=str, default="softmax",
                        help="surrogate loss type for learning to defer.")

    parser.add_argument("--ckp_dir", type=str, default="./Models",
                        help="directory name to save the checkpoints.")
    parser.add_argument("--experiment_name", type=str, default="classifier",
                        help="specify the experiment name. Checkpoints will be saved with this name.")

    config = parser.parse_args().__dict__

    # print(config)
    OneClassifier(config)
