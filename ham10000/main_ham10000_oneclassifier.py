
# To include lib
import sys

sys.path.insert(0, '../')

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Cifar 10 specific
from ham10000dataset import ham10000_expert
from models.baseline import ResNet34_oneclf
from models.resnet34 import ResNet34_defer
from torch.backends import cudnn

from lib.losses import Criterion
from lib.utils import AverageMeter, accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data_loader, loss_fn):
    correct = 0
    correct_sys = 0
    total = 0
    real_total = 0
    alone_correct = 0

    losses_log = []
    losses = AverageMeter()
    top1 = AverageMeter()
    with torch.no_grad():
        for data in data_loader:
            images, labels, hpred = data
            images, labels, hpred = images.to(device), labels.to(device), hpred
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size

            # One classifier loss ===
            log_output = torch.log(outputs + 1e-7)

            loss = loss_fn(log_output, labels)  # , collection_Ms, n_classes)
            losses_log.append(loss.item())

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, labels, topk=(1,))[0]
            losses.update(loss.data.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))

    cov = str(total) + str(" out of") + str(real_total)

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
                loss_fn):
    """ Train for one epoch """

    # Meters ===
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()
    epoch_train_loss = []

    for i, (input, target, hpred) in enumerate(train_loader):
        if iters < warmup_iters:
            lr = lrate * float(iters) / warmup_iters
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

        # One classifier loss ===
        # log_output = torch.log(output + 1e-7)
        log_output = F.log_softmax(output, dim=1)
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


def train(model, train_dataset, validation_dataset, config, seed=""):
    """
    General to all datasets.
    """

    # Data ===
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)

    # Model ===
    model = model.to(device)
    # Optimizer and loss ==
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), config["lr"],
                                 weight_decay=config["weight_decay"])
    loss_fn = nn.NLLLoss()
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
                                        loss_fn)

        metrics = evaluate(model, valid_loader, loss_fn)

        validation_loss = metrics["validation_loss"]

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            print("Saving the model with system accuracy {}".format(
                metrics['system_accuracy']), flush=True)
            save_path = os.path.join(config["ckp_dir"],
                                     config["experiment_name"] + '_' + str(config["n_experts"]) +
                                     '_experts' + '_seed_' + str(seed))
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


def one_classifier(config):
    config["ckp_dir"] = "./" + config["loss_type"] + "_classifier"
    os.makedirs(config["ckp_dir"], exist_ok=True)

    expert_fns = []

    seeds = ['', 948,  625]
    # seeds = [948, 625]
    for seed in seeds:
        print("run for seed {}".format(seed))
        if seed != '':
            set_seed(seed)

        print("One classifier | Seed {}".format(seed))

        # Model ===
        # model = model = ResNet34_oneclf(int(config["n_classes"]))
        model = model = ResNet34_defer(int(config["n_classes"]))

        # Data ===
        trainD, valD, _ = ham10000_expert.read(data_aug=True)
        # Train ===
        train(model, trainD, valD, config, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="scaling parameter for the loss function, default=1.0.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10,
                        help="number of patience steps for early stopping the training.")
    parser.add_argument("--expert_type", type=str, default="predict_biasedK",
                        help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.")
    parser.add_argument("--n_classes", type=int, default=7,
                        help="K for K class classification.")
    parser.add_argument("--k", type=int, default=0)
    # Dani experiments =====
    parser.add_argument("--n_experts", type=int, default=0)
    # Dani experiments =====
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
    print(config)

    one_classifier(config)
