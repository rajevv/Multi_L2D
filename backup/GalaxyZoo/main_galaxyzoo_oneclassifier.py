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
from torch.backends import cudnn

# Galaxy zoo specific
from galaxyzoodataset import GalaxyZooDataset
from models.resnet50 import ResNet50_defer

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

    losses = []
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
            loss = loss_fn(log_output, labels)
            losses.append(loss.item())

            for i in range(0, batch_size):
                prediction = predicted[i]
                alone_correct += (prediction == labels[i]).item()

                correct += (predicted[i] == labels[i]).item()
                correct_sys += (predicted[i] == labels[i]).item()
                total += 1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)

    # Add expert accuracies dict
    to_print = {"coverage": cov, "system_accuracy": 100 * correct_sys / real_total,
                "classifier_accuracy": 100 * correct / (total + 0.0001),
                "alone_classifier": 100 * alone_correct / real_total,
                "validation_loss": np.average(losses)}
    print(to_print, flush=True)
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
        output = F.softmax(output, dim=1)

        # One classifier loss ===
        log_output = torch.log(output + 1e-7)
        loss = loss_fn(log_output, target)
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
            print("Saving the model with classifier accuracy {}".format(
                metrics['classifier_accuracy']), flush=True)

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
    config["ckp_dir"] = "./" + config["loss_type"] + "_testclassifier"
    config["n_classes"] = 2  # Galaxy-Zoo
    os.makedirs(config["ckp_dir"], exist_ok=True)

    experts = np.arange(1, 11)
    experts = [4]

    seeds = [948, 625, 436]
    seeds = [948]
    for seed in seeds:
        set_seed(seed)
        for n in experts:
            print("One classifier | Seed {} | Experts {}".format(seed, n))

            config["n_experts"] = n
            # Model ===
            model = model = ResNet50_defer(int(config["n_classes"]))
            # Data ===
            trainD = GalaxyZooDataset()
            valD = GalaxyZooDataset(split='val')
            # Train ===
            train(model, trainD, valD, config, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="scaling parameter for the loss function, default=1.0.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20,
                        help="number of patience steps for early stopping the training.")
    parser.add_argument("--expert_type", type=str, default="predict_biasedK",
                        help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.")
    parser.add_argument("--n_classes", type=int, default=2,
                        help="K for K class classification.")
    parser.add_argument("--k", type=int, default=5)
    # Dani experiments =====
    parser.add_argument("--n_experts", type=int, default=2)
    # Dani experiments =====
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--loss_type", type=str, default="softmax",
                        help="surrogate loss type for learning to defer.")
    parser.add_argument("--ckp_dir", type=str, default="./Models",
                        help="directory name to save the checkpoints.")
    parser.add_argument("--experiment_name", type=str, default="multiple_experts",
                        help="specify the experiment name. Checkpoints will be saved with this name.")

    config = parser.parse_args().__dict__
    print(config)

    one_classifier(config)
