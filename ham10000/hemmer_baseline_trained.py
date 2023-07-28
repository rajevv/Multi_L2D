# To include lib
import sys

sys.path.insert(0, "../")

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
from ham10000dataset import ham10000_expert
from models.baseline import Network, ResNet34
from models.experts import synth_expert_hard_coded
from torch.autograd import Variable

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


def Hemmer_utils(input, target, hpred, allocation, clf_output, expert_fns, config):
    batch_size = clf_output.size()[0]  # batch_size
    exps_pred = []
    # We only support \alpha=1
    expert_predictions = []
    for idx, fn in enumerate(expert_fns):
        # We assume each expert function has access to the extra metadata, even if they don't use it.
        m = fn(input, target, hpred)
        expert_predictions.append(m)
        exp_pred = torch.zeros((batch_size, config["n_classes"]))
        for j in range(0, batch_size):
            exp_pred[j][int(m[j])] = 1
        exps_pred.append(exp_pred)

    exps_pred.append(clf_output.cpu())

    exps_pred = torch.stack(exps_pred).transpose(0, 1).to(device)
    allocation = allocation.unsqueeze(-1)
    p_team = torch.sum(allocation * exps_pred, dim=1)
    log_p_team = torch.log(p_team + 1e-7)
    return log_p_team, expert_predictions


def evaluate(model, expert_fns, loss_fn, n_classes, data_loader, config):
    """
    Computes metrics for deferal
    -----
    Arguments:
    net: model
    expert_fn: expert model
    n_classes: number of classes
    loader: data loader
    """
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
    losses = []
    feature_extractor, allocator, classifier = model[0], model[1], model[2]
    feature_extractor.eval()
    allocator.eval()
    classifier.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels, Z = data
            images, labels, Z = (
                images.to(device),
                labels.to(device),
                Z.float().to(device),
            )

            batch_features = feature_extractor(images)
            allocation = allocator(batch_features)
            # c_i i in {1, K} (from the paper)
            clf_pred = classifier(batch_features)

            # allocation = allocator(images)
            # clf_pred = classifier(images)

            _, clf_predictions = torch.max(clf_pred.data, 1)
            _, who_predicts = torch.max(allocation.data, 1)
            batch_size = clf_pred.size()[0]  # batch_size

            log_p_team, expert_predictions = Hemmer_utils(
                images, labels, Z, allocation, clf_pred, expert_fns, config
            )

            # print("len expert predictions {}".format(len(expert_predictions)))

            loss = loss_fn(log_p_team, labels)
            losses.append(loss.item())

            for i in range(0, batch_size):
                # r is true when non-ai expert has the maximum weight
                r = who_predicts[i].item() != len(expert_fns)
                clf_prediction = clf_predictions[i]
                alone_correct += (clf_prediction == labels[i]).item()
                if r == 0:  # the max is on the classifier
                    total += 1
                    correct += (clf_prediction == labels[i]).item()
                    correct_sys += (clf_prediction == labels[i]).item()
                if r == 1:

                    deferred_exp = (who_predicts[i]).item()

                    # print(len(expert_predictions), deferred_exp, batch_size)
                    exp_prediction = expert_predictions[deferred_exp][i]
                    #
                    # Deferral accuracy: No matter expert ===
                    exp += exp_prediction == labels[i].item()
                    exp_total += 1
                    # Individual Expert Accuracy ===
                    expert_correct_dic[deferred_exp] += (
                        exp_prediction == labels[i].item()
                    )
                    expert_total_dic[deferred_exp] += 1
                    #
                    correct_sys += exp_prediction == labels[i].item()
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)

    #  === Individual Expert Accuracies === #
    expert_accuracies = {
        "expert_{}".format(str(k)): 100
        * expert_correct_dic[k]
        / (expert_total_dic[k] + 0.0002)
        for k in range(len(expert_fns))
    }
    # Add expert accuracies dict
    to_print = {
        "coverage": cov,
        "system_accuracy": 100 * correct_sys / real_total,
        "expert_accuracy": 100 * exp / (exp_total + 0.0002),
        "classifier_accuracy": 100 * correct / (total + 0.0001),
        "alone_classifier": 100 * alone_correct / real_total,
        "validation_loss": np.average(losses),
        "n_experts": len(expert_fns),
        **expert_accuracies,
    }
    print(to_print, flush=True)
    return to_print


def train_epoch(
    iters,
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
    config,
):
    """Train for one epoch"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    feature_extractor, allocator, classifier = model[0], model[1], model[2]
    feature_extractor.train()
    allocator.train()
    classifier.train()

    end = time.time()

    epoch_train_loss = []

    for i, (input, target, Z) in enumerate(train_loader):
        if iters < warmup_iters:
            lr = lrate * float(iters) / warmup_iters
            print(iters, lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        target = target.to(device)
        input = input.to(device)
        Z = Z.float().to(device)

        # compute output
        # allocation w_j j in {1, ..., num_experts + 1} (from the paper)
        batch_features = feature_extractor(input)
        allocation = allocator(batch_features)
        # c_i i in {1, K} (from the paper)
        clf_output = classifier(batch_features)

        # get expert  predictions and costs
        batch_size = clf_output.size()[0]  # batch_size
        exps_pred = []
        # We only support \alpha=1

        log_p_team, _ = Hemmer_utils(
            input, target, Z, allocation, clf_output, expert_fns, config
        )

        loss = loss_fn(log_p_team, target)

        epoch_train_loss.append(loss.item())

        # measure accuracy and record loss
        prec1 = accuracy(log_p_team, target, topk=(1,))[0]
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
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                ),
                flush=True,
            )

    return iters, np.average(epoch_train_loss)


def train(model, train_dataset, validation_dataset, expert_fns, config, seed=""):

    n_classes = config["n_classes"] + len(expert_fns)
    kwargs = {"num_workers": 0, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        **kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        **kwargs
    )

    model = (model[0].to(device), model[1].to(device), model[2].to(device))
    # cudnn.benchmark = True
    optimizer = torch.optim.Adam(
        list(model[0].parameters())
        + list(model[1].parameters())
        + list(model[2].parameters()),
        config["lr"],
        weight_decay=config["weight_decay"],
    )

    criterion = Criterion()

    loss_fn = nn.NLLLoss()  # getattr(criterion, config["loss_type"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader) * config["epochs"]
    )
    best_validation_loss = np.inf
    patience = 0
    iters = 0
    warmup_iters = config["warmup_epochs"] * len(train_loader)
    lrate = config["lr"]

    for epoch in range(0, config["epochs"]):
        iters, train_loss = train_epoch(
            iters,
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
            config,
        )
        metrics = evaluate(model, expert_fns, loss_fn, n_classes, valid_loader, config)

        validation_loss = metrics["validation_loss"]

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            print(
                "Saving the model with classifier accuracy {}".format(
                    metrics["classifier_accuracy"]
                ),
                flush=True,
            )
            save_path = os.path.join(
                config["ckp_dir"],
                config["experiment_name"]
                + "_"
                + str(len(expert_fns))
                + "_experts"
                + "_seed_"
                + str(seed),
            )
            torch.save(
                {
                    "feature_extractor_state_dict": model[0].state_dict(),
                    "allocator_state_dict": model[1].state_dict(),
                    "classifier_state_dict": model[2].state_dict(),
                },
                save_path + ".pt",
            )
            # Additionally save the whole config dict
            with open(save_path + ".json", "w") as f:
                json.dump(config, f)
            patience = 0
        else:
            patience += 1

        if patience >= config["patience"]:
            print("Early Exiting Training.", flush=True)
            break


# === Experiment 1 === #
# TODO: Experts definition
# all_available_experts = ['MLPMixer', 'predict', 'predict_prob', 'predict_random']
# ham10000_label_dict = {'bkl':0, 'df':1, 'mel':2, 'nv':3, 'vasc':4, 'akiec':5, 'bcc':6}
# mal_dx = ["mel", "bcc", "akiec"]
# ben_dx = ["nv", "bkl", "df", "vasc"]
# Expert 1: Random
expert1 = synth_expert_hard_coded(p_in=0.10, p_out=1 / 7, k=["nv"], device=device)
# Expert 2: Malign low prob expert
expert2 = synth_expert_hard_coded(p_in=0.25, p_out=1 / 7, k=mal_dx, device=device)
# Expert 3: Benign low prob expert
expert3 = synth_expert_hard_coded(p_in=0.25, p_out=1 / 7, k=ben_dx, device=device)
# Expert 4: MLP Mixer
expert4 = synth_expert_hard_coded(p_in=0.5, p_out=1 / 7, k=["nv"], device=device)
# Expert 5: Vascular lession expert
expert5 = synth_expert_hard_coded(p_in=0.7, p_out=1 / 7, k=["vasc"], device=device)
# Expert 6: Melanoma Expert
expert6 = synth_expert_hard_coded(p_in=0.75, p_out=0.33, k=["mel"], device=device)
# Expert 7: Benign High prob expert
expert7 = synth_expert_hard_coded(p_in=0.75, p_out=0.25, k=ben_dx, device=device)
# Expert 8: Malign High prob expert
expert8 = synth_expert_hard_coded(p_in=0.75, p_out=0.5, k=mal_dx, device=device)
# Expert 9: Average dermatologist
expert9 = synth_expert_hard_coded(p_in=0.8, p_out=0.5, k=ben_dx + mal_dx, device=device)
# Expert 10: Experienced dermatologist
expert10 = synth_expert_hard_coded(
    p_in=0.8, p_out=0.6, k=ben_dx + mal_dx, device=device
)


experts = [
    getattr(expert1, "predict_random"),
    getattr(expert2, "predict_prob_ham10000_2"),
    getattr(expert3, "predict_prob_ham10000_2"),
    getattr(expert4, "predict_prob_ham10000_2"),
    getattr(expert5, "predict_prob_ham10000_2"),
    getattr(expert6, "predict_prob_ham10000_2"),
    getattr(expert7, "predict_prob_ham10000_2"),
    getattr(expert8, "MLPMixer"),
    getattr(expert9, "predict_prob_ham10000_2"),
    getattr(expert10, "predict_prob_ham10000_2"),
]


def increase_experts(config):
    config["ckp_dir"] = "./" + config["loss_type"] + "_increase_experts_trained"
    os.makedirs(config["ckp_dir"], exist_ok=True)

    experiment_experts = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for seed in [948, 625]:

        print("run for seed {}".format(seed))
        if seed != "":
            set_seed(seed)
        log = {"selected_experts": [], "selected_expert_fns": []}
        for i, n in enumerate(experiment_experts):
            print("Number of Experts: n is {}".format(n))
            num_experts = n

            expert_fns = [experts[j] for j in range(n)]

            # === HAM10000 models === #
            # print(len(expert_fns))
            feature_extractor = ResNet34(train_weights=True)

            classifier = Network(output_size=int(config["n_classes"]))
            allocator = Network(output_size=len(expert_fns) + 1)
            model = (feature_extractor, allocator, classifier)
            trainD, valD, _ = ham10000_expert.read(data_aug=True)

            train(model, trainD, valD, expert_fns, config, seed=seed)

        # pth = os.path.join(config['ckp_dir'], config['experiment_name'] + '_log_' + '_seed_' + str(seed))
        # with open(pth + '.json', 'w') as f:
        #     json.dump(log, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="scaling parameter for the loss function, default=1.0.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="number of patience steps for early stopping the training.",
    )
    parser.add_argument(
        "--expert_type",
        type=str,
        default="MLPMixer",
        help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.",
    )
    parser.add_argument(
        "--n_classes", type=int, default=7, help="K for K class classification."
    )
    parser.add_argument("--k", type=int, default=5)
    # Dani experiments =====
    parser.add_argument("--n_experts", type=int, default=2)
    # Dani experiments =====
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="hemmer",
        help="surrogate loss type for learning to defer.",
    )

    parser.add_argument(
        "--ckp_dir",
        type=str,
        default="./Models",
        help="directory name to save the checkpoints.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="multiple_experts_hardcoded",
        help="specify the experiment name. Checkpoints will be saved with this name.",
    )

    config = parser.parse_args().__dict__

    # print(config)
    increase_experts(config)
