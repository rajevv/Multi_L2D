from __future__ import division

from tqdm import tqdm
import argparse
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict

import hemmer_baseline_trained
import main_ham10000_oneclassifier
import main_increase_experts_hard_coded
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from losses.losses import *
from models.baseline import ResNet34, ResNet34_oneclf, Network
from models.experts import synth_expert_hard_coded
from models.resnet34 import ResNet34_defer
from scipy import stats
from torch.autograd import Variable
from utils import *

from data_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,  flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# === Experiment 1 === #
# TODO: Experts definition
# all_available_experts = ['MLPMixer', 'predict', 'predict_prob', 'predict_random']
# ham10000_label_dict = {'bkl':0, 'df':1, 'mel':2, 'nv':3, 'vasc':4, 'akiec':5, 'bcc':6}
# mal_dx = ["mel", "bcc", "akiec"]
# ben_dx = ["nv", "bkl", "df", "vasc"]
# Expert 1: Random
expert1 = synth_expert_hard_coded(
    p_in=0.10, p_out=1/7, k=["nv"], device=device)
# Expert 2: Malign low prob expert
expert2 = synth_expert_hard_coded(
    p_in=0.25, p_out=1/7, k=mal_dx, device=device)
# Expert 3: Benign low prob expert
expert3 = synth_expert_hard_coded(
    p_in=0.25, p_out=1/7, k=ben_dx, device=device)
# Expert 4: MLP Mixer
expert4 = synth_expert_hard_coded(p_in=0.5, p_out=1/7, k=["nv"], device=device)
# Expert 5: Vascular lession expert
expert5 = synth_expert_hard_coded(
    p_in=0.7, p_out=1/7, k=["vasc"], device=device)
# Expert 6: Melanoma Expert
expert6 = synth_expert_hard_coded(
    p_in=0.75, p_out=0.33, k=["mel"], device=device)
# Expert 7: Benign High prob expert
expert7 = synth_expert_hard_coded(
    p_in=0.75, p_out=0.25, k=ben_dx, device=device)
# Expert 8: Malign High prob expert
expert8 = synth_expert_hard_coded(
    p_in=0.75, p_out=0.5, k=mal_dx, device=device)
# Expert 9: Average dermatologist
expert9 = synth_expert_hard_coded(
    p_in=0.8, p_out=0.5, k=ben_dx + mal_dx, device=device)
# Expert 10: Experienced dermatologist
expert10 = synth_expert_hard_coded(
    p_in=0.8, p_out=0.6, k=ben_dx + mal_dx, device=device)


experts = [getattr(expert1, 'predict_random'),
           getattr(expert2, 'predict_prob_ham10000_2'),
           getattr(expert3, 'predict_prob_ham10000_2'),
           getattr(expert4, 'predict_prob_ham10000_2'),
           getattr(expert5, 'predict_prob_ham10000_2'),
           getattr(expert6, 'predict_prob_ham10000_2'),
           getattr(expert7, 'predict_prob_ham10000_2'),
           getattr(expert8, 'MLPMixer'),
           getattr(expert9, 'predict_prob_ham10000_2'),
           getattr(expert10, 'predict_prob_ham10000_2')]


def feed_result_dict_seed(result_dict_seed, result, i):
    experiment_expert_i = "experiment_expert_" + str(i)

    for key, val in result_dict_seed.items():
        if key == "coverage":  # coverage
            total_samples = int(
                result["test"]["coverage"].split()[-1].split("f")[-1])
            covered_samples = int(result["test"]["coverage"].split()[0])
            coverage = covered_samples / total_samples
            result_dict_seed[key].append(coverage)
        elif key == experiment_expert_i:
            for j in range(i):
                result_dict_seed[key]["expert_" +
                                      str(j)] = result["test"]["expert_"+str(j)]
        else:
            if "experiment" in key:
                continue  # already filled
            else:
                result_dict_seed[key].append(result["test"][key])
    return result_dict_seed


def feed_result_dict(result_dict, result_dict_seed):
    for key, val in result_dict.items():
        if "experiment" in key:
            for expert in result_dict[key].keys():
                result_dict[key][expert].append(result_dict_seed[key][expert])
        else:
            result_dict[key].append(result_dict_seed[key])
    return result_dict


def fill_result_dict_seed_experts_dict(result_dict_seed, experiment_experts):
    experiment_expert_i = "experiment_expert_" + str(experiment_experts)
    result_dict_seed[experiment_expert_i] = {
        "expert_" + str(i): None for i in range(experiment_experts)}
    return result_dict_seed


def print_results(result_dict):
    for key, val in result_dict.items():
        if "experiment" not in key:
            print("=== {} Mean and Standard Error===".format(key))
            print("Mean {}".format(np.mean(np.array(val), axis=0)))
            print("Standard Error {}".format(stats.sem(np.array(val), axis=0)))
        else:
            print("=== {} Experiment ===".format(key.split("_")[-1]))
            for expert, v in result_dict[key].items():
                print("{} Mean {}".format(expert, np.mean(np.array(v), axis=0)))
                print("{} Standard Error {}".format(
                    expert, stats.sem(np.array(v), axis=0)))
                print("==============")

    return


def main_validate_best_expert(testD, expert_fns, config):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_dl = torch.utils.data.DataLoader(
        testD, batch_size=config["batch_size"], shuffle=False, drop_last=True, **kwargs)
    expert_accs = {k: 0 for k in range(len(expert_fns))}
    for i, fn in enumerate(expert_fns):
        exp_acc = 0.0
        for j, data in enumerate(test_dl, 1):
            inp, lbl, Z = data
            inp, lbl, Z = inp.to(device), lbl, Z.to(device)
            # Cast to float to avoid errors
            Z = Z.float()

            exp_pred = fn(inp, lbl, Z)
            exp_pred = torch.tensor(exp_pred)
            exp_acc += torch.mean(lbl.eq(exp_pred).float())
        expert_accs[i] = exp_acc / j
    return expert_accs


def validate_best_expert(config):
    _, _, testD = ham10000_expert.read(data_aug=False)

    experiment_experts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # experiment_experts = [8, 9, 10]
    accuracy = []
    for seed in ['', 948,  625]:
        if seed != '':
            set_seed(seed)
        acc = []
        expert_fns = []
        for i, n in tqdm(enumerate(experiment_experts)):
            expert_fns = [experts[j] for j in range(n)]
            # expert_fns.append(experts[i])
            expert_accs = main_validate_best_expert(testD, expert_fns, config)
            temp = expert_accs.values()
            best = max(temp)
            # print(expert_accs)
            acc.append(best)
        # print(acc)
        accuracy.append(acc)

    print("===Mean and Standard Error===")
    print("Mean {}".format(np.mean(np.array(accuracy), axis=0)))
    print("Standard Error {}".format(stats.sem(np.array(accuracy), axis=0)))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def filter(dict_):
    d = {}
    for k, v in dict_.items():
        if torch.is_tensor(v):
            v = v.item()
        d[k] = v
    return d


def forward_surrogate(model, dataloader, expert_fns, config):
    confidence = []
    true = []
    expert_predictions = defaultdict(list)

    with torch.no_grad():
        for inp, lbl, Z in dataloader:
            inp = inp.to(device)
            lbl = lbl.to(device)
            Z = Z.to(device).float()
            conf = model(inp)
            for i, fn in enumerate(expert_fns):
                expert_pred1 = fn(inp, lbl, Z)
                expert_predictions[i].append(expert_pred1)
            confidence.append(conf.cpu())
            true.append(lbl.cpu())
            # break

    true = torch.stack(true, dim=0).view(-1)
    confidence = torch.stack(confidence, dim=0).view(-1,
                                                     config["n_classes"] + len(expert_fns))
    for k, v in expert_predictions.items():
        expert_predictions[k] = torch.stack(
            [torch.tensor(k) for k in v], dim=0).view(-1)

    return true, confidence, [v.numpy() for k, v in
                              expert_predictions.items()]


def main_validate_surrogate(model, testD, expert_fns, config, seed=''):
    def get(severity, dl):
        true, confidence, expert_predictions = forward_surrogate(
            model, dl, expert_fns, config)

        print("shapes: true labels {}, confidences {}, expert_predictions {}".format(
            true.shape, confidence.shape, np.array(expert_predictions).shape))

        criterion = Criterion()
        loss_fn = getattr(criterion, config["loss_type"])
        print("Evaluate...")
        result_ = main_increase_experts_hard_coded.evaluate(
            model, expert_fns, loss_fn, config["n_classes"]+len(expert_fns), dl, config)
        # print(result_)
        result[severity] = filter(result_)
        true_label[severity] = true.numpy()
        classifier_confidence[severity] = confidence.numpy()
        expert_preds[severity] = expert_predictions

    model_path = os.path.join(config["ckp_dir"],
                              config["experiment_name"] + '_' + str(len(expert_fns)) + '_experts' + '_seed_' + str(seed))
    model.load_state_dict(torch.load(model_path + '.pt', map_location=device))
    model = model.to(device)

    model_name = config["experiment_name"] + '_' + \
        str(len(expert_fns)) + '_experts' + '_seed_' + str(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_dl = torch.utils.data.DataLoader(
        testD, batch_size=config["batch_size"], shuffle=False, drop_last=True, **kwargs)

    result = {}
    classifier_confidence = {}
    true_label = {}
    expert_preds = {}

    get('test', test_dl)

    with open(config["ckp_dir"] + '/true_label_' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(true_label, cls=NumpyEncoder), f)

    with open(config["ckp_dir"] + '/confidence_' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(classifier_confidence, cls=NumpyEncoder), f)

    with open(config["ckp_dir"] + '/expert_predictions_' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)

    with open(config["ckp_dir"] + '/validation_results_' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(result, cls=NumpyEncoder), f)

    return result


def validate_surrogate(config):
    config["ckp_dir"] = "./" + config["loss_type"] + \
        "_increase_experts_select_hard_coded"
    experiment_experts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # experiment_experts = [8, 9]
    # experiment_experts = [2]

    # Result dict ===
    result_dict = {"system_accuracy": [],
                   "expert_accuracy": [],
                   "coverage": []}
    result_dict = {**result_dict, **{"experiment_expert_" +
                                     str(i): {"expert_"+str(j): [] for j in range(i)} for i in experiment_experts}}

    accuracy = []
    for seed in ['', 948,  625]:
    # for seed in ['']:

        if seed != '':
            set_seed(seed)

        result_dict_seed = {k: [] for k in result_dict.keys()}
        acc = []
        for i, n in tqdm(enumerate(experiment_experts)):

            result_dict_seed = fill_result_dict_seed_experts_dict(
                result_dict_seed, n)
            print("n is {}".format(n))
            num_experts = n

            expert_fns = [experts[j] for j in range(n)]

            model = model = ResNet34_defer(
                int(config["n_classes"])+num_experts)

            _, _, testD = ham10000_expert.read(data_aug=False)

            result = main_validate_surrogate(
                model, testD, expert_fns, config, seed=seed)

            result_dict_seed = feed_result_dict_seed(
                result_dict_seed, result, n)
            # acc.append(result['test']['system_accuracy'])

        # for key, val in result_dict.items():
        #         result_dict[key].append(result_dict_seed[key])
        result_dict = feed_result_dict(result_dict, result_dict_seed)
        # accuracy.append(acc)

    print("==={}===".format(config["loss_type"]))
    print_results(result_dict)
    with open(config["ckp_dir"] + '/results_' + config["ckp_dir"].split("/")[-1] + '.txt', 'w') as f:
        json.dump(json.dumps(result_dict, cls=NumpyEncoder), f)

    # print("=== Sys. Acc. Mean and Standard Error===")
    # print("Mean {}".format(np.mean(np.array(accuracy), axis=0)))
    # print("Standard Error {}".format(stats.sem(np.array(accuracy), axis=0)))


def forward_hemmer(model, dataloader, expert_fns):
    confidence = []
    true = []
    expert_predictions = defaultdict(list)

    with torch.no_grad():
        for inp, lbl, Z in dataloader:
            inp = inp.to(device)
            lbl = lbl.to(device)
            Z = Z.to(device).float()
            feature_extractor = model[0]
            allocator = model[1]  # experts confs
            bath_features = feature_extractor(inp)
            conf = allocator(bath_features)
            for i, fn in enumerate(expert_fns):
                expert_pred1 = fn(inp, lbl, Z)
                expert_predictions[i].append(expert_pred1)
            confidence.append(conf.cpu())
            true.append(lbl.cpu())
            # break

    true = torch.stack(true, dim=0).view(-1)
    confidence = torch.stack(confidence, dim=0).view(-1, len(expert_fns) + 1)
    for k, v in expert_predictions.items():
        expert_predictions[k] = torch.stack(
            [torch.tensor(k) for k in v], dim=0).view(-1)

    return true, confidence, [v.numpy() for k, v in
                              expert_predictions.items()]


def main_validate_hemmer(model, testD, expert_fns, config, seed=''):
    def get(severity, dl):
        true, confidence, expert_predictions = forward_hemmer(
            model, dl, expert_fns)

        print("shapes: true labels {}, confidences {}, expert_predictions {}".format(
            true.shape, confidence.shape, np.array(expert_predictions).shape))

        criterion = Criterion()
        loss_fn = nn.NLLLoss()
        print("Evaluate...")
        result_ = hemmer_baseline_trained.evaluate(
            model, expert_fns, loss_fn, config["n_classes"]+len(expert_fns), dl, config)
        # print(result_)
        result[severity] = filter(result_)
        true_label[severity] = true.numpy()
        allocator_confidence[severity] = confidence.numpy()
        expert_preds[severity] = expert_predictions

    model_path = os.path.join(config["ckp_dir"],
                              config["experiment_name"] + '_' + str(len(expert_fns)) + '_experts' + '_seed_' + str(seed))
    load_dict = torch.load(model_path + '.pt', map_location=device)
    feature_extractor, allocator, classifier = model[0], model[1], model[2]

    # print(type(load_dict['allocator_state_dict']),
    #       type(load_dict['classifier_state_dict']()))
    allocator.load_state_dict(load_dict['allocator_state_dict'])
    # import copy  # Careful with this. Actually I saved the method instead of the state_dict() for classifier
    # classifier.load_state_dict(copy.deepcopy(
    #     load_dict['classifier_state_dict']()))
    # feature_extractor.load_state_dict(copy.deepcopy(
    #     load_dict['feature_extractor_state_dict']()))

    classifier.load_state_dict(load_dict['classifier_state_dict'])
    feature_extractor.load_state_dict(
        load_dict['feature_extractor_state_dict'])
    feature_extractor, allocator, classifier = feature_extractor.to(
        device), allocator.to(device), classifier.to(device)
    model = (feature_extractor, allocator, classifier)

    model_name = config["experiment_name"] + '_' + \
        str(len(expert_fns)) + '_experts' + '_seed_' + str(seed)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_dl = torch.utils.data.DataLoader(
        testD, batch_size=config["batch_size"], shuffle=False, drop_last=True, **kwargs)

    result = {}
    allocator_confidence = {}
    true_label = {}
    expert_preds = {}

    get('test', test_dl)

    with open(config["ckp_dir"] + '/true_label_' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(true_label, cls=NumpyEncoder), f)

    with open(config["ckp_dir"] + '/confidence_' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(allocator_confidence, cls=NumpyEncoder), f)

    with open(config["ckp_dir"] + '/expert_predictions_' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)

    with open(config["ckp_dir"] + '/validation_results_' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(result, cls=NumpyEncoder), f)

    return result


# def validate_hemmer(config):
#     config["loss_type"] = "hemmer"
#     config["ckp_dir"] = "./" + config["loss_type"] + "_increase_experts"
#     config["experiment_name"] = "multiple_experts_hardcoded"
#     # experiment_experts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     experiment_experts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#     # experiment_experts = [8, 9]

#     # Result dict ===
#     result_dict = {"system_accuracy": [],
#                    "expert_accuracy": [],
#                    "coverage": []}
#     result_dict = {**result_dict, **{"experiment_expert_" +
#                                      str(i): {"expert_"+str(j): [] for j in range(i)} for i in experiment_experts}}

#     accuracy = []
#     # for seed in ['', 948,  625, 791, 436]:
#     for seed in ['', 948,  625]:
#         # for seed in ['']:

#         if seed != '':
#             set_seed(seed)
#         expert_fns = []
#         result_dict_seed = {k: [] for k in result_dict.keys()}
#         # acc = []
#         for i, n in tqdm(enumerate(experiment_experts)):

#             result_dict_seed = fill_result_dict_seed_experts_dict(
#                 result_dict_seed, n)

#             print("n is {}".format(n))
#             num_experts = n
#             # getattr(selected_expert, selected_expert_fn)
#             expert_fns = [experts[j] for j in range(n)]

#             # === Galaxy-Zoo models ===
#             # print(len(expert_fns))
#             feature_extractor = Resnet()
#             classifier = Network(output_size=int(config["n_classes"]))
#             allocator = Network(output_size=len(expert_fns)+1)
#             model = (feature_extractor, allocator, classifier)
#             testD = GalaxyZooDataset(split='val')
#             result = main_validate_hemmer(
#                 model, testD, expert_fns, config, seed=seed)

#             result_dict_seed = feed_result_dict_seed(
#                 result_dict_seed, result, n)
#             # acc.append(result['test']['system_accuracy'])

#         # for key, val in result_dict.items():
#         #         result_dict[key].append(result_dict_seed[key])
#         result_dict = feed_result_dict(result_dict, result_dict_seed)
#         # accuracy.append(acc)

#     print("===HEMMER===")
#     print_results(result_dict)
#     with open(config["ckp_dir"] + '/results_' + config["ckp_dir"].split("/")[-1] + '.txt', 'w') as f:
#         json.dump(json.dumps(result_dict, cls=NumpyEncoder), f)

#     # print("=== Sys. Acc. Mean and Standard Error===")
#     # print("Mean {}".format(np.mean(np.array(accuracy), axis=0)))
#     # print("Standard Error {}".format(stats.sem(np.array(accuracy), axis=0)))


def validate_hemmer_trained(config):
    config["loss_type"] = "hemmer"
    config["ckp_dir"] = "./" + config["loss_type"] + \
        "_increase_experts_trained"
    config["experiment_name"] = "multiple_experts_hardcoded"
    experiment_experts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # experiment_experts = [8]

    # Result dict ===
    result_dict = {"system_accuracy": [],
                   "expert_accuracy": [],
                   "coverage": []}
    result_dict = {**result_dict, **{"experiment_expert_" +
                                     str(i): {"expert_"+str(j): [] for j in range(i)} for i in experiment_experts}}

    accuracy = []
    # for seed in ['', 948,  625, 791, 436]:
    for seed in ['', 948,  625]:
        # for seed in ['']:

        if seed != '':
            set_seed(seed)
        expert_fns = []
        result_dict_seed = {k: [] for k in result_dict.keys()}

        # acc = []
        for i, n in tqdm(enumerate(experiment_experts)):

            result_dict_seed = fill_result_dict_seed_experts_dict(
                result_dict_seed, n)

            print("n is {}".format(n))
            num_experts = n
            # getattr(selected_expert, selected_expert_fn)
            expert_fns = [experts[j] for j in range(n)]

            # === Galaxy-Zoo models ===
            # print(len(expert_fns))
            feature_extractor = ResNet34(train_weights=True)
            classifier = Network(output_size=int(config["n_classes"]))
            allocator = Network(output_size=len(expert_fns)+1)
            model = (feature_extractor, allocator, classifier)
            _, _, testD = ham10000_expert.read(data_aug=False)
            result = main_validate_hemmer(
                model, testD, expert_fns, config, seed=seed)

            result_dict_seed = feed_result_dict_seed(
                result_dict_seed, result, n)
            # acc.append(result['test']['system_accuracy'])

        # for key, val in result_dict.items():
        #         result_dict[key].append(result_dict_seed[key])
        result_dict = feed_result_dict(result_dict, result_dict_seed)
        # accuracy.append(acc)

    print("===HEMMER TRAINED===")
    print_results(result_dict)
    with open(config["ckp_dir"] + '/results_' + config["ckp_dir"].split("/")[-1] + '.txt', 'w') as f:
        json.dump(json.dumps(result_dict, cls=NumpyEncoder), f)

    # print("=== Sys. Acc. Mean and Standard Error===")
    # print("Mean {}".format(np.mean(np.array(accuracy), axis=0)))
    # print("Standard Error {}".format(stats.sem(np.array(accuracy), axis=0)))


def main_validate_classifier(model, testD, expert_fns, config, seed=''):
    model_name = config["experiment_name"] + '_' + \
        str(len(expert_fns)) + '_experts' + '_seed_' + str(seed)
    model_path = config['ckp_dir'] + '/' + model_name + '.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_dl = torch.utils.data.DataLoader(
        testD, batch_size=config["batch_size"], shuffle=False, drop_last=True, **kwargs)

    result = {}
    result_ = main_ham10000_oneclassifier.evaluate(
        model, test_dl, nn.NLLLoss())
    result['test'] = filter(result_)
    return result


def validate_classifier(config):
    config["ckp_dir"] = "./" + config["loss_type"] + "_classifier"
    expert_fns = []
    accuracy = []
    # , 948,  625,  436,  791]: #, 1750,  812, 1331, 1617,  650, 1816]:
    # for seed in tqdm(['', 948, 625, 436, 791]):
    for seed in tqdm(['', 948, 625]):
    # for seed in tqdm([948]):
    # for seed in tqdm(['']):
        print("run for seed {}".format(seed))
        if seed != '':
            set_seed(seed)
        model = ResNet34_defer(int(config["n_classes"]))
        _, _, testD = ham10000_expert.read(data_aug=False)
        result = main_validate_classifier(
            model, testD, expert_fns, config, seed=seed)
        accuracy.append(result['test']['system_accuracy'])

    print("===Mean and Standard Error===")
    print("Mean {}".format(np.mean(np.array(accuracy))))
    print("Standard Error {}".format(stats.sem(np.array(accuracy))))


if __name__ == "__main__":
    # config surrogate loss methods

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="scaling parameter for the loss function, default=1.0.")
    parser.add_argument("--expert_type", type=str, default="predict_prob",
                        help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.")
    parser.add_argument("--n_classes", type=int, default=7,
                        help="K for K class classification.")
    parser.add_argument("--loss_type", type=str, default="ova",
                        help="surrogate loss type for learning to defer.")
    parser.add_argument("--ckp_dir", type=str, default="./Models",
                        help="directory name to save the checkpoints.")
    parser.add_argument("--experiment_name", type=str, default="multiple_experts",
                        help="specify the experiment name. Checkpoints will be saved with this name.")

    config = parser.parse_args().__dict__

    # config["loss_type"] = "softmax"

    # print("validate softmax surrogate loss method...")
    # validate_surrogate(config)

    config["loss_type"] = "ova"

    print("validate ova surrogate loss method...")
    validate_surrogate(config)

    # config["loss_type"] = "hemmer"

    # print("validate Hemmer TRAINED MoE baseline method...")
    # validate_hemmer_trained(config)

    # print("validate one classifier baseline...")
    # config["loss_type"] = "softmax"
    # config["experiment_name"] = "classifier"
    # validate_classifier(config)

    # print("validate best expert baseline...")
    # validate_best_expert(config)
