import json
import numpy as np
import torch.nn as nn
from reliability_diagram import *
# import seaborn as sns
# from matplotlib import rc
from scipy import stats

# global quantities
# seeds = [948,  625,  436 ,'']
seeds = [948,  625, '']
# experiment_experts = [1,2,3,4,5,6,7,8,9,10]
experiment_experts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
n_classes = 2


def read_json_file(path):
    with open(path, 'r') as f:
        out = json.loads(json.load(f))
    return out


def get_file_name(type_, prefix, experiment_name, num_experts, seed):
    return prefix + type_ + '_' + experiment_name + '_' + str(num_experts) + '_experts_seed_' + str(seed) + '.txt'


def Hemmer():
    path = './' + 'hemmer_increase_experts/'
    experiment_name = 'multiple_experts_hardcoded'
    ECE = []
    for seed in seeds:
        ece = []
        for i, n in enumerate(experiment_experts):
            confs = read_json_file(get_file_name(
                'confidence', path, experiment_name, n, seed))['test']
            true = read_json_file(get_file_name(
                'true_label', path, experiment_name, n, seed))['test']
            exps = read_json_file(get_file_name(
                'expert_predictions', path, experiment_name, n, seed))['test']

            c = torch.tensor(confs)

            prob = c
            true_label = torch.tensor(true)
            exp_prediction = torch.tensor(exps)

            # enumerate over all the experts
            eces = []
            for j in range(n):
                e_j = exp_prediction[j, :]
                t = true_label
                c_j = prob[:, j]
                acc_j = t.eq(e_j)
                log = compute_calibration(c_j, acc_j)
                eces.append(log['expected_calibration_error'].numpy())
            ece.append(np.average(eces))
        ECE.append(ece)

    return ECE

def Hemmer_trained():
    path = './' + 'hemmer_increase_experts_trained/'
    experiment_name = 'multiple_experts_hardcoded'
    ECE = []
    for seed in ['', 948,  625]:
        ece = []
        for i, n in enumerate(experiment_experts):
            confs = read_json_file(get_file_name(
                'confidence', path, experiment_name, n, seed))['test']
            true = read_json_file(get_file_name(
                'true_label', path, experiment_name, n, seed))['test']
            exps = read_json_file(get_file_name(
                'expert_predictions', path, experiment_name, n, seed))['test']

            c = torch.tensor(confs)

            prob = c
            true_label = torch.tensor(true)
            exp_prediction = torch.tensor(exps)

            # enumerate over all the experts
            eces = []
            for j in range(n):
                e_j = exp_prediction[j, :]
                t = true_label
                c_j = prob[:, j]
                acc_j = t.eq(e_j)
                log = compute_calibration(c_j, acc_j)
                eces.append(log['expected_calibration_error'].numpy())
            ece.append(np.average(eces))
        ECE.append(ece)

    return ECE


def Softmax():
    path = './' + 'softmax_increase_experts_select_hard_coded/'
    experiment_name = 'multiple_experts'

    ECE = []
    for seed in seeds:
        ece = []
        for i, n in enumerate(experiment_experts):
            confs = read_json_file(get_file_name(
                'confidence', path, experiment_name, n, seed))['test']
            true = read_json_file(get_file_name(
                'true_label', path, experiment_name, n, seed))['test']
            exps = read_json_file(get_file_name(
                'expert_predictions', path, experiment_name, n, seed))['test']

            c = torch.tensor(confs)
            c = c.softmax(dim=1)

            # normalization to get experts' confidences
            temp = 0
            for i in range(n):
                temp += c[:, n_classes + i]
            prob = c / (1.0 - temp).unsqueeze(-1)

            true_label = torch.tensor(true)
            exp_prediction = torch.tensor(exps)

            # enumerate over all the experts
            eces = []
            for j in range(n):
                e_j = exp_prediction[j, :]
                t = true_label
                c_j = prob[:, n_classes + j]
                ids_where_gt_one = torch.where(c_j > 1.0)
                c_j[ids_where_gt_one] = 1.0

                acc_j = t.eq(e_j)
                log = compute_calibration(c_j, acc_j)
                eces.append(log['expected_calibration_error'].numpy())
            ece.append(np.average(eces))
        ECE.append(ece)

    return ECE


def OvA():
    path = './' + 'ova_increase_experts_select_hard_coded/'
    experiment_name = 'multiple_experts'

    ECE = []
    for seed in seeds:
        ece = []
        for i, n in enumerate(experiment_experts):
            confs = read_json_file(get_file_name(
                'confidence', path, experiment_name, n, seed))['test']
            true = read_json_file(get_file_name(
                'true_label', path, experiment_name, n, seed))['test']
            exps = read_json_file(get_file_name(
                'expert_predictions', path, experiment_name, n, seed))['test']

            c = torch.tensor(confs)
            c = c.sigmoid()

            prob = c
            true_label = torch.tensor(true)
            exp_prediction = torch.tensor(exps)

            # enumerate over all the experts
            eces = []
            for j in range(n):
                e_j = exp_prediction[j, :]
                t = true_label
                c_j = prob[:, n_classes + j]
                acc_j = t.eq(e_j)
                log = compute_calibration(c_j, acc_j)
                eces.append(log['expected_calibration_error'].numpy())
            ece.append(np.average(eces))
        ECE.append(ece)

    return ECE


if __name__ == "__main__":

    # # Softmax
    # ECE_softmax = Softmax()
    # print("===Mean and Standard Error ECEs Softmax===")
    # #print("All \n {}".format(np.array(ECE_softmax)))
    # print("Mean {}".format(np.mean(np.array(ECE_softmax), axis=0)))
    # print("Standard Error {}".format(stats.sem(np.array(ECE_softmax), axis=0)))

    # OvA
    ECE_OvA = OvA()
    print("===Mean and Standard Error ECEs OvA===")
    #print("All \n {}".format(np.array(ECE_OvA)))
    print("Mean {}".format(np.mean(np.array(ECE_OvA), axis=0)))
    print("Standard Error {}".format(stats.sem(np.array(ECE_OvA), axis=0)))

    # # Hemmer
    # ECE_hemmer = Hemmer()
    # print("===Mean and Standard Error ECEs Hemmer===")
    # #print("All \n {}".format(np.array(ECE_hemmer)))
    # print("Mean {}".format(np.mean(np.array(ECE_hemmer), axis=0)))
    # print("Standard Error {}".format(stats.sem(np.array(ECE_hemmer), axis=0)))

    # Hemmer Trained
    ECE_hemmer = Hemmer_trained()
    print("===Mean and Standard Error ECEs Hemmer TRAINED===")
    #print("All \n {}".format(np.array(ECE_hemmer)))
    print("Mean {}".format(np.mean(np.array(ECE_hemmer), axis=0)))
    print("Standard Error {}".format(stats.sem(np.array(ECE_hemmer), axis=0)))
