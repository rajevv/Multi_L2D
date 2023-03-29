# To include lib
import sys

sys.path.insert(0, "../")

import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from lib.reliability_diagram import compute_calibration

# === Softmax ===
n_classes = 10
n_experts = 10
seed = 948
path = "softmax_increase_oracle/"
ECE = []
for n in range(n_classes):
    model_name = "_" + "k_" + str(n) + "seed_" + str(seed)  #
    with open(path + "confidence_multiple_experts" + model_name + ".txt", "r") as f:
        conf = json.loads(json.load(f))
    with open(
        path + "expert_predictions_multiple_experts" + model_name + ".txt", "r"
    ) as f:
        exp_pred = json.loads(json.load(f))
    with open(path + "true_label_multiple_experts" + model_name + ".txt", "r") as f:
        true_label = json.loads(json.load(f))
    # true.append(true_label['test'])
    # exps.append(exp_pred['test'])
    c = torch.tensor(conf["test"])
    # print(type(true_label['test']), type(exp_pred['test']), type(c))
    print(np.array(true_label["test"]).shape, np.array(exp_pred["test"]).shape, c.shape)

    c = c.softmax(dim=1)

    temp = 0
    for i in range(n_experts):
        temp += c[:, (n_classes + n_experts) - (i + 1)]
    prob = c / (1.0 - temp).unsqueeze(-1)

    true_label = torch.tensor(true_label["test"])
    exp_prediction = torch.tensor(
        exp_pred["test"]
    )  # exp_prediction shape [n, test_data_size]

    # ECEs ===

    # enumerate over all the experts
    eces = []
    for j in range(n_experts):
        e_j = exp_prediction[j, :]
        t = true_label
        c_j = prob[:, n_classes + j]
        ids_where_gt_one = torch.where(c_j > 1.0)
        c_j[ids_where_gt_one] = 1.0
        acc_j = t.eq(e_j)
        log = compute_calibration(c_j, acc_j)
        eces.append(log["expected_calibration_error"].numpy())
    # print(eces)
    ECE.append(np.average(eces))

print("Softmax Increasing # Experts ECE: {}".format(ECE))


# === OvA ===
path = "ova_increase_oracle/"
# n_experts = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
ECE = []
for n in range(n_classes):
    model_name = (
        "_" + "k" + "_" + str(n) + "seed_" + str(seed)
    )  #'_experts' #'_' + 'new' +
    with open(path + "confidence_multiple_experts" + model_name + ".txt", "r") as f:
        conf = json.loads(json.load(f))
    with open(
        path + "expert_predictions_multiple_experts" + model_name + ".txt", "r"
    ) as f:
        exp_pred = json.loads(json.load(f))
    with open(path + "true_label_multiple_experts" + model_name + ".txt", "r") as f:
        true_label = json.loads(json.load(f))
    # true.append(true_label['test'])
    # exps.append(exp_pred['test'])
    c = torch.tensor(conf["test"])
    c = c.sigmoid()

    prob = c

    true_label = torch.tensor(true_label["test"])
    exp_prediction = torch.tensor(
        exp_pred["test"]
    )  # exp_prediction shape [n, test_data_size]

    # ECEs ===

    # enumerate over all the experts
    eces = []
    for j in range(n_experts):
        e_j = exp_prediction[j, :]
        t = true_label
        c_j = prob[:, n_classes + j]
        # ids_where_gt_one = torch.where(c_j > 1.0)
        # c_j[ids_where_gt_one] = 1.0
        acc_j = t.eq(e_j)
        log = compute_calibration(c_j, acc_j)
        eces.append(log["expected_calibration_error"].numpy())
    # print(eces)
    ECE.append(np.average(eces))

print("OvA Increasing # Experts ECE: {}".format(ECE))
