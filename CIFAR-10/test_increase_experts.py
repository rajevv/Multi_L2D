import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from reliability_diagram import compute_calibration

# === Softmax ===
n_classes = 10
# confs = []
# exps = []
# true = []
path = "softmax_increase_experts/"
n_experts = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
ECE = []
for n in n_experts:
    model_name = '_' + 'new' + '_' + str(n) + '_experts' # 
    with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'r') as f:
        conf = json.loads(json.load(f))
    with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'r') as f:
        exp_pred = json.loads(json.load(f))
    with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'r') as f:
        true_label = json.loads(json.load(f))
    # true.append(true_label['test'])
    # exps.append(exp_pred['test'])
    c = torch.tensor(conf['test'])
    #print(type(true_label['test']), type(exp_pred['test']), type(c))
    print(np.array(true_label['test']).shape, np.array(exp_pred['test']).shape, c.shape)
    # DANI Correction ===
    c = c.softmax(dim=1)
    #c = c.sigmoid()
    # DANI Correction ===

    temp = 0
    for i in range(n):
        temp += c[:, (n_classes + n) - (i + 1)]
    prob = c / (1.0 - temp).unsqueeze(-1)
    # print(prob.shape)
    # confs.append(prob)

    true_label = torch.tensor(true_label['test'])
    exp_prediction = torch.tensor(exp_pred['test']) # exp_prediction shape [n, test_data_size]

    # ECEs ===

    # enumerate over all the experts
    eces = []
    for j in range(n):
        e_j = exp_prediction[j,:]
        t = true_label
        c_j = prob[:, n_classes + j]
        ids_where_gt_one = torch.where(c_j > 1.0)
        c_j[ids_where_gt_one] = 1.0
        acc_j = t.eq(e_j)
        log = compute_calibration(c_j, acc_j)
        eces.append(log['expected_calibration_error'].numpy())
    #print(eces)
    ECE.append(np.average(eces))

print("Softmax Increasing # Experts ECE: {}".format(ECE))


# === OvA ===
path = "ova_increase_experts/"
n_experts = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
ECE = []
for n in n_experts:
    model_name = '_' + 'new' + '_' + str(n) + '_experts' #'_' + 'new' + 
    with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'r') as f:
        conf = json.loads(json.load(f))
    with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'r') as f:
        exp_pred = json.loads(json.load(f))
    with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'r') as f:
        true_label = json.loads(json.load(f))
    # true.append(true_label['test'])
    # exps.append(exp_pred['test'])
    c = torch.tensor(conf['test'])
    #print(type(true_label['test']), type(exp_pred['test']), type(c))
    #print(np.array(true_label['test']).shape, np.array(exp_pred['test']).shape, c.shape)
    # DANI Correction ===
    #c = c.softmax(dim=1)
    c = c.sigmoid()
    # DANI Correction ===

    prob = c
    # temp = 0
    # for i in range(n):
    #     temp += c[:, (n_classes + n) - (i + 1)]
    # prob = c / (1.0 - temp).unsqueeze(-1)
    # print(prob.shape)
    # confs.append(prob)

    true_label = torch.tensor(true_label['test'])
    exp_prediction = torch.tensor(exp_pred['test']) # exp_prediction shape [n, test_data_size]

    # ECEs ===

    # enumerate over all the experts
    eces = []
    for j in range(n):
        e_j = exp_prediction[j,:]
        t = true_label
        c_j = prob[:, n_classes + j]
        #ids_where_gt_one = torch.where(c_j > 1.0)
        #c_j[ids_where_gt_one] = 1.0
        acc_j = t.eq(e_j)
        log = compute_calibration(c_j, acc_j)
        eces.append(log['expected_calibration_error'].numpy())
    #print(eces)
    ECE.append(np.average(eces))

print("OvA Increasing # Experts ECE: {}".format(ECE))




# ECEs = []
# for i in range(len(n_experts)):
#     c = confs[i]
#     e = exps[i]
#     t = torch.tensor(true[i])

#     eces = []
#     for j in range(len(e)):
#         e_j = e[j]
#         c_j = c[:, n_classes + j] #c.shape[1] - (j + 1)]
#         t_j = t
#         ids_where_gt_one = torch.where(c_j > 1.0)
#         c_j[ids_where_gt_one] = 1.0
#         acc_j = t_j.eq(torch.tensor(e_j))
#         log = compute_calibration(c_j, acc_j)
#         eces.append(log['expected_calibration_error'].numpy())
#     ECEs.append(eces)

# Y = []
# for l in ECEs:
#     Y.append(np.average(l))



# print("Softmax Increasing # Experts ECE: {}".format(Y))
# #plt.plot(Y)
# #plt.show()


# # === OvA ===
# n_classes = 10
# confs = []
# exps = []
# true = []
# path = "ova_increase_experts/"
# n_experts = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
# for n in n_experts:
#     model_name = '_' + 'new' + '_' + str(n) + '_experts'
#     with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'r') as f:
#         conf = json.loads(json.load(f))
#     with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'r') as f:
#         exp_pred = json.loads(json.load(f))
#     with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'r') as f:
#         true_label = json.loads(json.load(f))
#     true.append(true_label['test'])
#     exps.append(exp_pred['test'])
#     c = torch.tensor(conf['test'])
#     print(c.shape)
#     # DANI Correction ===
#     c = c.sigmoid()
#     # DANI Correction ===
#     confs.append(c)

# ECEs = []
# for i in range(len(n_experts)):
#     c = confs[i]
#     e = exps[i]
#     t = torch.tensor(true[i])
#     eces = []
#     for j in range(len(e)):
#         e_j = e[j]
#         c_j = c[:, n_classes + j] #c.shape[1] - (j + 1)]
#         t_j = t
#         ids_where_gt_one = torch.where(c_j > 1.0)
#         c_j[ids_where_gt_one] = 1.0
#         acc_j = t_j.eq(torch.tensor(e_j))
#         log = compute_calibration(c_j, acc_j)
#         eces.append(log['expected_calibration_error'].numpy())
#     ECEs.append(eces)

# Y = []
# for l in ECEs:
#     Y.append(np.average(l))

# print("OvA Increasing # Experts ECE: {}".format(Y))
#plt.plot(Y)
#plt.show()
