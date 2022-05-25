import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from reliability_diagram import compute_calibration

n_classes = 10
random_expert_idx = 0
confs = []
exps = []
true = []
path = "softmax_increase_confidence/"
n_experts = 4
p_experts = [0.2, 0.4, 0.6, 0.8, 0.95]
for p in p_experts:
    model_name = '_' + str(p) + '_confidence'
    with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'r') as f:
        conf = json.loads(json.load(f))
    with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'r') as f:
        exp_pred = json.loads(json.load(f))
    with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'r') as f:
        true_label = json.loads(json.load(f))
    true.append(true_label['test'])
    exps.append(exp_pred['test'])
    c = torch.tensor(conf['test'])
    print(c.shape)
    # DANI Correction ===
    c = c.softmax(dim=1)
    # DANI Correction ===

    temp = 0
    for i in range(n_experts):
        temp += c[:, (n_classes + n_experts) - (i + 1)]
    prob = c / (1.0 - temp).unsqueeze(-1)
    confs.append(prob)

ECEs = []
for i in range(len(p_experts)):
    c = confs[i]
    e = exps[i]
    t = torch.tensor(true[i])
    eces = []

    for j in range(n_experts):
        e_j = e[j]
        c_j = c[:, c.shape[1] - (j + 1)]
        t_j = t
        ids_where_gt_one = torch.where(c_j > 1.0)
        c_j[ids_where_gt_one] = 1.0
        acc_j = t_j.eq(torch.tensor(e_j))
        log = compute_calibration(c_j, acc_j)
        eces.append(log['expected_calibration_error'].numpy())
    ECEs.append(eces)

Y = []
# average on all experts
for l in ECEs:
    Y.append(np.average(l))

print("Average among Experts ECE: {}".format(Y))
plt.plot(Y)
plt.show()

Y_random = []
# random expert ECE
for l in ECEs:
    Y_random.append(l[random_expert_idx][0])

print("Random Expert ECE: {}".format(Y_random))
plt.plot(Y_random)
plt.show()

