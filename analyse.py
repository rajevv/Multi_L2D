# Analyze the confidences on test data

from collections import defaultdict
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



def forward(model, dataloader, num_experts, expert_fns, n_classes, n_experts):
    confidence  = []
    true = []
    expert_predictions = defaultdict(list)
    # density = []

    # dist1 = torch.distributions.normal.Normal(cluster1_mean, cluster1_var.sqrt())
    # dist2 = torch.distributions.normal.Normal(cluster2_mean, cluster2_var.sqrt())
    # dist3 = torch.distributions.normal.Normal(cluster3_mean, cluster3_var.sqrt())
    # dist4 = torch.distributions.normal.Normal(cluster4_mean, cluster4_var.sqrt())

    with torch.no_grad():
        for inp, lbl in dataloader:
            # for i in range(inp.shape[0]):
            #   if lbl[i] == 0:
            #           density.append(torch.sum(dist1.log_prob(inp[i])))
            #   elif lbl[i] == 1:
            #           density.append(torch.sum(dist2.log_prob(inp[i])))
            #   elif lbl[i] == 2:
            #           density.append(torch.sum(dist3.log_prob(inp[i])))
            #   elif lbl[i] == 3:
            #           density.append(torch.sum(dist4.log_prob(inp[i])))

            inp = inp.to(device)
            conf = model(inp)
            for i, fn in enumerate(expert_fns):
              expert_pred1 = fn(inp, lbl)
              expert_predictions[i].append(expert_pred1)
            confidence.append(conf.cpu())
            true.append(lbl)
            # expert_predictions1.append(torch.tensor(expert_pred1))
            # expert_predictions2.append(torch.tensor(expert_pred2))

    true = torch.stack(true, dim=0).view(-1)
    confidence = torch.stack(confidence, dim=0).view(-1, n_classes+n_experts)
    for k, v in expert_predictions.items():
      #print(type(v), len(v), v)
      expert_predictions[k] = torch.stack([torch.tensor(k) for k in v], dim=0).view(-1)

    #for k, v in expert_predictions.items():
      #print(type(v))
    print(true.shape, confidence.shape, [v.shape for k,v in expert_predictions.items()]) #,expert_predictions1.shape, expert_predictions2.shape) #, density.shape)
    return true, confidence, [v.numpy() for k,v in expert_predictions.items()] #(expert_predictions1, expert_predictions2) #, density


import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def validation(model_name, num_experts, expert_fns):

    def filter(dict_):
        d = {}
        for k, v in dict_.items():
            if torch.is_tensor(v):
                v = v.item()
            d[k] = v
        return d

    def get(severity, dl):
        true, confidence, expert_predictions = forward(model, dl, num_experts, expert_fns, n_dataset, n_expert)
        result_ = metrics_print(model, num_experts, expert_fns, n_dataset, dl)
        result[severity] = result_
        true_label[severity] = true.numpy()
        classifier_confidence[severity] = confidence.numpy()
        expert_preds[severity] = expert_predictions
        #inp_density[severity] = density.numpy()

    result = {}
    classifier_confidence = {}
    true_label = {}
    expert_preds = {}
    inp_density = {}

    n_dataset = 4
    batch_size = 64
    n_expert = num_experts
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    
    model = Net(dims=[32, 16, 8, n_dataset+num_experts])
    model.load_state_dict(torch.load(path + '/synthetic_mutiple_experts' + model_name + '.pt', map_location=device))
    model = model.to(device)

    get('test', test_dl)

    # ood_data = ['cifar']
    # for severity, d in enumerate(ood_data):
    #     OOD = Data(d[0], d[1])
    #     OOD_dl = torch.utils.data.DataLoader(OOD, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    #     get(str(severity+1), OOD_dl)

    

    # with open(path + 'validation.txt', 'w') as f:
    #     json.dump(result, f)

    with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(true_label, cls=NumpyEncoder), f)


    with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(classifier_confidence, cls=NumpyEncoder), f)

    with open(path + 'expert_predictions_multiple_experts' + model_name +'.txt', 'w') as f:
        json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)

    # with open(path + 'inp_log_density.txt', 'w') as f:
    #     json.dump(json.dumps(inp_density, cls=NumpyEncoder), f)


if __name__ == "__main__":
    alpha = 1.0
    n_dataset = 4
    for n in [2,4,6,8]:
      model_name = '_' + str(n) + '_experts'
      num_experts = n
      expert = synth_expert(n_dataset)
      # fill experts in the reverse order
      expert_fns = [expert.predict] * n
      validation(model_name, num_experts, expert_fns)