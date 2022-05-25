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
#from main import *
from utils import *
from data_utils import *
from models.wideresnet import *
from models.experts import *
from losses.losses import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def forward(model, dataloader, num_experts, expert_fns, n_classes, n_experts):
    confidence  = []
    true = []
    expert_predictions = defaultdict(list)

    with torch.no_grad():
        for inp, lbl in dataloader:
            inp = inp.to(device)
            conf = model(inp)
            for i, fn in enumerate(expert_fns):
              expert_pred1 = fn(inp, lbl)
              expert_predictions[i].append(expert_pred1)
            confidence.append(conf.cpu())
            true.append(lbl[:,0])


    true = torch.stack(true, dim=0).view(-1)
    confidence = torch.stack(confidence, dim=0).view(-1, n_classes+n_experts)
    for k, v in expert_predictions.items():
      expert_predictions[k] = torch.stack([torch.tensor(k) for k in v], dim=0).view(-1)
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
        #result_ = evaluate(model, expert_fns, n_dataset, dl)
        #result[severity] = result_
        true_label[severity] = true.numpy()
        classifier_confidence[severity] = confidence.numpy()
        expert_preds[severity] = expert_predictions
        return true, confidence, expert_predictions

    result = {}
    classifier_confidence = {}
    true_label = {}
    expert_preds = {}
    inp_density = {}

    n_dataset = 10
    batch_size = 64
    n_expert = num_experts
    kwargs = {'num_workers': 1, 'pin_memory': True}
    _, test_d = cifar.read(severity=0, slice_=-1, test=True, only_id=True)
    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    
    path = './Models'
    model = WideResNet(28, 3, n_dataset + num_experts, 4, dropRate=0.0)
    model.load_state_dict(torch.load(path + '/multiple_experts' + model_name + '.pt', map_location=device))
    model = model.to(device)

    true, confidence, expert_predictions = get('test', test_dl)


    pth = './validation/'
    os.makedirs(pth, exist_ok=True)
    with open(pth + 'true_label_multiple_experts' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(true_label, cls=NumpyEncoder), f)


    with open(pth + 'confidence_multiple_experts' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(classifier_confidence, cls=NumpyEncoder), f)

    with open(pth + 'expert_predictions_multiple_experts' + model_name +'.txt', 'w') as f:
        json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)


if __name__ == "__main__":
    alpha = 1.0
    n_classes = 10 # 10 classes for CIFAR-10
    k = 5
    for n in [1,2,4,8]:
      model_name = '_' + str(n) + '_experts'
      num_experts = n
      expert = synth_expert(k, n_classes)
      expert_fns = [expert.predict_biasedK] * n
      validation(model_name, num_experts, expert_fns)