# Analyze the confidences on test data

import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from models.wideresnet import WideResNet
from models.experts import synth_expert
from losses.losses import *

from data_utils import cifar
from main import metrics_print

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def forward(model, dataloader, num_experts, expert_fns, n_classes, n_experts):
    confidence = []
    true = []
    expert_predictions = defaultdict(list)
    # density = []

    with torch.no_grad():
        for inp, lbl in dataloader:
            inp = inp.to(device)
            conf = model(inp)
            for i, fn in enumerate(expert_fns):
                expert_pred1 = fn(inp, lbl)
                expert_predictions[i].append(expert_pred1)
            confidence.append(conf.cpu())
            true.append(lbl)

    true = torch.stack(true, dim=0).view(-1)
    confidence = torch.stack(confidence, dim=0).view(-1, n_classes + n_experts)
    for k, v in expert_predictions.items():
        expert_predictions[k] = torch.stack([torch.tensor(k) for k in v], dim=0).view(-1)

    # print(true.shape, confidence.shape, [v.shape for k, v in
    #                                      expert_predictions.items()])  # ,expert_predictions1.shape, expert_predictions2.shape) #, density.shape)
    return true, confidence, [v.numpy() for k, v in
                              expert_predictions.items()]  # (expert_predictions1, expert_predictions2) #, density



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
        # inp_density[severity] = density.numpy()

    result = {}
    classifier_confidence = {}
    true_label = {}
    expert_preds = {}
    # inp_density = {}

    n_dataset = 10
    batch_size = 64
    n_expert = num_experts
    # Data ===
    test_d = cifar.read(severity=0, slice_=-1, test=True, only_id=True)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

    # Model ===
    model = WideResNet(28, 3, n_dataset + num_experts, 4, dropRate=0.0)
    model_path = os.path.join(config["ckp_dir"], config["experiment_name"] + '_' + str(
        len(expert_fns)) + '_experts' + '.pt')
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

    with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)

    # with open(path + 'inp_log_density.txt', 'w') as f:
    #     json.dump(json.dumps(inp_density, cls=NumpyEncoder), f)


if __name__ == "__main__":
    alpha = 1.0
    n_dataset = 4
    for n in [2, 4, 6, 8]:
        model_name = '_' + str(n) + '_experts'
        num_experts = n
        expert = synth_expert(n_dataset)
        # fill experts in the reverse order
        expert_fns = [expert.predict] * n
        validation(model_name, num_experts, expert_fns)
