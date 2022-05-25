# Analyze the confidences on test data

import argparse
import json
import os
from collections import defaultdict

from data_utils import cifar
from losses.losses import *
from main_increase_experts import evaluate
from models.experts import synth_expert
from models.wideresnet import WideResNet

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)


def forward(model, dataloader, expert_fns, n_classes, n_experts):
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
            true.append(lbl[:, 0])

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


def validation(model_name, expert_fns, config):
    def filter(dict_):
        d = {}
        for k, v in dict_.items():
            if torch.is_tensor(v):
                v = v.item()
            d[k] = v
        return d

    def get(severity, dl):
        true, confidence, expert_predictions = forward(model, dl, expert_fns, n_dataset, n_expert)

        criterion = Criterion()
        loss_fn = getattr(criterion, config["loss_type"])
        n_classes = n_dataset
        # result_ = evaluate(model, expert_fns, loss_fn, n_classes, dl, config)
        # result_ = metrics_print(model, num_experts, expert_fns, n_dataset, dl)

        # result[severity] = result_
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
    batch_size = 1024
    n_expert = 4
    # Data ===
    ood_d, test_d = cifar.read(severity=0, slice_=-1, test=True, only_id=True)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

    # Model ===
    model = WideResNet(28, 3, n_dataset + n_experts, 4, dropRate=0.0)

    model_path = os.path.join(config["ckp_dir"],
                              config["experiment_name"] + '_' + str(config["p_in"]) + '_confidence' + '.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    get('test', test_dl)

    with open(config["ckp_dir"] + 'true_label_multiple_experts' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(true_label, cls=NumpyEncoder), f)

    with open(config["ckp_dir"] + 'confidence_multiple_experts' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(classifier_confidence, cls=NumpyEncoder), f)

    with open(config["ckp_dir"] + 'expert_predictions_multiple_experts' + model_name + '.txt', 'w') as f:
        json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)

    # with open(path + 'inp_log_density.txt', 'w') as f:
    #     json.dump(json.dumps(inp_density, cls=NumpyEncoder), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="scaling parameter for the loss function, default=1.0.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20,
                        help="number of patience steps for early stopping the training.")
    parser.add_argument("--expert_type", type=str, default="predict",
                        help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.")
    parser.add_argument("--n_classes", type=int, default=10,
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
    config["ckp_dir"] = "./" + config["loss_type"] + "_increase_experts/"
    print(config)

    alpha = 1.0
    n_dataset = 10
    n_experts = 4
    p_experts = [0.2, 0.4, 0.6, 0.8, 0.95]
    p_experts = [0.2, 0.4, 0.6, 0.8]

    for p_in in p_experts:
        model_name = '_' + str(p_in) + '_confidence'
        # Expert ===
        random_expert = synth_expert(config["k"], config["n_classes"])
        random_fn = random_expert.predict_random
        config["p_in"] = p_in
        increasing_expert = synth_expert(config["k"], config["n_classes"], p_in=p_in, p_out=0.2)
        increasing_fn = increasing_expert.predict_prob_cifar

        expert_fns = [random_fn] + [increasing_fn] * (n_experts - 1)

        validation(model_name, expert_fns, config)
