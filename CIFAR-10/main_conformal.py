import argparse
import json
import os
from collections import defaultdict

from data_utils import cifar
from losses.losses import *
from main_increase_experts import evaluate
from models.experts import *
from models.experts import synth_expert
from models.wideresnet import *
from models.wideresnet import WideResNet

# Analyze the confidences on test data

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
        n_classes = n_dataset + len(expert_fns)
        result_ = evaluate(model, expert_fns, loss_fn, n_classes, dl, config)
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
    n_expert = num_experts
    # Data ===
    ood_d, test_d = cifar.read(severity=0, slice_=-1, test=True, only_id=True)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

    # Model ===
    model = WideResNet(28, 3, n_dataset + num_experts, 4, dropRate=0.0)
    model_path = os.path.join(config["ckp_dir"], config["experiment_name"] + '_' + str(
        len(expert_fns)) + '_experts' + '.pt')
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
    # for n in [1, 2, 4, 6, 8]:
    for n in [4]:
        model_name = '_' + str(n) + '_experts'
        num_experts = n
        # Expert ===
        expert = synth_expert(config["k"], config["n_classes"])
        expert_fn = getattr(expert, config["expert_type"])
        expert_fns = [expert_fn] * n

        validation(model_name, expert_fns, config)




## 2. get Q_hat
probs_val = probs[:n_val, 10:]
experts_val = [exp[:n_val] for exp in experts]
y_true_val = y_true[:n_val]

# 2.b Sort J model outputs for experts
probs_experts = probs[:n_val, 10:]
sort, pi = probs_experts.sort(dim=1, descending=True)

# 2.c Test statistic S
S = 0


# Check if experts are correct
correct_exp = (np.array(experts_val) == np.array(y_true_val)).T
# Swap order to match confidence ordering
correct_exp = np.flip(correct_exp).copy()  # copy needed!

# idx for correct experts: [[0,1,2], [1,2], [], ...]
correct_exp_idx = [np.where(correct_exp_i)[0] for correct_exp_i in correct_exp]

# obtain the last expert to be retrieved. If empty, then add all values.
# indexes are not the real expert index, but the sorted indexes, e.g. [[1, 0 ,2],  [1,0], [], ...]
pi_corr_exp = [probs_experts[i, corr_exp].sort(descending=True)[1] for i, corr_exp in enumerate(correct_exp)]
pi_corr_exp_stop = [pi_corr_exp_i[-1] if len(pi_corr_exp_i)!=0 else -1 for pi_corr_exp_i in pi_corr_exp]  # last expert

# obtain real expert index back, e.g. [2,1,-1,...]
pi_stop = [correct_exp_idx[i][pi_corr_exp_stop_i] if len(correct_exp_idx[i])!=0 else -1 for i, pi_corr_exp_stop_i in enumerate(pi_corr_exp_stop)]


# =========
n_val = n_val
alpha = 0.95
scores = sort.cumsum(dim=1).gather(1, pi.argsort(1))[range(len(torch.tensor(pi_stop))), torch.tensor(pi_stop)]
qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)

qhat