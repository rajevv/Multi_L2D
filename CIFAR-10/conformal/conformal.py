import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from scipy import stats
# Load results functions ===
def load_results(path_confidence, path_experts, path_labels, model_name, seeds, exp_list, method="ova"):
    results = dict.fromkeys(seeds)
    for seed in seeds:
        # === OvA ===
        confs = []
        exps = []
        true = []

        for exp in exp_list:
            model_name = model_name.format(exp)  # model name for each specific experiment, .e.g  '_p_out_0.9'
            seed_path = "_seed_{}".format(seed)
            # Load ===
            full_conf_path = path_confidence + model_name + seed_path + '.txt'
            with open(full_conf_path, 'r') as f:
                conf = json.loads(json.load(f))

            full_experts_path = path_experts + model_name + seed_path + '.txt'
            with open(full_experts_path, 'r') as f:
                exp_pred = json.loads(json.load(f))

            full_true_path = path_labels + model_name + seed_path + '.txt'
            with open(full_true_path, 'r') as f:
                true_label = json.loads(json.load(f))

            true.append(true_label['test'])
            exps.append(exp_pred['test'])
            c = torch.tensor(conf['test'])
            if method == "ova":
                # logits to probs OvA -> sigmoid
                c = c.sigmoid()
            elif method == "softmax":
                # logits to probs Softmax -> Softmax
                c = c.softmax(dim=1)
            confs.append(c)

        seed_result = {"confs": confs,
                       "exps": exps,
                       "true": true
                       }
        # Fill dict for seed
        results[seed] = seed_result
        return results


def process_conformal_results(results, exp_list, exp_args, cal_percent=0.8, alpha=0.1):
    seeds = results.keys()
    metrics = dict.fromkeys(seeds)

    # Params ===
    n_classes = exp_args["n_classes"]
    n_experts = exp_args["n_experts"]
    n_classes_exp = n_classes + n_experts

    for seed in seeds:
        seed_dict = results[seed]  # confs, exps, true
        k_dict = {}
        for k, exp in enumerate(exp_list):
            confs_k = seed_dict["confs"][k]
            exps_k = seed_dict["exps"][k]
            true_k = seed_dict["true"][k]

            # 1. Split Calibration / Test ===
            n_cal = int(cal_percent * len(true_k))
            n_test = len(true_k) - n_cal

            # for shuffling
            idx = np.array([1] * n_cal + [0] * (len(true_k) - n_cal)) > 0
            assert int(sum(idx)) == n_cal
            np.random.shuffle(idx)

            idx_cal = idx
            idx_test = ~idx

            # 2. Get deferral ===
            r = get_deferral(confs_k, n_classes_exp, n_experts)

            # Deferral
            k_dict["deferral"] = r
            k_dict["idx_cal"] = idx_cal
            k_dict["idx_cal"] = idx_test

            # Model Coverage (non-deferred)
            k_dict["coverage_cal"] = (r[idx_cal] == 0).sum()/ n_cal
            k_dict["coverage_test"] = (r[idx_test] == 0).sum() / n_test

            # 2. Calculate Qhat on calibration
            qhat_k = get_qhat(confs_k, exps_k, true_k, r, idx_cal, n_classes, alpha=0.1)
            k_dict["qhat"] = qhat_k

            # TODO. WIP




    return


def get_qhat(confs, exps, true, deferral, idx_cal, n_classes, alpha=0.1):
    # Val/Calibration ===
    confs_experts_cal = confs[idx_cal, n_classes:]
    experts_cal = [np.array(exp)[idx_cal].tolist() for exp in exps]
    true_cal = np.array(true)[idx_cal].astype(list)
    r_cal = deferral[idx_cal]

    # Calculate Q_hat ===
    # Only on deferred samples !
    confs_experts_cal = confs_experts_cal[r_cal]
    experts_cal = [np.array(exp)[r_cal] for exp in experts_cal]
    true_cal = np.array(true_cal)[r_cal]

    # Model expert probs ===
    # Sort J model outputs for experts
    sort, pi = confs_experts_cal.sort(dim=1, descending=True)

    # Correctness experts ===
    # Check if experts are correct
    correct_exp = (np.array(experts_cal) == np.array(true_cal)).T
    # idx for correct experts: [[0,1,2], [1,2], [], ...]
    correct_exp_idx = [np.where(correct_exp_i)[0] for correct_exp_i in correct_exp]

    # obtain the last expert to be retrieved. If empty, then add all values.
    # indexes are not the real expert index, but the sorted indexes, e.g. [[1, 0 ,2],  [1,0], [], ...]
    pi_corr_exp = [confs_experts_cal[i, corr_exp].sort(descending=True)[1] for i, corr_exp in enumerate(correct_exp)]
    pi_corr_exp_stop = [pi_corr_exp_i[-1] if len(pi_corr_exp_i) != 0 else -1 for pi_corr_exp_i in
                        pi_corr_exp]  # last expert

    # obtain real expert index back, e.g. [2,1,-1,...]
    pi_stop = [correct_exp_idx[i][pi_corr_exp_stop_i] if len(correct_exp_idx[i]) != 0 else -1 for i, pi_corr_exp_stop_i
               in enumerate(pi_corr_exp_stop)]

    # Obtain quantile
    scores = sort.cumsum(dim=1).gather(1, pi.argsort(1))[range(len(torch.tensor(pi_stop))), torch.tensor(pi_stop)]
    n_quantile = r_cal.sum()
    qhat = torch.quantile(scores, np.ceil((n_quantile + 1) * (1 - alpha)) / n_quantile, interpolation="higher")
    return qhat


def get_deferral(probs, n_classes_exp, n_experts):
    _, predicted = torch.max(probs.data, 1)
    r = (predicted >= n_classes_exp - n_experts)
    return r

# Ensemble functions ===
# def get_expert_prediction(experts, prediction_set_i, method="voting"):  # TODO: Debug and prepare correectly.
#     r"""
#
#     Args:
#         experts:
#         prediction_set_i:
#         method:
#
#     Returns:
#
#     """
#     ensemble_expert_pred_i = np.array(experts)[prediction_set_i][:, i]
#     # Last ===
#     if method == "last":
#         exp_prediction = ensemble_expert_pred_i[-1] if len(ensemble_expert_pred_i) != 0 else []
#
#     # Random ===
#     if method == "random":
#         idx = np.random.randint(len(ensemble_expert_pred_i)) if len(ensemble_expert_pred_i) != 0 else -1
#         exp_prediction = ensemble_expert_pred_i[idx] if idx != -1 else []
#
#     # Fixed-size ensemble ===
#     if method == "ensemble":
#         # exp_prediction = # TODO
#
#     # Majority Voting ===
#     if method == "voting":
#         exp_prediction = stats.mode(ensemble_expert_pred_i).mode if len(ensemble_expert_pred_i) != 0 else []
#
#     return exp_prediction


# ======================================= #
# ====== Naive Conformal Inference ====== #
# ======================================= #




# ======================================= #
# === Regularized Conformal Inference === #
# ======================================= #


# utility functions
def get_kparam(paramtune_probs, paramtune_gt_labels, alpha):
    temp = paramtune_probs * paramtune_gt_labels
    flat = temp.reshape(-1)
    # indices where the temp is True
    non_zero_indices = torch.nonzero(flat)
    # confidence of the correct experts
    correct_experts_confs = flat[non_zero_indices]
    return torch.quantile(torch.sort(correct_experts_confs, descending=True)[0], 1 - alpha, interpolation='higher')


def false_negative_rate(prediction_set, gt_labels):
    return 1 - ((prediction_set * gt_labels).sum(axis=1) / gt_labels.sum(axis=1)).mean()


def get_lhat(calib_loss_table, lambdas, alpha, B=1):
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    lhat_idx = max(np.argmax(((n / (n + 1)) * rhat + B / (n + 1)) >= alpha) - 1, 0)  # Can't be -1.
    return lambdas[lhat_idx]


def conformal_risk_control(probs_experts, gt_labels, alpha=None, lower=0, upper=1):
    num_lam = 1500
    lambdas_example_table = np.linspace(lower, upper, num_lam)

    # Run the conformal risk control procedure
    loss_table = np.zeros((probs_experts.shape[0], num_lam))

    from tqdm import tqdm
    for j in range(num_lam):
        est_labels = probs_experts >= lambdas_example_table[j]
        loss = false_negative_rate(est_labels, gt_labels)
        loss_table[:, j] = loss

    lamhat = get_lhat(loss_table, lambdas_example_table, alpha)
    return lamhat

    # n = probs_experts.shape[0]
    # idx = torch.where(probs_experts > float(upper))
    # print(idx)
    # probs_experts[idx] = float(upper)
    # def lamhat_threshold(lam): return false_negative_rate(probs_experts>=(1-lam), gt_labels) - ((n+1)/n*alpha - 1/(n+1))
    # lamhat = ridder(lamhat_threshold, lower, upper)
    # return lamhat


class ConformalRiskControl(nn.Module):
    def __init__(self, calib_probs, calib_gt_labels, alpha, kparam=None, betaparam=None, prop=0.3, lower=0, upper=1,
                 model=None):
        super(ConformalRiskControl, self).__init__()
        self.model = model
        self.alpha = alpha

        if (kparam == None or betaparam == None):
            kparam, betaparam, calib_probs, calib_gt_labels = pick_parameters(calib_probs, calib_gt_labels, prop=prop,
                                                                              alpha=alpha)

        self.kparam = kparam
        self.betaparam = betaparam

        calib_probs = (1 + betaparam) * calib_probs - betaparam * kparam
        calib_gt_labels = calib_gt_labels

        self.lamhat = conformal_risk_control(calib_probs, calib_gt_labels, alpha=alpha, upper=upper, lower=lower)

    def forward(self, probs):
        with torch.no_grad():
            probs = (1 + self.betaparam) * probs - self.betaparam * self.kparam
            prediction_sets = probs >= self.lamhat
            return prediction_sets


def validate(conformal_model, probs, labels):
    with torch.no_grad():
        prediction_sets = conformal_model(probs)
        size = np.sum(prediction_sets.numpy(), axis=1)
        FNR = false_negative_rate(prediction_sets, labels)
    return {'size': size.tolist(), 'FNR': FNR}


def get_betaparam(paramtune_probs, paramtune_gt_labels, kstar, alpha):
    best_size = paramtune_probs.shape[1]  # Total number of experts

    for temp_beta in np.linspace(3.5, 1e-3, 50):
        conformal_model = ConformalRiskControl(paramtune_probs, paramtune_gt_labels, alpha, kparam=kstar,
                                               betaparam=temp_beta)
        metrics = validate(conformal_model, paramtune_probs, paramtune_gt_labels)
        mean_size = np.mean(metrics['size'])
        if mean_size < best_size:
            best_size = mean_size
            betastar = temp_beta
    return betastar


def pick_parameters(calibration_probs, gt_labels, prop=0.3, alpha=0.1):
    '''
    first split the calibration_probs into two separate datasets: one for hyperparam tuning and another for conformal risk
    calibration_probs: [N, E], N = deferred data points, E = no. of experts
    gt_labels: [N, E] Boolen tensor: True at the index where the expert is correct
    '''
    import torch.utils.data as tdata
    N = calibration_probs.shape[0]
    num_paramtune = int(np.ceil(prop * calibration_probs.shape[0]))
    idx = np.array([1] * num_paramtune + [0] * (N - num_paramtune)) > 0
    assert int(sum(idx)) == num_paramtune
    np.random.shuffle(idx)

    paramtune_probs, paramtune_gt_labels = calibration_probs[idx, :], gt_labels[idx, :]
    calib_probs, calib_gt_labels = calibration_probs[~idx, :], gt_labels[~idx, :]
    print("Number of hyperparameter tuning samples {}\n Number of conformal risk samples {}".format(
        paramtune_probs.shape[0], calib_probs.shape[0]))

    # pick kparam
    kstar = get_kparam(paramtune_probs, paramtune_gt_labels, alpha)

    # pick beta
    betastar = get_betaparam(paramtune_probs, paramtune_gt_labels, kstar, alpha)

    return kstar, betastar, calib_probs, calib_gt_labels


