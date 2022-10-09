import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from scipy import stats

# === Naïve Conformal Inference === #
# Ensemble functions ===
def get_expert_prediction(experts, prediction_set_i, method="voting"):
    ensemble_expert_pred_i = np.array(experts)[prediction_set_i][:, i]
    # Voting ===
    if method == "voting":
        exp_prediction = stats.mode(ensemble_expert_pred_i).mode if len(ensemble_expert_pred_i) != 0 else []

    # Last ===
    if method == "last":
        exp_prediction = ensemble_expert_pred_i[-1] if len(ensemble_expert_pred_i) != 0 else []

    # Random ===
    if method == "random":
        idx = np.random.randint(len(ensemble_expert_pred_i)) if len(ensemble_expert_pred_i) != 0 else -1
        exp_prediction = ensemble_expert_pred_i[idx] if idx != -1 else []

    return exp_prediction

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


