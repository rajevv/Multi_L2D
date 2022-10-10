import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

from scipy import stats

# Global variables ===
metric_methods = ["standard",  # standard L2D
                  "last", "random", "voting",  # conformal-based
                  "ensemble"]  # basic fixed-size ensemble


# Load results functions ===
def load_results(path_confidence, path_experts, path_labels, model_name, seed_name, seeds, exp_list, method="ova"):
    r"""
    Load results.
    Args:
        path_confidence: Path from the confidence values.
        path_experts: Path from the expert values.
        path_labels: Path from the expert values.
        model_name: Model name in the appropriate format.
        seeds: List containing the seeds.
        exp_list: List containing the experiment values ,e.g. prob_out = [0.1, 0.2, 0.3, ...]
        method: OvA or Softmax method.

    Returns:
        results: Dict containing confidences, expert predictions and true labels.
    """
    results = dict.fromkeys(seeds)
    print("\nLoad {} results".format(method))
    for seed in tqdm(seeds):
        # === OvA ===
        confs = []
        exps = []
        true = []

        for exp in exp_list:
            seed_path = seed_name.format(seed)
            # Load ===
            full_conf_path = path_confidence + model_name.format(exp) + seed_path + '.txt'
            with open(full_conf_path, 'r') as f:
                conf = json.loads(json.load(f))

            full_experts_path = path_experts + model_name.format(exp) + seed_path + '.txt'
            with open(full_experts_path, 'r') as f:
                exp_pred = json.loads(json.load(f))

            full_true_path = path_labels + model_name.format(exp)+ seed_path + '.txt'
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

            # Reset model name?
        seed_result = {"confs": confs,
                       "exps": exps,
                       "true": true
                       }
        # Fill dict for seed
        results[seed] = seed_result

    return results


def process_conformal_results(results, exp_list, exp_args, cal_percent=0.8, alpha=0.1,
                              metric_methods=metric_methods):
    seeds = results.keys()
    results_dict = dict.fromkeys(seeds)

    # Params ===
    n_classes = exp_args["n_classes"]
    n_experts = exp_args["n_experts"]
    n_classes_exp = n_classes + n_experts

    for seed in seeds:
        print("\n============================")
        print("\nResults Seed {}".format(seed))
        print("\n============================")
        seed_dict = results[seed]  # confs, exps, true

        results_dict[seed] = dict.fromkeys(exp_list)

        for k, exp in tqdm(enumerate(exp_list)):
            k_dict = {}
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
            k_dict["coverage_cal"] = (r[idx_cal] == 0).sum() / n_cal
            k_dict["coverage_test"] = (r[idx_test] == 0).sum() / n_test

            # ====== CONFORMAL ====== #
            # 2. Calculate Qhat on calibration
            qhat_k = get_qhat(confs_k, exps_k, true_k, r, idx_cal, n_classes, alpha=alpha)
            k_dict["qhat"] = qhat_k
            # ====== CONFORMAL ====== #

            # 3. Get metrics

            for method in metric_methods:
                metrics_dict = get_metrics(confs_k, exps_k, true_k, r, idx_test, n_classes, qhat=qhat_k, args=exp_args,
                                           method=method)
                k_dict[method] = metrics_dict
            results_dict[seed][exp] = k_dict

    return results_dict


def get_metrics(confs, exps, true, deferral, idx_test, n_classes, qhat, args, method="standard"):
    # Init Metrics
    correct = 0
    correct_sys = 0
    exp = 0
    alone_correct = 0

    # Test ===
    confs_test = confs[idx_test]  # model + experts
    confs_experts_test = confs_test[:, n_classes:]  # experts
    experts_test = [np.array(exp)[idx_test].tolist() for exp in exps]
    true_test = torch.tensor(true)[idx_test]
    r_test = deferral[idx_test]
    N_test = len(r_test)

    # Individual Expert Accuracies === # TODO
    # expert_correct_dic = {k: 0 for k in range(len(experts_test))}
    # expert_total_dic = {k: 0 for k in range(len(experts_test))}

    # Predicted value
    _, predicted = torch.max(confs_test.data, 1)
    _, model_prediction = torch.max(confs_test.data[:, :n_classes], 1)

    # r == 0 -> Not deferred ===========
    # Classifier alone ===
    alone_correct += (model_prediction[~r_test] == true_test[~r_test]).sum()
    correct_sys += alone_correct  # for non-deferred samples
    correct += alone_correct

    # r == 1 -> Deferred ===========
    # Filter by deferred
    experts_r_test = np.array(experts_test).T
    experts_r_test = experts_r_test[r_test]

    # Non Conformal prediction ===
    if method == "standard":  # Top-1
        top1_experts = predicted[r_test] - n_classes
        exp_prediction = torch.tensor([experts_r_test[i, top1] for i, top1 in enumerate(top1_experts)])

    # Conformal prediction ===
    conformal_dict = {}
    if method in ["voting", "last", "random"]:
        experts_conformal_mask, experts_conformal_sets = get_conformal_set(confs_experts_test[r_test], qhat=qhat)
        exp_prediction = get_expert_prediction(experts_r_test, experts_conformal_mask, method=method)

        # Conformal set sizes
        set_sizes = experts_conformal_mask.sum(axis=1)
        avg_set_size = set_sizes.numpy().mean()
        conformal_dict["set_sizes"] = set_sizes,
        conformal_dict["avg_set_size"] = avg_set_size

    # Naive Top-k ensemble, without conformal ====
    if method == "ensemble":
        exp_prediction, experts_ensemble_sets = get_fixed_ensemble(experts_r_test, confs_experts_test[r_test],
                                                                   ensemble_size=args["ensemble_size"])

    # Deferral accuracy: No matter expert ===
    exp += (exp_prediction == true_test[r_test]).sum()

    # Individual Expert Accuracy ===  # TODO
    # expert_correct_dic[deferred_exp] += (exp_prediction == labels[i].item())
    # expert_total_dic[deferred_exp] += 1

    # Total system accuracy
    correct_sys += (exp_prediction == true_test[r_test]).sum()

    # Metrics dict ===
    metrics_dict = {"alone_classifier": alone_correct / N_test,  # clf. acc. w.r.t all samples
                    "classifier_accuracy": correct / ((~r_test).sum() + 0.00001),  # acc. only for non-deferred samples
                    "expert_accuracy": exp / (r_test.sum() + 0.00001),  # on deferred samples
                    "system_accuracy": correct_sys / N_test,  # on all samples
                    **conformal_dict}  # for conformal methods

    return metrics_dict


# Obtain deferral r ===
def get_deferral(probs, n_classes_exp, n_experts):
    r"""
    Obtain deferral vector, with 1 indicating deferred samples to expert.
    Args:
        probs:
        n_classes_exp: Number of classes + number of experts.
        n_experts: Number of experts.

    Returns:
        r: Deferral vector.
    """
    _, predicted = torch.max(probs.data, 1)
    r = (predicted >= n_classes_exp - n_experts)
    return r


# Ensemble functions ===
def get_expert_prediction(experts_pred, experts_conformal_mask, method="voting", ensemble_size=5):
    r"""
    Obtain a vector with the expert
    Args:
        experts_pred: Experts predictions.
        experts_conformal_mask: Boolean mask indicating the experts in the conformal set.
        method: Ensemble method. It can be:
            - voting: majority voting among experts.
            - random: pick random expert from conformal set.
            - last: take last expert from the conformal set. It will be the worst one.
            - ensemble: take a fixed-size ensemble from the conformal expert set. If ensemble>size_conf_set, take whole
                conformal set.
    Returns:
        ensemble_final_pred: Tensor with the final expert prediction after the appropriate method.
    """
    N = len(experts_pred)
    ensemble_exp_pred = [experts_pred[i, experts_conformal_mask[i]] for i in range(N)]

    ensemble_final_pred = []
    for ensemble_exp_pred_i in ensemble_exp_pred:

        # If no set, wrong label.
        if len(ensemble_exp_pred_i) == 0:
            pred_i = -1
        else:
            # Last ===
            if method == "last":
                pred_i = ensemble_exp_pred_i[-1]

            # Random ===
            if method == "random":
                idx = np.random.randint(len(ensemble_exp_pred_i))
                pred_i = ensemble_exp_pred_i[idx]

            # Top-k ensemble ===
            if method == "ensemble":
                pred_i = ensemble_exp_pred_i[:ensemble_size]  # top-K.

            # Majority Voting ===
            if method == "voting":
                pred_i = torch.mode(torch.tensor(ensemble_exp_pred_i))[0]
        ensemble_final_pred.append(pred_i)
    return torch.tensor(ensemble_final_pred)


def get_fixed_ensemble(experts_pred, confs_exp, ensemble_size=5):
    r"""
    Naive ensemble method where we sort the experts according to the confidences of the output model and pick the top-k
    experts, where k is the ensemble size.
    Args:
        experts_pred: Expert predictions.
        confs_exp: Confidences output from the model for the experts.
        ensemble_size: Size of the ensemble k.

    Returns:
        prediction: Final prediction as the mode of all the experts in the ensemble.
        pi[:ensemble_size]: Expert belonging to the ensemble.
    """
    # Sort and get top-K ensemble prediction
    sort, pi = confs_exp.sort(descending=True)

    # Expert prediction from ensemble
    experts_pred_final = np.array([experts_pred[i, pi[i, :ensemble_size]] for i in range(len(pi))])
    prediction = torch.mode(torch.tensor(experts_pred_final))[0]
    return prediction, pi[:ensemble_size]


# ======================================= #
# ====== Naive Conformal Inference ====== #
# ======================================= #


def get_qhat(confs, exps, true, deferral, idx_cal, n_classes, alpha=0.1):
    r"""
    Obtain conformal quantile for a multi-expert scenario.
    We add to the conformal set until ALL correct experts are included.
    Args:
        confs: Confidence values from the model.
        exps: Expert predictions.
        true: Ground truth labels.
        deferral: Deferral boolean vector indicating deferred samples to experts.
        idx_cal: indexes for the calibration values.
        n_classes: Number of classes for the dataset.
        alpha: alpha value for the calculation of the quantile in the  conformal prediction.

    Returns:
        qhat: Conformal quantile.
    """
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


def get_conformal_set(confs_exp, qhat):
    r"""
    Obtain conformal mask indicating which experts are in the conformal set, and the conformal set.
    Args:
        confs_exp: Confidences output from the model for the experts.
        qhat: Conformal quantile.

    Returns:
        conformal_mask: Boolean mask indicating the experts in the conformal set.
        prediction_set: List with the conformal sets with the final experts.
    """
    # Sort J model outputs for experts. sorted probs and sorted indexes
    sort, pi = confs_exp.sort(descending=True)
    # Conformal set mask
    conformal_mask = (sort.cumsum(dim=1) <= qhat)
    # Get last sorted index to be below Q_hat
    prediction_set = [pi[i, conformal_mask[i]] for i in range(len(pi))]
    return conformal_mask, prediction_set


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
