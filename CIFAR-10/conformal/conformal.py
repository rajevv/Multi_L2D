import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy import stats


def get_expert_prediction(experts, prediction_set_i, method="voting"):
    ensemble_expert_pred_i = np.array(experts)[prediction_set_i][:, i]
    if method == "voting":
        exp_prediction = stats.mode(ensemble_expert_pred_i).mode if len(ensemble_expert_pred_i) != 0 else []

    if method == "last":
        exp_prediction = ensemble_expert_pred_i[-1] if len(ensemble_expert_pred_i) != 0 else []

    if method == "random":
        idx = np.random.randint(len(ensemble_expert_pred_i)) if len(ensemble_expert_pred_i) != 0 else -1
        exp_prediction = ensemble_expert_pred_i[idx] if idx != -1 else []

    return exp_prediction



