import os
import sys

sys.path.append("../")  # append for conformal function
from conformal import conformal, utils
from conformal.conformal_plots import plot_metric, compare_metric
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Experiment params ==============
# *** Change from here for other exps ***

conformal_type = "naive"

path_name = "increase_experts_select"
experiment_name = "increase_experts_hatespeech"
experiment_args = {"n_experts": 10,
                   "n_classes": 7,
                   "ensemble_size": 5}

exp_list = [1, 2, 4, 6, 8, 12, 16]
seeds = [948, 625, 436]
seed_name = "_seed_{}"
model_name = "_{}_experts"  # to include values in exp_list

# Plot params
xlabel = "Number of Experts"

# Experiment params ==============

# Ensemble methods ==============
metric_methods = ["standard",  # standard L2D
                  "last", "random", "voting",  # conformal-based
                  "ensemble"]  # basic fixed-size ensemble

# Metric depending on conformal type
metrics = ["system_accuracy", "expert_accuracy", "coverage_test", "avg_set_size", "qhat"]
if conformal_type == "regularized":
    metrics = ["system_accuracy", "expert_accuracy", "coverage_test", "avg_set_size", "lamhat"]

# Conformal params ==============
alpha = 0.1
cal_percent = 0.9

results_path = "../results/{}/{}/".format(experiment_name, conformal_type)
if not os.path.exists(results_path):
    os.makedirs(results_path)

ova_fig_path = "{}_ova_".format(experiment_name)
plot_args_ova = {"xlabel": xlabel,
                 "title": "HAM10000 OvA",
                 "fig_path": results_path + ova_fig_path + "{}.pdf"}

softmax_fig_path = "{}_softmax_".format(experiment_name)
plot_args_softmax = {"xlabel": xlabel,
                     "title": "HAM10000 Softmax",
                     "fig_path": results_path + softmax_fig_path + "{}.pdf"}

compare_fig_path = "{}_".format(experiment_name)
plot_args = {"xlabel": xlabel,
             "title": "HAM10000",
             "fig_path": results_path + compare_fig_path + "{}.pdf",
             "metric_path": results_path + compare_fig_path + "{}_{}.txt"}
# =================================
# FROM HERE, SAME FOR ALL EXPERIMENTS
# =================================
# =========== #
# === OvA === #
# =========== #
# Load data OvA ===
ova_path = "../../hatespeech/ova_{}/".format(path_name)
path_confidence_ova = ova_path + "confidence_multiple_experts"
path_experts_ova = ova_path + "expert_predictions_multiple_experts"
path_labels_ova = ova_path + "true_label_multiple_experts"

ova_results = conformal.load_results(path_confidence_ova, path_experts_ova, path_labels_ova, model_name, seed_name,
                                     seeds, exp_list, method="ova")

# Process Results ===
ova_metrics = conformal.process_conformal_results(ova_results, exp_list, experiment_args,
                                                  cal_percent=cal_percent,
                                                  alpha=alpha, metric_methods=metric_methods,
                                                  conformal_type=conformal_type)
for met in metrics:
    utils.save_metric(ova_metrics, met, metric_methods, plot_args["metric_path"].format(met, "ova"))
    f, ax = plot_metric(ova_metrics, metric_methods, met, plot_args_ova)

# =============== #
# === Softmax === #
# =============== #
# Load data Softmax ===
softmax_path = "../../hatespeech/softmax_{}/".format(path_name)
path_confidence_softmax = softmax_path + "confidence_multiple_experts"
path_experts_softmax = softmax_path + "expert_predictions_multiple_experts"
path_labels_softmax = softmax_path + "true_label_multiple_experts"

softmax_results = conformal.load_results(path_confidence_softmax, path_experts_softmax, path_labels_softmax,
                                         model_name, seed_name, seeds, exp_list, method="softmax")
# Process Results ===
softmax_metrics = conformal.process_conformal_results(softmax_results, exp_list, experiment_args,
                                                      cal_percent=cal_percent,
                                                      alpha=alpha, metric_methods=metric_methods,
                                                      conformal_type=conformal_type)
for met in metrics:
    utils.save_metric(softmax_metrics, met, metric_methods, plot_args["metric_path"].format(met, "softmax"))
    f, ax = plot_metric(softmax_metrics, metric_methods, met, plot_args_softmax)

# ======================= #
# === Compare results === #
# ======================= #
for met in metrics:
    f, ax = compare_metric(ova_metrics, softmax_metrics, metric_methods, met, plot_args)
