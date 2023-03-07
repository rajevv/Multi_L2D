import os
import sys

sys.path.append("../..")  # append for conformal function
from lib.conformal import conformal, utils
from lib.conformal.conformal_plots import plot_metric, compare_metric

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Experiment params ==============
# *** Change from here for other exps ***

conformal_type = "regularized"

experiment_name = "increase_oracle_v2"
experiment_args = {"n_experts": 10,
                   "n_classes": 10,
                   "ensemble_size": 5}

exp_list = [0, 1, 2, 3, 4, 5, 7, 9]
seeds = [436, 625, 948]
seed_name = "seed_{}"
model_name = "_k_{}"  # to include values in exp_list

# Plot params
xlabel = "Oracles"

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
cal_percent = 0.8

results_path = "../results/{}/{}/".format(experiment_name, conformal_type)
if not os.path.exists(results_path):
    os.makedirs(results_path)

ova_fig_path = "{}_ova_".format(experiment_name)
plot_args_ova = {"xlabel": xlabel,
                 "title": "CIFAR-10 OvA",
                 "fig_path": results_path + ova_fig_path + "{}.pdf"}

softmax_fig_path = "{}_softmax_".format(experiment_name)
plot_args_softmax = {"xlabel": xlabel,
                     "title": "CIFAR-10 Softmax",
                     "fig_path": results_path + softmax_fig_path + "{}.pdf"}

compare_fig_path = "{}_".format(experiment_name)
plot_args = {"xlabel": xlabel,
             "title": "CIFAR-10",
             "fig_path": results_path + compare_fig_path + "{}.pdf",
             "metric_path": results_path + compare_fig_path + "{}_{}.txt"}
# =================================
# FROM HERE, SAME FOR ALL EXPERIMENTS
# =================================
# =========== #
# === OvA === #
# =========== #
# Load data OvA ===
ova_path = "../ova_{}/".format(experiment_name)
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

# set_sizes = np.zeros((experiment_args["n_experts"], len(exp_list)))
# for seed in seeds:
#     for i, exp in enumerate(exp_list):
#         set_size = np.array(ova_metrics[seed][exp]["voting"]["set_sizes"][0])
#         counter = Counter(set_size)
#         idx = list(counter.keys())
#         idx = [j - 1 for j in idx]
#         counts = list(counter.values())
#         set_sizes[idx, i] += counts
#
# max_col = np.max(set_sizes, 0)
# renorm_by_col = np.tile(max_col, (experiment_args["n_experts"], 1))
# set_sizes_renorm = set_sizes/renorm_by_col
# f, ax = plt.subplots(1, 1)
# sns.heatmap(set_sizes_renorm, cmap="viridis")
# ax.set_xticks(np.array(exp_list)+0.5, exp_list)
# ax.set_yticks(np.arange(1,11) - 0.5, np.arange(1,11))
# ax.set_ylabel("Set sizes")
# ax.set_xlabel("Oracles")
# plt.title("{}".format(conformal_type))

for met in metrics:
    utils.save_metric(ova_metrics, met, metric_methods, plot_args["metric_path"].format(met, "ova"))
    f, ax = plot_metric(ova_metrics, metric_methods, met, plot_args_ova)

# =============== #
# === Softmax === #
# =============== #
# Load data Softmax ===
softmax_path = "../softmax_{}/".format(experiment_name)
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
