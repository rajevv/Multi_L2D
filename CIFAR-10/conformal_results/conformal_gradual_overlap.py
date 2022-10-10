import os
import sys

sys.path.append("../")  # append for conformal function
from conformal import conformal
from conformal.conformal_plots import plot_metric, compare_metric

# Experiment params ==============
# *** Change from here for other exps ***
experiment_name = "gradual_overlap"
experiment_args = {"n_experts": 10,
                   "n_classes": 10,
                   "ensemble_size": 5}
exp_list = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
seeds = [436, 625, 948]
model_name = "_p_out_{}"  # to include values in exp_list
# *** Change from here for other exps ***

# Ensemble methods ==============
metric_methods = ["standard",  # standard L2D
                  "last", "random", "voting",  # conformal-based
                  "ensemble"]  # basic fixed-size ensemble

# Conformal params ==============
alpha = 0.1
cal_percent = 0.8

results_path = "../results/{}/".format(experiment_name)
if not os.path.exists(results_path):
    os.makedirs(results_path)
# =================================

# =========== #
# === OvA === #
# =========== #
# Load data OvA ===
ova_path = "../ova_{}/".format(experiment_name)
path_confidence_ova = ova_path + "confidence_multiple_experts"
path_experts_ova = ova_path + "expert_predictions_multiple_experts"
path_labels_ova = ova_path + "true_label_multiple_experts"

ova_results = conformal.load_results(path_confidence_ova, path_experts_ova, path_labels_ova, model_name,
                                     seeds, exp_list, method="ova")

# Process Results ===
ova_metrics = conformal.process_conformal_results(ova_results, exp_list, experiment_args, cal_percent=cal_percent,
                                                  alpha=alpha, metric_methods=metric_methods)

# TODO
# 1. System Accuracy OK
# 2. Coverage
# 3. Avg. Conformal Set Size
# 4. Qhat
ova_fig_path = "{}_ova_".format(experiment_name)
plot_args_ova = {"xlabel": "Prob Experts",
                 "title": "CIFAR-10 OvA",
                 "fig_path": results_path + ova_fig_path + "{}.pdf"}

metrics = ["system_accuracy", "expert_accuracy", "coverage_test", "avg_set_size", "qhat"]
for met in metrics:
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
                                         model_name, seeds, exp_list, method="softmax")
# Process Results ===
softmax_metrics = conformal.process_conformal_results(softmax_results, exp_list, experiment_args,
                                                      cal_percent=cal_percent,
                                                      alpha=alpha, metric_methods=metric_methods)
softmax_fig_path = "{}_softmax_".format(experiment_name)
plot_args_softmax = {"xlabel": "Prob Experts",
                     "title": "CIFAR-10 Softmax",
                     "fig_path": results_path + softmax_fig_path + "{}.pdf"}
metrics = ["system_accuracy", "expert_accuracy", "coverage_test", "avg_set_size", "qhat"]
for met in metrics:
    f, ax = plot_metric(softmax_metrics, metric_methods, met, plot_args_softmax)


# ======================= #
# === Compare results === #
# ======================= #
compare_fig_path = "{}_".format(experiment_name)
plot_args = {"xlabel": "Prob Experts",
             "title": "CIFAR-10",
             "fig_path": results_path + compare_fig_path + "{}.pdf"}
metrics = ["system_accuracy", "expert_accuracy", "coverage_test", "avg_set_size", "qhat"]
for met in metrics:
    f, ax = compare_metric(ova_metrics, softmax_metrics, metric_methods, met, plot_args)

