import os
import sys

sys.path.append("../")  # append for conformal function
from conformal import conformal
from conformal.conformal_plots import plot_sys_acc

# Process Results ===
experiment_args = {"n_experts": 10,
                   "n_classes": 10,
                   "ensemble_size": 5}

metric_methods = ["standard",  # standard L2D
                  "last", "random", "voting",  # conformal-based
                  "ensemble"]  # basic fixed-size ensemble

results_path = "../results/gradual_overlap/"

if not os.path.exists(results_path):
    os.makedirs(results_path)


# =========== #
# === OvA === #
# =========== #
# Load data OvA ===
ova_path = "../ova_gradual_overlap/"
path_confidence_ova = ova_path + "confidence_multiple_experts"
path_experts_ova = ova_path + "expert_predictions_multiple_experts"
path_labels_ova = ova_path + "true_label_multiple_experts"
model_name = "_p_out_{}"  # to include values in exp_list
exp_list = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
seeds = [436, 625, 948]

ova_results = conformal.load_results(path_confidence_ova, path_experts_ova, path_labels_ova, model_name,
                                     seeds, exp_list, method="ova")

# Process Results ===
ova_metrics = conformal.process_conformal_results(ova_results, exp_list, experiment_args, cal_percent=0.8,
                                                  alpha=0.1, metric_methods=metric_methods)

plot_args = {"xlabel": "Prob Experts",
             "title": "CIFAR-10 OvA",
             "fig_path": results_path + "gradual_overlap_ova_{}.pdf"}
f, ax = plot_sys_acc(ova_metrics, metric_methods, plot_args)

# =============== #
# === Softmax === #
# =============== #
# Load data Softmax ===
softmax_path = "../softmax_gradual_overlap/"
path_confidence_softmax = softmax_path + "confidence_multiple_experts"
path_experts_softmax = softmax_path + "expert_predictions_multiple_experts"
path_labels_softmax = softmax_path + "true_label_multiple_experts"
model_name = "_p_out_{}"  # to include values in exp_list
exp_list = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
seeds = [436, 625, 948]

softmax_results = conformal.load_results(path_confidence_softmax, path_experts_softmax, path_labels_softmax,
                                         model_name, seeds, exp_list, method="softmax")
# Process Results ===
softmax_metrics = conformal.process_conformal_results(softmax_results, exp_list, experiment_args, cal_percent=0.8,
                                                      alpha=0.1, metric_methods=metric_methods)

# ==================== #
# === Plot results === #
# ==================== #
