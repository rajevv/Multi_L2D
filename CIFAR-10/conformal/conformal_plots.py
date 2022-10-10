import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc
from conformal import utils
from plots import utils_plots

Y_TITLES = {"system_accuracy": r"System Acc. ($\%$)",
            "expert_accuracy": r"Expert Acc. ($\%$)",
            "coverage_test": r"Model Coverage. ($\%$)",
            "avg_set_size": r"Avg. Set Size",
            "qhat": r"$\hat{Q}$"}


# === Plotting functions === #
def plot_metric(results, method_list, metric, plot_args):
    r"""
    Plot system accuracy for one L2D formulation type.
    Args:
        results:
        method_list:

    Returns:

    """
    # Set style
    utils_plots.set_aistats2023_style()
    method_list_cp = method_list.copy()
    # Lists
    seeds_list = list(results.keys())
    exp_list = list(results[seeds_list[0]].keys())

    # Retrieve metric
    # Sanity check
    if metric == "avg_set_size":
        method_list_cp.remove("standard")
        method_list_cp.remove("ensemble")

    metric_dict = {method: utils.get_metric(results, seeds_list, exp_list, metric, method)
                       for method in method_list_cp}

    f, ax = plt.subplots(1, 1)
    for i, (method, acc_np) in enumerate(metric_dict.items()):
        method_metric_mean = acc_np.mean(axis=0)
        method_metric_std = acc_np.std(axis=0)
        label = r"{}"
        plt.errorbar(exp_list, method_metric_mean, yerr=method_metric_std,
                     alpha=0.75, label=label.format(method), marker="o")

    # plt.xticks(np.linspace(min(exp_list), max(exp_list), len(exp_list)), exp_list)  # TODO: Don't use
    plt.xticks(exp_list, exp_list)

    plt.yticks(list(plt.yticks()[0])[::2])
    plt.ylabel(Y_TITLES[metric])
    plt.xlabel(r'{}'.format(plot_args["xlabel"]))
    plt.title(r'{}'.format(plot_args["title"]))
    plt.legend(loc="best")
    plt.grid()
    f.set_tight_layout(True)
    plt.legend()
    plt.savefig(plot_args["fig_path"].format(metric))
    return f, ax


def compare_metric(results_ova, results_softmax, method_list, metric, plot_args):
    r"""
    Plot system accuracy for one L2D formulation type.
    Args:
        results:
        method_list:

    Returns:

    """
    # Set style
    utils_plots.set_aistats2023_style()
    cmap = sns.color_palette()
    method_list_cp = method_list.copy()

    # Lists
    seeds_list = list(results_ova.keys())
    exp_list = list(results_ova[seeds_list[0]].keys())

    # Retrieve metric
    # Sanity check
    if metric == "avg_set_size":
        method_list_cp.remove("standard")
        method_list_cp.remove("ensemble")

    metric_dict_ova = {method: utils.get_metric(results_ova, seeds_list, exp_list, metric, method)
                       for method in method_list_cp}
    utils.save_dict_as_txt(metric_dict_ova, "ova_{}.txt".format(metric))

    metric_dict_softmax = {method: utils.get_metric(results_softmax, seeds_list, exp_list, metric, method)
                          for method in method_list_cp}
    utils.save_dict_as_txt(metric_dict_softmax, "softmax_{}.txt".format(metric))

    f, ax = plt.subplots(1, 1)
    for i, method in enumerate(method_list_cp):
        # Get method accuracies
        metric_np_ova = metric_dict_ova[method]
        metric_np_softmax = metric_dict_softmax[method]

        # Get mean and std
        metric_ova_mean = metric_np_ova.mean(axis=0)
        metric_ova_std = metric_np_ova.std(axis=0)
        metric_softmax_mean = metric_np_softmax.mean(axis=0)
        metric_softmax_std = metric_np_softmax.std(axis=0)

        # OvA ===
        ova_label = r"{}"
        plt.errorbar(exp_list, metric_ova_mean, yerr=metric_ova_std, color=cmap[i],
                     alpha=0.75, label=ova_label.format(method), marker="o", linestyle="-")
        # Softmax ===
        softmax_label = r"{}"
        plt.errorbar(exp_list, metric_softmax_mean, yerr=metric_softmax_std, color=cmap[i],
                     alpha=0.75,  # label=softmax_label.format(method),
                     marker="s", linestyle="--")

    plt.xticks(exp_list, exp_list)
    plt.yticks(list(plt.yticks()[0])[::2])
    plt.ylabel(Y_TITLES[metric])
    plt.xlabel(r'{}'.format(plot_args["xlabel"]))
    plt.title(r'{}'.format(plot_args["title"]))
    plt.legend(loc="best")
    plt.grid()
    f.set_tight_layout(True)
    plt.legend()
    plt.savefig(plot_args["fig_path"].format(metric))
    return f, ax


def plot_exp_acc():  # TODO
    pass


def plot_qhat():  # TODO
    pass


def plot_coverage():  # TODO
    pass
