import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc
from conformal import utils

# === Latex Options === #
rc('font', family='serif')
rc('text', usetex=True)

# === Matplotlib Options === #
cm = plt.cm.get_cmap('tab10')
global_plot_args = {"marker": "o",
                    "markeredgecolor": "k",
                    "markersize": 10,
                    "linewidth": 8
                    }
sns.set_context("talk", font_scale=1.3)
fig_size = (7, 7)


# === Plotting functions === #
def plot_sys_acc(results, method_list, plot_args):
    r"""
    Plot system accuracy for one L2D formulation type.
    Args:
        results:
        method_list:

    Returns:

    """
    seeds_list = list(results.keys())
    exp_list = list(results[seeds_list[0]].keys())
    # Retrieve accuracies

    accuracies_dict = {method: utils.get_metric(results, seeds_list, exp_list, "system_accuracy", method)
                       for method in method_list}

    f, ax = plt.subplots(1, 1, figsize=fig_size)
    for i, (method, acc_np) in enumerate(accuracies_dict.items()):
        method_acc_mean = acc_np.mean(axis=0)
        method_acc_std = acc_np.std(axis=0)
        label = r"{}"
        ax.plot(exp_list, method_acc_mean, "-", label=label.format(method), color=cm(i), **global_plot_args)
        plt.errorbar(exp_list, method_acc_mean, yerr=method_acc_std)

    plt.xticks(exp_list, exp_list)
    plt.yticks(list(plt.yticks()[0])[::2])
    plt.ylabel(r'System Acc. ($\%$)')
    plt.xlabel(r'{}'.format(plot_args["xlabel"]))
    plt.title(r'{}'.format(plot_args["title"]))
    plt.legend(loc="best")
    plt.grid()
    f.set_tight_layout(True)
    plt.legend()
    plt.savefig(plot_args["fig_path"].format("system_accuracy"))
    return f, ax


def plot_exp_acc():  # TODO
    pass


def plot_qhat():  # TODO
    pass


def plot_coverage():  # TODO
    pass


