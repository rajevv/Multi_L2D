import os

import matplotlib.pyplot as plt
import numpy as np
from tueplots import figsizes, fontsizes, fonts, axes

from conformal.utils import load_dict_txt
from plots.utils_plots import set_aistats2023_style

paper_results_path = "paper/"
if not os.path.exists(paper_results_path):
    os.makedirs(paper_results_path)


# Experiment 1: Multi-expert accuracies and calibration


def experiment1():
    plt.rcParams.update(figsizes.aistats2023_half(nrows=1, ncols=2, height_to_width_ratio=1))

    plt.rcParams.update(fonts.aistats2022_tex(family="serif"))
    plt.rcParams.update(fontsizes.aistats2023(default_smaller=2.5))
    plt.rcParams.update(axes.lines(line_base_ratio=5))
    plt.rcParams.update(axes.grid(grid_alpha=0.5))  # custom grid. alpha=0-1, for transparency
    plt.rcParams.update({"lines.markersize": 3,
                         "xtick.labelsize": 5})

    f, ax = plt.subplots(nrows=1, ncols=2, sharex=True)

    ax[0] = increase_experts_accuracy(f, ax[0])
    ax[1] = increase_experts_calibration(f, ax[1])  # TODO Calibration

    ax[0].legend(loc="best")
    # f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_experts_sys_accuracy.pdf")
    return f, axes


def increase_experts_accuracy(f, ax):
    exp_list = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
    exp_path = "increase_experts/naive/"
    ova_accuracies = load_dict_txt(exp_path + "increase_experts_expert_accuracy_ova.txt")
    softmax_accuracies = load_dict_txt(exp_path + "increase_experts_expert_accuracy_softmax.txt")

    ova_accuracies = np.array(ova_accuracies["standard"])
    softmax_accuracies = np.array(softmax_accuracies["standard"])

    # OvA
    ova_acc_mean = ova_accuracies.mean(axis=0)
    ova_acc_std = ova_accuracies.std(axis=0)
    ax.errorbar(exp_list, ova_acc_mean, yerr=ova_acc_std,
                alpha=0.75, label="OvA", marker="o")

    # Softmax
    softmax_acc_mean = softmax_accuracies.mean(axis=0)
    softmax_acc_std = softmax_accuracies.std(axis=0)
    ax.errorbar(exp_list, softmax_acc_mean, yerr=softmax_acc_std,
                alpha=0.75, label="Softmax", marker="s")
    ax.set_xticks(exp_list, exp_list)
    # plt.yticks(list(plt.yticks()[0])[::2])
    ax.set_ylabel(r"System Accuracy $(\%)$")
    ax.set_xlabel(r"Number of Experts")
    ax.grid()
    return ax


def increase_experts_calibration(f, ax):  # TODO
    exp_list = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
    exp_path = "increase_experts/naive/"
    ova_accuracies = load_dict_txt(exp_path + "increase_experts_expert_accuracy_ova.txt")
    softmax_accuracies = load_dict_txt(exp_path + "increase_experts_expert_accuracy_softmax.txt")

    ova_accuracies = np.array(ova_accuracies["standard"])
    softmax_accuracies = np.array(softmax_accuracies["standard"])

    # OvA
    ova_acc_mean = ova_accuracies.mean(axis=0)
    ova_acc_std = ova_accuracies.std(axis=0)
    # ax.errorbar(exp_list, ova_acc_mean, yerr=ova_acc_std,
    #             alpha=0.75, label="OvA", marker="o")

    # Softmax
    softmax_acc_mean = softmax_accuracies.mean(axis=0)
    softmax_acc_std = softmax_accuracies.std(axis=0)
    # ax.errorbar(exp_list, softmax_acc_mean, yerr=softmax_acc_std,
    #             alpha=0.75, label="Softmax", marker="s")
    ax.set_xticks(exp_list, exp_list)
    # plt.yticks(list(plt.yticks()[0])[::2])
    ax.set_ylabel(r"Average ECE $(\%)$")
    ax.set_xlabel(r"Number of Experts")
    ax.grid()
    return ax


def experiment2():
    plt.rcParams.update(figsizes.aistats2023_half(nrows=1, ncols=2, height_to_width_ratio=1))

    plt.rcParams.update(fonts.aistats2022_tex(family="serif"))
    plt.rcParams.update(fontsizes.aistats2023(default_smaller=2.5))
    plt.rcParams.update(axes.lines(line_base_ratio=5))
    plt.rcParams.update(axes.grid(grid_alpha=0.5))  # custom grid. alpha=0-1, for transparency
    plt.rcParams.update({"lines.markersize": 3,
                         "xtick.labelsize": 5})

    f, ax = plt.subplots(nrows=1, ncols=2, sharex=True)


if __name__ == '__main__':

    set_aistats2023_style()
    # Experiment 1: Multi-expert accuracies and calibration
    experiment1()
    experiment2()
