import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tueplots import figsizes, axes

from conformal.utils import load_dict_txt
from plots.utils_plots import set_aistats2023_style

paper_results_path = "paper/"
if not os.path.exists(paper_results_path):
    os.makedirs(paper_results_path)


# Experiment 1: Multi-expert accuracies and calibration


def experiment1():
    def increase_experts_accuracy(f, ax):
        cmap = sns.color_palette()
        ova_args = {"color": cmap[0],
                    "marker": "o"}
        softmax_args = {"color": cmap[1],
                        "marker": "s"}
        exp_list = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
        exp_path = "increase_experts/naive/"
        ova_accuracies = load_dict_txt(exp_path + "increase_experts_expert_accuracy_ova.txt")
        softmax_accuracies = load_dict_txt(exp_path + "increase_experts_expert_accuracy_softmax.txt")

        ova_accuracies = np.array(ova_accuracies["standard"]) * 100
        softmax_accuracies = np.array(softmax_accuracies["standard"]) * 100

        # OvA
        ova_acc_mean = ova_accuracies.mean(axis=0)
        ova_acc_std = ova_accuracies.std(axis=0)
        ax.errorbar(exp_list, ova_acc_mean, yerr=ova_acc_std,
                    alpha=0.75, label="OvA", **ova_args)

        # Softmax
        softmax_acc_mean = softmax_accuracies.mean(axis=0)
        softmax_acc_std = softmax_accuracies.std(axis=0)
        ax.errorbar(exp_list, softmax_acc_mean, yerr=softmax_acc_std,
                    alpha=0.75, label="Softmax", **softmax_args)
        ax.set_xticks(exp_list, exp_list)
        plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Number of Experts")
        ax.grid()

        ova_leg = mlines.Line2D([], [], linestyle='-', label='OvA', **ova_args)
        softmax_leg = mlines.Line2D([], [], linestyle='-', label='Softmax', **softmax_args)
        ax.legend(handles=[ova_leg, softmax_leg], loc="best")
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
        ax.errorbar(exp_list, ova_acc_mean, yerr=ova_acc_std,
                    alpha=0.75, label="OvA", marker="o")

        # Softmax
        softmax_acc_mean = softmax_accuracies.mean(axis=0)
        softmax_acc_std = softmax_accuracies.std(axis=0)
        ax.errorbar(exp_list, softmax_acc_mean, yerr=softmax_acc_std,
                    alpha=0.75, label="Softmax", marker="s")
        ax.set_xticks(exp_list, exp_list)
        # plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average ECE $(\%)$")
        ax.set_xlabel(r"Number of Experts")
        ax.grid()
        return ax

    # # Figures aesthetics ===
    # plt.rcParams.update(figsizes.aistats2023_half(nrows=1, ncols=2, height_to_width_ratio=1))
    #
    # plt.rcParams.update(fonts.aistats2022_tex(family="serif"))
    # plt.rcParams.update(fontsizes.aistats2023(default_smaller=2.5))
    plt.rcParams.update(axes.lines(line_base_ratio=7))
    # plt.rcParams.update(axes.grid(grid_alpha=0.5))  # custom grid. alpha=0-1, for transparency
    # plt.rcParams.update({"lines.markersize": 3,
    #                      "xtick.labelsize": 5})
    #
    # f, ax = plt.subplots(nrows=1, ncols=2, sharex=True)
    #
    # ax[0] = increase_experts_accuracy(f, ax[0])
    # ax[1] = increase_experts_calibration(f, ax[1])  # TODO Calibration
    #
    # ax[0].legend(loc="best")
    # # f.set_tight_layout(True)
    # plt.show()
    # f.savefig(paper_results_path + "increase_experts_sys_accuracy.pdf")

    # Figures aesthetics ===
    plt.rcParams.update(figsizes.aistats2023_half(height_to_width_ratio=1))

    # plt.rcParams.update(fonts.aistats2022_tex(family="serif"))
    # plt.rcParams.update(fontsizes.aistats2023(default_smaller=))
    # plt.rcParams.update(axes.lines(line_base_ratio=5))
    # plt.rcParams.update(axes.grid(grid_alpha=0.5))  # custom grid. alpha=0-1, for transparency
    # plt.rcParams.update({"lines.markersize": 3,
    #                      "xtick.labelsize": 5})
    f, ax = plt.subplots(1, 1)

    ax = increase_experts_accuracy(f, ax)

    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_experts_sys_accuracy.pdf")
    return f, ax


def experiment2():
    # plt.rcParams.update(figsizes.aistats2023_half(height_to_width_ratio=1))
    #
    # plt.rcParams.update(fonts.aistats2022_tex(family="serif"))
    # plt.rcParams.update(fontsizes.aistats2023())
    # plt.rcParams.update(axes.lines(line_base_ratio=5))
    # plt.rcParams.update(axes.grid(grid_alpha=0.5))  # custom grid. alpha=0-1, for transparency
    # plt.rcParams.update({"lines.markersize": 3,
    #                      "xtick.labelsize": 5})

    f, ax = plt.subplots(1, 1)
    ax = plot_avg_set_size(f, ax)

    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "avg_set_size_nonrandomized.pdf")
    return f, axes


def plot_avg_set_size(f, ax):
    cmap = sns.color_palette()
    exp_list = [1, 3, 5, 7, 9]
    ova_args = {"color": cmap[0],
                "marker": "o"}
    softmax_args = {"color": cmap[1],
                    "marker": "s"}
    # Naive ===
    exp_path_naive = "increase_oracle_v2/naive/"
    ova_setsize_naive = load_dict_txt(exp_path_naive + "increase_oracle_v2_avg_set_size_ova.txt")
    ova_setsize_naive = np.array(ova_setsize_naive["voting"])
    softmax_setsize_naive = load_dict_txt(exp_path_naive + "increase_oracle_v2_avg_set_size_softmax.txt")
    softmax_setsize_naive = np.array(softmax_setsize_naive["voting"])

    # Regularized ===
    exp_path_reg = "increase_oracle_v2/regularized/"
    ova_setsize_reg = load_dict_txt(exp_path_reg + "increase_oracle_v2_avg_set_size_ova.txt")
    ova_setsize_reg = np.array(ova_setsize_reg["voting"])
    softmax_setsize_reg = load_dict_txt(exp_path_reg + "increase_oracle_v2_avg_set_size_softmax.txt")
    softmax_setsize_reg = np.array(softmax_setsize_reg["voting"])

    # OvA ===
    # naive
    ova_setsize_naive_mean = ova_setsize_naive.mean(axis=0)
    ova_setsize_naive_std = ova_setsize_naive.std(axis=0)
    ax.errorbar(exp_list, ova_setsize_naive_mean, yerr=ova_setsize_naive_std, linestyle="-", alpha=0.75,
                label="OvA Naive Conformal", **ova_args)
    ova_naive_leg = mlines.Line2D([], [], linestyle='-', label="OvA Conformal", **ova_args)

    # Softmax ===
    # naive
    softmax_setsize_naive_mean = softmax_setsize_naive.mean(axis=0)
    softmax_setsize_naive_std = softmax_setsize_naive.std(axis=0)
    ax.errorbar(exp_list, softmax_setsize_naive_mean, yerr=softmax_setsize_naive_std, linestyle="-", alpha=0.75,
                label="Softmax Naive Conformal", **softmax_args)
    softmax_naive_leg = mlines.Line2D([], [], linestyle='-', label="Softmax Conformal", **softmax_args)

    # OvA ===
    # reg
    ova_setsize_reg_mean = ova_setsize_reg.mean(axis=0)
    ova_setsize_reg_std = ova_setsize_reg.std(axis=0)
    ax.errorbar(exp_list, ova_setsize_reg_mean, yerr=ova_setsize_reg_std, linestyle="--", alpha=0.75,
                label="OvA Reg. Conformal", **ova_args)
    ova_reg_leg = mlines.Line2D([], [], linestyle=(0.5,(1,3)), label="OvA Reg. Conformal", **ova_args)

    # Softmax ===
    # reg
    softmax_setsize_reg_mean = softmax_setsize_reg.mean(axis=0)
    softmax_setsize_reg_std = softmax_setsize_reg.std(axis=0)
    ax.errorbar(exp_list, softmax_setsize_reg_mean, yerr=softmax_setsize_reg_std, linestyle="--", alpha=0.75,
                label="Softmax Reg. Conformal", **softmax_args)
    softmax_reg_leg = mlines.Line2D([], [], linestyle=(0.5,(1,3)), label="Softmax Reg. Conformal", **softmax_args)

    ax.set_xticks(exp_list, exp_list)
    # plt.yticks(list(plt.yticks()[0])[::2])
    ax.set_ylabel(r"Average Set Size")
    ax.set_xlabel(r"Oracles")
    ax.grid()

    ax.legend(handles=[ova_naive_leg, softmax_naive_leg, ova_reg_leg, softmax_reg_leg],
              loc="best", fontsize=7)
              # prop = {"size"})

    return ax


if __name__ == '__main__':
    set_aistats2023_style()
    # Experiment 1: Multi-expert accuracies and calibration
    experiment1()
    experiment2()
