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
    cmap = sns.color_palette()
    ova_args = {"color": cmap[0],
                "marker": "o"}
    softmax_args = {"color": cmap[1],
                    "marker": "s"}

    def increase_experts_accuracy(f, ax):
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
        exp_list = [1, 2, 4, 6, 8, 10, 14, 18]

        softmax_ece_mean = np.array(
            [0.05178541, 0.03142016, 0.02538094, 0.03436143, 0.0277727, 0.02835129, 0.02579127, 0.02650053]) * 100
        softmax_ece_std = np.array(
            [0.01946501, 0.00332035, 0.00089578, 0.00231578, 0.00197512, 0.0007479, 0.00126423, 0.00166975]) * 100
        ova_ece_mean = np.array(
            [0.02493655, 0.03006555, 0.02682439, 0.02946591, 0.02868607, 0.0301291, 0.02640719, 0.02986932]) * 100
        ova_ece_std = np.array(
            [0.00375583, 0.00056767, 0.00241768, 0.00096937, 0.00122256, 0.00015699, 0.00201291, 0.00055201]) * 100

        # OvA
        ax.errorbar(exp_list, ova_ece_mean, yerr=ova_ece_std,
                    alpha=0.75, label="OvA", marker="o")
        # Softmax
        ax.errorbar(exp_list, softmax_ece_mean, yerr=softmax_ece_std,
                    alpha=0.75, label="Softmax", marker="s")

        ax.set_xticks(exp_list, exp_list)
        # plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average ECE $(\%)$")
        ax.set_xlabel(r"Number of Experts")
        ax.grid()

        ova_leg = mlines.Line2D([], [], linestyle='-', label='OvA', **ova_args)
        softmax_leg = mlines.Line2D([], [], linestyle='-', label='Softmax', **softmax_args)
        ax.legend(handles=[ova_leg, softmax_leg], loc="best")
        return ax

    def increase_confidence_calibration_average(f, ax):  # TODO
        exp_list = [0.2, 0.4, 0.6, 0.8, 0.95]

        softmax_ece_mean = np.array(
            [0.03834392, 0.04669855, 0.0517783, 0.0499814, 0.0400186]) * 100
        softmax_ece_std = np.array(
            [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 2.6341782e-09, 0.0000000e+00]) * 100
        ova_ece_mean = np.array(
            [0.01724488, 0.02530832, 0.02194638, 0.02296229, 0.0179429]) * 100
        ova_ece_std = np.array(
            [0., 0., 0., 0., 0.]) * 100

        # OvA
        ax.errorbar(exp_list, ova_ece_mean, yerr=ova_ece_std,
                    alpha=0.75, label="OvA", marker="o")
        # Softmax
        ax.errorbar(exp_list, softmax_ece_mean, yerr=softmax_ece_std,
                    alpha=0.75, label="Softmax", marker="s")

        ax.set_xticks(exp_list, exp_list)
        # plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average ECE $(\%)$")
        ax.set_xlabel(r"Probability of Experts")
        ax.grid()

        ova_leg = mlines.Line2D([], [], linestyle='-', label='OvA', **ova_args)
        softmax_leg = mlines.Line2D([], [], linestyle='-', label='Softmax', **softmax_args)
        ax.legend(handles=[ova_leg, softmax_leg], loc="best")
        return ax

    def increase_confidence_calibration_random(f, ax):  # TODO
        exp_list = [0.2, 0.4, 0.6, 0.8, 0.95]

        softmax_ece_mean = np.array(
            [0.01336952, 0.02237696, 0.02346867, 0.01674813, 0.02441374]) * 100
        softmax_ece_std = np.array(
            [6.5854455e-10, 0.0000000e+00, 1.3170891e-09, 0.0000000e+00, 0.0000000e+00]) * 100
        ova_ece_mean = np.array(
            [0.00998215, 0.01230554, 0.01088819, 0.00944444, 0.00641519]) * 100
        ova_ece_std = np.array(
            [0., 0., 0., 0., 0.]) * 100

        # OvA
        ax.errorbar(exp_list, ova_ece_mean, yerr=ova_ece_std,
                    alpha=0.75, label="OvA", marker="o")
        # Softmax
        ax.errorbar(exp_list, softmax_ece_mean, yerr=softmax_ece_std,
                    alpha=0.75, label="Softmax", marker="s")

        ax.set_xticks(exp_list, exp_list)
        # plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average ECE $(\%)$")
        ax.set_xlabel(r"Probability of Experts")
        ax.grid()

        ova_leg = mlines.Line2D([], [], linestyle='-', label='OvA', **ova_args)
        softmax_leg = mlines.Line2D([], [], linestyle='-', label='Softmax', **softmax_args)
        ax.legend(handles=[ova_leg, softmax_leg], loc="best")
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

    f, ax = plt.subplots(1, 1)
    ax = increase_experts_calibration(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_experts_calibration.pdf")

    f, ax = plt.subplots(1, 1)
    ax = increase_confidence_calibration_average(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_confidence_calibration_average.pdf")

    f, ax = plt.subplots(1, 1)
    ax = increase_confidence_calibration_random(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_confidence_calibration_randomexpert.pdf")

    return f, ax


def experiment2():
    cmap = sns.color_palette()
    exp_list = [1, 3, 5, 7, 9]
    ova_args = {"color": cmap[0],
                "marker": "o"}
    softmax_args = {"color": cmap[1],
                    "marker": "s"}

    # Naive ===
    def plot_avg_set_size(f, ax):
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
        ova_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA Reg. Conformal", **ova_args)

        # Softmax ===
        # reg
        softmax_setsize_reg_mean = softmax_setsize_reg.mean(axis=0)
        softmax_setsize_reg_std = softmax_setsize_reg.std(axis=0)
        ax.errorbar(exp_list, softmax_setsize_reg_mean, yerr=softmax_setsize_reg_std, linestyle="--", alpha=0.75,
                    label="Softmax Reg. Conformal", **softmax_args)
        softmax_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax Reg. Conformal", **softmax_args)

        ax.set_xticks(exp_list, exp_list)
        # plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average Set Size")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(handles=[ova_naive_leg, softmax_naive_leg, ova_reg_leg, softmax_reg_leg],
                  loc="best", fontsize=7)
        # prop = {"size"})

        return ax

    def plot_sys_acc_naive(f, ax):
        # Naive ===
        exp_path = "increase_oracle_v2/naive/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_oracle_v2_system_accuracy_ova.txt")
        ova_sys_acc_voting = np.array(ova_sys_acc["voting"])
        ova_sys_acc_ensem = np.array(ova_sys_acc["ensemble"])

        softmax_sys_acc = load_dict_txt(exp_path + "increase_oracle_v2_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = np.array(softmax_sys_acc["voting"])
        softmax_sys_acc_ensem = np.array(softmax_sys_acc["ensemble"])

        # # Regularized ===
        # exp_path_reg = "increase_oracle_v2/regularized/"
        # ova_setsize_reg = load_dict_txt(exp_path_reg + "increase_oracle_v2_avg_set_size_ova.txt")
        # ova_setsize_reg = np.array(ova_setsize_reg["voting"])
        # softmax_setsize_reg = load_dict_txt(exp_path_reg + "increase_oracle_v2_avg_set_size_softmax.txt")
        # softmax_setsize_reg = np.array(softmax_setsize_reg["voting"])

        # OvA ===
        # naive
        ova_sys_acc_voting_mean = ova_sys_acc_voting.mean(axis=0)
        ova_sys_acc_voting_std = ova_sys_acc_voting.std(axis=0)
        ax.errorbar(exp_list, ova_sys_acc_voting_mean, yerr=ova_sys_acc_voting_std, linestyle="-", alpha=0.75,
                    **ova_args)
        ova_sys_acc_vot_leg = mlines.Line2D([], [], linestyle='-', label="OvA Conformal", **ova_args)

        ova_sys_acc_ensem_mean = ova_sys_acc_ensem.mean(axis=0)
        ova_sys_acc_ensem_std = ova_sys_acc_ensem.std(axis=0)
        ax.errorbar(exp_list, ova_sys_acc_ensem_mean, yerr=ova_sys_acc_ensem_std, linestyle="--", alpha=0.75,
                    **ova_args)
        ova_sys_acc_ensem_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA Fixed-Size ($k=5$)",
                                              **ova_args)

        # Softmax ===
        # naive
        softmax_sys_acc_voting_mean = softmax_sys_acc_voting.mean(axis=0)
        softmax_sys_acc_voting_std = softmax_sys_acc_voting.std(axis=0)
        ax.errorbar(exp_list, softmax_sys_acc_voting_mean, yerr=softmax_sys_acc_voting_std, linestyle="-", alpha=0.75,
                    **softmax_args)
        softmax_sys_acc_vot_leg = mlines.Line2D([], [], linestyle='-', label="Softmax Conformal", **softmax_args)

        softmax_sys_acc_ensem_mean = softmax_sys_acc_ensem.mean(axis=0)
        softmax_sys_acc_ensem_std = softmax_sys_acc_ensem.std(axis=0)
        ax.errorbar(exp_list, softmax_sys_acc_ensem_mean, yerr=softmax_sys_acc_ensem_std, linestyle="--", alpha=0.75,
                    **softmax_args)
        softmax_sys_acc_ensem_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax Fixed-Size ($k=5$)",
                                                  **softmax_args)
        # # OvA ===
        # # reg
        # ova_setsize_reg_mean = ova_setsize_reg.mean(axis=0)
        # ova_setsize_reg_std = ova_setsize_reg.std(axis=0)
        # ax.errorbar(exp_list, ova_setsize_reg_mean, yerr=ova_setsize_reg_std, linestyle="--", alpha=0.75,
        #             label="OvA Reg. Conformal", **ova_args)
        # ova_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA Reg. Conformal", **ova_args)
        #
        # # Softmax ===
        # # reg
        # softmax_setsize_reg_mean = softmax_setsize_reg.mean(axis=0)
        # softmax_setsize_reg_std = softmax_setsize_reg.std(axis=0)
        # ax.errorbar(exp_list, softmax_setsize_reg_mean, yerr=softmax_setsize_reg_std, linestyle="--", alpha=0.75,
        #             label="Softmax Reg. Conformal", **softmax_args)
        # softmax_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax Reg. Conformal", **softmax_args)

        ax.set_xticks(exp_list, exp_list)
        # plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_ensem_leg, softmax_sys_acc_vot_leg, softmax_sys_acc_ensem_leg],
            loc="best", fontsize=7)
        # prop = {"size"})

        return ax

    def plot_sys_acc_reg(f, ax):
        # Naive ===
        exp_path = "increase_oracle_v2/regularized/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_oracle_v2_system_accuracy_ova.txt")
        ova_sys_acc_voting = np.array(ova_sys_acc["voting"])
        ova_sys_acc_ensem = np.array(ova_sys_acc["ensemble"])

        softmax_sys_acc = load_dict_txt(exp_path + "increase_oracle_v2_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = np.array(softmax_sys_acc["voting"])
        softmax_sys_acc_ensem = np.array(softmax_sys_acc["ensemble"])

        # # Regularized ===
        # exp_path_reg = "increase_oracle_v2/regularized/"
        # ova_setsize_reg = load_dict_txt(exp_path_reg + "increase_oracle_v2_avg_set_size_ova.txt")
        # ova_setsize_reg = np.array(ova_setsize_reg["voting"])
        # softmax_setsize_reg = load_dict_txt(exp_path_reg + "increase_oracle_v2_avg_set_size_softmax.txt")
        # softmax_setsize_reg = np.array(softmax_setsize_reg["voting"])

        # OvA ===
        # naive
        ova_sys_acc_voting_mean = ova_sys_acc_voting.mean(axis=0)
        ova_sys_acc_voting_std = ova_sys_acc_voting.std(axis=0)
        ax.errorbar(exp_list, ova_sys_acc_voting_mean, yerr=ova_sys_acc_voting_std, linestyle="-", alpha=0.75,
                    **ova_args)
        ova_sys_acc_vot_leg = mlines.Line2D([], [], linestyle='-', label="OvA Conformal", **ova_args)

        ova_sys_acc_ensem_mean = ova_sys_acc_ensem.mean(axis=0)
        ova_sys_acc_ensem_std = ova_sys_acc_ensem.std(axis=0)
        ax.errorbar(exp_list, ova_sys_acc_ensem_mean, yerr=ova_sys_acc_ensem_std, linestyle="--", alpha=0.75,
                    **ova_args)
        ova_sys_acc_ensem_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA Fixed-Size ($k=5$)",
                                              **ova_args)

        # Softmax ===
        # naive
        softmax_sys_acc_voting_mean = softmax_sys_acc_voting.mean(axis=0)
        softmax_sys_acc_voting_std = softmax_sys_acc_voting.std(axis=0)
        ax.errorbar(exp_list, softmax_sys_acc_voting_mean, yerr=softmax_sys_acc_voting_std, linestyle="-", alpha=0.75,
                    **softmax_args)
        softmax_sys_acc_vot_leg = mlines.Line2D([], [], linestyle='-', label="Softmax Conformal", **softmax_args)

        softmax_sys_acc_ensem_mean = softmax_sys_acc_ensem.mean(axis=0)
        softmax_sys_acc_ensem_std = softmax_sys_acc_ensem.std(axis=0)
        ax.errorbar(exp_list, softmax_sys_acc_ensem_mean, yerr=softmax_sys_acc_ensem_std, linestyle="--", alpha=0.75,
                    **softmax_args)
        softmax_sys_acc_ensem_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax Fixed-Size ($k=5$)",
                                                  **softmax_args)
        # # OvA ===
        # # reg
        # ova_setsize_reg_mean = ova_setsize_reg.mean(axis=0)
        # ova_setsize_reg_std = ova_setsize_reg.std(axis=0)
        # ax.errorbar(exp_list, ova_setsize_reg_mean, yerr=ova_setsize_reg_std, linestyle="--", alpha=0.75,
        #             label="OvA Reg. Conformal", **ova_args)
        # ova_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA Reg. Conformal", **ova_args)
        #
        # # Softmax ===
        # # reg
        # softmax_setsize_reg_mean = softmax_setsize_reg.mean(axis=0)
        # softmax_setsize_reg_std = softmax_setsize_reg.std(axis=0)
        # ax.errorbar(exp_list, softmax_setsize_reg_mean, yerr=softmax_setsize_reg_std, linestyle="--", alpha=0.75,
        #             label="Softmax Reg. Conformal", **softmax_args)
        # softmax_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax Reg. Conformal", **softmax_args)

        ax.set_xticks(exp_list, exp_list)
        # plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_ensem_leg, softmax_sys_acc_vot_leg, softmax_sys_acc_ensem_leg],
            loc="best", fontsize=7)
        # prop = {"size"})

        return ax

    # plt.rcParams.update(figsizes.aistats2023_half(height_to_width_ratio=1))
    #
    # plt.rcParams.update(fonts.aistats2022_tex(family="serif"))
    # plt.rcParams.update(fontsizes.aistats2023())
    # plt.rcParams.update(axes.lines(line_base_ratio=5))
    # plt.rcParams.update(axes.grid(grid_alpha=0.5))  # custom grid. alpha=0-1, for transparency
    # plt.rcParams.update({"lines.markersize": 3,
    #                      "xtick.labelsize": 5})

    # f, ax = plt.subplots(1, 1)
    # ax = plot_avg_set_size(f, ax)
    #
    # f.set_tight_layout(True)
    # plt.show()
    # f.savefig(paper_results_path + "avg_set_size_nonrandomized.pdf")
    #

    f, ax = plt.subplots(1, 1)
    ax = plot_sys_acc_naive(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "system_accuracy_nonrandomized_naive.pdf")

    f, ax = plt.subplots(1, 1)
    ax = plot_sys_acc_reg(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "system_accuracy_nonrandomized_reg.pdf")
    return f, axes


if __name__ == '__main__':
    set_aistats2023_style()
    # Experiment 1: Multi-expert accuracies and calibration
    # experiment1()
    experiment2()
