import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tueplots import figsizes, axes, fontsizes

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
        # exp_list = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
        exp_list = [2, 4, 6, 8, 10, 14, 18]
        exp_path = "increase_experts/naive/"
        ova_accuracies = load_dict_txt(exp_path + "increase_experts_system_accuracy_ova.txt")
        softmax_accuracies = load_dict_txt(exp_path + "increase_experts_system_accuracy_softmax.txt")

        ova_accuracies = np.array(ova_accuracies["standard"]) * 100
        softmax_accuracies = np.array(softmax_accuracies["standard"]) * 100

        ova_accuracies = ova_accuracies[:, 1:]
        softmax_accuracies = softmax_accuracies[:, 1:]
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
        ax.set_yticks(list(plt.yticks()[0])[::2])
        # ax.set_ylim((ax.get_ylim()[0], 100))
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Number of Experts")
        ax.grid()

        ova_leg = mlines.Line2D([], [], linestyle='-', label='OvA', **ova_args)
        softmax_leg = mlines.Line2D([], [], linestyle='-', label='Softmax', **softmax_args)
        ax.legend(handles=[ova_leg, softmax_leg], loc="best")
        return ax

    def increase_experts_calibration(f, ax):  # TODO
        exp_list = [2, 4, 6, 8, 10, 14, 18]

        softmax_ece_mean = np.array(
            [0.05178541, 0.03142016, 0.02538094, 0.03436143, 0.0277727, 0.02835129, 0.02579127, 0.02650053]) * 100
        softmax_ece_std = np.array(
            [0.01946501, 0.00332035, 0.00089578, 0.00231578, 0.00197512, 0.0007479, 0.00126423, 0.00166975]) * 100
        ova_ece_mean = np.array(
            [0.02493655, 0.03006555, 0.02682439, 0.02946591, 0.02868607, 0.0301291, 0.02640719, 0.02986932]) * 100
        ova_ece_std = np.array(
            [0.00375583, 0.00056767, 0.00241768, 0.00096937, 0.00122256, 0.00015699, 0.00201291, 0.00055201]) * 100

        softmax_ece_mean = softmax_ece_mean[1:]
        softmax_ece_std = softmax_ece_std[1:]
        ova_ece_mean = ova_ece_mean[1:]
        ova_ece_std = ova_ece_std[1:]
        # OvA
        ax.errorbar(exp_list, ova_ece_mean, yerr=ova_ece_std,
                    alpha=0.75, label="OvA", marker="o")
        # Softmax
        ax.errorbar(exp_list, softmax_ece_mean, yerr=softmax_ece_std,
                    alpha=0.75, label="Softmax", marker="s")

        ax.set_xticks(exp_list, exp_list)
        plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average ECE $(\%)$")
        ax.set_xlabel(r"Number of Experts")
        ax.grid()

        ova_leg = mlines.Line2D([], [], linestyle='-', label='OvA', **ova_args)
        softmax_leg = mlines.Line2D([], [], linestyle='-', label='Softmax', **softmax_args)
        ax.legend(handles=[ova_leg, softmax_leg], loc="best")
        return ax

    def increase_experts_accuracy_70(f, ax):
        # exp_list = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
        exp_list = [4, 8, 12, 16, 20]
        exp_path = "increase_experts_prob/naive/"
        ova_accuracies = load_dict_txt(exp_path + "increase_experts_prob_system_accuracy_ova.txt")
        softmax_accuracies = load_dict_txt(exp_path + "increase_experts_prob_system_accuracy_softmax.txt")

        ova_accuracies = np.array(ova_accuracies["standard"]) * 100
        softmax_accuracies = np.array(softmax_accuracies["standard"]) * 100

        ova_accuracies = ova_accuracies[:, 1:]
        softmax_accuracies = softmax_accuracies[:, 1:]
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
        ax.set_yticks(list(plt.yticks()[0])[::2])
        # ax.set_ylim((ax.get_ylim()[0], 100))
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Number of Experts")
        ax.grid()

        ova_leg = mlines.Line2D([], [], linestyle='-', label='OvA', **ova_args)
        softmax_leg = mlines.Line2D([], [], linestyle='-', label='Softmax', **softmax_args)
        ax.legend(handles=[ova_leg, softmax_leg], loc="best")
        return ax

    def increase_experts_calibration_70(f, ax):  # TODO
        exp_list = [1, 4, 8, 12, 16, 20]
        exp_list = [4, 8, 12, 16, 20]

        softmax_ece_mean = np.array(
            [0.06080247, 0.04876373, 0.05128332, 0.04765288, 0.04044338, 0.04623997]) * 100
        softmax_ece_std = np.array(
            [0.00093865, 0.00088569, 0.00080867, 0.00198443, 0.00044131, 0.00067968]) * 100
        ova_ece_mean = np.array(
            [0.02295815, 0.02530994, 0.02178339, 0.02150761, 0.0202533, 0.02191817]) * 100
        ova_ece_std = np.array(
            [0.0051681, 0.00122212, 0.00204974, 0.00030093, 0.0013439, 0.00031387]) * 100

        softmax_ece_mean = softmax_ece_mean[1:]
        softmax_ece_std = softmax_ece_std[1:]
        ova_ece_mean = ova_ece_mean[1:]
        ova_ece_std = ova_ece_std[1:]
        # OvA
        ax.errorbar(exp_list, ova_ece_mean, yerr=ova_ece_std,
                    alpha=0.75, label="OvA", marker="o")
        # Softmax
        ax.errorbar(exp_list, softmax_ece_mean, yerr=softmax_ece_std,
                    alpha=0.75, label="Softmax", marker="s")

        ax.set_xticks(exp_list, exp_list)
        ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average ECE $(\%)$")
        ax.set_xlabel(r"Number of Experts")
        ax.grid()

        ova_leg = mlines.Line2D([], [], linestyle='-', label='OvA', **ova_args)
        softmax_leg = mlines.Line2D([], [], linestyle='-', label='Softmax', **softmax_args)
        ax.legend(handles=[ova_leg, softmax_leg], loc="best")
        return ax

    def increase_confidence_calibration_average(f, ax):  # TODO
        exp_list = [0.2, 0.4, 0.6, 0.8, 0.95]
        x_ticks = [int(100 * e) for e in exp_list]
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

        ax.set_xticks(exp_list, x_ticks)
        ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average ECE $(\%)$")
        ax.set_xlabel(r"Expert Correctness ($\%$)")
        ax.grid()

        ova_leg = mlines.Line2D([], [], linestyle='-', label='OvA', **ova_args)
        softmax_leg = mlines.Line2D([], [], linestyle='-', label='Softmax', **softmax_args)
        ax.legend(handles=[ova_leg, softmax_leg], loc="best")
        return ax

    def increase_confidence_calibration_random(f, ax):  # TODO
        exp_list = [0.2, 0.4, 0.6, 0.8, 0.95]
        x_ticks = [int(100 * e) for e in exp_list]

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

        ax.set_xticks(exp_list, x_ticks)
        ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Random Expert ECE $(\%)$")
        ax.set_xlabel(r"Expert Correctness ($\%$)")
        ax.grid()

        ova_leg = mlines.Line2D([], [], linestyle='-', label='OvA', **ova_args)
        softmax_leg = mlines.Line2D([], [], linestyle='-', label='Softmax', **softmax_args)
        ax.legend(handles=[ova_leg, softmax_leg], loc="best")
        return ax

    f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    ax = increase_experts_accuracy(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_experts_sys_accuracy.pdf")

    f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    ax = increase_experts_accuracy_70(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_experts_sys_accuracy_70.pdf")

    f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    ax = increase_experts_calibration(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_experts_calibration.pdf")

    f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    ax = increase_experts_calibration_70(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_experts_calibration_70.pdf")

    f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    ax = increase_confidence_calibration_average(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_confidence_calibration_average.pdf")

    f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    ax = increase_confidence_calibration_random(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "increase_confidence_calibration_randomexpert.pdf")


# GRADUAL OVERLAP ===
def experiment2():
    # Non-randomized experts ===
    cmap = sns.color_palette()
    exp_list = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    exp_list = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
    ova_args = {"color": cmap[0],
                "marker": "o"}
    softmax_args = {"color": cmap[1],
                    "marker": "s"}

    # Naive ===
    def plot_avg_set_size(f, ax):
        # Naive ===
        exp_path_naive = "gradual_overlap/naive/"
        ova_setsize_naive = load_dict_txt(exp_path_naive + "gradual_overlap_avg_set_size_ova.txt")
        ova_setsize_naive = np.array(ova_setsize_naive["voting"])[:, :-1]
        softmax_setsize_naive = load_dict_txt(exp_path_naive + "gradual_overlap_avg_set_size_softmax.txt")
        softmax_setsize_naive = np.array(softmax_setsize_naive["voting"])[:, :-1]

        # Regularized ===
        exp_path_reg = "gradual_overlap/regularized/"
        ova_setsize_reg = load_dict_txt(exp_path_reg + "gradual_overlap_avg_set_size_ova.txt")
        ova_setsize_reg = np.array(ova_setsize_reg["voting"])[:, :-1]
        softmax_setsize_reg = load_dict_txt(exp_path_reg + "gradual_overlap_avg_set_size_softmax.txt")
        softmax_setsize_reg = np.array(softmax_setsize_reg["voting"])[:, :-1]

        # OvA ===
        # naive
        ova_setsize_naive_mean = ova_setsize_naive.mean(axis=0)
        ova_setsize_naive_std = ova_setsize_naive.std(axis=0)
        ax.errorbar(exp_list, ova_setsize_naive_mean, yerr=ova_setsize_naive_std, linestyle="-", alpha=0.75,
                    label="OvA Naive Conformal", **ova_args)
        ova_naive_leg = mlines.Line2D([], [], linestyle='-', label="OvA, naive", **ova_args)

        # Softmax ===
        # naive
        softmax_setsize_naive_mean = softmax_setsize_naive.mean(axis=0)
        softmax_setsize_naive_std = softmax_setsize_naive.std(axis=0)
        ax.errorbar(exp_list, softmax_setsize_naive_mean, yerr=softmax_setsize_naive_std, linestyle="-", alpha=0.75,
                    label="Softmax Naive Conformal", **softmax_args)
        softmax_naive_leg = mlines.Line2D([], [], linestyle='-', label="Softmax, naive", **softmax_args)

        # OvA ===
        # reg
        ova_setsize_reg_mean = ova_setsize_reg.mean(axis=0)
        ova_setsize_reg_std = ova_setsize_reg.std(axis=0)
        ax.errorbar(exp_list, ova_setsize_reg_mean, yerr=ova_setsize_reg_std, linestyle="--", alpha=0.75,
                    label="OvA Reg. Conformal", **ova_args)
        ova_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA, regularized", **ova_args)

        # Softmax ===
        # reg
        softmax_setsize_reg_mean = softmax_setsize_reg.mean(axis=0)
        softmax_setsize_reg_std = softmax_setsize_reg.std(axis=0)
        ax.errorbar(exp_list, softmax_setsize_reg_mean, yerr=softmax_setsize_reg_std, linestyle="--", alpha=0.75,
                    label="Softmax Reg. Conformal", **softmax_args)
        softmax_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax, regularized", **softmax_args)

        ax.set_ylabel(r"Average Set Size")
        ax.set_xlabel(r"Overlap probability ($\%$)")
        xtick_labels = [str(int(i * 100)) for i in exp_list]
        ax.set_xticks(exp_list, xtick_labels)
        ax.set_xticklabels(xtick_labels)
        ax.grid()

        ax.legend(handles=[ova_naive_leg, softmax_naive_leg, ova_reg_leg, softmax_reg_leg],
                  loc="best")
        # prop = {"size"})

        return ax

    def plot_sys_acc_naive(f, ax):
        # Naive ===
        exp_path = "gradual_overlap/naive/"
        ova_sys_acc = load_dict_txt(exp_path + "gradual_overlap_system_accuracy_ova.txt")
        ova_sys_acc_voting = (np.array(ova_sys_acc["voting"]) * 100)[:, :-1]
        ova_sys_acc_ensem = (np.array(ova_sys_acc["ensemble"]) * 100)[:, :-1]

        softmax_sys_acc = load_dict_txt(exp_path + "gradual_overlap_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = (np.array(softmax_sys_acc["voting"]) * 100)[:, :-1]
        softmax_sys_acc_ensem = (np.array(softmax_sys_acc["ensemble"]) * 100)[:, :-1]

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

        xtick_labels = [str(int(i * 100)) for i in exp_list]
        ax.set_xticks(exp_list, xtick_labels)
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Overlap Probability ($\%$)")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_ensem_leg, softmax_sys_acc_vot_leg, softmax_sys_acc_ensem_leg],
            loc="best")

        return ax

    def plot_sys_acc_reg(f, ax):
        # Naive ===
        exp_path = "gradual_overlap/regularized/"
        ova_sys_acc = load_dict_txt(exp_path + "gradual_overlap_system_accuracy_ova.txt")
        ova_sys_acc_voting = (np.array(ova_sys_acc["voting"]) * 100)[:, :-1]
        ova_sys_acc_ensem = (np.array(ova_sys_acc["ensemble"]) * 100)[:, :-1]

        softmax_sys_acc = load_dict_txt(exp_path + "gradual_overlap_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = (np.array(softmax_sys_acc["voting"]) * 100)[:, :-1]
        softmax_sys_acc_ensem = (np.array(softmax_sys_acc["ensemble"]) * 100)[:, :-1]

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

        xtick_labels = [str(int(i * 100)) for i in exp_list]
        ax.set_xticks(exp_list, xtick_labels)
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Overlap Probability ($\%$)")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_ensem_leg, softmax_sys_acc_vot_leg, softmax_sys_acc_ensem_leg],
            loc="best")

        return ax

    f, ax = plt.subplots(1, 1)
    ax = plot_avg_set_size(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "avg_set_size_gradual_overlap.pdf")

    f, ax = plt.subplots(1, 1)
    ax = plot_sys_acc_naive(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "system_accuracy_gradual_overlap_naive.pdf")

    f, ax = plt.subplots(1, 1)
    ax = plot_sys_acc_reg(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "system_accuracy_gradual_overlap_reg.pdf")


# RANDOMIZED ===
def experiment3():
    # Non-randomized experts ===
    cmap = sns.color_palette()
    exp_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_ticks = [e + 1 for e in exp_list]

    ova_args = {"color": cmap[0],
                "marker": "o"}
    softmax_args = {"color": cmap[1],
                    "marker": "s"}

    # Naive ===
    def plot_avg_set_size(f, ax):
        # Naive ===
        exp_path_naive = "increase_oracle/naive/"
        ova_setsize_naive = load_dict_txt(exp_path_naive + "increase_oracle_avg_set_size_ova.txt")
        ova_setsize_naive = np.array(ova_setsize_naive["voting"])[:, 1:]
        softmax_setsize_naive = load_dict_txt(exp_path_naive + "increase_oracle_avg_set_size_softmax.txt")
        softmax_setsize_naive = np.array(softmax_setsize_naive["voting"])[:, 1:]

        # Regularized ===
        exp_path_reg = "increase_oracle/regularized/"
        ova_setsize_reg = load_dict_txt(exp_path_reg + "increase_oracle_avg_set_size_ova.txt")
        ova_setsize_reg = np.array(ova_setsize_reg["voting"])[:, 1:]
        softmax_setsize_reg = load_dict_txt(exp_path_reg + "increase_oracle_avg_set_size_softmax.txt")
        softmax_setsize_reg = np.array(softmax_setsize_reg["voting"])[:, 1:]

        # OvA ===
        # naive
        ova_setsize_naive_mean = ova_setsize_naive.mean(axis=0)
        ova_setsize_naive_std = ova_setsize_naive.std(axis=0)
        ax.errorbar(exp_list, ova_setsize_naive_mean, yerr=ova_setsize_naive_std, linestyle="-", alpha=0.75,
                    label="OvA Naive Conformal", **ova_args)
        ova_naive_leg = mlines.Line2D([], [], linestyle='-', label="OvA, naive", **ova_args)

        # Softmax ===
        # naive
        softmax_setsize_naive_mean = softmax_setsize_naive.mean(axis=0)
        softmax_setsize_naive_std = softmax_setsize_naive.std(axis=0)
        ax.errorbar(exp_list, softmax_setsize_naive_mean, yerr=softmax_setsize_naive_std, linestyle="-", alpha=0.75,
                    label="Softmax Naive Conformal", **softmax_args)
        softmax_naive_leg = mlines.Line2D([], [], linestyle='-', label="Softmax, naive", **softmax_args)

        # OvA ===
        # reg
        ova_setsize_reg_mean = ova_setsize_reg.mean(axis=0)
        ova_setsize_reg_std = ova_setsize_reg.std(axis=0)
        ax.errorbar(exp_list, ova_setsize_reg_mean, yerr=ova_setsize_reg_std, linestyle="--", alpha=0.75,
                    label="OvA Reg. Conformal", **ova_args)
        ova_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA, regularized", **ova_args)

        # Softmax ===
        # reg
        softmax_setsize_reg_mean = softmax_setsize_reg.mean(axis=0)
        softmax_setsize_reg_std = softmax_setsize_reg.std(axis=0)
        ax.errorbar(exp_list, softmax_setsize_reg_mean, yerr=softmax_setsize_reg_std, linestyle="--", alpha=0.75,
                    label="Softmax Reg. Conformal", **softmax_args)
        softmax_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax, regularized", **softmax_args)

        ax.set_xticks(exp_list, x_ticks)
        # ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average Set Size")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(handles=[ova_naive_leg, softmax_naive_leg, ova_reg_leg, softmax_reg_leg],
                  loc="best")
        # prop = {"size"})

        return ax

    def plot_sys_acc_naive(f, ax):
        # Naive ===
        exp_path = "increase_oracle/naive/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_oracle_system_accuracy_ova.txt")
        ova_sys_acc_voting = np.array(ova_sys_acc["voting"])[:, 1:] * 100
        ova_sys_acc_ensem = np.array(ova_sys_acc["ensemble"])[:, 1:] * 100

        softmax_sys_acc = load_dict_txt(exp_path + "increase_oracle_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = np.array(softmax_sys_acc["voting"])[:, 1:] * 100
        softmax_sys_acc_ensem = np.array(softmax_sys_acc["ensemble"])[:, 1:] * 100

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
        ax.set_xticks(exp_list, x_ticks)
        ax.set_ylim((ax.get_ylim()[0], 100))
        ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_ensem_leg, softmax_sys_acc_vot_leg, softmax_sys_acc_ensem_leg],
            loc="best")
        # prop = {"size"})

        return ax

    def plot_sys_acc_reg(f, ax):
        # Naive ===
        exp_path = "increase_oracle/regularized/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_oracle_system_accuracy_ova.txt")
        ova_sys_acc_voting = np.array(ova_sys_acc["voting"])[:, 1:] * 100
        ova_sys_acc_ensem = np.array(ova_sys_acc["ensemble"])[:, 1:] * 100

        softmax_sys_acc = load_dict_txt(exp_path + "increase_oracle_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = np.array(softmax_sys_acc["voting"])[:, 1:] * 100
        softmax_sys_acc_ensem = np.array(softmax_sys_acc["ensemble"])[:, 1:] * 100

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

        ax.set_xticks(exp_list, x_ticks)
        ax.set_ylim((ax.get_ylim()[0], 100))
        ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_ensem_leg, softmax_sys_acc_vot_leg, softmax_sys_acc_ensem_leg],
            loc="best")
        return ax

    def plot_sys_acc_standard_vs_conformal_naive(f, ax):
        # Naive ===
        exp_path = "increase_oracle/naive/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_oracle_system_accuracy_ova.txt")
        ova_sys_acc_voting = np.array(ova_sys_acc["voting"])[:, 1:] * 100
        ova_sys_acc_standard = np.array(ova_sys_acc["standard"])[:, 1:] * 100

        softmax_sys_acc = load_dict_txt(exp_path + "increase_oracle_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = np.array(softmax_sys_acc["voting"])[:, 1:] * 100
        softmax_sys_acc_standard = np.array(softmax_sys_acc["standard"])[:, 1:] * 100

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

        ova_sys_acc_standard_mean = ova_sys_acc_standard.mean(axis=0)
        ova_sys_acc_standard_std = ova_sys_acc_standard.std(axis=0)
        ax.errorbar(exp_list, ova_sys_acc_standard_mean, yerr=ova_sys_acc_standard_std, linestyle="--", alpha=0.75,
                    **ova_args)
        ova_sys_acc_standard_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA Standard",
                                                 **ova_args)

        # Softmax ===
        # naive
        softmax_sys_acc_voting_mean = softmax_sys_acc_voting.mean(axis=0)
        softmax_sys_acc_voting_std = softmax_sys_acc_voting.std(axis=0)
        ax.errorbar(exp_list, softmax_sys_acc_voting_mean, yerr=softmax_sys_acc_voting_std, linestyle="-", alpha=0.75,
                    **softmax_args)
        softmax_sys_acc_vot_leg = mlines.Line2D([], [], linestyle='-', label="Softmax Conformal", **softmax_args)

        softmax_sys_acc_standard_mean = softmax_sys_acc_standard.mean(axis=0)
        softmax_sys_acc_standard_std = softmax_sys_acc_standard.std(axis=0)
        ax.errorbar(exp_list, softmax_sys_acc_standard_mean, yerr=softmax_sys_acc_standard_std, linestyle="--",
                    alpha=0.75,
                    **softmax_args)
        softmax_sys_acc_standard_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax Standard",
                                                     **softmax_args)

        ax.set_xticks(exp_list, x_ticks)
        ax.set_ylim(ax.get_ylim())
        # ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_standard_leg, softmax_sys_acc_vot_leg,
                     softmax_sys_acc_standard_leg],
            loc="best")
        # prop = {"size"})

        return ax

    def plot_sys_acc_standard_vs_conformal_reg(f, ax):
        # Naive ===
        exp_path = "increase_oracle/regularized/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_oracle_system_accuracy_ova.txt")
        ova_sys_acc_voting = np.array(ova_sys_acc["voting"])[:, 1:] * 100
        ova_sys_acc_standard = np.array(ova_sys_acc["standard"])[:, 1:] * 100

        softmax_sys_acc = load_dict_txt(exp_path + "increase_oracle_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = np.array(softmax_sys_acc["voting"])[:, 1:] * 100
        softmax_sys_acc_standard = np.array(softmax_sys_acc["standard"])[:, 1:] * 100

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

        ova_sys_acc_standard_mean = ova_sys_acc_standard.mean(axis=0)
        ova_sys_acc_standard_std = ova_sys_acc_standard.std(axis=0)
        ax.errorbar(exp_list, ova_sys_acc_standard_mean, yerr=ova_sys_acc_standard_std, linestyle="--", alpha=0.75,
                    **ova_args)
        ova_sys_acc_standard_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA Standard",
                                                 **ova_args)

        # Softmax ===
        # naive
        softmax_sys_acc_voting_mean = softmax_sys_acc_voting.mean(axis=0)
        softmax_sys_acc_voting_std = softmax_sys_acc_voting.std(axis=0)
        ax.errorbar(exp_list, softmax_sys_acc_voting_mean, yerr=softmax_sys_acc_voting_std, linestyle="-", alpha=0.75,
                    **softmax_args)
        softmax_sys_acc_vot_leg = mlines.Line2D([], [], linestyle='-', label="Softmax Conformal", **softmax_args)

        softmax_sys_acc_standard_mean = softmax_sys_acc_standard.mean(axis=0)
        softmax_sys_acc_standard_std = softmax_sys_acc_standard.std(axis=0)
        ax.errorbar(exp_list, softmax_sys_acc_standard_mean, yerr=softmax_sys_acc_standard_std, linestyle="--",
                    alpha=0.75,
                    **softmax_args)
        softmax_sys_acc_standard_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax Standard",
                                                     **softmax_args)

        ax.set_xticks(exp_list, x_ticks)
        ax.set_ylim(ax.get_ylim())
        # ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_standard_leg, softmax_sys_acc_vot_leg,
                     softmax_sys_acc_standard_leg],
            loc="best")
        # prop = {"size"})

        return ax

    # f, ax = plt.subplots(1, 1)
    # ax = plot_avg_set_size(f, ax)
    # f.set_tight_layout(True)
    # plt.show()
    # f.savefig(paper_results_path + "avg_set_size_randomized.pdf")
    #
    # f, ax = plt.subplots(1, 1)
    # ax = plot_sys_acc_naive(f, ax)
    # f.set_tight_layout(True)
    # plt.show()
    # f.savefig(paper_results_path + "system_accuracy_randomized_naive.pdf")
    #
    # f, ax = plt.subplots(1, 1)
    # ax = plot_sys_acc_reg(f, ax)
    # f.set_tight_layout(True)
    # plt.show()
    # f.savefig(paper_results_path + "system_accuracy_randomized_reg.pdf")

    f, ax = plt.subplots(1, 1)
    ax = plot_sys_acc_standard_vs_conformal_naive(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "system_accuracy_randomized_std_vs_conformal_naive.pdf")

    f, ax = plt.subplots(1, 1)
    ax = plot_sys_acc_standard_vs_conformal_reg(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "system_accuracy_randomized_std_vs_conformal_reg.pdf")


# NON-RANDOMIZED ===
def experiment4():
    # Non-randomized experts ===
    cmap = sns.color_palette()
    exp_list = [0, 1, 2, 3, 4, 5, 7, 9]
    x_ticks = [e + 1 for e in exp_list]
    # exp_list = [1, 2, 3, 4, 5]
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
        ova_naive_leg = mlines.Line2D([], [], linestyle='-', label="OvA, naive", **ova_args)

        # Softmax ===
        # naive
        softmax_setsize_naive_mean = softmax_setsize_naive.mean(axis=0)
        softmax_setsize_naive_std = softmax_setsize_naive.std(axis=0)
        ax.errorbar(exp_list, softmax_setsize_naive_mean, yerr=softmax_setsize_naive_std, linestyle="-", alpha=0.75,
                    label="Softmax Naive Conformal", **softmax_args)
        softmax_naive_leg = mlines.Line2D([], [], linestyle='-', label="Softmax, naive", **softmax_args)

        # OvA ===
        # reg
        ova_setsize_reg_mean = ova_setsize_reg.mean(axis=0)
        ova_setsize_reg_std = ova_setsize_reg.std(axis=0)
        ax.errorbar(exp_list, ova_setsize_reg_mean, yerr=ova_setsize_reg_std, linestyle="--", alpha=0.75,
                    label="OvA Reg. Conformal", **ova_args)
        ova_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA, regularized", **ova_args)

        # Softmax ===
        # reg
        softmax_setsize_reg_mean = softmax_setsize_reg.mean(axis=0)
        softmax_setsize_reg_std = softmax_setsize_reg.std(axis=0)
        ax.errorbar(exp_list, softmax_setsize_reg_mean, yerr=softmax_setsize_reg_std, linestyle="--", alpha=0.75,
                    label="Softmax Reg. Conformal", **softmax_args)
        softmax_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax, regularized", **softmax_args)

        ax.set_xticks(exp_list, x_ticks)
        # plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average Set Size")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(handles=[ova_naive_leg, softmax_naive_leg, ova_reg_leg, softmax_reg_leg],
                  loc="best")
        # prop = {"size"})

        return ax

    def plot_sys_acc_naive(f, ax):
        # Naive ===
        exp_path = "increase_oracle_v2/naive/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_oracle_v2_system_accuracy_ova.txt")
        ova_sys_acc_voting = np.array(ova_sys_acc["voting"]) * 100
        ova_sys_acc_ensem = np.array(ova_sys_acc["ensemble"]) * 100

        softmax_sys_acc = load_dict_txt(exp_path + "increase_oracle_v2_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = np.array(softmax_sys_acc["voting"]) * 100
        softmax_sys_acc_ensem = np.array(softmax_sys_acc["ensemble"]) * 100

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
        ax.set_xticks(exp_list, x_ticks)
        # ax.set_ylim((ax.get_ylim()[0], 100))
        # ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_ensem_leg, softmax_sys_acc_vot_leg, softmax_sys_acc_ensem_leg],
            loc="best")
        # prop = {"size"})

        return ax

    def plot_sys_acc_reg(f, ax):
        # Naive ===
        exp_path = "increase_oracle_v2/regularized/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_oracle_v2_system_accuracy_ova.txt")
        ova_sys_acc_voting = np.array(ova_sys_acc["voting"]) * 100
        ova_sys_acc_ensem = np.array(ova_sys_acc["ensemble"]) * 100

        softmax_sys_acc = load_dict_txt(exp_path + "increase_oracle_v2_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = np.array(softmax_sys_acc["voting"]) * 100
        softmax_sys_acc_ensem = np.array(softmax_sys_acc["ensemble"]) * 100

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

        ax.set_xticks(exp_list, x_ticks)
        ax.set_ylim(ax.get_ylim())
        # ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Oracles")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_ensem_leg, softmax_sys_acc_vot_leg, softmax_sys_acc_ensem_leg],
            loc="best")
        # prop = {"size"})

        return ax

    f, ax = plt.subplots(1, 1)
    ax = plot_avg_set_size(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "avg_set_size_nonrandomized.pdf")

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


def ham10000():
    # Non-randomized experts ===
    cmap = sns.color_palette()
    exp_list = [1, 2, 4, 6, 8, 12, 16]
    x_ticks = exp_list
    # exp_list = [1, 2, 3, 4, 5]
    ova_args = {"color": cmap[0],
                "marker": "o"}
    softmax_args = {"color": cmap[1],
                    "marker": "s"}

    # Naive ===
    def plot_avg_set_size(f, ax):
        # Naive ===
        exp_path_naive = "increase_experts_ham10000/naive/"
        ova_setsize_naive = load_dict_txt(exp_path_naive + "increase_experts_select_avg_set_size_ova.txt")
        ova_setsize_naive = np.array(ova_setsize_naive["voting"])
        softmax_setsize_naive = load_dict_txt(exp_path_naive + "increase_experts_select_avg_set_size_softmax.txt")
        softmax_setsize_naive = np.array(softmax_setsize_naive["voting"])

        # Regularized ===
        exp_path_reg = "increase_experts_ham10000/regularized/"
        ova_setsize_reg = load_dict_txt(exp_path_reg + "increase_experts_select_avg_set_size_ova.txt")
        ova_setsize_reg = np.array(ova_setsize_reg["voting"])
        softmax_setsize_reg = load_dict_txt(exp_path_reg + "increase_experts_select_avg_set_size_softmax.txt")
        softmax_setsize_reg = np.array(softmax_setsize_reg["voting"])

        # OvA ===
        # naive
        ova_setsize_naive_mean = ova_setsize_naive.mean(axis=0)
        ova_setsize_naive_std = ova_setsize_naive.std(axis=0)
        ax.errorbar(exp_list, ova_setsize_naive_mean, yerr=ova_setsize_naive_std, linestyle="-", alpha=0.75,
                    label="OvA Naive Conformal", **ova_args)
        ova_naive_leg = mlines.Line2D([], [], linestyle='-', label="OvA, naive", **ova_args)

        # Softmax ===
        # naive
        softmax_setsize_naive_mean = softmax_setsize_naive.mean(axis=0)
        softmax_setsize_naive_std = softmax_setsize_naive.std(axis=0)
        ax.errorbar(exp_list, softmax_setsize_naive_mean, yerr=softmax_setsize_naive_std, linestyle="-", alpha=0.75,
                    label="Softmax Naive Conformal", **softmax_args)
        softmax_naive_leg = mlines.Line2D([], [], linestyle='-', label="Softmax, naive", **softmax_args)

        # OvA ===
        # reg
        ova_setsize_reg_mean = ova_setsize_reg.mean(axis=0)
        ova_setsize_reg_std = ova_setsize_reg.std(axis=0)
        ax.errorbar(exp_list, ova_setsize_reg_mean, yerr=ova_setsize_reg_std, linestyle="--", alpha=0.75,
                    label="OvA Reg. Conformal", **ova_args)
        ova_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA, regularized", **ova_args)

        # Softmax ===
        # reg
        softmax_setsize_reg_mean = softmax_setsize_reg.mean(axis=0)
        softmax_setsize_reg_std = softmax_setsize_reg.std(axis=0)
        ax.errorbar(exp_list, softmax_setsize_reg_mean, yerr=softmax_setsize_reg_std, linestyle="--", alpha=0.75,
                    label="Softmax Reg. Conformal", **softmax_args)
        softmax_reg_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax, regularized", **softmax_args)

        ax.set_xticks(exp_list, x_ticks)
        # plt.yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"Average Set Size")
        ax.set_xlabel(r"Number of experts")
        ax.grid()

        ax.legend(handles=[ova_naive_leg, softmax_naive_leg, ova_reg_leg, softmax_reg_leg],
                  loc="best")
        # prop = {"size"})

        return ax

    def plot_sys_acc_naive(f, ax):
        # Naive ===
        exp_path = "increase_experts_ham10000/naive/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_experts_select_system_accuracy_ova.txt")
        ova_sys_acc_voting = np.array(ova_sys_acc["voting"]) * 100
        ova_sys_acc_ensem = np.array(ova_sys_acc["ensemble"]) * 100

        softmax_sys_acc = load_dict_txt(exp_path + "increase_experts_select_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = np.array(softmax_sys_acc["voting"]) * 100
        softmax_sys_acc_ensem = np.array(softmax_sys_acc["ensemble"]) * 100

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
        ax.set_xticks(exp_list, x_ticks)
        # ax.set_ylim((ax.get_ylim()[0], 100))
        # ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Number of experts")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_ensem_leg, softmax_sys_acc_vot_leg, softmax_sys_acc_ensem_leg],
            loc="best")
        # prop = {"size"})

        return ax

    def plot_sys_acc_reg(f, ax):
        # Naive ===
        exp_path = "increase_experts_ham10000/regularized/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_experts_select_system_accuracy_ova.txt")
        ova_sys_acc_voting = np.array(ova_sys_acc["voting"]) * 100
        ova_sys_acc_ensem = np.array(ova_sys_acc["ensemble"]) * 100

        softmax_sys_acc = load_dict_txt(exp_path + "increase_experts_select_system_accuracy_softmax.txt")
        softmax_sys_acc_voting = np.array(softmax_sys_acc["voting"]) * 100
        softmax_sys_acc_ensem = np.array(softmax_sys_acc["ensemble"]) * 100

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
        ax.errorbar(exp_list, ova_sys_acc_voting_mean, yerr=ova_sys_acc_voting_std, linestyle="", alpha=0.7,
                    **ova_args)
        ax.plot(exp_list, ova_sys_acc_voting_mean, linestyle="-", alpha=0.45, color=cmap[0])
        ova_sys_acc_vot_leg = mlines.Line2D([], [], linestyle='-', label="OvA Conformal", **ova_args)

        ova_sys_acc_ensem_mean = ova_sys_acc_ensem.mean(axis=0)
        ova_sys_acc_ensem_std = ova_sys_acc_ensem.std(axis=0)
        ax.errorbar(exp_list, ova_sys_acc_ensem_mean, yerr=ova_sys_acc_ensem_std, linestyle="", alpha=0.7,
                    **ova_args)
        ax.plot(exp_list, ova_sys_acc_ensem_mean, linestyle="--", alpha=0.45, color=cmap[0])
        ova_sys_acc_ensem_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="OvA Fixed-Size ($k=5$)",
                                              **ova_args)

        # Softmax ===
        # naive
        softmax_sys_acc_voting_mean = softmax_sys_acc_voting.mean(axis=0)
        softmax_sys_acc_voting_std = softmax_sys_acc_voting.std(axis=0)
        ax.errorbar(exp_list, softmax_sys_acc_voting_mean, yerr=softmax_sys_acc_voting_std, linestyle="", alpha=0.7,
                    **softmax_args)
        ax.plot(exp_list, softmax_sys_acc_voting_mean, linestyle="-", alpha=0.45, color=cmap[1])
        softmax_sys_acc_vot_leg = mlines.Line2D([], [], linestyle='-', label="Softmax Conformal", **softmax_args)

        softmax_sys_acc_ensem_mean = softmax_sys_acc_ensem.mean(axis=0)
        softmax_sys_acc_ensem_std = softmax_sys_acc_ensem.std(axis=0)
        ax.errorbar(exp_list, softmax_sys_acc_ensem_mean, yerr=softmax_sys_acc_ensem_std, linestyle="", alpha=0.7,
                    **softmax_args)
        ax.plot(exp_list, softmax_sys_acc_ensem_mean, linestyle="--", alpha=0.45, color=cmap[1])
        softmax_sys_acc_ensem_leg = mlines.Line2D([], [], linestyle=(0.5, (1, 3)), label="Softmax Fixed-Size ($k=5$)",
                                                  **softmax_args)

        ax.set_xticks(exp_list, x_ticks)
        ax.set_ylim(ax.get_ylim())
        ax.set_yticks(list(plt.yticks()[0])[::2])
        ax.set_ylabel(r"System Accuracy $(\%)$")
        ax.set_xlabel(r"Number of experts")
        ax.grid()

        ax.legend(
            handles=[ova_sys_acc_vot_leg, ova_sys_acc_ensem_leg, softmax_sys_acc_vot_leg, softmax_sys_acc_ensem_leg],
            loc="best")
        # prop = {"size"})

        return ax

    def get_sys_acc_standard():
        # Naive ===
        exp_path = "increase_experts_ham10000/naive/"
        ova_sys_acc = load_dict_txt(exp_path + "increase_experts_select_system_accuracy_ova.txt")
        ova_sys_acc = np.array(ova_sys_acc["standard"]) * 100

        softmax_sys_acc = load_dict_txt(exp_path + "increase_experts_select_system_accuracy_softmax.txt")
        softmax_sys_acc = np.array(softmax_sys_acc["standard"]) * 100

        # OvA ===
        # naive
        ova_sys_acc_mean = ova_sys_acc.mean(axis=0)
        ova_sys_acc__std = ova_sys_acc.std(axis=0)

        softmax_sys_acc_mean = softmax_sys_acc.mean(axis=0)
        softmax_sys_acc_std = softmax_sys_acc.std(axis=0)
        print("OvA ======")
        print("Mean: {}".format(ova_sys_acc_mean))
        print("Std: {}".format(ova_sys_acc__std))
        print("Softmax ======")
        print("Mean: {}".format(softmax_sys_acc_mean))
        print("Std: {}".format(softmax_sys_acc_std))

    f, ax = plt.subplots(1, 1)
    ax = plot_avg_set_size(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "avg_set_size_ham10000.pdf")

    f, ax = plt.subplots(1, 1)
    ax = plot_sys_acc_naive(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "system_accuracy_ham10000_naive.pdf")

    f, ax = plt.subplots(1, 1)
    ax = plot_sys_acc_reg(f, ax)
    f.set_tight_layout(True)
    plt.show()
    f.savefig(paper_results_path + "system_accuracy_ham10000_reg.pdf")

    get_sys_acc_standard()


def galaxyzoo():
    # TODO
    pass


def hatespeech():
    # TODO
    softmax_sys_acc_mean = np.array([80.57617188, 80.53222656, 80.47851562, 83.18847656, 83.046875,
                                     82.890625, 85.20019531, 89.65332031, 90.03417969, 90.26855469])
    softmax_sys_acc_std = np.array([0.14435298, 0.22250307, 0.2896541, 0.29180678, 0.33799151,
                                    0.41721626, 0.187909, 0.36165789, 0.32985196, 0.0921934])

    ova_sys_acc_mean = np.array([81.12792969, 81.06933594, 80.98632812, 84.43847656,
                                 84.41894531, 84.12109375, 85.703125, 90.33203125, 91.0859375, 91.18410156])
    ova_sys_acc_std = np.array([0.24535829, 0.12367114, 0.08293598, 0.22766686, 0.10224801,
                                0.10907377, 0.30538079, 0.12638847, 0.05960232, 0.05725605])

    hemmer_sys_acc_mean = np.array([81.32324219, 81.31347656, 81.328125, 81.2109375, 82.02636719,
                                    80.3515625, 85.21972656, 90.28320312, 90.28320312, 90.1949219])
    hemmer_sys_acc_std = np.array([0.09296598, 0.07933631, 0.19491537, 0.52371436, 0.5194748,
                                   0.24698055, 0.776797, 0., 0., 0.04907166])

    oneclf_sys_ac_mean = np.array([80.6884765625] * 10)
    oneclf_sys_acc_std = np.array([0.047591769261762513] * 10)

    best_expert_sys_acc_mean = np.array([0.3350586, 0.39033204, 0.4868164, 0.7583984, 0.76240236,
                                         0.7614746, 0.81767577, 0.90283203, 0.90283203, 0.90283203])
    best_expert_sys_acc_std = np.array([0.00236275, 0.00213312, 0.00400048, 0.00252966, 0.00064778,
                                        0.00283876, 0.0003654, 0., 0., 0.])


if __name__ == '__main__':
    set_aistats2023_style()
    # plt.rcParams.update(fontsizes.aistats2023(default_smaller=0))
    # plt.rcParams.update(figsizes.aistats2023_half(tight_layout=True, height_to_width_ratio=1))  # make square fig.

    # experiment1()  # INCREASE NUMBER OF EXPERTS AND CONFIDENCE
    # experiment2()  # GRADUAL OVERLAP ===
    experiment3()  # RANDOMIZED ===
    # experiment4()  # NON-RANDOMIZED ===
    # ham10000()
