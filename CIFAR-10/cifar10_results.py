import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc


def plot_ece(softmax_ece, ova_ece, experiment_range):
    # # === Latex Options === #
    rc('font', family='serif')
    rc('text', usetex=True)

    # === Matplotlib Options === #
    plot_args = {"linestyle": "-",
                 "marker": "o",
                 "markeredgecolor": "k",
                 "markersize": 10,
                 "linewidth": 4
                 }

    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 30})
    f, ax = plt.subplots(1, 1)
    ax.plot(experiment_range, softmax_ece, label=r"Softmax", **plot_args)
    ax.plot(experiment_range, ova_ece, label=r"OvA", **plot_args)

    plt.xticks(experiment_range, experiment_range)
    plt.ylabel(r'Average ECE $\%$')
    plt.xlabel(r'Number of Experts')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return f


# # === Latex Options === #
rc('font', family='serif')
rc('text', usetex=True)

# === Matplotlib Options === #
cm = plt.cm.get_cmap('tab10')
plot_args = {"linestyle": "-",
             "marker": "o",
             "markeredgecolor": "k",
             "markersize": 10,
             "linewidth": 4
             }

# ====== Experiment 1: Increase number of experts ======
n_experts = [1, 2, 4, 6, 8]
# Percentage
ece_softmax = np.array([0.04212178, 0.03236192, 0.04614954, 0.048385993, 0.046213374]) * 100
ece_ova = np.array([0.033160426, 0.027528169, 0.029059665, 0.028813586, 0.03282707]) * 100

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 30})
# sns.set_style("whitegrid")
f, ax = plt.subplots(1, 1)
ax.plot(n_experts, ece_softmax, label=r"Softmax", **plot_args)
ax.plot(n_experts, ece_ova, label=r"OvA", **plot_args)
plt.xticks(n_experts, n_experts)
plt.ylabel(r'Average ECE $\%$')
plt.xlabel(r'Number of Experts')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("cifar_increase_experts.pdf")


# # ====== Experiment 2: Increase confidence for 3 experts ======
p_experts = [0.2, 0.4, 0.6, 0.8, 0.95]
# Percentage
ece_softmax = np.array([0.20560052, 0.3304271, 0.43687683, 0.5407775, 0.5562724]) * 100
ece_ova = np.array([0.007408993, 0.009338078, 0.009841489, 0.008472728, 0.011463852]) * 100

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 30})

f, ax = plt.subplots(1, 1)
ax.plot(p_experts, ece_softmax, label=r"Softmax", **plot_args)
ax.plot(p_experts, ece_ova, label=r"OvA", **plot_args)
plt.xticks(p_experts, p_experts)
plt.ylabel(r'ECE ($\%$) for Random Expert')
plt.xlabel(r'Probability of Expert Correctness')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("cifar_increase_confidence.pdf")
