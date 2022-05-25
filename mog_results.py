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
# No Percentage
# ece_softmax = np.array([0.035688892, 0.0393773, 0.042222522, 0.07813597, 0.13169229])
# ece_ova = np.array([0.031140802, 0.038146082, 0.023401938, 0.03507, 0.027209044])

# Percentage
ece_softmax = np.array([0.035688892, 0.0393773, 0.042222522, 0.07813597, 0.13169229]) * 100
ece_ova = np.array([0.031140802, 0.038146082, 0.023401938, 0.03507, 0.027209044]) * 100

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
plt.savefig("mog_increase_experts.pdf")
