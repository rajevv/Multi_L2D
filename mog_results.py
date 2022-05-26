import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc

# # === Latex Options === #
rc('font', family='serif')
rc('text', usetex=True)

# === Matplotlib Options === #
cm = plt.cm.get_cmap('tab10')
plot_args = {"linestyle": "-",
             "marker": "o",
             "markeredgecolor": "k",
             "markersize": 10,
             "linewidth": 8,

             }
sns.set_context("talk", font_scale=1.3)
fig_size = (5, 4)

# ====== Experiment 1: Increase number of experts ======

n_experts = [1, 2, 4, 6, 8]
# No Percentage
# ece_softmax = np.array([0.035688892, 0.0393773, 0.042222522, 0.07813597, 0.13169229])
# ece_ova = np.array([0.031140802, 0.038146082, 0.023401938, 0.03507, 0.027209044])

# Percentage
ece_softmax = np.array([0.035688892, 0.0393773, 0.042222522, 0.07813597, 0.13169229]) * 100
ece_ova = np.array([0.031140802, 0.038146082, 0.023401938, 0.03507, 0.027209044]) * 100

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(n_experts, ece_softmax, label=r"Softmax", **plot_args)
ax.plot(n_experts, ece_ova, label=r"OvA", **plot_args)
plt.xticks(n_experts, n_experts)
plt.yticks(list(plt.yticks()[0][::2]))
plt.ylabel(r'Average ECE ($\%$)')
plt.xlabel(r'Number of Experts')
# plt.title(r"MoG")
# plt.legend(loc="best")
plt.grid()
# plt.tight_layout()
f.set_tight_layout(True)
plt.savefig("mog_increase_experts.pdf", bbox_inches='tight')

# ====== Experiment 2: Increase confidence for 3 experts ======
p_experts = [25, 45, 65, 75, 85, 95]
# Percentage

ece_softmax = np.array([0.2047, 0.2982, 0.3592, 0.3907, 0.4194, 0.4165]) * 100
ece_ova = np.array([0.0793, 0.0691, 0.0567, 0.0517, 0.0527, 0.0561]) * 100

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(p_experts, ece_softmax, label=r"Softmax", **plot_args)
ax.plot(p_experts, ece_ova, label=r"OvA", **plot_args)
plt.xticks(p_experts, p_experts)
plt.yticks(list(plt.yticks()[0][::2]))
plt.ylabel(r'ECE ($\%$) Rand. Expert')
plt.xlabel(r'Expert Correctness ($\%$)')
# plt.title(r"MoG")
# plt.legend()
plt.grid()
f.set_tight_layout(True)
plt.savefig("mog_increase_confidence.pdf", bbox_inches='tight')

# # ====== Accuracies: Increasing # Experts ======
n_experts = [1, 2, 4, 6, 8]
acc_softmax = np.array([91.416, 90.730, 90.599, 90.614, 78.75])
acc_ova = np.array([91.785, 90.955, 91.135, 90.970, 90.995])

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(n_experts, acc_softmax, label=r"Softmax", **plot_args)
ax.plot(n_experts, acc_ova, label=r"OvA", **plot_args)
plt.xticks(n_experts, n_experts)

plt.ylabel(r'System Accuracy ($\%$)')
plt.xlabel(r'Number of Experts')
# plt.title(r"MoG")
# plt.legend(loc="best")
plt.grid()
f.set_tight_layout(True)
plt.savefig("mog_acc_increase_experts.pdf", bbox_inches='tight')
