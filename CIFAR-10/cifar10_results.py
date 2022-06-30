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
             "linewidth": 8
             }
sns.set_context("talk", font_scale=1.3)
fig_size = (5, 4)

# ====== Experiment 1: Increase number of experts ======
n_experts = [1, 2, 4, 6, 8]
# Percentage
ece_softmax = np.array([0.04212178, 0.03236192, 0.04614954, 0.048385993, 0.046213374]) * 100
ece_ova = np.array([0.033160426, 0.027528169, 0.029059665, 0.028813586, 0.03282707]) * 100

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(n_experts, ece_softmax, label=r"Softmax", **plot_args)
ax.plot(n_experts, ece_ova, label=r"OvA", **plot_args)
plt.xticks(n_experts, n_experts)
plt.yticks(list(plt.yticks()[0])[::2])
plt.ylabel(r'Average ECE ($\%$)')
plt.xlabel(r'Number of Experts')
# plt.title(r"CIFAR-10")
# plt.legend(loc="best")
plt.grid()
f.set_tight_layout(True)
plt.savefig("cifar_increase_experts.pdf", bbox_inches='tight')

# # ====== Experiment 2: Increase confidence for 3 experts ======
p_experts = [20, 40, 60, 80, 95]
# Percentage
ece_softmax = np.array([0.20560052, 0.3304271, 0.43687683, 0.5407775, 0.5562724]) * 100
ece_ova = np.array([0.007408993, 0.009338078, 0.009841489, 0.008472728, 0.011463852]) * 100

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(p_experts, ece_softmax, label=r"Softmax", **plot_args)
ax.plot(p_experts, ece_ova, label=r"OvA", **plot_args)
plt.ylim([-1, np.max(ece_softmax) + 2])
plt.yticks(list(plt.yticks()[0])[1:])
plt.xticks(p_experts, p_experts)
plt.ylabel(r'ECE ($\%$) Rand. Expert')
plt.xlabel(r'Expert Correctness ($\%$)')
# plt.title(r"CIFAR-10")
# plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("cifar_increase_confidence.pdf", bbox_inches='tight')

# ====== Accuracies: Increasing # Experts ======

n_experts = [1, 2, 4, 6, 8]

acc_softmax = np.array([92.317, 92.730, 90.09, 90.2126, 91.5907])
acc_ova = np.array([92.9145, 93.4678, 93.6631, 93.4678, 92.6757])

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(n_experts, acc_softmax, label=r"Softmax", **plot_args)
ax.plot(n_experts, acc_ova, label=r"OvA", **plot_args)
plt.xticks(n_experts, n_experts)
plt.ylabel(r'System Accuracy ($\%$)')
plt.xlabel(r'Number of Experts')
# plt.title(r"CIFAR-10")
# plt.legend()
plt.grid()
f.set_tight_layout(True)
plt.savefig("cifar_acc_increase_experts.pdf", bbox_inches='tight')

# # ====== Experiment 3:Overlapping Experts ======
n_experts = [1, 2, 3, 4, 5]
# Percentage
ece_softmax = np.array([0.06, 0.084, 0.072, 0.094, 0.112]) * 100
ece_ova = np.array([0.043, 0.049, 0.038, 0.051, 0.045]) * 100

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(n_experts, ece_softmax, label=r"Softmax", **plot_args)
ax.plot(n_experts, ece_ova, label=r"OvA", **plot_args)
plt.xticks(n_experts, n_experts)
plt.yticks(list(plt.yticks()[0])[::2])
plt.ylabel(r'Average ECE ($\%$)')
plt.xlabel(r'Number of Experts')
# plt.title(r"CIFAR-10")
# plt.legend(loc="best")
plt.grid()
f.set_tight_layout(True)
plt.savefig("cifar_overlapping_ece.pdf", bbox_inches='tight')

# # ====== Accuracy Experiment 3:Overlapping Experts ======
n_experts = [1, 2, 3, 4, 5]
# Percentage
acc_softmax = np.array([87.22, 88.34, 87.71, 87.02, 88.43])
acc_ova = np.array([87.84, 90.917, 89.35, 87.08, 89.73])

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(n_experts, acc_softmax, label=r"Softmax", **plot_args)
ax.plot(n_experts, acc_ova, label=r"OvA", **plot_args)
plt.xticks(n_experts, n_experts)
plt.yticks(list(plt.yticks()[0])[::2])
plt.ylabel(r'System Accuracy ($\%$)')
plt.xlabel(r'Number of Experts')
# plt.title(r"CIFAR-10")
# plt.legend(loc="best")
plt.grid()
f.set_tight_layout(True)
plt.savefig("cifar_overlapping_acc.pdf", bbox_inches='tight')
