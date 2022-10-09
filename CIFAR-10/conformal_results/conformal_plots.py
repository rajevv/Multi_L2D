import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc

# === Latex Options === #
rc('font', family='serif')
rc('text', usetex=True)

# === Matplotlib Options === #
cm = plt.cm.get_cmap('tab10')
plot_args = {"marker": "o",
             "markeredgecolor": "k",
             "markersize": 10,
             "linewidth": 8
             }
sns.set_context("talk", font_scale=1.3)
fig_size = (7,7)


# === Plotting functions === #
def plot_sys_acc():
    pass


def plot_exp_acc():
    pass


def plot_qhat():
    pass


def plot_coverage():
    pass