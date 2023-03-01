import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

# === Matplotlib Options === #
cm = plt.cm.get_cmap('tab10')
plot_args = {"linestyle": "-",
             "marker": "o",
             "markeredgecolor": "k",
             "markersize": 10,
             "linewidth": 8
             }
sns.set_context("poster", font_scale=1.3)

cluster1 = np.load("cluster1.npy")
cluster2 = np.load("cluster2.npy")
cluster3 = np.load("cluster3.npy")
cluster4 = np.load("cluster4.npy")

sns.set_theme()
with sns.axes_style("ticks"):
    fig = plt.figure(figsize=(3, 3))

    plt.scatter(cluster1[:, 0], cluster1[:, 1], alpha=0.5)  # zeroth class
    plt.scatter(cluster2[:, 0], cluster2[:, 1], alpha=0.5)  # oneth class
    plt.scatter(cluster3[:, 0], cluster3[:, 1], alpha=0.5)  # second class
    plt.scatter(cluster4[:, 0], cluster4[:, 1], alpha=0.5)  # third class
    sns.despine()

    fig.set_tight_layout(True)
    plt.savefig("mog_plot.pdf", bbox_inches='tight')
