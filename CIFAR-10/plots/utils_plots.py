import matplotlib.pyplot as plt

import seaborn as sns

from tueplots import bundles
from tueplots import figsizes, fonts, axes, fontsizes, markers

PALETTE = "deep"
# PALETTE = "tab10"
# PALETTE = "colorblind"

# === OLD FIGURE OPTIONS === #
# # Latex Options === #
# rc('font', family='serif')
# rc('text', usetex=True)

# # Matplotlib Options === #
# cm = sns.color_palette("deep")
# global_plot_args = {"marker": "o",
#                     "markeredgecolor": "k",
#                     "markersize": 10,
#                     "linewidth": 8
#                     }
# sns.set_context("notebook", font_scale=1.3)
# fig_size = (7, 7)

# # ============================ #
# # === AISTATS 2023 OPTIONS === #
# # ============================ #
# # === Seaborn Color ===
# sns.set_palette(sns.color_palette("tab10"))
# # sns.set_palette(sns.color_palette("deep"))
# # sns.set_palette(sns.color_palette("colorblind"))
#
# # === TUEPLOTS CONFIG ===
# # Increase resolution ===
# plt.rcParams.update({"figure.dpi": 150})
# # Load bundle ===
# # plt.rcParams.update(bundles.aistats2023())
#
# # Update figsize ===
# # full =
# # plt.rcParams.update(figsizes.aistats2023_full())
# # half =
# plt.rcParams.update(figsizes.aistats2023_half(constrained_layout=False,
#                                               tight_layout=True,
#                                               height_to_width_ratio=1))  # make square fig.
#
# # FONT ===
# # Update font
# # font family =
# plt.rcParams.update(fonts.aistats2022_tex(family="serif"))
# # font size =
# plt.rcParams.update(fontsizes.aistats2023(default_smaller=0))
# # custom fontsize
# # plt.rcParams.update({"font.size": 10})
#
# # AXES ===
# plt.rcParams.update(axes.lines(base_width=1,
#                                line_base_ratio=4))  # increase base_width for thicker lines
# plt.rcParams.update(axes.grid(grid_alpha=0.5))  # custom grid. alpha=0-1, for transparency
# # plt.rcParams.update(axes.lines())  # increase base_width for thicker lines
#
# # OTHERS ===
# # Markers
# # plt.rcParams.update(markers.with_edge())
# # Error bars capsize
# plt.rcParams.update({"errorbar.capsize": 2})
# plt.rcParams.update(markers.with_edge())
#


def set_aistats2023_style():
    # Color
    sns.set_palette(sns.color_palette(PALETTE))
    # === TUEPLOTS CONFIG ===
    # Increase resolution
    plt.rcParams.update({"figure.dpi": 150})
    # Figsize
    plt.rcParams.update(figsizes.aistats2023_half(constrained_layout=False,
                                                  tight_layout=True,
                                                  height_to_width_ratio=1))  # make square fig.

    # Font
    plt.rcParams.update(fonts.aistats2022_tex(family="serif"))
    # Font size
    plt.rcParams.update(fontsizes.aistats2023(default_smaller=0))
    # Axes ===
    plt.rcParams.update(axes.lines(base_width=1,  # base width for line params.
                                   line_base_ratio=4))  # increase only linewidth
    plt.rcParams.update(axes.grid(grid_alpha=0.5))  # custom grid. alpha=0-1, for transparency
    # Markers
    plt.rcParams.update({"errorbar.capsize": 2})  # error bars capsize
    plt.rcParams.update(markers.with_edge())  # set markers with black edge
