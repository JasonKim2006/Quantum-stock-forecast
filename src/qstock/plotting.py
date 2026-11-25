"""Shared matplotlib style so every figure in the repo reads as one set."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots, always assigned in this order (validated for colour-vision
# deficiency). Each series also gets its own marker so colour is never the only cue.
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600"]
MARKERS = ["o", "s", "^", "D"]
INK = "#191919"
INK_2 = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "font.family": "sans-serif",
        }
    )
