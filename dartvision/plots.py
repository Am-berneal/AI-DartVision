"""Plotting helpers: dartboard drawing, landing maps, distributions, heatmaps, confusion matrices."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, Wedge

from .board import LABELS, RADII, SECTORS, SECTOR_WIDTH_DEG

C_DARK, C_LIGHT, C_RED, C_GREEN = "#1d1f22", "#f3e9d2", "#c0392b", "#2e7d4f"


def draw_board(ax=None, numbers: bool = True, alpha: float = 1.0):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Circle((0, 0), 1.12, color="#111", alpha=alpha, zorder=0))
    rings = [
        (RADII["outer_bull"], RADII["triple_inner"], "single"),
        (RADII["triple_inner"], RADII["triple_outer"], "ring"),
        (RADII["triple_outer"], RADII["double_inner"], "single"),
        (RADII["double_inner"], RADII["double_outer"], "ring"),
    ]
    for i, num in enumerate(SECTORS):
        centre = 90.0 - i * SECTOR_WIDTH_DEG
        t1, t2 = centre - SECTOR_WIDTH_DEG / 2, centre + SECTOR_WIDTH_DEG / 2
        dark = i % 2 == 0
        for r0, r1, kind in rings:
            if kind == "single":
                col = C_DARK if dark else C_LIGHT
            else:
                col = C_RED if dark else C_GREEN
            ax.add_patch(Wedge((0, 0), r1, t1, t2, width=r1 - r0, facecolor=col,
                               edgecolor="#999", linewidth=0.4, alpha=alpha, zorder=1))
        if numbers:
            a = np.radians(centre)
            ax.text(1.06 * np.cos(a), 1.06 * np.sin(a), str(num), ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold", zorder=3)
    ax.add_patch(Circle((0, 0), RADII["outer_bull"], facecolor=C_GREEN, edgecolor="#999", lw=0.4, alpha=alpha, zorder=2))
    ax.add_patch(Circle((0, 0), RADII["inner_bull"], facecolor=C_RED, edgecolor="#999", lw=0.4, alpha=alpha, zorder=2))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def landing_map(df: pd.DataFrame, hue: str | None = None, title: str = "Landing coordinates", ax=None, s=10):
    ax = draw_board(ax, alpha=0.55)
    if hue and hue in df:
        # draw the widest-spread groups first so tight groups stay visible on top
        spread = df.groupby(hue)[["x", "y"]].std().sum(axis=1)
        groups = list(spread.sort_values(ascending=False).index)
        palette = sns.color_palette("tab10", len(groups))
        for g, c in zip(groups, palette):
            d = df[df[hue] == g]
            ax.scatter(d["x"], d["y"], s=s, color=c, edgecolor="white", linewidth=0.3, label=str(g), zorder=5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    else:
        ax.scatter(df["x"], df["y"], s=s, color="#1f77b4", edgecolor="white", linewidth=0.3, zorder=5)
    ax.set_title(title)
    return ax


def region_distribution(df: pd.DataFrame, title="Region distribution", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4))
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    ax.bar(counts.index, counts.values, color=[C_RED if l.startswith(("T", "D")) else "#2e6f73" for l in counts.index])
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    return ax


def score_distribution(df: pd.DataFrame, title="Score distribution", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    counts = df["score"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#2e6f73")
    ax.set_title(title)
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    return ax


def density_heatmap(df: pd.DataFrame, bins: int = 40, title="Spatial density", ax=None):
    ax = draw_board(ax, alpha=0.35, numbers=True)
    h, xe, ye = np.histogram2d(df["x"], df["y"], bins=bins, range=[[-1.15, 1.15], [-1.15, 1.15]])
    h = np.ma.masked_where(h == 0, h)
    ax.imshow(h.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="Reds", alpha=0.8, zorder=4)
    ax.set_title(title)
    return ax


def confusion_heatmap(cm: np.ndarray, title="Confusion matrix", ax=None, normalise=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 11))
    m = cm.astype(float)
    if normalise:
        row = m.sum(axis=1, keepdims=True)
        m = np.divide(m, row, out=np.zeros_like(m), where=row > 0)
    sns.heatmap(m, xticklabels=LABELS, yticklabels=LABELS, cmap="Reds", ax=ax, cbar_kws={"label": "row share" if normalise else "count"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.tick_params(labelsize=6)
    return ax


def model_comparison(metrics: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    d = metrics.melt(id_vars=["model", "feature_set"], value_vars=["accuracy", "macro_f1"], var_name="metric")
    d["label"] = d["model"]
    sns.barplot(data=d[d["metric"] == "macro_f1"], x="model", y="value", hue="feature_set", ax=ax, palette=["#9fb7b9", C_RED])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1 (held-out test)")
    ax.set_xlabel("")
    ax.set_title("Model comparison: raw (x, y) vs engineered features")
    ax.tick_params(axis="x", rotation=15)
    return ax


def save(fig_or_ax, path):
    fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)