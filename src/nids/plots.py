"""Render figures/ from the CSV/JSON files in results/.

    python -m nids.plots --results results --out figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from .models import DISPLAY_NAMES  # noqa: E402

# Colour tokens (validated categorical palette, light mode). Each model keeps the
# same colour in every figure.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MODEL_COLORS = {"hist_gb": "#2a78d6", "random_forest": "#eb6834", "logreg": "#1baf7a"}
MODEL_ORDER = ["hist_gb", "random_forest", "logreg"]
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "font.family": "DejaVu Sans",
    }
)


def _header(ax, title, subtitle, legend_cols=0):
    """Title, then subtitle, then (optionally) a one-row legend, stacked above the plot."""
    top = 4
    if legend_cols:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=legend_cols, borderaxespad=0.4,
                  handletextpad=0.3, columnspacing=1.4)
        top = 24
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, top + 4),
                textcoords="offset points", fontsize=9.5, color=INK_2, va="bottom")
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, top + 22),
                textcoords="offset points", fontsize=13, fontweight="bold", color=INK, va="bottom")


def _save(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


def per_attack_detection(results: Path, out: Path):
    det = pd.read_csv(results / "per_attack_detection.csv", index_col="attack")
    models = [m for m in MODEL_ORDER if m in det.columns]
    det = det.sort_values([models[0], "n_test"], ascending=[True, False])  # hardest first = top row
    y = np.arange(len(det))

    fig, ax = plt.subplots(figsize=(8.2, 0.5 * len(det) + 1.6))
    # Small vertical dodge so models with identical scores (often 100%) stay visible.
    offsets = np.linspace(-0.2, 0.2, len(models)) if len(models) > 1 else [0.0]
    for m, dy in zip(models, offsets):
        ax.scatter(det[m] * 100, y + dy, s=48, color=MODEL_COLORS[m], edgecolor=SURFACE, linewidth=1.5,
                   label=DISPLAY_NAMES[m], zorder=3)
    for yy in y[:-1]:
        ax.axhline(yy + 0.5, color=GRID, linewidth=0.6, zorder=1)
    ax.set_yticks(y, [f"{a}  (n={n:,})" for a, n in zip(det.index, det["n_test"])])
    ax.set_xlim(-2, 102)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of that attack's test flows flagged as malicious (binary detector)")
    ax.set_ylim(len(det) - 0.5, -0.5)
    _header(ax, "Detection rate by attack type",
            "Hardest attacks at the top. n = deduplicated test flows of that type.", legend_cols=len(models))
    _save(fig, out / "per_attack_detection.png")


def error_tradeoff(results: Path, out: Path):
    """DET-style plot: missed attacks vs. false alarms, both on log axes.

    A plain ROC curve hides the differences between strong detectors (they all hug
    100%); on log-log axes the gap between them is visible. Lower-left is better.
    """
    roc = pd.read_csv(results / "roc_curves.csv")
    metrics = pd.read_csv(results / "binary_metrics.csv", index_col="model")
    floor = 1e-4
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for m in [m for m in MODEL_ORDER if m in roc["model"].unique()]:
        r = roc[(roc["model"] == m) & (roc["fpr"] >= floor) & (roc["tpr"] < 1)]  # line ends once nothing is missed
        miss = 1 - r["tpr"]
        ax.plot(r["fpr"], miss, color=MODEL_COLORS[m], linewidth=2,
                label=f"{DISPLAY_NAMES[m]}  (ROC-AUC {metrics.loc[m, 'roc_auc']:.5f})")
        op = metrics.loc[m]
        ax.scatter([op["false_positive_rate"]], [1 - op["detection_rate"]], s=48, color=MODEL_COLORS[m],
                   edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.axvline(1e-3, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(1.12e-3, 0.5, "0.1% FPR", color=INK_2, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(floor, 1)
    ax.set_ylim(5e-6, 1)
    pct = FuncFormatter(lambda v, _: f"{v * 100:g}%")
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("False-positive rate: benign flows flagged (log)")
    ax.set_ylabel("Miss rate: attack flows not flagged (log)")
    _header(ax, "Missed attacks vs. false alarms",
            "Lower-left is better. Dots mark each model at the default 0.5 threshold.")
    ax.legend(loc="upper right")
    _save(fig, out / "error_tradeoff.png")


def confusion_matrix(results: Path, out: Path):
    cm = pd.read_csv(results / "multiclass_confusion_best.csv", index_col=0)
    best = (results / "multiclass_best_model.txt").read_text().strip()
    support = cm.sum(axis=1)
    norm = cm.div(support.replace(0, np.nan), axis=0).fillna(0)
    labels = list(cm.index)

    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    ax.imshow(norm.values, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = norm.values[i, j]
            if v >= 0.005:
                txt = f"{v * 100:.0f}" if v >= 0.015 else "1"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if v > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(labels)), [f"{l}  ({n:,})" for l, n in zip(labels, support)])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class  (test flows)")
    _header(ax, f"Attack-type confusion matrix: {DISPLAY_NAMES.get(best, best)}",
            "Each row sums to 100%: cell = % of that true class predicted as the column class. Blank < 0.5%.")
    _save(fig, out / "confusion_matrix.png")


def feature_importance(results: Path, out: Path, top: int = 15):
    imp = pd.read_csv(results / "feature_importance_rf.csv", index_col=0)["importance"].head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * top + 1.3))
    ax.barh(range(len(imp)), imp.values * 100, height=0.62, color=MODEL_COLORS["random_forest"])
    ax.set_yticks(range(len(imp)), imp.index)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of total impurity reduction")
    _header(ax, f"Top {top} features, Random Forest (binary)",
            "Impurity-based importance; correlated features split credit between them.")
    _save(fig, out / "feature_importance.png")


def class_distribution(results: Path, out: Path):
    s = json.loads((results / "dataset_summary.json").read_text())
    raw = pd.Series(s["raw_label_counts"])
    clean = pd.Series(s["clean_label_counts"]).reindex(raw.index).fillna(0)
    order = raw.sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(order) + 1.5))
    ax.hlines(y, clean[order], raw[order], color=GRID, linewidth=2, zorder=1)
    ax.scatter(raw[order], y, s=48, color=MUTED, edgecolor=SURFACE, linewidth=1.5, label="Raw flows", zorder=3)
    ax.scatter(clean[order], y, s=48, color=MODEL_COLORS["hist_gb"], edgecolor=SURFACE, linewidth=1.5,
               label="After cleaning + dedup", zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{c}  ({int(r):,} \u2192 {int(k):,})" for c, r, k in zip(order, raw[order], clean[order])])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Flows (log scale)")
    _header(ax, "Class sizes before and after cleaning",
            "Dropping destination port, then exact duplicates, collapses repetitive attacks like PortScan.",
            legend_cols=2)
    _save(fig, out / "class_distribution.png")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    a = p.parse_args(argv)
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    class_distribution(results, out)
    per_attack_detection(results, out)
    error_tradeoff(results, out)
    if (results / "feature_importance_rf.csv").exists():
        feature_importance(results, out)
    if (results / "multiclass_confusion_best.csv").exists():
        confusion_matrix(results, out)


if __name__ == "__main__":
    main()
