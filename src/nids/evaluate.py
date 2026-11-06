"""Metrics for the binary (benign vs. attack) and multiclass (attack type) tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from .data import BENIGN

FPR_BUDGET = 0.001  # 0.1%: roughly 1 false alarm per 1,000 benign flows


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Detection metrics with attack as the positive class.

    false_positive_rate = benign flows flagged as attacks / all benign flows.
    detection_rate is recall on attack flows. dr_at_fpr_0.1pct is the best
    detection rate any threshold achieves while keeping FPR <= 0.1%.
    """
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr_curve, tpr_curve, _ = roc_curve(y_true, proba)
    within_budget = fpr_curve <= FPR_BUDGET
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "detection_rate": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": f1_score(y_true, pred),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "dr_at_fpr_0.1pct": float(tpr_curve[within_budget].max()) if within_budget.any() else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def roc_points(y_true: np.ndarray, proba: np.ndarray, n: int = 400) -> pd.DataFrame:
    """Down-sampled ROC curve for plotting (log-spaced on FPR so the low-FPR region is kept)."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    grid = np.unique(np.concatenate([[0.0], np.logspace(-6, 0, n)]))
    tpr_on_grid = np.interp(grid, fpr, tpr)
    return pd.DataFrame({"fpr": grid, "tpr": tpr_on_grid})


def per_attack_detection(labels: pd.Series, pred_attack: np.ndarray) -> pd.DataFrame:
    """For each attack type: how many of its test flows the binary detector flagged."""
    df = pd.DataFrame({"label": labels.to_numpy(), "flagged": pred_attack.astype(int)})
    out = df.groupby("label")["flagged"].agg(n_test="size", detected="sum")
    out["detection_rate"] = out["detected"] / out["n_test"]
    return out.drop(index=BENIGN, errors="ignore").sort_values("detection_rate")


def multiclass_tables(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]):
    """Per-class precision/recall/F1, the confusion matrix, and each class's most common mistake."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    per_class = pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s})

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=pd.Index(classes, name="true"), columns=pd.Index(classes, name="predicted"))

    off = cm.copy()
    np.fill_diagonal(off, 0)
    off_diag = pd.DataFrame(off, index=classes, columns=classes)
    per_class["most_confused_with"] = [
        off_diag.loc[c].idxmax() if off_diag.loc[c].sum() else "" for c in classes
    ]
    per_class["n_confused"] = [int(off_diag.loc[c].max()) for c in classes]

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
    }
    return per_class, cm_df, summary
