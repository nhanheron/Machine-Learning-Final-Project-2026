"""
Evaluation utilities: AUC-ROC, AUC-PR, sensitivity @ specificity, and plots.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .dataset import TARGET_LABELS


# -----------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------

def compute_aurocs(
    labels: np.ndarray,
    preds: np.ndarray,
    target_labels: Optional[list[str]] = None,
) -> dict[str, float]:
    """AUC-ROC per pathology, skipping columns with only one class present."""
    target_labels = target_labels or TARGET_LABELS
    aurocs = {}
    for i, name in enumerate(target_labels):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        valid = ~np.isnan(y_true) & (y_true != -1.0)
        y_true_v = y_true[valid]
        y_pred_v = y_pred[valid]
        if len(np.unique(y_true_v)) < 2:
            aurocs[name] = float("nan")
        else:
            aurocs[name] = roc_auc_score(y_true_v, y_pred_v)
    return aurocs


def compute_auprcs(
    labels: np.ndarray,
    preds: np.ndarray,
    target_labels: Optional[list[str]] = None,
) -> dict[str, float]:
    """AUC-PR per pathology."""
    target_labels = target_labels or TARGET_LABELS
    auprcs = {}
    for i, name in enumerate(target_labels):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        valid = ~np.isnan(y_true) & (y_true != -1.0)
        y_true_v = y_true[valid]
        y_pred_v = y_pred[valid]
        if len(np.unique(y_true_v)) < 2:
            auprcs[name] = float("nan")
        else:
            auprcs[name] = average_precision_score(y_true_v, y_pred_v)
    return auprcs


def sensitivity_at_specificity(
    labels: np.ndarray,
    preds: np.ndarray,
    target_specificity: float = 0.90,
    target_labels: Optional[list[str]] = None,
) -> dict[str, float]:
    """Sensitivity (recall) at a given specificity operating point."""
    target_labels = target_labels or TARGET_LABELS
    result = {}
    for i, name in enumerate(target_labels):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        valid = ~np.isnan(y_true) & (y_true != -1.0)
        y_true_v = y_true[valid]
        y_pred_v = y_pred[valid]

        if len(np.unique(y_true_v)) < 2:
            result[name] = float("nan")
            continue

        fpr, tpr, _ = roc_curve(y_true_v, y_pred_v)
        specificity = 1.0 - fpr
        # Find the tpr at the closest specificity >= target
        idx = np.where(specificity >= target_specificity)[0]
        if len(idx) == 0:
            result[name] = float("nan")
        else:
            result[name] = tpr[idx[-1]]
    return result


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_roc_curves(
    results: list[dict],
    target_labels: Optional[list[str]] = None,
    figsize: tuple = (18, 4),
) -> plt.Figure:
    """One ROC subplot per pathology with all strategies overlaid."""
    target_labels = target_labels or TARGET_LABELS
    n = len(target_labels)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for col_idx, (ax, name) in enumerate(zip(axes, target_labels)):
        for res in results:
            y_true = res["labels"][:, col_idx]
            y_pred = res["preds"][:, col_idx]
            valid = ~np.isnan(y_true) & (y_true != -1.0)
            y_true_v = y_true[valid]
            y_pred_v = y_pred[valid]

            if len(np.unique(y_true_v)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true_v, y_pred_v)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{res['strategy']} ({roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(name)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=7)

    fig.suptitle("ROC Curves by Strategy", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_auc_comparison(
    results: list[dict],
    target_labels: Optional[list[str]] = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Grouped bar chart of AUC-ROC per strategy per pathology."""
    target_labels = target_labels or TARGET_LABELS
    strategies = [r["strategy"] for r in results]
    n_strategies = len(strategies)
    n_labels = len(target_labels)

    auc_matrix = np.zeros((n_strategies, n_labels))
    for i, res in enumerate(results):
        for j, name in enumerate(target_labels):
            auc_matrix[i, j] = res["aurocs"].get(name, float("nan"))

    x = np.arange(n_labels)
    width = 0.8 / n_strategies

    fig, ax = plt.subplots(figsize=figsize)
    for i, strat in enumerate(strategies):
        offset = (i - n_strategies / 2 + 0.5) * width
        bars = ax.bar(x + offset, auc_matrix[i], width, label=strat)
        for bar, val in zip(bars, auc_matrix[i]):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC Comparison Across Strategies")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    return fig


def plot_sensitivity_comparison(
    results: list[dict],
    target_labels: Optional[list[str]] = None,
    target_specificity: float = 0.90,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Grouped bar chart of sensitivity @ given specificity."""
    target_labels = target_labels or TARGET_LABELS
    strategies = [r["strategy"] for r in results]
    n_strategies = len(strategies)
    n_labels = len(target_labels)

    sens_matrix = np.zeros((n_strategies, n_labels))
    for i, res in enumerate(results):
        sens = sensitivity_at_specificity(
            res["labels"], res["preds"], target_specificity, target_labels
        )
        for j, name in enumerate(target_labels):
            sens_matrix[i, j] = sens.get(name, float("nan"))

    x = np.arange(n_labels)
    width = 0.8 / n_strategies

    fig, ax = plt.subplots(figsize=figsize)
    for i, strat in enumerate(strategies):
        offset = (i - n_strategies / 2 + 0.5) * width
        ax.bar(x + offset, sens_matrix[i], width, label=strat)

    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)
    ax.set_ylabel(f"Sensitivity @ {target_specificity:.0%} Specificity")
    ax.set_title("Clinical Safety: Sensitivity at High Specificity")
    ax.legend()
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    return fig


def build_summary_table(
    results: list[dict],
    target_labels: Optional[list[str]] = None,
) -> str:
    """Return a markdown table summarising AUC-ROC, AUC-PR, and Sens@90%Spec."""
    target_labels = target_labels or TARGET_LABELS
    lines = []
    header = "| Strategy | " + " | ".join(target_labels) + " | Mean AUC |"
    sep = "|---|" + "|".join(["---"] * len(target_labels)) + "|---|"
    lines.append(header)
    lines.append(sep)

    for res in results:
        row = f"| {res['strategy']} |"
        aucs = []
        for name in target_labels:
            v = res["aurocs"].get(name, float("nan"))
            aucs.append(v)
            row += f" {v:.3f} |"
        row += f" {np.nanmean(aucs):.3f} |"
        lines.append(row)

    return "\n".join(lines)
