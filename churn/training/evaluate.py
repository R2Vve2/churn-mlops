"""Metrics, threshold selection and diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "expected_cost": float(fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE),
    }


def choose_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> dict[str, float]:
    """Pick the probability cut-off that minimises expected business cost, and the F1-optimal one.

    Should be called on *out-of-fold* probabilities so the threshold is not tuned on data the
    model has already seen.
    """
    grid = np.linspace(0.05, 0.95, 181)
    costs = []
    f1s = []
    for t in grid:
        pred = proba >= t
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        costs.append(fn * cost_fn + fp * cost_fp)
        f1s.append(f1_score(y_true, pred, zero_division=0))
    return {
        "cost_optimal": float(grid[int(np.argmin(costs))]),
        "f1_optimal": float(grid[int(np.argmax(f1s))]),
        "min_expected_cost": float(min(costs)),
    }


def save_plots(y_true: np.ndarray, proba: np.ndarray, threshold: float, importances: dict[str, float], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, proba):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"), ax.set_ylabel("True positive rate"), ax.legend(loc="lower right")
    ax.set_title("ROC curve (test set)")
    paths.append(out_dir / "roc_curve.png"), fig.tight_layout(), fig.savefig(paths[-1], dpi=120), plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"PR AUC = {average_precision_score(y_true, proba):.3f}")
    ax.axhline(y_true.mean(), color="k", ls="--", lw=0.8, label=f"base rate = {y_true.mean():.2f}")
    ax.set_xlabel("Recall"), ax.set_ylabel("Precision"), ax.legend(loc="upper right"), ax.set_title("Precision-recall (test set)")
    paths.append(out_dir / "pr_curve.png"), fig.tight_layout(), fig.savefig(paths[-1], dpi=120), plt.close(fig)

    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, 9)
    mean_pred = [proba[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    frac_pos = [y_true[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred, frac_pos, "o-", label=f"Brier = {brier_score_loss(y_true, proba):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    ax.set_xlabel("Mean predicted probability"), ax.set_ylabel("Observed churn rate"), ax.legend(), ax.set_title("Calibration")
    paths.append(out_dir / "calibration.png"), fig.tight_layout(), fig.savefig(paths[-1], dpi=120), plt.close(fig)

    top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:12][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([k for k, _ in top], [v for _, v in top])
    ax.set_xlabel("Permutation importance (drop in ROC AUC)"), ax.set_title("Feature importance (test set)")
    paths.append(out_dir / "feature_importance.png"), fig.tight_layout(), fig.savefig(paths[-1], dpi=120), plt.close(fig)

    cm = confusion_matrix(y_true, (proba >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black" if v < cm.max() / 2 else "white")
    ax.set_xticks([0, 1]), ax.set_yticks([0, 1]), ax.set_xticklabels(["stay", "churn"]), ax.set_yticklabels(["stay", "churn"])
    ax.set_xlabel("Predicted"), ax.set_ylabel("Actual"), ax.set_title(f"Confusion matrix @ {threshold:.2f}")
    paths.append(out_dir / "confusion_matrix.png"), fig.tight_layout(), fig.savefig(paths[-1], dpi=120), plt.close(fig)
    return paths
