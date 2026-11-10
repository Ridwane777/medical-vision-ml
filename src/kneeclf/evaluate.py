"""Threshold selection on validation, metrics, comparison table and figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, fbeta_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def decision_scores(model, X):
    """Continuous score, higher = more likely positive (decision_function or P(y=1))."""
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(X), dtype=float)
        except AttributeError:
            pass
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def choose_threshold(y, scores, mode="target_recall", target_recall=0.85, beta=2.0):
    """Pick a decision threshold on validation scores.

    target_recall: the highest threshold whose recall is still >= target (best precision at that recall).
    f2: the threshold maximizing F-beta (beta=2 weighs recall twice as much as precision).
    """
    y = np.asarray(y)
    prec, rec, thr = precision_recall_curve(y, scores)
    prec, rec = prec[:-1], rec[:-1]     # aligned with thr (ascending)
    if mode == "target_recall":
        ok = np.flatnonzero(rec >= target_recall)
        i = ok[-1] if len(ok) else 0
    elif mode == "f2":
        f = (1 + beta ** 2) * prec * rec / np.maximum(beta ** 2 * prec + rec, 1e-12)
        i = int(np.argmax(f))
    else:
        raise ValueError(f"unknown threshold mode {mode!r}")
    return float(thr[i])


def binary_metrics(y, scores, threshold):
    y = np.asarray(y)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two = len(np.unique(y)) == 2
    m = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": fbeta_score(y, pred, beta=1, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y, scores) if two else float("nan"),
        "pr_auc": average_precision_score(y, scores) if two else float("nan"),
    }
    m = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}
    m["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return m


def comparison_table(rows):
    """rows: list of dicts (one per candidate) -> DataFrame sorted by validation ROC-AUC."""
    df = pd.DataFrame(rows)
    return df.sort_values("val_roc_auc", ascending=False).reset_index(drop=True)


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def plot_curves(y, scores, threshold, title, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fpr, tpr, _ = roc_curve(y, scores)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, scores):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    pred = np.asarray(scores) >= threshold
    y = np.asarray(y)
    axes[0].scatter([pred[y == 0].mean()], [pred[y == 1].mean()], color="red", zorder=3, label="chosen threshold")
    axes[0].set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (recall)", title="ROC")
    axes[0].legend(loc="lower right")
    prec, rec, _ = precision_recall_curve(y, scores)
    axes[1].plot(rec, prec, label=f"AP = {average_precision_score(y, scores):.3f}")
    axes[1].axhline(y.mean(), ls="--", color="grey", lw=0.8, label="prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    axes[1].legend(loc="lower left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return fig


def plot_confusion(cm, class_names, title, path):
    import matplotlib.pyplot as plt
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], [f"pred {c}" for c in class_names], fontsize=8)
    ax.set_yticks([0, 1], [f"true {c}" for c in class_names], fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return fig
