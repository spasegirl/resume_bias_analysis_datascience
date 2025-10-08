
"""
fairness_metrics.py
--------------------
Lightweight, dependency-minimal fairness metrics helpers for binary classifiers.
Computes per-group rates (TPR/FPR/SelectionRate), Equalized Odds gaps,
Demographic Parity gaps, and Calibration-by-Group (ECE, Brier, curves).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, f1_score

# -----------------------
# Utility / Math helpers
# -----------------------
def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0

def _ensure_numpy(x) -> np.ndarray:
    return x.values if hasattr(x, "values") else np.asarray(x)

# -----------------------------------
# Core: confusion counts per submask
# -----------------------------------
def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1) -> Tuple[int, int, int, int]:
    y_true = _ensure_numpy(y_true)
    y_pred = _ensure_numpy(y_pred)
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    tn = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))
    fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
    return tp, fp, tn, fn

# -----------------------------------
# Public: per-group rate computation
# -----------------------------------
def compute_group_rates(
    y_true,
    y_pred,
    y_prob,
    group,
    pos_label: int = 1,
    auc_if_possible: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per group containing:
    n, pos_rate_true, selection_rate, TPR, FPR, TNR, FNR, Precision, NPV, AUC (if possible), Brier.
    """
    y_true = _ensure_numpy(y_true)
    y_pred = _ensure_numpy(y_pred)
    y_prob = _ensure_numpy(y_prob) if y_prob is not None else None
    group  = _ensure_numpy(group)

    rows = []
    for g in pd.unique(group):
        m = (group == g)
        yt, yp = y_true[m], y_pred[m]
        tp, fp, tn, fn = _confusion_counts(yt, yp, pos_label=pos_label)

        p = int(np.sum(yt == pos_label))
        n = int(np.sum(yt != pos_label))
        total = int(m.sum())

        tpr = _safe_div(tp, p)           # recall for positive class
        fpr = _safe_div(fp, n)
        tnr = _safe_div(tn, n)
        fnr = _safe_div(fn, p)
        sel = _safe_div(tp + fp, total)  # P(ŷ=1)
        pos_rate_true = _safe_div(p, total)
        prec = _safe_div(tp, (tp + fp))
        npv  = _safe_div(tn, (tn + fn))

        auc = np.nan
        brier = np.nan
        if (y_prob is not None) and total > 1 and (len(np.unique(yt)) == 2):
            try:
                auc = float(roc_auc_score(yt, y_prob[m]))
            except Exception:
                auc = np.nan
            try:
                brier = float(brier_score_loss(yt, y_prob[m]))
            except Exception:
                brier = np.nan

        rows.append(dict(
            group=g, n=total, pos_rate_true=pos_rate_true, selection_rate=sel,
            TPR=tpr, FPR=fpr, TNR=tnr, FNR=fnr, Precision=prec, NPV=npv,
            AUC=auc, Brier=brier
        ))
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)

# -----------------------------------
# Equalized Odds & Demographic Parity
# -----------------------------------
def equalized_odds_gaps(group_rates: pd.DataFrame) -> Dict[str, float]:
    """
    Returns {'TPR_gap': max-min, 'FPR_gap': max-min} across groups.
    """
    tpr_gap = float(group_rates["TPR"].max() - group_rates["TPR"].min()) if "TPR" in group_rates else np.nan
    fpr_gap = float(group_rates["FPR"].max() - group_rates["FPR"].min()) if "FPR" in group_rates else np.nan
    return {"TPR_gap": tpr_gap, "FPR_gap": fpr_gap}

def demographic_parity_gap(group_rates: pd.DataFrame) -> float:
    """
    Returns SelectionRate gap = max-min across groups.
    """
    return float(group_rates["selection_rate"].max() - group_rates["selection_rate"].min())

# -------------------------
# Calibration by Group (ECE)
# -------------------------
def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Computes ECE with equal-width bins on [0,1].
    """
    y_true = _ensure_numpy(y_true).astype(int)
    y_prob = _ensure_numpy(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(m):
            continue
        conf = float(np.mean(y_prob[m]))
        acc  = float(np.mean(y_true[m]))
        w    = float(np.mean(m))  # bin weight
        ece += w * abs(acc - conf)
    return ece

def calibration_by_group(y_true, y_prob, group, n_bins: int = 10) -> Tuple[Dict, pd.DataFrame]:
    """
    Returns (curves_by_group, summary_df)

    curves_by_group: dict[group] -> DataFrame with columns
        ['bin_lower','bin_upper','bin_center','mean_prob','frac_pos','count']

    summary_df: one row per group with columns ['group','ECE','Brier','n']
    """
    y_true = _ensure_numpy(y_true).astype(int)
    y_prob = _ensure_numpy(y_prob).astype(float)
    group  = _ensure_numpy(group)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    curves = {}
    summary_rows = []

    for g in pd.unique(group):
        m = (group == g)
        yt, yp = y_true[m], y_prob[m]
        rows = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            in_bin = (yp >= lo) & (yp < hi) if i < n_bins - 1 else (yp >= lo) & (yp <= hi)
            cnt = int(np.sum(in_bin))
            if cnt == 0:
                rows.append(dict(bin_lower=lo, bin_upper=hi, bin_center=(lo+hi)/2, mean_prob=np.nan, frac_pos=np.nan, count=0))
                continue
            mean_prob = float(np.mean(yp[in_bin]))
            frac_pos  = float(np.mean(yt[in_bin]))
            rows.append(dict(bin_lower=lo, bin_upper=hi, bin_center=(lo+hi)/2, mean_prob=mean_prob, frac_pos=frac_pos, count=cnt))
        df_curve = pd.DataFrame(rows)
        curves[g] = df_curve
        ece = expected_calibration_error(yt, yp, n_bins=n_bins)
        try:
            brier = float(brier_score_loss(yt, yp))
        except Exception:
            brier = np.nan
        summary_rows.append(dict(group=g, ECE=ece, Brier=brier, n=int(m.sum())))

    return curves, pd.DataFrame(summary_rows).sort_values("group").reset_index(drop=True)

# -------------------------
# Threshold sweep utilities
# -------------------------
def threshold_sweep(
    y_true,
    y_prob,
    group,
    thresholds: Iterable[float] = np.linspace(0.0, 1.0, 21),
    pos_label: int = 1,
) -> pd.DataFrame:
    """
    For each threshold, computes global metrics (accuracy, macro-f1) and fairness gaps (TPR/FPR/Selection).
    Returns DataFrame with one row per threshold.
    """
    y_true = _ensure_numpy(y_true).astype(int)
    y_prob = _ensure_numpy(y_prob).astype(float)
    group  = _ensure_numpy(group)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro"))
        gr = compute_group_rates(y_true, y_pred, y_prob, group, pos_label=pos_label)
        eo = equalized_odds_gaps(gr)
        dp = demographic_parity_gap(gr)
        rows.append(dict(
            threshold=float(t),
            accuracy=acc,
            macro_f1=f1m,
            TPR_gap=eo["TPR_gap"],
            FPR_gap=eo["FPR_gap"],
            Selection_gap=dp
        ))
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

# -------------------------
# Plotting (matplotlib-only)
# -------------------------
import matplotlib.pyplot as plt

def plot_calibration_by_group(curves_by_group: Dict, title: str = "Calibration by Group"):
    """
    Plots reliability curves per group (one figure). Avoids explicit color settings.
    """
    plt.figure(figsize=(8, 6))
    xs = np.linspace(0, 1, 100)
    plt.plot(xs, xs, linestyle="--", label="Perfectly calibrated")
    for g, df_curve in curves_by_group.items():
        # Drop empty bins to avoid jagged lines
        dfp = df_curve.dropna(subset=["mean_prob","frac_pos"])
        if not dfp.empty:
            plt.plot(dfp["mean_prob"], dfp["frac_pos"], marker="o", label=str(g))
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()
