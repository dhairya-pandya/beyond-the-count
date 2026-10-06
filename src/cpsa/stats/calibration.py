"""Calibration of the full model's out-of-fold probabilities: Brier, ECE, reliability bins, Platt recalibration."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_summary(y: np.ndarray, p: np.ndarray, n_bins: int = 10, seed: int = 0) -> dict:
    """Brier (+ skill vs. prevalence), ECE, calibration slope, CV-Platt-recalibrated Brier, reliability bins."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    brier_ref = float(np.mean((y.mean() - y) ** 2))
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bins.append(dict(bin=b, n=int(m.sum()), mean_pred=float(p[m].mean()), frac_pos=float(y[m].mean())))
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    z = _logit(p).reshape(-1, 1)
    slope, brier_recal = float("nan"), float("nan")
    if 0 < y.sum() < len(y):
        lr = LogisticRegression(C=1e6).fit(z, y)
        slope = float(lr.coef_[0, 0])
        k = int(min(5, y.sum(), (1 - y).sum()))
        if k >= 2:
            pr = cross_val_predict(LogisticRegression(C=1e6), z, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
            brier_recal = float(np.mean((pr - y) ** 2))
    return dict(brier=brier, brier_skill=1 - brier / brier_ref if brier_ref > 0 else float("nan"), ece=float(ece),
                slope=slope, brier_recalibrated=brier_recal, bins=bins)


def platt_oof(y: np.ndarray, p: np.ndarray, seed: int = 0) -> np.ndarray:
    """Cross-validated Platt recalibration of out-of-fold scores (monotone, so ranking/AUROC are preserved)."""
    y = np.asarray(y).astype(int)
    k = int(min(5, y.sum(), (1 - y).sum()))
    if k < 2:
        return np.asarray(p, dtype=float)
    z = _logit(np.asarray(p, dtype=float)).reshape(-1, 1)
    return cross_val_predict(LogisticRegression(C=1e6), z, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
