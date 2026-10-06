"""Per-endpoint grouped cross-validation, paired bootstrap of delta-AUROC, and calibration inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (ties get average rank)."""
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def make_model(y_train: np.ndarray, seed: int) -> XGBClassifier:
    """Hyper-parameters follow the source paper's classifier (150 trees, lr 0.05, scale_pos_weight=neg/pos)."""
    pos = max(int((y_train == 1).sum()), 1)
    return XGBClassifier(
        objective="binary:logistic", n_estimators=150, learning_rate=0.05, tree_method="hist",
        scale_pos_weight=(y_train == 0).sum() / pos, n_jobs=1, random_state=seed, verbosity=0,
    )


def oof_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int, n_repeats: int, seed: int, shuffle: bool = True) -> np.ndarray:
    """Out-of-fold probabilities, shape (n_repeats, n). Folds are grouped (no group in train and test) and stratified.

    ``shuffle=False`` reproduces the source paper's protocol (plain unshuffled StratifiedKFold on row order); it needs one
    row per group and is used only for the reproduction check."""
    if not shuffle and len(np.unique(groups)) != len(groups):
        raise ValueError("unshuffled StratifiedKFold ignores groups; it requires exactly one row per group")
    Xv = X.to_numpy(dtype="float32")
    y = np.asarray(y)
    out = np.zeros((n_repeats, len(y)))
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r) if shuffle else StratifiedKFold(n_splits)
        for tr, te in cv.split(Xv, y, groups):
            m = make_model(y[tr], seed + r)
            m.fit(Xv[tr], y[tr])
            out[r, te] = m.predict_proba(Xv[te])[:, 1]
    return out


def paired_bootstrap_delta(y: np.ndarray, s_full: np.ndarray, s_ref: np.ndarray, n_boot: int = 1000, seed: int = 0, floor: float | None = None) -> dict:
    """Bootstrap compounds; delta = AUROC(full) - AUROC(ref). One-sided p for delta > 0: (1 + #{delta* <= 0}) / (B + 1).

    ``floor``: if given, the reference AUROC is replaced by max(AUROC(ref), floor) (use 0.5 so a below-chance baseline,
    which is cross-validation noise, cannot inflate delta)."""
    def ref_auc(yy, ss):
        a = auroc(yy, ss)
        return a if floor is None else max(a, floor)

    y = np.asarray(y)
    if y.min() == y.max():
        raise ValueError("paired_bootstrap_delta needs both classes present")
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    while len(deltas) < n_boot:
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        deltas.append(auroc(yb, s_full[idx]) - ref_auc(yb, s_ref[idx]))
    deltas = np.asarray(deltas)
    return dict(
        delta=auroc(y, s_full) - ref_auc(y, s_ref),
        ci_lo=float(np.percentile(deltas, 2.5)), ci_hi=float(np.percentile(deltas, 97.5)),
        p=float((1 + (deltas <= 0).sum()) / (n_boot + 1)),
    )


def evaluate_endpoint(
    data: pd.DataFrame, y: pd.Series, models: dict[str, list[str]], *, n_splits: int = 5, n_repeats: int = 3,
    n_boot: int = 1000, seed: int = 0, groups: np.ndarray | None = None,
) -> dict:
    """Evaluate several feature sets on one endpoint with identical folds.

    ``models`` maps a model name to its feature columns; the model named ``full`` is compared against
    every other model. Returns pooled-OOF AUROC/PR-AUC per model (OOF predictions averaged over repeats)
    and ``delta[ref]`` = paired-bootstrap result for full vs ``ref``.
    """
    yv = y.to_numpy().astype(int)
    g = np.asarray(data.index) if groups is None else groups
    oof = {name: oof_predict(data[cols], yv, g, n_splits, n_repeats, seed) for name, cols in models.items()}
    pooled = {name: p.mean(axis=0) for name, p in oof.items()}
    metrics = {
        name: dict(
            auroc=auroc(yv, pooled[name]),
            prauc=float(average_precision_score(yv, pooled[name])),
            auroc_repeat_sd=float(np.std([auroc(yv, oof[name][r]) for r in range(n_repeats)])),
        )
        for name in models
    }
    delta = {}
    if "full" in models:
        for ref in models:
            if ref != "full":
                delta[ref] = paired_bootstrap_delta(yv, pooled["full"], pooled[ref], n_boot=n_boot, seed=seed)
    return dict(metrics=metrics, delta=delta, oof=pooled, y=yv, ids=np.asarray(data.index))
