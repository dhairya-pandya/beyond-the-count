"""Audit of the 'Cell Painting is X-fold more sensitive than cytotoxicity assays' claim.

The paper's Cell Painting 'bioactivity POD' is the LOWEST point of departure over up to 19 feature-category endpoints per compound, while
LDH / MT / cell count each contribute one POD. A minimum over many correlated tests is biased low, so part of the fold advantage can be selection.
We therefore report paired fold differences for the paper's definition, for each single endpoint, and for single-endpoint representations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_fold(pod_a: pd.Series, pod_b: pd.Series, n_boot: int = 1000, seed: int = 0) -> dict:
    """Geometric-mean ratio pod_a / pod_b over compounds that have both PODs, with a percentile-bootstrap CI over compounds."""
    both = pd.concat([pod_a.rename("a"), pod_b.rename("b")], axis=1).dropna()
    both = both[(both["a"] > 0) & (both["b"] > 0)]
    if len(both) < 3:
        return dict(n_pairs=len(both), fold=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    lr = np.log(both["a"].to_numpy() / both["b"].to_numpy())
    rng = np.random.default_rng(seed)
    boots = [lr[rng.integers(0, len(lr), len(lr))].mean() for _ in range(n_boot)]
    return dict(n_pairs=len(both), fold=float(np.exp(lr.mean())), ci_lo=float(np.exp(np.percentile(boots, 2.5))), ci_hi=float(np.exp(np.percentile(boots, 97.5))))
