"""Synthetic phenotypic-profiling data with known ground truth, to validate the audit itself.

Latent structure: a cell-survival factor ``count`` (what a cell-count baseline sees) and several morphology factors ``z`` that a cell-count
model cannot see. Profile features encode both; baseline features are noisy copies of ``count`` only. Endpoint kinds:

* ``null``       labels independent of everything (true delta = 0, baseline AUROC = 0.5)
* ``shortcut``   labels depend on ``count`` only (true delta <= 0: the baseline already has everything)
* ``adds``       labels depend on ``count`` and on a morphology factor (true delta > 0)
* ``morph_only`` labels depend on morphology factors only (true delta > 0)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EFFECTS = (0.5, 1.0, 1.5)


@dataclass
class SimData:
    profile: pd.DataFrame
    baseline: pd.DataFrame
    labels: pd.DataFrame
    truth: pd.DataFrame  # endpoint_id, kind, effect, prevalence


def _labels_from_logit(rng: np.random.Generator, eta: np.ndarray, prevalence: float) -> np.ndarray:
    lo, hi = -15.0, 15.0
    for _ in range(40):  # bisection on the intercept so that the mean probability hits the target prevalence
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(eta + mid)))).mean() > prevalence:
            hi = mid
        else:
            lo = mid
    return (rng.random(len(eta)) < 1 / (1 + np.exp(-(eta + (lo + hi) / 2)))).astype(float)


def make_dataset(n: int = 600, n_features: int = 120, endpoints: dict[str, int] | None = None, prevalence: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
                 seed: int = 0) -> SimData:
    """``endpoints`` maps kind -> number of endpoints of that kind (cycled over ``prevalence`` and, for adds/morph_only, over EFFECTS)."""
    endpoints = endpoints or {"null": 20, "shortcut": 20, "adds": 30, "morph_only": 10}
    rng = np.random.default_rng(seed)
    count = rng.normal(size=n)
    z = rng.normal(size=(n, 8))
    n_dens, n_morph = n_features // 6, n_features // 3
    cols = []
    for _ in range(n_dens):  # density-encoding features
        cols.append(rng.normal() * count + 0.8 * rng.normal(size=n))
    for _ in range(n_morph):  # morphology features: mixtures of the factors
        w = rng.normal(size=8) * (rng.random(8) < 0.4)
        cols.append(z @ w + 0.8 * rng.normal(size=n))
    cols += [rng.normal(size=n) for _ in range(n_features - n_dens - n_morph)]
    profile = pd.DataFrame(np.column_stack(cols), columns=[f"f{i:03d}" for i in range(n_features)])
    baseline = pd.DataFrame({f"cc{i}": count + 0.35 * rng.normal(size=n) for i in range(5)})
    ids = [f"S{i:04d}" for i in range(n)]
    profile.index = baseline.index = ids
    lab, truth = {}, []
    for kind, k in endpoints.items():
        for j in range(k):
            prev = prevalence[j % len(prevalence)]
            eff = EFFECTS[(j // len(prevalence)) % len(EFFECTS)] if kind in ("adds", "morph_only") else 0.0
            g = z[:, 0] + 0.7 * z[:, 1] - 0.5 * z[:, 2]
            eta = {"null": 0 * count, "shortcut": 1.5 * count, "adds": 1.0 * count + eff * 2.0 * g, "morph_only": eff * 2.0 * g}[kind]
            name = f"{kind}_{j:03d}"
            lab[name] = _labels_from_logit(rng, eta, prev)
            truth.append(dict(endpoint_id=name, kind=kind, effect=eff, prevalence=prev))
    return SimData(profile, baseline, pd.DataFrame(lab, index=ids), pd.DataFrame(truth))
