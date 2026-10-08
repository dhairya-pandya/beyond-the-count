"""How large an advantage could an endpoint's data have detected?

The 15-active / 15-non-hit rule is a count heuristic. What determines whether a verdict is informative is the width of the
uncertainty around the AUROC difference. From the paired-bootstrap interval we recover the standard error of the difference,
widen it by the label-permutation null sd (the raw bootstrap understates the true sampling variability), and report the
smallest true advantage that the test would have found with the chosen power (the minimum detectable effect).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from cpsa.stats.null_calibration import Z95


def detectable_effect(ci_lo, ci_hi, null_sd: float = 1.0, alpha: float = 0.05, power: float = 0.8):
    """Minimum detectable true difference in AUROC for a one-sided test at ``alpha`` with the given ``power``."""
    se = (np.asarray(ci_hi, dtype=float) - np.asarray(ci_lo, dtype=float)) / (2 * Z95)
    se = np.where(se > 0, se, np.nan)  # a zero-width interval (identical constant predictions) carries no information about precision
    return (norm.isf(alpha) + norm.ppf(power)) * null_sd * se
