"""Power check: endpoints with too few actives or inactives are reported as indeterminate."""
from __future__ import annotations

MIN_ACTIVE = 15
MIN_INACTIVE = 15


def power_flag(n_active: int, n_inactive: int, min_active: int = MIN_ACTIVE, min_inactive: int = MIN_INACTIVE) -> bool:
    """True if the endpoint has enough compounds in both classes to support a delta-AUROC test."""
    return bool(n_active >= min_active and n_inactive >= min_inactive)
