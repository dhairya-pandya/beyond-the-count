"""Load well-level Cell Painting profiles published with Ewald et al. (Zenodo 10.5281/zenodo.17067683)."""
from __future__ import annotations

import pandas as pd

from cpsa import RAW

FEATURE_SETS = ("cellprofiler", "cpcnn", "dino")
CONTROL = "DMSO"


def raw_path(feature_set: str):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
    return RAW / f"{feature_set}_raw.parquet"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("Metadata_")]


def load_raw(feature_set: str) -> pd.DataFrame:
    """Well-level raw profiles (metadata columns prefixed ``Metadata_``)."""
    return pd.read_parquet(raw_path(feature_set))
