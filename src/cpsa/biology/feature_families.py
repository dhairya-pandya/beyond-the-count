"""Attribute full-model importance to interpretable CellProfiler feature families (compartment / feature type / channel)."""
from __future__ import annotations

import pandas as pd

CHANNELS = ("AGP", "DNA", "ER", "Mito", "RNA", "Brightfield")
COMPARTMENTS = ("Cells", "Nuclei", "Cytoplasm", "Image")


def parse_feature_name(name: str) -> tuple[str, str, str]:
    """(compartment, feature group, channel(s)) from a CellProfiler feature name; non-CellProfiler names -> 'unparsed'."""
    toks = name.split("_")
    if toks[0] not in COMPARTMENTS or len(toks) < 2:
        return ("unparsed",) * 3
    ch = [t for t in toks[2:] if t in CHANNELS]
    return toks[0], toks[1], "+".join(ch) if ch else "none"


def aggregate_importance(importance: pd.DataFrame, level: str = "group") -> pd.DataFrame:
    """importance: endpoint x feature (gain). Returns endpoint x family, each row normalised to sum to 1.
    level in {'compartment', 'group', 'channel'}."""
    idx = {"compartment": 0, "group": 1, "channel": 2}[level]
    fam = pd.Series([parse_feature_name(c)[idx] for c in importance.columns], index=importance.columns)
    out = importance.T.groupby(fam).sum().T
    return out.div(out.sum(axis=1), axis=0)
