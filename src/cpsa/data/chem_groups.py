"""Chemical-similarity groups for leakage-aware cross-validation.

Per-compound grouping stops replicate wells leaking, but structurally related compounds (salts, isomers, analog series) can still sit in
train and test. Grouping by Butina clusters of ECFP4 Tanimoto similarity keeps whole analog series together in a fold.
"""
from __future__ import annotations

import pandas as pd

from cpsa.data.load_labels import ANNOT


def load_smiles() -> pd.Series:
    a = pd.read_csv(ANNOT / "v5_oasis_03Sept2024_simple.csv", usecols=["OASIS_ID", "SMILES"]).dropna(subset=["OASIS_ID"]).drop_duplicates("OASIS_ID")
    return a.set_index("OASIS_ID")["SMILES"]


def cluster_smiles(smiles: pd.Series, similarity: float = 0.4, radius: int = 2, n_bits: int = 2048) -> pd.Series:
    """Cluster id per index entry; molecules with Tanimoto similarity >= ``similarity`` to a cluster centroid share its cluster (Butina).
    Missing / unparsable SMILES get their own singleton cluster."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, keep = [], []
    for key, smi in smiles.items():
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is not None:
            fps.append(gen.GetFingerprint(mol))
            keep.append(key)
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(fps), 1.0 - similarity, isDistData=True) if len(fps) > 1 else [(0,)] * len(fps)
    out = {}
    for cid, members in enumerate(clusters):
        for m in members:
            out[keep[m]] = cid
    nxt = len(clusters)
    for key in smiles.index:
        if key not in out:
            out[key] = nxt
            nxt += 1
    return pd.Series(out).reindex(smiles.index).rename("chem_cluster")
