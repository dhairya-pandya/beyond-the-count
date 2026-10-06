import pandas as pd

from cpsa.data.chem_groups import cluster_smiles


def test_close_analogs_share_a_cluster_and_unrelated_molecules_do_not():
    smiles = pd.Series({
        "ethyl_benzoate": "CCOC(=O)c1ccccc1", "methyl_benzoate": "COC(=O)c1ccccc1", "propyl_benzoate": "CCCOC(=O)c1ccccc1",
        "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "theophylline": "Cn1c2nc[nH]c2c(=O)n(C)c1=O",
        "hexane": "CCCCCC", "bad": "not-a-smiles", "missing": None,
    })
    g = cluster_smiles(smiles, similarity=0.4)
    assert g["ethyl_benzoate"] == g["methyl_benzoate"] == g["propyl_benzoate"]
    assert g["caffeine"] == g["theophylline"] != g["ethyl_benzoate"]
    assert g["hexane"] not in (g["caffeine"], g["ethyl_benzoate"])
    assert g["bad"] != g["missing"]  # unparsable / missing structures become singleton groups, never merged
    assert g.index.tolist() == smiles.index.tolist()
