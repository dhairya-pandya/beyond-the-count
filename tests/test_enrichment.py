import numpy as np
import pandas as pd

from cpsa.biology.enrichment import enrichment_table


def _audit(n_x=12, n_y=30, adv_x=10, adv_y=3):
    rows = []
    for fam, n, adv in (("X", n_x, adv_x), ("Y", n_y, adv_y)):
        for i in range(n):
            rows.append(dict(endpoint_id=f"{fam}{i}", assay_target_family=fam, category="cellbased",
                             verdict="morphology_advantage" if i < adv else "no_advantage"))
    rows += [dict(endpoint_id="ind", assay_target_family="X", category="cellbased", verdict="indeterminate"),
             dict(endpoint_id="ctl", assay_target_family="X", category="native", verdict="positive_control")]
    return pd.DataFrame(rows)


def test_enriched_family_detected_and_indeterminate_or_control_excluded():
    t = enrichment_table(_audit(), by=["assay_target_family"], min_group_size=5)
    x = t[t.group == "X"].iloc[0]
    assert x.n_in_group == 12 and x.n_adv_in_group == 10  # indeterminate + control rows not counted
    assert x.odds_ratio > 5 and x.p < 0.01 and x.q < 0.05
    y = t[t.group == "Y"].iloc[0]
    assert y.q > 0.5  # depleted, one-sided enrichment test -> not significant


def test_null_when_advantage_is_random_across_families():
    rng = np.random.default_rng(0)
    rows = [dict(endpoint_id=f"e{i}", assay_target_family=f"F{i % 6}", category="cellbased",
                 verdict=rng.choice(["morphology_advantage", "no_advantage"], p=[0.3, 0.7])) for i in range(120)]
    t = enrichment_table(pd.DataFrame(rows), by=["assay_target_family"], min_group_size=5)
    assert (t.q > 0.05).all()


def test_small_groups_are_skipped():
    t = enrichment_table(_audit(n_x=3, adv_x=3), by=["assay_target_family"], min_group_size=5)
    assert "X" not in set(t.group)


def test_cluster_permutation_is_more_cautious_when_one_assay_drives_the_group():
    from cpsa.biology.enrichment import assay_cluster
    rows = []
    for i in range(12):  # family X: all endpoints from ONE assay cluster, all advantage
        rows.append(dict(endpoint_id=f"AX_one_{i}", assay_target_family="X", verdict="morphology_advantage"))
    for j in range(20):  # family Y: 20 separate assays, 3 advantage
        rows.append(dict(endpoint_id=f"AY_{j}_a", assay_target_family="Y", verdict="morphology_advantage" if j < 3 else "no_advantage"))
    a = pd.DataFrame(rows)
    a["cluster"] = assay_cluster(a["endpoint_id"])
    t = enrichment_table(a, by=["assay_target_family"], min_group_size=5, cluster_col="cluster", n_perm=3000)
    x = t[t.group == "X"].iloc[0]
    assert x.p < 0.001 and 0.02 < x.p_cluster < 0.15  # naive test counts 12 independent endpoints; one assay among ~21 can reach p of about 1/21 at best


def test_assay_cluster_takes_first_two_tokens():
    from cpsa.biology.enrichment import assay_cluster
    s = assay_cluster(pd.Series(["BSK_SAg_Eselectin", "TOX21_ERa_BLA_Agonist_ratio", "MT"]))
    assert s.tolist() == ["BSK_SAg", "TOX21_ERa", "MT"]
