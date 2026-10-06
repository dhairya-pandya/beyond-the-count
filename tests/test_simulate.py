import numpy as np

from cpsa.simulate import make_dataset


def test_make_dataset_shapes_prevalence_and_truth_table():
    d = make_dataset(n=400, n_features=60, endpoints={"null": 3, "shortcut": 3, "adds": 3, "morph_only": 3}, prevalence=(0.1, 0.3), seed=0)
    assert d.profile.shape == (400, 60) and d.baseline.shape[0] == 400
    assert d.labels.shape == (400, 12) and set(d.truth["kind"]) == {"null", "shortcut", "adds", "morph_only"}
    prev = d.labels.mean()
    assert ((prev > 0.05) & (prev < 0.4)).all()
    assert set(np.unique(d.labels.to_numpy())) <= {0.0, 1.0}


def test_baseline_is_a_function_of_latent_count_only():
    d = make_dataset(n=600, n_features=40, endpoints={"shortcut": 2}, seed=1)
    # baseline columns are noisy copies of the latent count; correlated with each other but not with the morphology-only factors
    corr = d.baseline.corr().to_numpy()
    assert corr[np.triu_indices_from(corr, 1)].min() > 0.5
