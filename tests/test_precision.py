import numpy as np
import pytest

from cpsa.stats.precision import detectable_effect


def test_detectable_effect_scales_with_interval_width_and_null_sd():
    narrow = detectable_effect(-0.02, 0.02)
    wide = detectable_effect(-0.10, 0.10)
    assert wide == pytest.approx(5 * narrow)
    assert detectable_effect(-0.05, 0.05, null_sd=1.4) == pytest.approx(1.4 * detectable_effect(-0.05, 0.05))


def test_detectable_effect_known_value():
    # 95% interval of +-0.0392 -> se 0.02; (1.645 + 0.8416) * 0.02 = 0.0497
    assert detectable_effect(-0.03920, 0.03920) == pytest.approx(0.0497, abs=1e-3)


def test_nan_interval_gives_nan():
    assert np.isnan(detectable_effect(np.nan, 0.1))


def test_zero_width_interval_is_not_reported_as_perfectly_precise():
    assert np.isnan(detectable_effect(0.0, 0.0))
