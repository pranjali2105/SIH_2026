"""Tests for the scalar speed-fusion Kalman filter.

Pure numpy, no dependency on the (currently missing) `data` package or a
trained model -- everything here is synthetic, same pattern as
`test_mount_calibration.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion.speed_filter import SpeedFilterConfig, fuse_speed_sequence


def _regime_biased_drive(seed: int, n: int = 120):
    """A speed profile with the SAME shape of model error findings.md
    documents: large over-prediction at low speed, near-accurate at high
    speed -- not a flat constant offset. `a_fwd` is derived from the TRUE
    speed, so it carries none of the model's regime-dependent bias.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    true_v = 15.0 + 8.0 * np.sin(2 * np.pi * k / 40.0)
    true_a = np.gradient(true_v, 1.0)
    bias = np.where(true_v < 12.0, 6.0, 0.3)
    mu = true_v + bias + rng.normal(0, 1.5, n)
    sigma_mu = np.full(n, 1.5)
    a_fwd = true_a + rng.normal(0, 0.05, n)
    return true_v, mu, sigma_mu, a_fwd


def test_reduces_drift_under_regime_dependent_model_bias():
    """The documented failure mode (results/mapmatch.md: map_matched_model
    carries a systematic +118 m / 60 s bias) is regime-dependent, not a flat
    offset -- CAE varies sharply by speed bucket (results/final_scoring.md).
    An accelerometer signal that does not share that regime dependence
    should let the filter claw back some of the cumulative drift.
    """
    true_v, mu, sigma_mu, a_fwd = _regime_biased_drive(seed=1)
    filt = fuse_speed_sequence(mu, sigma_mu, a_fwd)

    raw_drift = float(np.sum(mu - true_v))
    filt_drift = float(np.sum(filt - true_v))
    assert abs(filt_drift) < 0.8 * abs(raw_drift)


def test_reduces_rmse_against_noisy_model_output():
    true_v, mu, sigma_mu, a_fwd = _regime_biased_drive(seed=2)
    filt = fuse_speed_sequence(mu, sigma_mu, a_fwd)

    raw_rmse = float(np.sqrt(np.mean((mu - true_v) ** 2)))
    filt_rmse = float(np.sqrt(np.mean((filt - true_v) ** 2)))
    assert filt_rmse < raw_rmse


def test_a_flat_constant_bias_is_not_fully_removed():
    """Honesty check on the module's own claim: the accelerometer supplies
    only RATE information, no absolute speed reference, so a perfectly
    constant model offset (as opposed to the regime-dependent one above)
    should survive mostly intact -- this filter is not magic, and the
    docstring must not claim more than the math supports.
    """
    rng = np.random.default_rng(3)
    n = 60
    k = np.arange(n)
    true_v = 15.0 + 3.0 * np.sin(2 * np.pi * k / 20.0)
    true_a = np.gradient(true_v, 1.0)
    mu = true_v + 3.0 + rng.normal(0, 2.0, n)      # flat +3 m/s, every step
    sigma_mu = np.full(n, 2.0)
    a_fwd = true_a + rng.normal(0, 0.05, n)

    filt = fuse_speed_sequence(mu, sigma_mu, a_fwd)
    raw_bias = float(np.mean(mu - true_v))
    filt_bias = float(np.mean(filt - true_v))
    assert filt_bias == pytest.approx(raw_bias, rel=0.25)


def test_without_an_accelerometer_signal_still_smooths():
    """All-NaN `a_fwd`: the predict step is skipped every second, so the
    filter degrades to smoothing `mu` against its own recent history
    instead of crashing or silently ignoring the missing input.
    """
    rng = np.random.default_rng(4)
    n = 40
    mu = 10.0 + rng.normal(0, 3.0, n)
    sigma_mu = np.full(n, 3.0)
    a_fwd = np.full(n, np.nan)
    filt = fuse_speed_sequence(mu, sigma_mu, a_fwd)
    assert np.isfinite(filt).all()
    assert np.std(filt) < np.std(mu)               # still smooths


def test_output_never_negative():
    mu = np.array([-5.0, -1.0, 0.0, 3.0])
    sigma_mu = np.full(4, 1.0)
    a_fwd = np.full(4, -10.0)
    filt = fuse_speed_sequence(mu, sigma_mu, a_fwd)
    assert (filt >= 0.0).all()


def test_nan_sigma_falls_back_to_least_trust_without_crashing():
    mu = np.array([10.0, 12.0, 9.0])
    sigma_mu = np.array([1.0, np.nan, 2.0])
    a_fwd = np.array([np.nan, 0.5, -0.2])
    filt = fuse_speed_sequence(mu, sigma_mu, a_fwd)
    assert np.isfinite(filt).all()


def test_empty_input_returns_empty():
    out = fuse_speed_sequence(np.array([]), np.array([]), np.array([]))
    assert out.shape == (0,)


def test_config_is_respected():
    """`max_sigma_mu` is a CAP on how untrustworthy a reported sigma can
    make the model measurement -- lowering the cap FORCES more trust in mu
    (a smaller effective R), it does not reduce it. Model reports itself as
    very unsure (sigma=1000) at step 1; capping that down to 2.0 should pull
    the fused estimate noticeably closer to that step's mu than leaving the
    reported 1000 untouched does.
    """
    mu = np.array([10.0, 50.0, 10.0])
    sigma_mu = np.array([1.0, 1000.0, 1.0])         # step 1: model very unsure
    a_fwd = np.array([np.nan, 0.0, 0.0])
    uncapped = fuse_speed_sequence(mu, sigma_mu, a_fwd,
                                   cfg=SpeedFilterConfig(max_sigma_mu=1000.0))
    capped = fuse_speed_sequence(mu, sigma_mu, a_fwd,
                                 cfg=SpeedFilterConfig(max_sigma_mu=2.0))
    assert capped[1] > uncapped[1]
