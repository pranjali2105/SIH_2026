"""Tests for the windowed dataset builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.windows import (
    CHANNELS,
    STRIDE_SAMPLES,
    WINDOW_SAMPLES,
    LeakageError,
    SessionWindows,
    apply_normalisation,
    fit_normalisation,
    verify_no_leakage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
needs_data = pytest.mark.skipif(not DATA_ROOT.exists(), reason="dataset absent")


def _stub(session: str, role: str, n: int = 5) -> SessionWindows:
    rng = np.random.default_rng(abs(hash(session)) % 2**32)
    return SessionWindows(
        session=session, role=role, group="S",
        X=rng.normal(size=(n, WINDOW_SAMPLES, len(CHANNELS))),
        y=rng.normal(10, 2, size=n),
        is_stationary=np.zeros(n, bool),
        yaw_rate_true=np.zeros(n),
        wheel_speed=np.full(n, np.nan),
        t0=np.arange(n, dtype=float),
        n_rejected=0, label_source="gps")


# -- leakage ---------------------------------------------------------------

def test_duplicate_session_under_two_roles_raises():
    built = [_stub("S-M", "train"), _stub("S-M", "validate")]
    with pytest.raises(LeakageError, match="two roles"):
        verify_no_leakage(built)


def test_session_windowed_twice_raises():
    built = [_stub("S-M", "train"), _stub("S-M", "train")]
    with pytest.raises(LeakageError, match="more than once"):
        verify_no_leakage(built)


def test_wrong_window_length_raises():
    b = _stub("S-M", "train")
    b.X = b.X[:, :50, :]
    with pytest.raises(LeakageError, match="window length"):
        verify_no_leakage([b])


def test_non_monotonic_window_starts_raise():
    """Out-of-order starts would mean windows were concatenated across files."""
    b = _stub("S-M", "train")
    b.t0 = np.array([0.0, 5.0, 2.0, 7.0, 9.0])
    with pytest.raises(LeakageError, match="not increasing"):
        verify_no_leakage([b])


# -- normalisation ---------------------------------------------------------

def test_normalisation_is_fitted_on_train_only():
    train = [_stub("S-A", "train")]
    stats = fit_normalisation(train)
    assert stats["fitted_on"] == "train"
    assert len(stats["mean"]) == len(CHANNELS)
    stacked = train[0].X.reshape(-1, len(CHANNELS))
    assert np.allclose(stats["mean"], stacked.mean(axis=0))


def test_normalisation_standardises_train():
    train = [_stub("S-A", "train", n=50)]
    stats = fit_normalisation(train)
    out = apply_normalisation(train[0].X, stats)
    assert np.allclose(out.reshape(-1, len(CHANNELS)).mean(axis=0), 0, atol=1e-6)
    assert np.allclose(out.reshape(-1, len(CHANNELS)).std(axis=0), 1, atol=1e-6)


def test_zero_variance_channel_does_not_divide_by_zero():
    b = _stub("S-A", "train")
    b.X[:, :, 0] = 3.0
    stats = fit_normalisation([b])
    assert stats["std"][0] == 1.0
    assert np.isfinite(apply_normalisation(b.X, stats)).all()


def test_fit_normalisation_requires_train():
    with pytest.raises(ValueError, match="no train windows"):
        fit_normalisation([])


# -- real data -------------------------------------------------------------

@needs_data
def test_real_session_has_expected_shape_and_stride():
    from data.windows import build_session
    b = build_session("S-A5")
    assert b is not None
    assert b.X.shape[1:] == (WINDOW_SAMPLES, len(CHANNELS))
    assert b.y.shape[0] == b.X.shape[0]
    assert np.isfinite(b.X).all(), "windows must not contain NaN"
    assert np.isfinite(b.y).all(), "labels must not contain NaN"
    gaps = np.diff(b.t0)
    assert np.allclose(gaps[gaps < 5], STRIDE_SAMPLES / 10.0, atol=1e-6)


@needs_data
def test_labels_respect_the_plausibility_guard():
    from data.sanity import MAX_PLAUSIBLE_SPEED_MS
    from data.windows import build_session
    b = build_session("S-A5")
    assert b is not None
    assert (b.y >= 0).all()
    assert (b.y <= MAX_PLAUSIBLE_SPEED_MS * 1.0).all()


@needs_data
def test_inflated_positions_are_rejected_at_session_level():
    """S-T1's GPS path is 3.5x the integral of its own reported speed.

    The per-window 60 m/s guard only trims the tail; the session-level check
    refuses the session outright.
    """
    from data.windows import InflatedPositionsError, build_session, path_vs_speed_ratio
    from data.loader import load_session

    s = load_session("S-T1", check_rate=False)
    assert path_vs_speed_ratio(s) > 3.0
    with pytest.raises(InflatedPositionsError, match="positions jump"):
        build_session("S-T1")


@needs_data
def test_clean_session_passes_the_path_ratio_check():
    from data.windows import path_vs_speed_ratio
    from data.loader import load_session
    for name in ("S-T2", "S-A5", "S-M"):
        r = path_vs_speed_ratio(load_session(name, check_rate=False))
        assert abs(r - 1.0) <= 0.10, f"{name} ratio {r:.3f}"


@needs_data
def test_gps_is_the_default_label_source():
    """Train, validate and test must share a label source."""
    from data.windows import build_session
    b = build_session("S-M")
    assert b.label_source == "gps"
    e = build_session("S-M", prefer_ecu=True)
    assert e.label_source == "ecu_wheel_speed"
    # The ECU path is retained as an ablation and must still work.
    assert e.y.mean() > b.y.mean(), "ECU labels ran ~2% high in measurement"


@needs_data
def test_yaw_rate_true_is_nan_at_low_speed():
    """GPS heading is meaningless at rest, so the true yaw rate must be absent."""
    from data.windows import build_session
    b = build_session("S-M")
    assert b is not None
    if b.is_stationary.any():
        assert not np.isfinite(b.yaw_rate_true[b.is_stationary]).any()
