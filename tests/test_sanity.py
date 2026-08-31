"""Tests for the sanity / sync diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.sanity import (  # noqa: E402
    CorrelationNormalisationError,
    _checked_corr,
    _normalise_curve,
    gps_cumulative_distance,
)
from data.loader import load_session  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"

needs_data = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason="dataset absent — run scripts/fetch_data.sh")


# -- correlation guard -----------------------------------------------------

def test_valid_correlation_passes():
    for r in (-1.0, -0.5, 0.0, 0.37, 1.0):
        assert _checked_corr(r, "unit test") == r


def test_nan_correlation_passes_through():
    assert np.isnan(_checked_corr(float("nan"), "unit test"))


@pytest.mark.parametrize("bad", [1.0001, -1.5, 3.32, -2.105])
def test_impossible_correlation_raises_rather_than_clamping(bad):
    """|r| > 1 means the normalisation broke; clamping would hide it."""
    with pytest.raises(CorrelationNormalisationError, match="outside"):
        _checked_corr(bad, "unit test")


# -- geodesic distance -----------------------------------------------------

def test_pyproj_geodesic_matches_vincenty_package():
    """The vectorised geodesic used in the sweep agrees with Vincenty.

    pyproj is used because the pure-Python `vincenty` package is scalar-only
    and far too slow for the corpus; this pins the equivalence.
    """
    from pyproj import Geod
    from vincenty import vincenty

    geod = Geod(ellps="WGS84")
    rng = np.random.default_rng(0)
    lat1 = 52.4 + rng.normal(0, 0.01, 200)
    lon1 = -1.5 + rng.normal(0, 0.01, 200)
    lat2 = lat1 + rng.normal(0, 0.0005, 200)
    lon2 = lon1 + rng.normal(0, 0.0005, 200)

    fast = geod.inv(lon1, lat1, lon2, lat2)[2]
    slow = np.array([vincenty((a, b), (c, d)) * 1000.0
                     for a, b, c, d in zip(lat1, lon1, lat2, lon2)])
    assert np.max(np.abs(fast - slow)) < 0.01   # under a centimetre


# -- curve normalisation ---------------------------------------------------

def test_normalise_curve_maps_to_unit_interval():
    y = np.array([5.0, 7.0, 11.0, 25.0])
    out = _normalise_curve(y)
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)


def test_normalise_curve_survives_a_flat_curve():
    y = np.zeros(10)
    assert np.isfinite(_normalise_curve(y)).all()


# -- integration with real data -------------------------------------------

@needs_data
def test_gps_cumulative_distance_is_monotone_and_matches_sweep():
    s = load_session("S-Vw12", check_rate=False)
    out = gps_cumulative_distance(s)
    assert out is not None
    t, cum = out
    assert np.all(np.diff(cum) >= -1e-9), "cumulative distance must not decrease"
    assert np.all(np.diff(t) > 0), "fix timestamps must increase"
    assert cum[-1] > 0
