"""Tests for the metrics and outage harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval.metrics import aeps, cae, crse, crse_scalar, geodesic_distance_m

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
needs_data = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason="dataset absent")


def test_crse_is_the_norm_of_the_accumulated_error_vector():
    # Three seconds, each 1 m east and 1 m north.
    e = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    assert crse(e) == pytest.approx(np.hypot(3.0, 3.0))


def test_crse_vector_form_differs_from_the_scalar_sum():
    """Errors that cancel in direction must not accumulate.

    This is exactly why the scalar reading inverted the published ordering:
    it cannot see a track bending away from truth.
    """
    e = np.array([[3.0, 0.0], [-3.0, 0.0]])
    assert crse(e) == pytest.approx(0.0)
    assert crse_scalar([3.0, -3.0]) == pytest.approx(6.0)


def test_cae_is_signed_and_cancels():
    assert cae([5.0, -5.0]) == pytest.approx(0.0)
    assert crse_scalar([5.0, -5.0]) == pytest.approx(10.0)


def test_aeps_is_the_mean_absolute_error():
    assert aeps([1.0, -3.0, 2.0]) == pytest.approx(2.0)


def test_aeps_ignores_nan():
    assert aeps([2.0, np.nan, 4.0]) == pytest.approx(3.0)


def test_geodesic_distance_is_symmetric_and_zero_on_identity():
    assert geodesic_distance_m(52.4, -1.5, 52.4, -1.5) == pytest.approx(0.0)
    a = geodesic_distance_m(52.4, -1.5, 52.5, -1.6)
    b = geodesic_distance_m(52.5, -1.6, 52.4, -1.5)
    assert a == pytest.approx(b)


@needs_data
def test_harness_refuses_a_stationary_start():
    """GPS heading is meaningless at rest, so such outages must be skipped."""
    from data.loader import load_session
    from eval.harness import OutageSkipped, run_outage
    from baseline.ins_dr import INSDeadReckoning

    s = load_session("V-Vw12", check_rate=False)
    p = INSDeadReckoning(s)

    class Stationary:
        name = "stub"

        def predict(self, t0, duration):
            n = int(duration)
            return {"displacements": np.zeros(n), "headings": np.zeros(n)}

    # Force the speed below the gate and confirm the harness refuses.
    s.df["speed_ms"] = 0.0
    with pytest.raises(OutageSkipped, match="below"):
        run_outage(s, float(s.df["time_s"].iloc[0]) + 1.0, 10.0, Stationary())


@needs_data
def test_predictor_interface_is_swappable():
    """The harness must score any object exposing predict()."""
    from data.loader import load_session
    from eval.harness import run_outage

    s = load_session("V-Vw12", check_rate=False)

    class Oracle:
        """Predicts the truth: zero displacement error by construction."""
        name = "oracle"

        def __init__(self, session):
            self.session = session

        def predict(self, t0, duration):
            n = int(duration)
            return {"displacements": np.full(n, 25.0), "headings": np.zeros(n)}

    t0 = float(s.df["time_s"].iloc[0]) + 5.0
    r = run_outage(s, t0, 10.0, Oracle(s))
    assert r.predictor == "oracle"
    assert r.displacement_errors.size == 10
    assert np.isfinite(r.final_position_error)


@needs_data
def test_motorway_reproduces_the_published_figure():
    """The Stage 0 known-answer test: V-Vw12, nine 10 s outages."""
    from data.loader import load_session
    from eval.harness import run_session
    from baseline.ins_dr import INSDeadReckoning

    s = load_session("V-Vw12", check_rate=False)
    results, _ = run_session(s, INSDeadReckoning(s), 10.0)
    assert len(results) == 9, "the paper reports nine sequences on this subset"
    max_crse = max(r.metrics()["crse"] for r in results)
    # Published: 30.11 m. Allow 30%.
    assert 0.7 * 30.11 <= max_crse <= 1.3 * 30.11, f"max CRSE {max_crse:.2f}"


# -- speed-stratified reporting -------------------------------------------

def test_buckets_partition_the_speed_range():
    from eval.buckets import BUCKET_NAMES, assign_buckets
    idx = assign_buckets([0.0, 4.9, 5.0, 14.9, 15.0, 24.9, 25.0, 100.0])
    assert list(idx) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(BUCKET_NAMES) == 4


def test_early_stopping_metric_is_unweighted_across_buckets():
    """A dominant bucket must not be able to carry the metric.

    Two buckets: one with 1000 windows and zero error, one with 40 windows and
    an error of 10. Pooled MAE is ~0.38; the bucket mean is 5.0.
    """
    from eval.buckets import early_stopping_metric, pooled_mae
    y = np.concatenate([np.full(1000, 20.0), np.full(40, 2.0)])
    p = np.concatenate([np.full(1000, 20.0), np.full(40, 12.0)])
    assert pooled_mae(y, p) == pytest.approx(10 * 40 / 1040, rel=1e-6)
    assert early_stopping_metric(y, p) == pytest.approx(5.0)


def test_sparse_buckets_are_excluded_from_the_mean():
    """A handful of windows must not swing the stopping metric."""
    from eval.buckets import early_stopping_metric
    y = np.concatenate([np.full(500, 20.0), np.full(5, 2.0)])
    p = np.concatenate([np.full(500, 20.0), np.full(5, 1000.0)])
    assert early_stopping_metric(y, p) == pytest.approx(0.0)


def test_bucketing_uses_truth_not_prediction():
    """A bad model must not be able to relabel its hard cases into an easy bucket."""
    from eval.buckets import bucketed_mae
    y = np.full(100, 30.0)          # all in the >25 bucket
    p = np.full(100, 1.0)           # predicts them all as 0-5
    df = bucketed_mae(y, p)
    assert int(df.loc[df.bucket == ">25", "n_windows"].iloc[0]) == 100
    assert int(df.loc[df.bucket == "0-5", "n_windows"].iloc[0]) == 0
