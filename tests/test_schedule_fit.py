"""The scheduled-sampling curriculum and the fitter's search.

Synthetic only -- no graph, no checkpoint, no session data.
"""

from __future__ import annotations

import numpy as np
import pytest

from mapmatch.schedule_fit import (BOUNDS, ScheduleConfig,
                                   ScheduledSamplingFitter, apply,
                                   teacher_forcing_prob)
from mapmatch.viterbi import ViterbiConfig


# -- the schedule ----------------------------------------------------------

@pytest.mark.parametrize("decay", ["inverse_sigmoid", "linear"])
def test_eps_starts_fully_taught_and_ends_fully_free(decay):
    """The endpoints are the whole point: epoch 0 is teacher-forced, the last
    epoch is deployment exactly, so its objective is directly comparable to a
    reported drift number."""
    cfg = ScheduleConfig(epochs=6, decay=decay)
    assert teacher_forcing_prob(0, cfg) == pytest.approx(1.0)
    assert teacher_forcing_prob(5, cfg) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("decay", ["inverse_sigmoid", "linear"])
def test_eps_is_monotone(decay):
    cfg = ScheduleConfig(epochs=8, decay=decay)
    e = [teacher_forcing_prob(i, cfg) for i in range(8)]
    assert all(e[i] >= e[i + 1] for i in range(7))
    assert all(0.0 <= v <= 1.0 for v in e)


def test_inverse_sigmoid_holds_the_teacher_longer_early_then_crosses():
    """Why it is the default: the EARLY free-running rollouts carry no signal,
    so the schedule should not hand them over early. Normalised, it is
    S-shaped and meets linear at the midpoint -- it is not uniformly slower,
    and asserting that it is would be asserting something false."""
    inv = ScheduleConfig(epochs=9, decay="inverse_sigmoid")
    lin = ScheduleConfig(epochs=9, decay="linear")
    assert teacher_forcing_prob(2, inv) > teacher_forcing_prob(2, lin)
    assert teacher_forcing_prob(4, inv) == pytest.approx(
        teacher_forcing_prob(4, lin), abs=1e-9)


def test_single_epoch_is_free_running():
    """One epoch means no curriculum, and the reported objective must still be
    a deployment number -- so it is the free-running end, not the taught one."""
    assert teacher_forcing_prob(0, ScheduleConfig(epochs=1)) == pytest.approx(0.0)


# -- the search ------------------------------------------------------------

def _fitter(monkeypatch, surface):
    """A fitter whose rollout is a known analytic surface, so the search can
    be tested without a road graph."""
    f = ScheduledSamplingFitter.__new__(ScheduledSamplingFitter)
    f.m = type("M", (), {"cfg": ViterbiConfig(backend="graph")})()
    f.cfg = ScheduleConfig(epochs=3, trials=5)
    f._truth = {}

    def fake_objective(params, starts, eps, seed):
        return surface(params)
    f.objective = fake_objective
    return f


def test_search_moves_downhill_towards_the_optimum():
    """A quadratic bowl in sigma_turn_rad with its minimum well away from the
    hand-set default: the fit must move toward it."""
    target = np.deg2rad(30.0)
    f = _fitter(None, lambda p: (p["sigma_turn_rad"] - target) ** 2)
    start = ViterbiConfig(backend="graph").sigma_turn_rad
    out = f.fit([0.0], verbose=False)
    got = out["params"]["sigma_turn_rad"]
    assert abs(got - target) < abs(start - target), "search did not improve"


def test_search_never_leaves_the_physical_bounds():
    """A surface that rewards ever-larger sigma must still be clipped: the
    bounds are physical, not search latitude."""
    f = _fitter(None, lambda p: -p["sigma_turn_rad"] - p["sigma_scale"])
    out = f.fit([0.0], verbose=False)
    for name, (lo, hi) in BOUNDS.items():
        assert lo <= out["params"][name] <= hi, f"{name} escaped its bounds"


def test_objective_never_increases_across_the_epochs_of_a_fixed_surface():
    """With a stationary surface the recorded objective is monotone: the
    search only accepts strict improvements."""
    f = _fitter(None, lambda p: (p["sigma_scale"] - 2.0) ** 2)
    hist = f.fit([0.0], verbose=False)["history"]
    vals = [h["objective"] for h in hist]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


def test_history_records_eps_alongside_the_objective():
    """Without eps on the row, a fitted number cannot be told apart from a
    teacher-forced one -- which is the mistake the whole module exists to
    avoid making."""
    f = _fitter(None, lambda p: 1.0)
    hist = f.fit([0.0], verbose=False)["history"]
    assert [h["epoch"] for h in hist] == [0, 1, 2]
    assert hist[0]["eps"] == pytest.approx(1.0)
    assert hist[-1]["eps"] == pytest.approx(0.0, abs=1e-9)
    assert set(BOUNDS) <= set(hist[0])


# -- applying the result ---------------------------------------------------

def test_apply_returns_a_new_config_and_does_not_mutate_the_old():
    base = ViterbiConfig(backend="graph")
    out = apply(base, {"sigma_turn_rad": 0.5, "min_sigma_m": 2.0})
    assert out.sigma_turn_rad == 0.5 and out.min_sigma_m == 2.0
    assert base.sigma_turn_rad != 0.5, "apply must not mutate in place"
    assert out.backend == base.backend, "unrelated fields must survive"


def test_apply_ignores_keys_the_config_does_not_have():
    out = apply(ViterbiConfig(backend="graph"), {"not_a_field": 1.0})
    assert not hasattr(out, "not_a_field")
