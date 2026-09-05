"""The HMM observation model and the Viterbi decode.

Synthetic only -- no OSRM, no network, no session data.
"""

from __future__ import annotations

import numpy as np
import pytest

from mapmatch.graph import EdgePosition, RoadGraph
from mapmatch.predictor import OutageTrace
from mapmatch.viterbi import (CLASS_MAX_MS, LOG_EPS, ViterbiConfig,
                              ViterbiMapMatcher, _State)


@pytest.fixture
def crossroads() -> RoadGraph:
    """Plus-shaped junction; node 1 centre, 2 N, 3 S, 4 E, 5 W."""
    coords = {1: (51.5000, -2.5000), 2: (51.5010, -2.5000),
              3: (51.4990, -2.5000), 4: (51.5000, -2.4986),
              5: (51.5000, -2.5014)}
    return RoadGraph.from_ways(coords, [([3, 1, 2], False), ([5, 1, 4], False)])


def _m(graph, **kw) -> ViterbiMapMatcher:
    """A matcher with graph + cfg only; no session, no OSRM."""
    p = ViterbiMapMatcher.__new__(ViterbiMapMatcher)
    p.graph = graph
    p.cfg = ViterbiConfig(backend="graph", **kw)
    p.backend = "graph"
    return p


# -- emission --------------------------------------------------------------

def test_emission_is_maximal_when_the_gyro_agrees_with_the_road(crossroads):
    """The whole point: a turn the road explains costs nothing."""
    m = _m(crossroads)
    a = EdgePosition(0, 10.0)
    b = EdgePosition(0, 60.0)          # same edge, so the road turn is ~0
    straight = m._emission_logp(a, b, observed_turn=0.0)
    assert straight == pytest.approx(0.0, abs=1e-6)


def test_emission_penalises_a_turn_the_road_does_not_explain(crossroads):
    m = _m(crossroads)
    a, b = EdgePosition(0, 10.0), EdgePosition(0, 60.0)
    agree = m._emission_logp(a, b, 0.0)
    disagree = m._emission_logp(a, b, np.deg2rad(40.0))
    assert disagree < agree
    # 40 deg against a 12 deg sigma is ~3.3 sigma: about -5.5 in log space.
    assert -8.0 < disagree < -3.0


def test_emission_is_monotone_in_the_mismatch(crossroads):
    m = _m(crossroads)
    a, b = EdgePosition(0, 10.0), EdgePosition(0, 60.0)
    vals = [m._emission_logp(a, b, np.deg2rad(x)) for x in (0, 5, 10, 20, 40)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_emission_is_floored_not_infinite(crossroads):
    """A hopeless hypothesis must be dead, not NaN-poisoning the decode."""
    m = _m(crossroads)
    a, b = EdgePosition(0, 10.0), EdgePosition(0, 60.0)
    assert m._emission_logp(a, b, np.pi) >= LOG_EPS


# -- transition ------------------------------------------------------------

def test_transition_is_maximal_when_distance_matches_the_prediction(crossroads):
    m = _m(crossroads)
    assert m._transition_logp(25.0, 25.0, 3.0) == pytest.approx(0.0, abs=1e-9)


def test_transition_penalises_a_route_that_does_not_fit_the_displacement(crossroads):
    m = _m(crossroads)
    close = m._transition_logp(26.0, 25.0, 3.0)
    far = m._transition_logp(40.0, 25.0, 3.0)
    assert far < close < 0.0


def test_transition_respects_the_models_own_uncertainty(crossroads):
    """A window the model is unsure about should constrain the route less."""
    m = _m(crossroads)
    confident = m._transition_logp(31.0, 25.0, 2.0)
    unsure = m._transition_logp(31.0, 25.0, 10.0)
    assert unsure > confident, "high sigma must flatten the transition term"


def test_transition_floors_sigma(crossroads):
    """An over-confident sigma must not let one window dictate the route."""
    m = _m(crossroads, min_sigma_m=1.5)
    a = m._transition_logp(30.0, 25.0, 1e-6)
    b = m._transition_logp(30.0, 25.0, 1.5)
    assert a == pytest.approx(b)


# -- road-class prior ------------------------------------------------------

def test_class_prior_is_free_below_the_plausible_speed(crossroads):
    m = _m(crossroads)
    m.graph.highway = np.array(["residential", "residential"])
    assert m._class_logp(0, 10.0) == 0.0


def test_class_prior_penalises_motorway_speed_on_a_service_road(crossroads):
    m = _m(crossroads)
    m.graph.highway = np.array(["service", "service"])
    assert m._class_logp(0, 32.0) < -1.0


def test_class_prior_is_silent_for_unknown_classes(crossroads):
    m = _m(crossroads)
    m.graph.highway = np.array(["something_new", "something_new"])
    assert m._class_logp(0, 99.0) == 0.0
    assert set(CLASS_MAX_MS) and "motorway" in CLASS_MAX_MS


# -- state dedup -----------------------------------------------------------

def test_states_within_a_quantum_collapse():
    a = _State(EdgePosition(4, 10.0), 0.0)
    b = _State(EdgePosition(4, 14.0), -1.0)
    c = _State(EdgePosition(4, 30.0), -1.0)
    assert a.key(8.0) == b.key(8.0), "same edge, same quantum -> one state"
    assert a.key(8.0) != c.key(8.0)
    assert a.key(8.0) != _State(EdgePosition(5, 10.0), 0.0).key(8.0)


# -- the decode ------------------------------------------------------------

def test_backtrack_follows_pointers_not_the_final_score():
    """The decoded path is the best PATH, which need not be the path the
    winning state was following step by step."""
    l0 = [_State(EdgePosition(0, 0.0), 0.0), _State(EdgePosition(2, 0.0), -0.1)]
    l1 = [_State(EdgePosition(0, 10.0), -0.5, back=1),
          _State(EdgePosition(2, 10.0), -0.2, back=0)]
    l2 = [_State(EdgePosition(0, 20.0), -3.0, back=0),
          _State(EdgePosition(2, 20.0), -0.4, back=1)]
    path = ViterbiMapMatcher._backtrack([l0, l1, l2])
    assert [s.pos.edge for s in path] == [0, 2, 2]
    assert len(path) == 3


def test_backtrack_handles_an_empty_lattice():
    assert ViterbiMapMatcher._backtrack([]) == []
    assert ViterbiMapMatcher._backtrack([[]]) == []


def test_along_distance_is_exact_within_an_edge_and_falls_back_across_one(crossroads):
    m = _m(crossroads)
    same = m._along_distance(EdgePosition(0, 10.0), EdgePosition(0, 35.0), 99.0)
    assert same == pytest.approx(25.0)
    crossed = m._along_distance(EdgePosition(0, 10.0), EdgePosition(2, 5.0), 42.0)
    assert crossed == pytest.approx(42.0), "across a junction, use the command"


def test_config_inherits_the_backend_switch():
    """Viterbi must run on the in-process backend like everything else."""
    cfg = ViterbiConfig(backend="graph")
    assert cfg.backend == "graph"
    assert cfg.sigma_turn_rad > 0 and cfg.beam_width >= 2
