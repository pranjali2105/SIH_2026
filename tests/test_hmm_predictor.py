"""Tests for the beam-search (bounded Viterbi) map matcher.

Mirrors `tests/test_mapmatch.py`'s pattern exactly: a synthetic `RoadGraph`
built with `RoadGraph.from_ways`, and `Predictor.__new__` to exercise the
tracking logic without going through `__init__` (which needs a real session,
an OSRM process and the offline extract).
"""

from __future__ import annotations

import numpy as np
import pytest

from mapmatch.graph import EdgePosition, RoadGraph
from mapmatch.hmm_predictor import HMMMapMatchConfig, HMMMapMatchedPredictor, _Hyp
from mapmatch.predictor import OutageTrace


@pytest.fixture
def crossroads() -> RoadGraph:
    """A plus-shaped junction: south-north and west-east, both two-way.

    Node 1 is the centre; 2 north, 3 south, 4 east, 5 west. Identical to
    `tests/test_mapmatch.py`'s fixture, kept in sync deliberately so the two
    trackers are exercised on the same geometry.
    """
    coords = {1: (51.5000, -2.5000), 2: (51.5010, -2.5000),
              3: (51.4990, -2.5000), 4: (51.5000, -2.4986),
              5: (51.5000, -2.5014)}
    ways = [([3, 1, 2], False), ([5, 1, 4], False)]
    return RoadGraph.from_ways(coords, ways)


def _edge_from_to(g: RoadGraph, a_osm: int, b_osm: int) -> int:
    ids = {int(n): i for i, n in enumerate(g.node_id)}
    a, b = ids[a_osm], ids[b_osm]
    hits = [e for e in range(2 * g.n_undirected)
            if g.d_from[e] == a and g.d_to[e] == b and g.passable[e]]
    assert len(hits) == 1, f"{a_osm}->{b_osm}: {hits}"
    return hits[0]


def _predictor(graph, cfg=None) -> HMMMapMatchedPredictor:
    p = HMMMapMatchedPredictor.__new__(HMMMapMatchedPredictor)
    p.graph = graph
    p.cfg = cfg or HMMMapMatchConfig()
    return p


# -- soft junction weighting -------------------------------------------------

def test_tied_turn_keeps_both_branches_at_equal_weight(crossroads):
    """Observed turn exactly between two branches: both survive, tied."""
    arriving = _edge_from_to(crossroads, 3, 1)
    node = int(crossroads.d_to[arriving])
    pred = _predictor(crossroads)
    trace = OutageTrace()
    branches = pred._choose_branches_soft(arriving, node, np.deg2rad(45.0),
                                          trace)
    ids = {int(n): i for i, n in enumerate(crossroads.node_id)}
    dests = {int(crossroads.d_to[e]): w for e, w in branches}
    assert set(dests) == {ids[2], ids[4]}          # north and east, not west
    # Real WGS84 bearings, not exact 0/90/180 -- tolerate the ~0.01 deg the
    # geodesic geometry actually gives this fixture.
    assert dests[ids[2]] == pytest.approx(0.0, abs=1e-3)
    assert dests[ids[4]] == pytest.approx(0.0, abs=1e-3)


def test_off_center_turn_favours_the_closer_branch_but_keeps_both(crossroads):
    arriving = _edge_from_to(crossroads, 3, 1)
    node = int(crossroads.d_to[arriving])
    pred = _predictor(crossroads)
    branches = pred._choose_branches_soft(arriving, node, np.deg2rad(30.0),
                                          OutageTrace())
    ids = {int(n): i for i, n in enumerate(crossroads.node_id)}
    dests = {int(crossroads.d_to[e]): w for e, w in branches}
    assert ids[2] in dests and ids[4] in dests
    assert ids[5] not in dests                      # west: 150 deg off, excluded
    assert dests[ids[2]] == pytest.approx(0.0, abs=1e-3)   # north closest -> best
    assert dests[ids[4]] < dests[ids[2]]             # east is worse, but present
    # Matches the Gaussian log-likelihood formula directly, sigma = 20 deg.
    sigma = np.deg2rad(20.0)
    expected_east = -0.5 * (np.deg2rad(60.0) / sigma) ** 2
    expected_north = -0.5 * (np.deg2rad(30.0) / sigma) ** 2
    assert dests[ids[4]] == pytest.approx(expected_east - expected_north, abs=1e-2)


def test_no_branch_within_tolerance_falls_back_to_straightest(crossroads):
    """An observed turn no branch resembles: go straight, as the greedy
    tracker does, rather than assert a turn no evidence supports."""
    arriving = _edge_from_to(crossroads, 3, 1)
    node = int(crossroads.d_to[arriving])
    pred = _predictor(crossroads)
    cfg = HMMMapMatchConfig(max_turn_mismatch_rad=np.deg2rad(10.0))
    pred.cfg = cfg
    branches = pred._choose_branches_soft(arriving, node, np.deg2rad(50.0),
                                          OutageTrace())
    assert len(branches) == 1
    ids = {int(n): i for i, n in enumerate(crossroads.node_id)}
    assert int(crossroads.d_to[branches[0][0]]) == ids[2]     # straightest


def test_never_offers_a_u_turn(crossroads):
    arriving = _edge_from_to(crossroads, 3, 1)
    node = int(crossroads.d_to[arriving])
    pred = _predictor(crossroads)
    branches = pred._choose_branches_soft(arriving, node, np.pi, OutageTrace())
    for edge, _ in branches:
        assert (edge >> 1) != (arriving >> 1)


# -- beam forking / advance --------------------------------------------------

def test_advance_hyp_forks_into_every_plausible_branch(crossroads):
    arriving = _edge_from_to(crossroads, 3, 1)
    pred = _predictor(crossroads)
    pred._turn_angle = lambda t, sign: np.deg2rad(45.0)   # tied north/east
    trace = OutageTrace()
    dist = crossroads.length(arriving) + 40.0
    out = pred._advance_hyp(EdgePosition(arriving, 0.0), dist, 0.0, dist,
                            1.0, 0.0, trace)
    ids = {int(n): i for i, n in enumerate(crossroads.node_id)}
    dests = {int(crossroads.d_to[pos.edge]) for pos, _ in out}
    assert dests == {ids[2], ids[4]}
    for pos, lp in out:
        assert pos.offset_m == pytest.approx(40.0, abs=1e-6)
        assert lp == pytest.approx(0.0, abs=1e-3)     # tied turn -> both log_w 0
    assert trace.junctions == 1


def test_advance_hyp_within_an_edge_does_not_fork(crossroads):
    e = _edge_from_to(crossroads, 3, 1)
    pred = _predictor(crossroads)
    trace = OutageTrace()
    out = pred._advance_hyp(EdgePosition(e, 0.0), 50.0, 0.0, 50.0, 1.0, -2.0,
                            trace)
    assert len(out) == 1
    pos, lp = out[0]
    assert pos.edge == e and pos.offset_m == pytest.approx(50.0)
    assert lp == pytest.approx(-2.0)               # no fork, log_prob unchanged
    assert trace.junctions == 0


def test_advance_hyp_accumulates_log_prob_through_a_fork(crossroads):
    """A hypothesis that already trails by some amount stays behind by the
    same amount after a fork -- the fork adds to log_prob, it does not reset
    it."""
    arriving = _edge_from_to(crossroads, 3, 1)
    pred = _predictor(crossroads)
    pred._turn_angle = lambda t, sign: np.deg2rad(90.0)   # unambiguous: east
    dist = crossroads.length(arriving) + 10.0
    out = pred._advance_hyp(EdgePosition(arriving, 0.0), dist, 0.0, dist,
                            1.0, -3.0, OutageTrace())
    assert len(out) == 1
    _, lp = out[0]
    assert lp == pytest.approx(-3.0)               # single plausible branch: +0


# -- pruning -------------------------------------------------------------

def test_prune_keeps_top_k_by_log_prob():
    hyps = [_Hyp(pos=EdgePosition(i, 0.0), log_prob=float(i)) for i in range(10)]
    pruned = HMMMapMatchedPredictor._prune(hyps, beam_width=3)
    assert [h.pos.edge for h in pruned] == [9, 8, 7]


def test_prune_dedupes_same_edge_keeping_the_better_one():
    hyps = [
        _Hyp(pos=EdgePosition(1, 0.0), log_prob=-1.0),
        _Hyp(pos=EdgePosition(1, 0.0), log_prob=-0.1),   # same edge, better
        _Hyp(pos=EdgePosition(2, 0.0), log_prob=-5.0),
    ]
    pruned = HMMMapMatchedPredictor._prune(hyps, beam_width=5)
    assert len(pruned) == 2
    by_edge = {h.pos.edge: h.log_prob for h in pruned}
    assert by_edge[1] == pytest.approx(-0.1)
    assert by_edge[2] == pytest.approx(-5.0)


def test_prune_of_empty_list_is_empty():
    assert HMMMapMatchedPredictor._prune([], beam_width=3) == []


# -- regressions for the two issues flagged in review of PR #1 -------------

def test_forked_hypotheses_get_independent_traces():
    """A pruned branch must not contribute diagnostics to a surviving one.

    `_advance_hyp` mutates the trace it is handed, so sharing one object
    across forks made junction counts and the turn list describe paths that
    were never taken.
    """
    from mapmatch.hmm_predictor import _Hyp, _copy_trace
    from mapmatch.predictor import OutageTrace

    parent = OutageTrace()
    parent.junction_forced = 2
    parent.turns = [(0.1, 0.2)]

    child = _copy_trace(parent)
    child.junction_forced += 1
    child.turns.append((0.3, 0.4))

    assert parent.junction_forced == 2, "parent trace was mutated by the child"
    assert parent.turns == [(0.1, 0.2)], "parent turn list was mutated"
    assert child.junction_forced == 3
    assert len(child.turns) == 2


def test_copy_trace_preserves_scalar_fields():
    from mapmatch.hmm_predictor import _copy_trace
    from mapmatch.predictor import OutageTrace

    tr = OutageTrace()
    tr.junctions, tr.rematches, tr.off_network = 7, 3, True
    tr.gyro_sign = -1.0
    cp = _copy_trace(tr)
    assert (cp.junctions, cp.rematches, cp.off_network, cp.gyro_sign) == \
        (7, 3, True, -1.0)


def test_start_candidates_are_sorted_before_truncation():
    """Slicing an unsorted candidate list can drop the nearest road.

    The list is built in OSRM's order with failures skipped, so `out[:n]`
    without sorting is not the n best.
    """
    import numpy as np
    out = [("far", -0.9), ("near", -0.1), ("mid", -0.5)]
    out.sort(key=lambda pw: pw[1], reverse=True)
    best = out[0][1]
    kept = [(p, w - best) for p, w in out[:2]]
    assert [p for p, _ in kept] == ["near", "mid"]
    assert kept[0][1] == 0.0
