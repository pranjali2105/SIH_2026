"""The in-process (graph) matching backend: no server, no HTTP, no network.

Every test here must pass on a machine with no `osrm-routed` binary and no
network. That is the point of the backend, so it is what the tests assert.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mapmatch.graph import RoadGraph
from mapmatch.predictor import (MapMatchConfig, MapMatchedPredictor,
                                _resolve_backend)

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_CACHE = REPO_ROOT / "data" / "osm" / "road_graph.npz"

needs_map = pytest.mark.skipif(not GRAPH_CACHE.exists(),
                               reason="road_graph.npz not built")


@pytest.fixture
def crossroads() -> RoadGraph:
    """A plus-shaped junction: south-north and west-east, both two-way.

    Node 1 is the centre; 2 north, 3 south, 4 east, 5 west. Mirrors the
    fixture in tests/test_mapmatch.py.
    """
    coords = {1: (51.5000, -2.5000), 2: (51.5010, -2.5000),
              3: (51.4990, -2.5000), 4: (51.5000, -2.4986),
              5: (51.5000, -2.5014)}
    ways = [([3, 1, 2], False), ([5, 1, 4], False)]
    return RoadGraph.from_ways(coords, ways)


def _pred(graph: RoadGraph, **cfg_kw) -> MapMatchedPredictor:
    """A predictor with graph + cfg only -- deliberately NO `osrm` attribute.

    Bypassing __init__ mirrors tests/test_mapmatch.py and means any stray
    `self.osrm` reference on this path fails loudly with AttributeError
    instead of silently working because a client happened to be present.
    """
    p = MapMatchedPredictor.__new__(MapMatchedPredictor)
    p.graph = graph
    p.cfg = MapMatchConfig(backend="graph", **cfg_kw)
    p.backend = "graph"
    return p


# -- the core guarantee ----------------------------------------------------

def test_no_osrm_attribute_is_ever_touched(crossroads):
    """The load-bearing test for "no server process at all"."""
    p = _pred(crossroads)
    assert not hasattr(p, "osrm")

    # 40 m north of centre, heading north.
    lat, lon, hd = 51.50036, -2.5000, 0.0
    assert p._candidates(lat, lon, hd)          # would raise if osrm touched
    pos, _ = p._match(lat, lon, hd)
    assert pos is not None


def test_module_import_does_not_pull_in_the_osrm_client():
    """The library import graph must not carry subprocess/urllib.

    `predictor.py` imports OSRMClient only under TYPE_CHECKING; if that
    regresses to a runtime import, embedding the matcher drags a server
    client along with it.
    """
    import ast
    src = (REPO_ROOT / "src" / "mapmatch" / "predictor.py").read_text()
    tree = ast.parse(src)
    runtime_osrm = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "osrm"
        # under TYPE_CHECKING the ImportFrom sits inside an `if`, so a
        # module-level (col_offset 0) import is the regression we care about
        and n.col_offset == 0
    ]
    assert not runtime_osrm, "OSRMClient is imported at runtime, not type-only"


# -- candidate generation --------------------------------------------------

def test_candidates_are_nearest_first_with_metre_distances(crossroads):
    p = _pred(crossroads)
    # ~40 m north of the centre, on the south-north road.
    cands = p._candidates(51.50036, -2.5000, 0.0)
    assert cands
    dists = [d for _, d in cands]
    assert dists == sorted(dists), "candidates must be nearest first"
    assert dists[0] < 5.0, f"on-road point should snap tight, got {dists[0]:.1f} m"


def test_bearing_gate_selects_direction_of_travel(crossroads):
    """Heading picks BETWEEN the two directions of the same road."""
    p = _pred(crossroads)
    north, _ = p._match(51.50036, -2.5000, 0.0)
    south, _ = p._match(51.50036, -2.5000, np.pi)
    assert north is not None and south is not None
    assert (north.edge >> 1) == (south.edge >> 1), "same physical road"
    assert north.edge != south.edge, "opposite directions of travel"


def test_bearing_gate_uses_the_configured_half_window(crossroads):
    """The gate must be the configured window, not locate()'s 90 deg default.

    Heading east on the south-north road, with the radius tight enough that
    the east-west road (39.8 m away) is out of range -- otherwise it is
    legitimately admitted and the north-south rejection is invisible.
    """
    p = _pred(crossroads)
    raw = p.graph.nearest_edges(51.50036, -2.5000, radius_m=25.0, limit=12)
    assert raw, "the south-north road should be in range"

    tight = p._directed(raw, np.pi / 2, np.radians(60))
    assert not tight, "60 deg window must reject a road 90 deg off heading"

    wide = p._directed(raw, np.pi / 2, np.radians(120))
    assert wide, "120 deg window should admit it"


def test_bearing_gate_falls_back_unrestricted(crossroads):
    """A filter admitting nothing is a failure of the filter, not evidence
    there is no road -- same rule the OSRM path uses."""
    p = _pred(crossroads, osrm_bearing_range_deg=5)
    cands = p._candidates(51.50036, -2.5000, np.pi / 2)
    assert cands, "should fall back to an unrestricted search"


# -- ambiguity -------------------------------------------------------------

def test_junction_centre_is_ambiguous(crossroads):
    """Two genuinely competing roads meet here; the caller must stand still."""
    p = _pred(crossroads)
    _, ambiguous = p._match(51.5000, -2.5000, 0.0)
    assert ambiguous


def test_mid_edge_is_not_ambiguous(crossroads):
    """Well along one road with the crossing road out of range."""
    p = _pred(crossroads, snap_radius_m=25.0)
    _, ambiguous = p._match(51.50072, -2.5000, 0.0)
    assert not ambiguous


def test_no_road_within_radius_returns_none(crossroads):
    """`predict` must skip the outage rather than snap to nothing."""
    p = _pred(crossroads, snap_radius_m=40.0)
    pos, ambiguous = p._match(51.6000, -2.5000, 0.0)
    assert pos is None and ambiguous is False


# -- backend validation ----------------------------------------------------

def test_backend_osrm_without_client_is_rejected():
    with pytest.raises(ValueError, match="needs an OSRMClient"):
        _resolve_backend(MapMatchConfig(backend="osrm"), None)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        _resolve_backend(MapMatchConfig(backend="magic"), None)


def test_graph_backend_needs_no_client():
    assert _resolve_backend(MapMatchConfig(backend="graph"), None) == "graph"


# -- HMM inherits the backend for free -------------------------------------

def test_hmm_start_candidates_on_graph_backend(crossroads):
    from mapmatch.hmm_predictor import (HMMMapMatchConfig,
                                        HMMMapMatchedPredictor)
    p = HMMMapMatchedPredictor.__new__(HMMMapMatchedPredictor)
    p.graph = crossroads
    p.cfg = HMMMapMatchConfig(backend="graph", n_start_hypotheses=2)
    p.backend = "graph"
    assert not hasattr(p, "osrm")

    out = p._match_candidates(51.5000, -2.5000, 0.0)
    assert 0 < len(out) <= 2
    weights = [w for _, w in out]
    assert weights == sorted(weights, reverse=True), "must be sorted"
    assert weights[0] == pytest.approx(0.0), "best candidate is the reference"


# -- against the real graph ------------------------------------------------

@needs_map
def test_candidates_on_the_real_graph():
    """Real fixes from a test session snap without OSRM."""
    graph = RoadGraph.load(GRAPH_CACHE)
    p = _pred(graph, snap_radius_m=60.0)
    # A point on the S-A5 route (Somerset), from the session bbox.
    cands = p._candidates(51.2, -2.9, None)
    for _, d in cands:
        assert d <= 60.0, "candidate outside the snap radius"
