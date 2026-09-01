"""Tests for offline map matching and along-road tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mapmatch.extent import BBox, session_bbox
from mapmatch.build_map import select_regions
from mapmatch.graph import EdgePosition, RoadGraph, bearing_rad, wrap_pi
from mapmatch.osrm import OSRMClient, OfflineViolation
from mapmatch.predictor import MapMatchConfig, MapMatchedPredictor, OutageTrace

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
MAP_DIR = REPO_ROOT / "data" / "osm"

needs_data = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason="dataset absent — run scripts/fetch_data.sh")
needs_map = pytest.mark.skipif(
    not (MAP_DIR / "road_graph.npz").exists()
    and not (MAP_DIR / "extent.osm.pbf").exists(),
    reason="offline map absent — run `python -m mapmatch.build_map`")


def _open_graph() -> RoadGraph:
    """The built graph, from its cache when the extract has been reclaimed."""
    cache = MAP_DIR / "road_graph.npz"
    if cache.exists():
        return RoadGraph.load(cache)
    return RoadGraph.build(MAP_DIR / "extent.osm.pbf", cache)


# -- geometry --------------------------------------------------------------

def test_bearing_cardinal_directions():
    assert bearing_rad(51.0, -2.0, 52.0, -2.0) == pytest.approx(0.0, abs=1e-9)
    assert bearing_rad(51.0, -2.0, 51.0, -1.0) == pytest.approx(np.pi / 2,
                                                                abs=1e-2)


def test_wrap_pi_folds_into_the_half_open_interval():
    assert float(wrap_pi(3 * np.pi)) == pytest.approx(-np.pi)
    assert float(wrap_pi(-1.5 * np.pi)) == pytest.approx(0.5 * np.pi)
    assert float(wrap_pi(0.25 * np.pi)) == pytest.approx(0.25 * np.pi)


# -- bbox ------------------------------------------------------------------

def test_bbox_padding_widens_both_axes():
    b = BBox(-2.0, 51.0, -1.0, 52.0)
    p = b.padded(10.0)
    assert p.min_lon < b.min_lon and p.max_lon > b.max_lon
    assert p.min_lat < b.min_lat and p.max_lat > b.max_lat


def test_bbox_intersects():
    a = BBox(-2.0, 51.0, -1.0, 52.0)
    assert a.intersects(BBox(-1.5, 51.5, 0.0, 53.0))
    assert not a.intersects(BBox(0.0, 51.0, 1.0, 52.0))


@needs_data
def test_extent_covers_every_test_session_fix():
    """The map extent must contain the GPS it was derived from."""
    import pandas as pd
    from data.loader import _find, load_session
    box = session_bbox()
    for name in ("S-A5", "S-A6", "S-A7", "S-A8"):
        s = load_session(name, check_rate=False)
        la = pd.to_numeric(s.df[_find(s.df, r"LATITUDE")], errors="coerce")
        lo = pd.to_numeric(s.df[_find(s.df, r"LONGITUDE")], errors="coerce")
        ok = la.notna() & lo.notna() & (la != 0) & (lo != 0)
        assert box.min_lat <= la[ok].min() and la[ok].max() <= box.max_lat
        assert box.min_lon <= lo[ok].min() and lo[ok].max() <= box.max_lon


def test_select_regions_takes_leaves_not_parents():
    """A parent and its children both intersect; downloading both is waste."""
    def feat(rid, parent, box):
        lo, la, hi, ha = box
        return {"properties": {"id": rid, "parent": parent,
                               "urls": {"pbf": f"http://x/{rid}.pbf"}},
                "geometry": {"type": "Polygon",
                             "coordinates": [[[lo, la], [hi, la], [hi, ha],
                                              [lo, ha], [lo, la]]]}}
    index = {"features": [
        feat("united-kingdom", None, (-8, 49, 2, 61)),
        feat("england", "united-kingdom", (-6, 50, 2, 56)),
        feat("devon", "england", (-4.7, 50.2, -2.9, 51.3)),
        feat("kent", "england", (0.0, 50.9, 1.5, 51.5)),
        feat("france", None, (-5, 42, 8, 51)),
    ]}
    picked = dict(select_regions(index, BBox(-4.0, 50.4, -3.0, 51.0)))
    assert set(picked) == {"devon"}          # not england, not united-kingdom


# -- synthetic graph -------------------------------------------------------

@pytest.fixture
def crossroads() -> RoadGraph:
    """A plus-shaped junction: south-north and west-east, both two-way.

    Node 1 is the centre; 2 north, 3 south, 4 east, 5 west.
    """
    coords = {1: (51.5000, -2.5000), 2: (51.5010, -2.5000),
              3: (51.4990, -2.5000), 4: (51.5000, -2.4986),
              5: (51.5000, -2.5014)}
    ways = [([3, 1, 2], False), ([5, 1, 4], False)]
    return RoadGraph.from_ways(coords, ways)


def _edge_from_to(g: RoadGraph, a_osm: int, b_osm: int) -> int:
    """The directed edge running from OSM node `a_osm` to `b_osm`."""
    ids = {int(n): i for i, n in enumerate(g.node_id)}
    a, b = ids[a_osm], ids[b_osm]
    hits = [e for e in range(2 * g.n_undirected)
            if g.d_from[e] == a and g.d_to[e] == b and g.passable[e]]
    assert len(hits) == 1, f"{a_osm}->{b_osm}: {hits}"
    return hits[0]


def test_ways_split_at_the_junction(crossroads):
    # Two ways, each cut at node 1, gives four undirected edges.
    assert crossroads.n_undirected == 4
    assert crossroads.passable.all()          # nothing is one-way


def test_edge_length_matches_geometry(crossroads):
    e = _edge_from_to(crossroads, 3, 1)
    # 0.001 degrees of latitude is about 111 m.
    assert crossroads.length(e) == pytest.approx(111.2, rel=0.02)


def test_position_interpolates_along_the_polyline(crossroads):
    e = _edge_from_to(crossroads, 3, 1)
    half = crossroads.length(e) / 2
    lat, lon = crossroads.position(EdgePosition(e, half))
    assert lat == pytest.approx(51.4995, abs=1e-5)
    assert lon == pytest.approx(-2.5000, abs=1e-6)


@pytest.mark.parametrize("turn_deg, expect_node", [
    (0.0, 2),        # straight on, north
    (90.0, 4),       # right, east
    (-90.0, 5),      # left, west
])
def test_junction_branch_follows_the_gyro(crossroads, turn_deg, expect_node):
    """The observed turn angle selects the branch, and nothing else does."""
    arriving = _edge_from_to(crossroads, 3, 1)
    node = int(crossroads.d_to[arriving])
    pred = MapMatchedPredictor.__new__(MapMatchedPredictor)
    pred.graph, pred.cfg = crossroads, MapMatchConfig()
    trace = OutageTrace()
    chosen = pred._choose_branch(arriving, node, np.deg2rad(turn_deg), trace)
    ids = {int(n): i for i, n in enumerate(crossroads.node_id)}
    assert int(crossroads.d_to[chosen]) == ids[expect_node]


def test_junction_never_makes_a_u_turn(crossroads):
    """Reversing back down the edge just travelled is not a branch."""
    arriving = _edge_from_to(crossroads, 3, 1)
    node = int(crossroads.d_to[arriving])
    pred = MapMatchedPredictor.__new__(MapMatchedPredictor)
    pred.graph, pred.cfg = crossroads, MapMatchConfig()
    # A 180 degree observed turn still cannot pick the reverse edge.
    chosen = pred._choose_branch(arriving, node, np.pi, OutageTrace())
    assert (chosen >> 1) != (arriving >> 1)


def test_advance_crosses_the_junction_onto_the_chosen_branch(crossroads):
    arriving = _edge_from_to(crossroads, 3, 1)
    pred = MapMatchedPredictor.__new__(MapMatchedPredictor)
    pred.graph, pred.cfg = crossroads, MapMatchConfig()
    pred._turn_angle = lambda t, sign: np.deg2rad(90.0)      # turn east
    trace = OutageTrace()
    start = EdgePosition(arriving, 0.0)
    end = pred._advance(start, crossroads.length(arriving) + 40.0, 0.0, 1.0,
                        trace)
    assert trace.junctions == 1
    assert end.offset_m == pytest.approx(40.0, abs=1e-6)
    ids = {int(n): i for i, n in enumerate(crossroads.node_id)}
    assert int(crossroads.d_to[end.edge]) == ids[4]          # heading east


def test_advance_within_an_edge_touches_no_junction(crossroads):
    e = _edge_from_to(crossroads, 3, 1)
    pred = MapMatchedPredictor.__new__(MapMatchedPredictor)
    pred.graph, pred.cfg = crossroads, MapMatchConfig()
    trace = OutageTrace()
    end = pred._advance(EdgePosition(e, 0.0), 50.0, 0.0, 1.0, trace)
    assert trace.junctions == 0
    assert end.edge == e and end.offset_m == pytest.approx(50.0)


def test_oneway_blocks_the_reverse_direction():
    coords = {1: (51.5, -2.5), 2: (51.501, -2.5)}
    g = RoadGraph.from_ways(coords, [([1, 2], True)])
    assert g.passable[0] and not g.passable[1]


def test_locate_picks_the_direction_of_travel(crossroads):
    """Same road, opposite headings, must give opposite directed edges."""
    north = crossroads.locate(51.4995, -2.5000, heading=0.0, radius_m=50)
    south = crossroads.locate(51.4995, -2.5000, heading=np.pi, radius_m=50)
    assert north is not None and south is not None
    assert (north.edge >> 1) == (south.edge >> 1)     # the same road
    assert north.edge != south.edge                   # opposite directions


def test_locate_node_pair_anchors_to_the_named_segment(crossroads):
    pos = crossroads.locate_node_pair(3, 1, 51.4995, -2.5000, heading=0.0)
    assert pos is not None
    assert int(crossroads.d_to[pos.edge]) == \
        {int(n): i for i, n in enumerate(crossroads.node_id)}[1]


# -- harness contract ------------------------------------------------------

def test_chords_reproduce_the_track_under_harness_accumulation():
    """The harness sums d * [cos psi, sin psi]; that must retrace our path.

    This is the whole reason the predictor emits chords rather than a heading
    it integrated: the harness's own arithmetic has to land on the positions
    the tracker actually visited.
    """
    from eval.metrics import geodesic_distance_m
    rng = np.random.default_rng(0)
    lat = 51.5 + np.cumsum(rng.normal(0, 2e-4, 30))
    lon = -2.5 + np.cumsum(rng.normal(0, 2e-4, 30))
    track = np.column_stack([lat, lon])

    out = MapMatchedPredictor._to_harness(track, OutageTrace())
    north = np.cumsum(out["displacements"] * np.cos(out["headings"]))
    east = np.cumsum(out["displacements"] * np.sin(out["headings"]))

    lat0, lon0 = lat[0], lon[0]
    m_lat = geodesic_distance_m(lat0, lon0, lat0 + 1e-4, lon0) / 1e-4
    m_lon = geodesic_distance_m(lat0, lon0, lat0, lon0 + 1e-4) / 1e-4
    true_n = (lat[1:] - lat0) * m_lat
    true_e = (lon[1:] - lon0) * m_lon

    err = np.hypot(north - true_n, east - true_e)
    # Sub-centimetre over ~1 km of track: the residual is the local flat-earth
    # conversion the harness uses, not the chord representation.
    assert err.max() < 0.05


def test_zero_length_second_holds_the_previous_bearing():
    """A stopped second has no bearing; NaN would poison the accumulation."""
    track = np.array([[51.5, -2.5], [51.501, -2.5], [51.501, -2.5]])
    out = MapMatchedPredictor._to_harness(track, OutageTrace())
    assert np.isfinite(out["headings"]).all()
    assert out["headings"][1] == pytest.approx(out["headings"][0])


# -- offline guarantee -----------------------------------------------------

def test_non_loopback_host_is_refused():
    """Inference must not leave the machine, and this is what enforces it."""
    c = OSRMClient.__new__(OSRMClient)
    c.host, c.port = "example.com", 5000
    with pytest.raises(OfflineViolation):
        c._get("nearest", "0,0", {})


# -- end to end ------------------------------------------------------------

@needs_data
@needs_map
def test_graph_snaps_a_real_test_session_fix():
    from data.loader import load_session, _find
    import pandas as pd
    g = _open_graph()
    s = load_session("S-A5", check_rate=False)
    la = pd.to_numeric(s.df[_find(s.df, r"LATITUDE")], errors="coerce")
    lo = pd.to_numeric(s.df[_find(s.df, r"LONGITUDE")], errors="coerce")
    ok = (la.notna() & lo.notna() & (la != 0) & (lo != 0)).to_numpy()
    idx = np.flatnonzero(ok)[::5000][:10]
    hits = sum(g.locate(float(la.iloc[i]), float(lo.iloc[i]),
                        radius_m=60.0) is not None for i in idx)
    assert hits >= len(idx) - 1        # allow one fix in a car park


@needs_map
def test_node_pair_lookup_round_trips_on_the_real_graph():
    """Anchoring by OSM node id must find the edge those nodes belong to.

    The packed-index key is the part that could silently break on a larger
    extract, so it is exercised against the built graph, not only the
    synthetic one.
    """
    g = _open_graph()
    rng = np.random.default_rng(0)
    for e in rng.integers(0, g.n_undirected, 200):
        e = int(e)
        a, b = int(g._ptr[e]), int(g._ptr[e + 1])
        i = int(rng.integers(a, b - 1))
        u, v = int(g.node_id[g._seq[i]]), int(g.node_id[g._seq[i + 1]])
        assert g._edge_for_node_pair(u, v) == e
        assert g._edge_for_node_pair(v, u) == e
