"""Drivable road graph: polylines and junction connectivity.

Built from the SAME cropped extract `osrm-extract` consumed, so the geometry
the tracker walks along and the geometry OSRM snaps to cannot disagree.

Why a graph at all, when OSRM is already here: OSRM's HTTP API answers "where
am I on the road network" (`/nearest`) and "route me from A to B" (`/route`).
Neither answers "I am 40 m from the end of this segment, which branches leave
the junction ahead and at what angle" -- which is the whole of along-road
tracking during an outage, when the destination is by definition unknown. So
OSRM does the matching, as specified, and this does the tracking.

Reading the PBF. osmium's OPL text format is line-oriented and carries way
node references, which `osmium export` drops. Parsing it directly keeps this
module free of a compiled OSM binding; the parse runs once and caches to an
.npz, so the cost is paid at setup, not at inference.

Geometry convention throughout: bearings are radians CLOCKWISE FROM NORTH,
matching `eval.harness`, which accumulates position as
`d * [cos(psi), sin(psi)]` into (north, east).
"""

from __future__ import annotations

import subprocess
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Geod

# Tags whose ways a car can drive. `service` and `living_street` are included
# because outages do end up in car parks and estates; footways, cycleways and
# tracks are not.
DRIVABLE = frozenset({
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street", "service", "road",
})

# Ways that are one-directional even without an explicit oneway tag.
IMPLICIT_ONEWAY_HIGHWAY = frozenset({"motorway", "motorway_link"})
IMPLICIT_ONEWAY_JUNCTION = frozenset({"roundabout", "circular"})

# WGS84, the same ellipsoid `eval.metrics` measures ground truth on. Using a
# spherical approximation here instead costs ~0.3% on east-west distance,
# which sounds negligible and is not: it is a SYSTEMATIC along-track bias, so
# over a 1 km outage it becomes a metres-scale error that grows with every
# outage second and would be indistinguishable from a model bias.
_GEOD = Geod(ellps="WGS84")


def geodesic_m(lat1, lon1, lat2, lon2):
    """Geodesic distance in metres. Scalars or arrays."""
    return _GEOD.inv(lon1, lat1, lon2, lat2)[2]


def bearing_rad(lat1, lon1, lat2, lon2):
    """Forward azimuth, radians clockwise from north."""
    az = _GEOD.inv(lon1, lat1, lon2, lat2)[0]
    return np.radians(az)


def wrap_pi(a):
    """Wrap an angle to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class EdgePosition:
    """A point on the network: how far along one DIRECTED edge.

    `edge` indexes the directed edge array; `offset_m` is distance from that
    edge's start along its polyline.
    """
    edge: int
    offset_m: float

    def __repr__(self) -> str:
        return f"EdgePosition(edge={self.edge}, offset={self.offset_m:.1f} m)"


class GraphBuildError(RuntimeError):
    """The road graph could not be built from the extract."""


class RoadGraph:
    """Directed drivable graph with polyline geometry.

    Directed edges come in pairs: edge `2e` traverses undirected edge `e`
    forwards, `2e+1` backwards. A one-way edge simply has its reverse marked
    impassable rather than removed, so geometry indices stay symmetric.
    """

    def __init__(self, lat, lon, ptr, seq, cum, passable, node_id):
        self.node_lat = lat            # (n_pts,) shape-point latitudes
        self.node_lon = lon
        self._ptr = ptr                # (n_edges+1,) into seq/cum
        self._seq = seq                # concatenated shape-point indices
        self._cum = cum                # cumulative metres along each polyline
        self.passable = passable       # (2*n_edges,) bool
        self.node_id = node_id         # (n_pts,) OSM node ids
        self.n_undirected = len(ptr) - 1
        self._build_topology()

    # -- construction ------------------------------------------------------

    @staticmethod
    def _filter_roads(pbf: Path, out: Path) -> Path:
        """Reduce the extract to drivable ways and the nodes they reference.

        Cuts the OPL stream the parse has to read by roughly an order of
        magnitude -- buildings, landuse and footpaths dominate an OSM extract
        and none of them are drivable.
        """
        expr = "w/highway=" + ",".join(sorted(DRIVABLE))
        cmd = ["osmium", "tags-filter", "--overwrite", str(pbf), expr,
               "-o", str(out)]
        print(f"  $ osmium tags-filter ... -o {out.name}", file=sys.stderr)
        subprocess.run(cmd, check=True)
        return out

    @staticmethod
    def _parse_opl(pbf: Path):
        """Stream `osmium cat -f opl` into flat arrays.

        Arrays, not dictionaries. A `{node_id: (lat, lon)}` dict over the
        millions of nodes in this extract costs several GB of Python object
        overhead; the same data as three typed arrays is a few hundred MB, and
        way references resolve against it with a sorted `searchsorted` instead
        of millions of hash lookups.

        OPL emits nodes before ways, so a single pass suffices.
        """
        cmd = ["osmium", "cat", "-f", "opl", str(pbf)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True,
                                bufsize=1 << 20)
        nid = array("q")
        nlat, nlon = array("d"), array("d")
        way_refs, way_ptr, way_oneway = array("q"), array("q", [0]), array("b")
        assert proc.stdout is not None
        for line in proc.stdout:
            kind = line[0]
            if kind == "n":
                x = y = None
                fields = line.rstrip("\n").split(" ")
                for f in fields:
                    c = f[:1]
                    if c == "x":
                        x = f[1:]
                    elif c == "y":
                        y = f[1:]
                if not x or not y:
                    continue
                nid.append(int(fields[0][1:]))
                nlat.append(float(y))
                nlon.append(float(x))
            elif kind == "w":
                tags = refs = ""
                for f in line.rstrip("\n").split(" "):
                    c = f[:1]
                    if c == "T":
                        tags = f[1:]
                    elif c == "N":
                        refs = f[1:]
                if not refs or "highway=" not in tags:
                    continue
                tagmap = {}
                for kv in tags.split(","):
                    k, _, v = kv.partition("=")
                    tagmap[k] = v
                hw = tagmap.get("highway", "")
                if hw not in DRIVABLE or tagmap.get("access") in ("no", "private"):
                    continue
                nodes = [int(r[1:]) for r in refs.split(",") if r[:1] == "n"]
                if len(nodes) < 2:
                    continue
                ow = tagmap.get("oneway", "")
                oneway = (ow in ("yes", "true", "1")
                          or hw in IMPLICIT_ONEWAY_HIGHWAY
                          or tagmap.get("junction", "") in IMPLICIT_ONEWAY_JUNCTION)
                if ow == "-1":
                    nodes.reverse()      # tagged against the drawn direction
                    oneway = True
                way_refs.extend(nodes)
                way_ptr.append(len(way_refs))
                way_oneway.append(1 if oneway else 0)
        proc.stdout.close()
        if proc.wait() != 0:
            raise GraphBuildError(f"`{' '.join(cmd)}` failed")
        return (np.frombuffer(nid, dtype=np.int64).copy(),
                np.frombuffer(nlat, dtype=np.float64).copy(),
                np.frombuffer(nlon, dtype=np.float64).copy(),
                np.frombuffer(way_ptr, dtype=np.int64).copy(),
                np.frombuffer(way_refs, dtype=np.int64).copy(),
                np.frombuffer(way_oneway, dtype=np.int8).astype(bool))

    @classmethod
    def build(cls, pbf: Path, cache: Path | None = None,
              keep_filtered: bool = False) -> "RoadGraph":
        """Parse `pbf` into a graph, splitting ways at junctions."""
        pbf = Path(pbf)
        roads = pbf.with_name("roads.osm.pbf")
        made = False
        if not roads.exists():
            cls._filter_roads(pbf, roads)
            made = True
        try:
            print(f"parsing {roads.name} ...", file=sys.stderr)
            parsed = cls._parse_opl(roads)
        finally:
            if made and not keep_filtered:
                roads.unlink(missing_ok=True)
        if parsed[3].size < 2:
            raise GraphBuildError(f"no drivable ways found in {pbf}")
        print(f"  {parsed[3].size - 1} drivable ways, {parsed[0].size} nodes",
              file=sys.stderr)
        g = cls._assemble(*parsed)
        print(f"  {g.n_undirected} edges, {int(g.passable.sum())} directed",
              file=sys.stderr)
        if cache is not None:
            g.save(Path(cache))
        return g

    @classmethod
    def from_ways(cls, coords: dict, ways: list) -> "RoadGraph":
        """Assemble from `{osm_node_id: (lat, lon)}` and `[(refs, oneway)]`.

        The convenient form, for tests and small inputs; `build` goes through
        the array path instead because the corpus extract is too large for a
        dict of node coordinates.
        """
        keys = sorted(coords)
        nid = np.asarray(keys, dtype=np.int64)
        nlat = np.array([coords[k][0] for k in keys], dtype=np.float64)
        nlon = np.array([coords[k][1] for k in keys], dtype=np.float64)
        refs, ptr, ow = [], [0], []
        for nodes, oneway in ways:
            refs.extend(nodes)
            ptr.append(len(refs))
            ow.append(bool(oneway))
        return cls._assemble(nid, nlat, nlon,
                             np.asarray(ptr, dtype=np.int64),
                             np.asarray(refs, dtype=np.int64),
                             np.asarray(ow, dtype=bool))

    @classmethod
    def _assemble(cls, nid, nlat, nlon, way_ptr, way_refs, way_oneway):
        """Split every way at its junctions and index the result."""
        order = np.argsort(nid, kind="stable")
        nid_sorted = nid[order]
        pos = np.searchsorted(nid_sorted, way_refs)
        pos = np.clip(pos, 0, nid_sorted.size - 1)
        known = nid_sorted[pos] == way_refs        # refs outside the extract
        ref_idx = order[pos]                       # index into nlat/nlon

        # A node is a junction if two or more ways use it, or it terminates a
        # way. Ways are split there, so every edge runs junction to junction.
        counts = np.bincount(ref_idx[known], minlength=nid.size)
        is_junction = counts >= 2
        starts, stops = way_ptr[:-1], way_ptr[1:] - 1
        is_junction[ref_idx[starts[known[starts]]]] = True
        is_junction[ref_idx[stops[known[stops]]]] = True

        ptr, seq, oneway_flags = [0], [], []
        for w in range(way_ptr.size - 1):
            a, b = way_ptr[w], way_ptr[w + 1]
            nodes = ref_idx[a:b][known[a:b]]
            if nodes.size < 2:
                continue
            ow = bool(way_oneway[w])
            run = [int(nodes[0])]
            for n in nodes[1:]:
                n = int(n)
                run.append(n)
                if is_junction[n] and len(run) >= 2:
                    seq.extend(run)
                    ptr.append(len(seq))
                    oneway_flags.append(ow)
                    run = [n]
            if len(run) >= 2:              # trailing piece with no junction end
                seq.extend(run)
                ptr.append(len(seq))
                oneway_flags.append(ow)

        ptr = np.asarray(ptr, dtype=np.int64)
        seq = np.asarray(seq, dtype=np.int64)
        if ptr.size < 2:
            raise GraphBuildError("no edges survived junction splitting")

        # Cumulative length along every polyline, in one vectorised pass.
        step = geodesic_m(nlat[seq[:-1]], nlon[seq[:-1]],
                          nlat[seq[1:]], nlon[seq[1:]])
        run = np.concatenate([[0.0], np.cumsum(step)])
        # `run` accumulates across edge boundaries too; subtracting each edge's
        # own start both zeroes its first point and cancels the meaningless
        # step that spans the boundary.
        cum = run - np.repeat(run[ptr[:-1]], np.diff(ptr))

        n_e = ptr.size - 1
        passable = np.ones(2 * n_e, dtype=bool)
        passable[1::2] = ~np.asarray(oneway_flags, dtype=bool)
        return cls(nlat, nlon, ptr, seq, cum, passable, nid)

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, lat=self.node_lat, lon=self.node_lon, ptr=self._ptr,
            seq=self._seq, cum=self._cum, passable=self.passable,
            node_id=self.node_id)
        print(f"cached graph -> {path}", file=sys.stderr)

    @classmethod
    def load(cls, path: Path) -> "RoadGraph":
        z = np.load(path)
        return cls(z["lat"], z["lon"], z["ptr"], z["seq"], z["cum"],
                   z["passable"], z["node_id"])

    @classmethod
    def open(cls, pbf: Path, cache: Path) -> "RoadGraph":
        """Load the cache, building it from `pbf` the first time."""
        cache = Path(cache)
        if cache.exists():
            return cls.load(cache)
        return cls.build(Path(pbf), cache)

    # -- topology ----------------------------------------------------------

    def _build_topology(self):
        ptr, seq = self._ptr, self._seq
        n_e = self.n_undirected
        self.edge_len = self._cum[ptr[1:] - 1]

        # Directed endpoints: 2e runs first->last, 2e+1 last->first.
        first, last = seq[ptr[:-1]], seq[ptr[1:] - 1]
        self.d_from = np.empty(2 * n_e, dtype=np.int64)
        self.d_to = np.empty(2 * n_e, dtype=np.int64)
        self.d_from[0::2], self.d_to[0::2] = first, last
        self.d_from[1::2], self.d_to[1::2] = last, first

        # Outgoing directed edges per node, CSR-style.
        order = np.argsort(self.d_from, kind="stable")
        self._out_edges = order
        counts = np.bincount(self.d_from, minlength=self.node_lat.size)
        self._out_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        self._seg_cells = None
        self._pair_key = None
        self._nid_order = None

    def out_edges(self, node: int) -> np.ndarray:
        """Passable directed edges leaving `node`."""
        lo, hi = self._out_ptr[node], self._out_ptr[node + 1]
        cand = self._out_edges[lo:hi]
        return cand[self.passable[cand]]

    # -- geometry ----------------------------------------------------------

    def shape(self, d_edge: int) -> np.ndarray:
        """(k, 2) lat/lon along a directed edge, in travel order."""
        e, rev = d_edge >> 1, d_edge & 1
        s = self._seq[self._ptr[e]:self._ptr[e + 1]]
        if rev:
            s = s[::-1]
        return np.column_stack([self.node_lat[s], self.node_lon[s]])

    def _cum_along(self, d_edge: int) -> np.ndarray:
        e, rev = d_edge >> 1, d_edge & 1
        c = self._cum[self._ptr[e]:self._ptr[e + 1]]
        return (c[-1] - c[::-1]) if rev else c

    def length(self, d_edge: int) -> float:
        return float(self.edge_len[d_edge >> 1])

    def position(self, pos: EdgePosition) -> tuple[float, float]:
        """(lat, lon) at an along-edge offset, interpolated on the polyline."""
        pts = self.shape(pos.edge)
        cum = self._cum_along(pos.edge)
        d = float(np.clip(pos.offset_m, 0.0, cum[-1]))
        i = int(np.searchsorted(cum, d, side="right")) - 1
        i = max(0, min(i, len(cum) - 2))
        span = cum[i + 1] - cum[i]
        f = 0.0 if span <= 0 else (d - cum[i]) / span
        return (float(pts[i, 0] + f * (pts[i + 1, 0] - pts[i, 0])),
                float(pts[i, 1] + f * (pts[i + 1, 1] - pts[i, 1])))

    def bearing_at(self, pos: EdgePosition) -> float:
        """Direction of travel at an along-edge offset."""
        pts = self.shape(pos.edge)
        cum = self._cum_along(pos.edge)
        d = float(np.clip(pos.offset_m, 0.0, cum[-1]))
        i = int(np.searchsorted(cum, d, side="right")) - 1
        i = max(0, min(i, len(cum) - 2))
        return float(bearing_rad(pts[i, 0], pts[i, 1], pts[i + 1, 0], pts[i + 1, 1]))

    def entry_bearing(self, d_edge: int) -> float:
        """Direction of travel arriving at the END of a directed edge."""
        pts = self.shape(d_edge)
        return float(bearing_rad(pts[-2, 0], pts[-2, 1], pts[-1, 0], pts[-1, 1]))

    def exit_bearing(self, d_edge: int) -> float:
        """Direction of travel leaving the START of a directed edge."""
        pts = self.shape(d_edge)
        return float(bearing_rad(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1]))

    # -- spatial lookup ----------------------------------------------------

    def _ensure_index(self, cell_deg: float = 0.01):
        """Bucket every polyline SEGMENT into a lat/lon grid.

        Segments, not vertices. Indexing by vertex looks equivalent and is
        not: a motorway can run hundreds of metres between shape points, so a
        query beside the middle of such a segment is far from both of its
        endpoints and a vertex index returns nothing at all -- the road is
        invisible precisely where it is longest and straightest.

        A segment is filed under the cells of both its endpoints. Cells are
        ~1.1 x 0.7 km, far longer than almost every segment, and the query
        widens its search when a cell yields nothing, so a segment longer than
        a cell is still found.
        """
        if getattr(self, "_seg_cells", None) is not None:
            return
        self._cell = cell_deg
        ptr, seq = self._ptr, self._seq

        # Segment j spans shape points (j, j+1); the last point of each edge
        # starts no segment.
        keep = np.ones(seq.size, dtype=bool)
        keep[ptr[1:] - 1] = False
        js = np.flatnonzero(keep)
        self._seg_pt = js                                   # first point index
        self._seg_edge = np.repeat(np.arange(self.n_undirected),
                                   np.diff(ptr) - 1).astype(np.int64)
        a, b = seq[js], seq[js + 1]
        self._seg_a, self._seg_b = a, b

        def key(node_idx):
            gx = np.floor(self.node_lon[node_idx] / cell_deg).astype(np.int64)
            gy = np.floor(self.node_lat[node_idx] / cell_deg).astype(np.int64)
            return gy * 1000003 + gx

        ids = np.arange(js.size, dtype=np.int64)
        keys = np.concatenate([key(a), key(b)])
        vals = np.concatenate([ids, ids])
        order = np.argsort(keys, kind="stable")
        self._seg_cells = keys[order]
        self._seg_ids = vals[order]
        self._cell_start = {}
        uniq, first = np.unique(self._seg_cells, return_index=True)
        stop = np.append(first[1:], self._seg_cells.size)
        self._cell_start = {int(k): (int(i), int(j))
                            for k, i, j in zip(uniq, first, stop)}

    def _segments_near(self, lat: float, lon: float, span: int) -> np.ndarray:
        cell = self._cell
        gx0 = int(np.floor(lon / cell))
        gy0 = int(np.floor(lat / cell))
        out = []
        for dy in range(-span, span + 1):
            for dx in range(-span, span + 1):
                got = self._cell_start.get((gy0 + dy) * 1000003 + (gx0 + dx))
                if got:
                    out.append(self._seg_ids[got[0]:got[1]])
        if not out:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(out))

    def _segment_distances(self, segs: np.ndarray, lat: float, lon: float):
        """Perpendicular distance to each segment, and the fraction along it.

        Local equirectangular metres about the query point. Over the tens of
        metres this is used for, the distortion is orders of magnitude below
        the GPS accuracy being snapped (9-10 m median on these sessions).
        """
        kx = 111320.0 * np.cos(np.radians(lat))
        ky = 110540.0
        a, b = self._seg_a[segs], self._seg_b[segs]
        ax = (self.node_lon[a] - lon) * kx
        ay = (self.node_lat[a] - lat) * ky
        bx = (self.node_lon[b] - lon) * kx
        by = (self.node_lat[b] - lat) * ky
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(L2 > 0, -(ax * vx + ay * vy) / L2, 0.0)
        t = np.clip(t, 0.0, 1.0)
        d = np.hypot(ax + t * vx, ay + t * vy)
        return d, t

    def nearest_edges(self, lat: float, lon: float, radius_m: float = 60.0,
                      limit: int = 12):
        """Undirected edges with a point within `radius_m`, nearest first.

        Returns [(edge, distance_m, offset_m)], the offset being how far along
        that edge's FORWARD direction the closest point lies.
        """
        self._ensure_index()
        span = max(1, int(np.ceil(radius_m / (self._cell * 111320.0 * 0.6))))
        for attempt in range(4):          # widen rather than give up
            segs = self._segments_near(lat, lon, span * (attempt + 1))
            if segs.size:
                d, t = self._segment_distances(segs, lat, lon)
                hit = d <= radius_m
                if hit.any():
                    segs, d, t = segs[hit], d[hit], t[hit]
                    order = np.argsort(d)
                    out, seen = [], set()
                    for i in order:
                        e = int(self._seg_edge[segs[i]])
                        if e in seen:
                            continue
                        seen.add(e)
                        j = int(self._seg_pt[segs[i]])
                        base = self._cum[j] - self._cum[self._ptr[e]]
                        seg_len = self._cum[j + 1] - self._cum[j]
                        out.append((e, float(d[i]),
                                    float(base + t[i] * seg_len)))
                        if len(out) >= limit:
                            break
                    return out
        return []

    def locate(self, lat: float, lon: float, heading: float | None = None,
               radius_m: float = 60.0,
               heading_tol: float = np.pi / 2) -> EdgePosition | None:
        """Snap a fix to a directed edge, honouring the direction of travel.

        `heading` (radians clockwise from north) selects BETWEEN the two
        directions of the same road, which is the difference between tracking
        forwards and tracking backwards down it. Without it the choice would
        be arbitrary.
        """
        best, best_score = None, np.inf
        for e, dist, offset in self.nearest_edges(lat, lon, radius_m):
            total = float(self.edge_len[e])
            for d_edge in (2 * e, 2 * e + 1):
                if not self.passable[d_edge]:
                    continue
                off = offset if (d_edge & 1) == 0 else total - offset
                pos = EdgePosition(int(d_edge), float(np.clip(off, 0.0, total)))
                score = dist
                if heading is not None:
                    delta = abs(float(wrap_pi(self.bearing_at(pos) - heading)))
                    if delta > heading_tol:
                        continue
                    score += 20.0 * (delta / np.pi)   # gentle tie-break, metres
                if score < best_score:
                    best_score, best = score, pos
        return best

    def _edge_segments(self, edge: int) -> np.ndarray:
        """Segment indices of one edge.

        Contiguous by construction: segments are generated edge by edge and
        each edge of k points contributes k-1 of them, so edge `e` starts at
        `ptr[e] - e`. Scanning `_seg_edge == edge` instead is a full pass over
        every segment in the country -- correct, and roughly 13 million
        comparisons per snap.
        """
        lo = int(self._ptr[edge]) - edge
        hi = int(self._ptr[edge + 1]) - (edge + 1)
        return np.arange(lo, hi, dtype=np.int64)

    def project_on_edge(self, edge: int, lat: float, lon: float):
        """Closest point on ONE undirected edge. Returns (distance, offset)."""
        self._ensure_index()
        segs = self._edge_segments(edge)
        if segs.size == 0:
            return None
        d, t = self._segment_distances(segs, lat, lon)
        i = int(np.argmin(d))
        j = int(self._seg_pt[segs[i]])
        base = self._cum[j] - self._cum[self._ptr[edge]]
        seg_len = self._cum[j + 1] - self._cum[j]
        return float(d[i]), float(base + t[i] * seg_len)

    def _ensure_pair_index(self):
        """Sorted (node, node) -> segment table for OSRM node-pair anchoring.

        Keyed on INTERNAL node indices packed into one int64, not on the OSM
        ids: OSM ids run past 2^33 and will not pack in pairs, while the
        internal indices are under 2^23 here. The alternative -- a Python dict
        of every ordered node pair -- is 26 million entries and gigabytes of
        object overhead for a lookup that a sorted array answers in a
        binary search.
        """
        if self._pair_key is not None:
            return
        self._ensure_index()
        if self.node_lat.size >= (1 << 24):
            raise GraphBuildError(
                f"{self.node_lat.size} nodes exceeds the 2^24 the node-pair "
                "key packs two indices into; widen the shift or key on a "
                "structured array before extending the extract")
        a, b = self._seg_a, self._seg_b
        lo = np.minimum(a, b).astype(np.int64)
        hi = np.maximum(a, b).astype(np.int64)
        key = (lo << np.int64(24)) | hi
        order = np.argsort(key, kind="stable")
        self._pair_key = key[order]
        self._pair_seg = order.astype(np.int64)
        self._nid_order = np.argsort(self.node_id, kind="stable")
        self._nid_sorted = self.node_id[self._nid_order]

    def _node_index(self, osm_id: int) -> int | None:
        i = int(np.searchsorted(self._nid_sorted, osm_id))
        if i >= self._nid_sorted.size or int(self._nid_sorted[i]) != osm_id:
            return None
        return int(self._nid_order[i])

    def _edge_for_node_pair(self, n1: int, n2: int) -> int | None:
        """The undirected edge carrying the segment between two OSM nodes."""
        self._ensure_pair_index()
        u, v = self._node_index(n1), self._node_index(n2)
        if u is None or v is None:
            return None
        key = (min(u, v) << 24) | max(u, v)
        i = int(np.searchsorted(self._pair_key, key))
        if i >= self._pair_key.size or int(self._pair_key[i]) != key:
            return None
        return int(self._seg_edge[self._pair_seg[i]])

    def locate_node_pair(self, n1: int, n2: int, lat: float, lon: float,
                         heading: float | None = None) -> EdgePosition | None:
        """Locate onto the edge carrying OSM segment (n1, n2), if known.

        This is how an OSRM `/nearest` result is anchored: the response names
        the two OSM nodes of the segment it snapped to, which pins the match
        to an exact edge rather than re-deriving it from the coordinates and
        possibly landing on a different road that happens to be as close.
        """
        e = self._edge_for_node_pair(int(n1), int(n2))
        if e is None:
            return None
        got = self.project_on_edge(e, lat, lon)
        if got is None:
            return None
        dist, offset = got
        total = float(self.edge_len[e])
        best, best_score = None, np.inf
        for d_edge in (2 * e, 2 * e + 1):
            if not self.passable[d_edge]:
                continue
            off = offset if (d_edge & 1) == 0 else total - offset
            pos = EdgePosition(int(d_edge), float(np.clip(off, 0.0, total)))
            score = dist
            if heading is not None:
                score += 40.0 * abs(float(wrap_pi(
                    self.bearing_at(pos) - heading))) / np.pi
            if score < best_score:
                best_score, best = score, pos
        return best
