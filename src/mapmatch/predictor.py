"""Along-road tracking through a GNSS outage.

The idea, and why it is not dead reckoning. Classical DR integrates heading,
so a heading error at second 3 bends the whole remaining track and the
position error grows roughly with the square of outage duration. Here the
vehicle's state is a SCALAR: how far along a road polyline it is. The road
supplies the shape. Heading is never integrated -- it is read off the map at
whatever point the scalar has reached, so a heading error cannot accumulate,
only a distance error can, and distance error grows linearly.

The cost is that the vehicle must be on a road we know about and we must
choose correctly at junctions. Both are handled explicitly below.

Per the brief (Gomes & Costa's rules, adapted to this dataset):

* GPS position and heading are read at t0 only, and the outage is SKIPPED if
  the fix at t0 is poor -- a bad initial fix puts the vehicle on the wrong
  road immediately, which is where their error growth came from.
* The start position is matched to a road segment through OSRM.
* Each second the model's predicted displacement advances the scalar along
  the polyline.
* At junctions, accumulated gyro yaw over the preceding few seconds picks the
  branch whose turn angle is closest. Gyro bias is fatal over a minute but
  negligible over the ~2 s of one turn, which is the only span it is used for.
* A re-match runs every 10 s, not every step.
* If more than one match is plausible, the estimate is left unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import Session, _find
from eval.harness import OutageSkipped

from .graph import (EdgePosition, RoadGraph, bearing_rad, geodesic_m,
                    wrap_pi)
from .osrm import OSRMClient

# Levelled gyro channel 5 is rotation about the vertical. Its sign relative to
# a compass heading depends on which way the phone faces in its mount, and the
# levelling in `data.windows.levelled_channels` fixes roll and pitch but not
# that. Measured against 1 s GPS heading changes on the four test sessions,
# the slope is -0.94, -0.91, -0.72, -0.91 (correlation -0.84, -0.86, -0.55,
# -0.80): consistently negative, magnitude near 1, so levelling is right and
# only the sign needs flipping. Compared at 1 s, NOT at 10 Hz -- at 10 Hz the
# held 1 Hz GPS heading differences into a spike train and the correlation
# collapses to -0.02..-0.27, which is the differencing artefact CLAUDE.md
# warns about, not a real disagreement.
CORPUS_GYRO_SIGN = -1.0

# Minimum pre-outage history needed to calibrate the sign from this session's
# own GPS rather than falling back to the corpus value.
MIN_CALIBRATION_S = 120.0


def _unwrap_filled(angles: np.ndarray) -> np.ndarray:
    """Unwrap an angle series, interpolating across missing samples."""
    a = np.asarray(angles, dtype=float)
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros_like(a)
    idx = np.arange(a.size)
    filled = np.interp(idx, idx[ok], a[ok])
    return np.unwrap(filled)


@dataclass
class MapMatchConfig:
    """Every tunable, with the brief's values as defaults."""

    # A bad initial fix is the documented failure mode, so this is the guard.
    # Test-session GPS accuracy is 9-10 m median, 12-13 m at p90, so 20 m
    # admits the ordinary fixes and rejects the excursions (S-A6 reaches
    # 3911 m, S-A8 122 m).
    max_gps_accuracy_m: float = 20.0

    # Re-match cadence. Gomes & Costa swept this and 10 s was optimal;
    # continuous snapping fights the model's own predictions.
    rematch_every_s: float = 10.0

    # Accumulated-gyro window for the junction turn decision.
    turn_window_s: float = 4.0
    # Where that window sits relative to the junction crossing. 0 is the
    # brief's "preceding"; the vehicle physically turns while traversing the
    # junction, so a lead is available (the IMU keeps running) and is swept as
    # an ablation rather than assumed.
    turn_lead_s: float = 0.0
    # A branch further than this from the observed turn is not a candidate;
    # if that leaves nothing, the straightest branch is taken.
    max_turn_mismatch_rad: float = np.deg2rad(75.0)

    # Matching geometry.
    snap_radius_m: float = 40.0
    osrm_bearing_range_deg: int = 60
    n_candidates: int = 3
    # Two candidates count as plausible-and-competing when the second is
    # within this factor of the first's distance AND names a different road.
    ambiguity_ratio: float = 1.5
    ambiguity_margin_m: float = 15.0


@dataclass
class OutageTrace:
    """What the tracker did, for diagnosis. Never read by the harness."""
    junctions: int = 0
    junction_forced: int = 0          # only one branch, gyro not consulted
    rematches: int = 0
    rematch_applied: int = 0
    rematch_ambiguous: int = 0
    off_network: bool = False
    start_ambiguous: bool = False
    gyro_sign: float = float("nan")
    turns: list = field(default_factory=list)


class MapMatchedPredictor:
    """Harness predictor: a displacement model tracked along the road network.

    `inner` supplies per-second displacement and is used for NOTHING else --
    its headings are discarded, because the map supplies heading. Any harness
    predictor works: the model, or constant-velocity DR to isolate the map's
    contribution from the model's.
    """

    name = "map_matched"

    def __init__(self, inner, session: Session, graph: RoadGraph,
                 osrm: OSRMClient, cfg: MapMatchConfig | None = None):
        from data.windows import GRID_HZ, levelled_channels
        from data.sanity import gps_cumulative_distance

        self.inner = inner
        self.session = session
        self.graph = graph
        self.osrm = osrm
        self.cfg = cfg or MapMatchConfig()
        self.hz = GRID_HZ

        df = session.df
        self.t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)
        lat_c, lon_c = _find(df, r"LATITUDE"), _find(df, r"LONGITUDE")
        self.lat = pd.to_numeric(df[lat_c], errors="coerce").to_numpy(float)
        self.lon = pd.to_numeric(df[lon_c], errors="coerce").to_numpy(float)
        acc_c = _find(df, r"GPS ACCURACY")
        self.gps_acc = (pd.to_numeric(df[acc_c], errors="coerce").to_numpy(float)
                        if acc_c is not None
                        else np.full(self.t.shape, np.nan))
        head_c = _find(df, r"^HEADING") or _find(df, r"GPS ORIENTATION")
        self.heading = (np.radians(pd.to_numeric(df[head_c], errors="coerce")
                                   .to_numpy(float))
                        if head_c is not None else np.zeros_like(self.t))
        # Unwrapped once, and gaps filled by interpolation rather than by
        # nan_to_num: replacing a missing heading with zero injects a jump of
        # up to 2*pi that `np.unwrap` then propagates through the whole rest
        # of the session.
        self.heading_unwrapped = _unwrap_filled(self.heading)

        # Levelled IMU on a uniform grid; channel 5 is yaw rate about vertical.
        gps = gps_cumulative_distance(session)
        lo, hi = float(gps[0].min()), float(gps[0].max())
        self.grid = np.arange(lo, hi, 1.0 / GRID_HZ)
        chan = levelled_channels(session, self.grid)
        self.yaw_rate = (chan[:, 5] if chan is not None
                         else np.zeros_like(self.grid))

        self._primed: dict[tuple[float, float], np.ndarray] = {}
        self._sign_cache: dict[float, float] = {}
        # Running totals over every outage this predictor has served. Purely
        # diagnostic; the harness never reads them.
        self.stats = {"outages": 0, "skipped_gps": 0, "skipped_nomatch": 0,
                      "junctions": 0, "junction_forced": 0, "rematches": 0,
                      "rematch_applied": 0, "rematch_ambiguous": 0,
                      "start_ambiguous": 0, "off_network": 0}

    # -- inner displacements ----------------------------------------------

    def prime(self, t0s, duration: float) -> None:
        """Pre-compute the inner model's displacements for many outages.

        Purely a speed optimisation: `ModelPredictor.predict_many` batches all
        of an outage's forward passes, and batching across outages too is
        exact because the network reads only raw sensor windows -- no
        predicted value is ever fed back into its input.
        """
        t0s = [float(t) for t in t0s]
        if hasattr(self.inner, "predict_many"):
            outs = self.inner.predict_many(t0s, duration)
        else:
            outs = [self.inner.predict(t, duration) for t in t0s]
        for t0, o in zip(t0s, outs):
            self._primed[(t0, float(duration))] = np.asarray(
                o["displacements"], dtype=float)

    def _displacements(self, t0: float, duration: float) -> np.ndarray:
        key = (float(t0), float(duration))
        if key in self._primed:
            return self._primed[key]
        out = self.inner.predict(t0, duration)
        return np.asarray(out["displacements"], dtype=float)

    # -- gyro --------------------------------------------------------------

    def _gyro_sign(self, t0: float) -> float:
        """Mount-orientation sign, from GPS BEFORE t0 only.

        Deployable: the calibration a vehicle would have performed while it
        still had a fix. Leak-free: nothing at or after t0 is read. Falls back
        to the corpus value when there is too little history.
        """
        key = round(t0, 3)
        if key in self._sign_cache:
            return self._sign_cache[key]
        sign = CORPUS_GYRO_SIGN
        t_start = float(self.t[np.isfinite(self.t)].min())
        if t0 - t_start >= MIN_CALIBRATION_S:
            g0, g1 = self.grid[0], t0
            sel = (self.grid >= g0) & (self.grid < g1)
            n = int(sel.sum()) // int(self.hz) * int(self.hz)
            if n >= int(self.hz) * 60:
                idx = np.flatnonzero(sel)[:n]
                # 1 s blocks: integrated gyro against the GPS heading change
                # over the same second. Compared at 1 s deliberately -- 10 Hz
                # differencing of held GPS heading is a spike train.
                gy = self.yaw_rate[idx].reshape(-1, int(self.hz)).sum(1) / self.hz
                marks = self.grid[idx].reshape(-1, int(self.hz))[:, 0]
                h = np.interp(marks, self.t, self.heading_unwrapped)
                dh = wrap_pi(np.diff(h))
                spd = np.interp(marks[:-1], self.t,
                                np.nan_to_num(self._speed()))
                m = np.isfinite(dh) & np.isfinite(gy[:-1]) & (spd > 8.0)
                if m.sum() >= 60 and np.std(gy[:-1][m]) > 1e-6:
                    slope = float(np.polyfit(gy[:-1][m], dh[m], 1)[0])
                    if abs(slope) > 0.3:
                        sign = float(np.sign(slope))
        self._sign_cache[key] = sign
        return sign

    def _speed(self) -> np.ndarray:
        if "speed_ms" in self.session.df.columns:
            return pd.to_numeric(self.session.df["speed_ms"],
                                 errors="coerce").to_numpy(float)
        return np.full(self.t.shape, np.nan)

    def _turn_angle(self, t_junction: float, sign: float) -> float:
        """Accumulated yaw across the window around a junction crossing.

        Radians, positive clockwise (a right turn in the UK), matching the
        map bearings this is compared against.
        """
        w = self.cfg.turn_window_s
        lo = t_junction - w + self.cfg.turn_lead_s
        hi = t_junction + self.cfg.turn_lead_s
        i0, i1 = np.searchsorted(self.grid, [lo, hi])
        if i1 <= i0:
            return 0.0
        seg = self.yaw_rate[i0:i1]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            return 0.0
        return float(sign * seg.sum() / self.hz)

    # -- matching ----------------------------------------------------------

    def _match(self, lat: float, lon: float, heading: float | None):
        """OSRM `/nearest`, anchored onto our own graph.

        Returns (EdgePosition or None, ambiguous). `ambiguous` is true when a
        second candidate is a genuinely competing road -- comparably close and
        a DIFFERENT edge of the network. That is the cheap defence against
        wrong-street failure: when it fires, the caller leaves the estimate
        alone rather than gambling on which street it is.
        """
        cfg = self.cfg
        deg = None if heading is None else float(np.degrees(heading)) % 360.0
        wps = self.osrm.nearest(lat, lon, number=cfg.n_candidates, bearing=deg,
                                bearing_range=cfg.osrm_bearing_range_deg,
                                radius_m=cfg.snap_radius_m)
        if not wps:
            # Retry unrestricted: a bearing filter that admits nothing is a
            # failure of the filter, not evidence there is no road.
            wps = self.osrm.nearest(lat, lon, number=cfg.n_candidates,
                                    radius_m=cfg.snap_radius_m)
        if not wps:
            return None, False

        # Resolve every candidate onto our own graph before comparing them.
        # Comparing by street NAME alone is not enough: a large share of the
        # segments OSRM returns here are unnamed (service roads, slip roads,
        # unclassified lanes), and treating two unnamed roads as "the same
        # road" would suppress exactly the ambiguity this exists to catch.
        resolved = []
        for w in wps:
            loc = w.get("location") or [lon, lat]
            wl, wn = float(loc[1]), float(loc[0])
            nodes = w.get("nodes") or []
            pos = None
            if len(nodes) >= 2 and nodes[0] and nodes[1]:
                pos = self.graph.locate_node_pair(nodes[0], nodes[1], wl, wn,
                                                  heading)
            if pos is None:
                pos = self.graph.locate(wl, wn, heading,
                                        radius_m=cfg.snap_radius_m)
            resolved.append((w, pos))

        best, best_pos = resolved[0]
        d0 = float(best.get("distance", 0.0))
        ambiguous = False
        for other, pos in resolved[1:]:
            d1 = float(other.get("distance", np.inf))
            if d1 > max(d0 * cfg.ambiguity_ratio, d0 + cfg.ambiguity_margin_m):
                continue                      # not plausible; not a competitor
            same_edge = (best_pos is not None and pos is not None
                         and (best_pos.edge >> 1) == (pos.edge >> 1))
            if not same_edge:
                ambiguous = True
                break

        # OSRM's car profile and our tag filter can disagree at the margins,
        # in which case `locate` above already fell back to our own snap.
        return best_pos, ambiguous

    # -- junctions ---------------------------------------------------------

    def _choose_branch(self, arriving: int, node: int, observed_turn: float,
                       trace: OutageTrace) -> int | None:
        """Pick the outgoing edge whose turn angle best matches the gyro."""
        cand = [int(c) for c in self.graph.out_edges(node)]
        # The reverse of the edge we arrived on is a U-turn; drop it unless it
        # is all there is (a dead end, where turning round is what happens).
        forward = [c for c in cand if (c >> 1) != (arriving >> 1)]
        if not forward:
            if not cand:
                trace.off_network = True
                return None
            trace.junction_forced += 1
            return cand[0]
        if len(forward) == 1:
            trace.junction_forced += 1
            return forward[0]

        entry = self.graph.entry_bearing(arriving)
        angles = np.array([float(wrap_pi(self.graph.exit_bearing(c) - entry))
                           for c in forward])
        err = np.abs(wrap_pi(angles - observed_turn))
        k = int(np.argmin(err))
        if err[k] > self.cfg.max_turn_mismatch_rad:
            # No branch resembles the observed turn: the gyro is telling us
            # something the map does not offer. Go straightest rather than
            # commit to a turn no evidence supports.
            k = int(np.argmin(np.abs(angles)))
        trace.turns.append((float(observed_turn), float(angles[k])))
        return forward[k]

    def _advance(self, pos: EdgePosition, dist: float, t_start: float,
                 sign: float, trace: OutageTrace) -> EdgePosition:
        """Move `dist` metres along the network over the second from `t_start`.

        The junction crossing time is interpolated from how much of the
        second's travel had been used on arrival, rather than charged to the
        end of the second. With a 4 s gyro window a whole second of timing
        error is a quarter of the window, which is enough to blur a turn into
        the straight running either side of it.
        """
        g = self.graph
        remaining = float(dist)
        guard = 0
        while remaining > 0:
            guard += 1
            if guard > 200:               # pathological micro-edge loop
                break
            edge_len = g.length(pos.edge)
            room = edge_len - pos.offset_m
            if remaining < room:
                return EdgePosition(pos.edge, pos.offset_m + remaining)
            # Reached the junction at the far end of this edge.
            remaining -= max(room, 0.0)
            node = int(g.d_to[pos.edge])
            trace.junctions += 1
            frac = 1.0 - remaining / dist if dist > 0 else 1.0
            turn = self._turn_angle(t_start + frac, sign)
            nxt = self._choose_branch(pos.edge, node, turn, trace)
            if nxt is None:
                trace.off_network = True
                return EdgePosition(pos.edge, edge_len)   # dead end: stop here
            pos = EdgePosition(nxt, 0.0)
        return pos

    # -- harness interface -------------------------------------------------

    def predict(self, t0: float, duration: float) -> dict:
        cfg = self.cfg
        n = int(round(duration))

        acc = float(np.interp(t0, self.t, np.nan_to_num(self.gps_acc,
                                                        nan=np.inf)))
        if not np.isfinite(acc) or acc > cfg.max_gps_accuracy_m:
            self.stats["skipped_gps"] += 1
            raise OutageSkipped(
                f"GPS accuracy {acc:.1f} m at t0 exceeds "
                f"{cfg.max_gps_accuracy_m:.0f} m — a bad initial fix matches "
                "to the wrong road, and along-road tracking never recovers")

        lat0 = float(np.interp(t0, self.t, self.lat))
        lon0 = float(np.interp(t0, self.t, self.lon))
        h0 = float(np.interp(t0, self.t, self.heading_unwrapped))

        trace = OutageTrace()
        pos, trace.start_ambiguous = self._match(lat0, lon0, h0)
        if pos is None:
            self.stats["skipped_nomatch"] += 1
            raise OutageSkipped(
                f"no road within {cfg.snap_radius_m:.0f} m of the start fix "
                f"({lat0:.5f}, {lon0:.5f}) — outside the offline extract, or "
                "off the drivable network")

        sign = self._gyro_sign(t0)
        trace.gyro_sign = sign
        disp = self._displacements(t0, duration)
        if disp.size < n:
            raise OutageSkipped("inner predictor returned too few seconds")

        # The track, as latitude/longitude, one point per second mark. The
        # start point is the MATCHED position, not the raw fix: the vehicle is
        # on the road, and the road is what we then follow.
        #
        # The snap therefore translates the whole track by the few metres
        # between the fix and the carriageway. That translation is neither
        # charged nor credited, because the harness scores every predictor on
        # displacement RELATIVE to its own start. What it does not forgive is
        # snapping to the wrong road: that changes the SHAPE of the track, and
        # shape is exactly what the harness measures.
        track = np.empty((n + 1, 2), dtype=float)
        track[0] = self.graph.position(pos)
        next_rematch = cfg.rematch_every_s

        for k in range(n):
            pos = self._advance(pos, float(disp[k]), t0 + k, sign, trace)
            if (k + 1) >= next_rematch:
                next_rematch += cfg.rematch_every_s
                trace.rematches += 1
                lat_p, lon_p = self.graph.position(pos)
                bear = self.graph.bearing_at(pos)
                cand, ambiguous = self._match(lat_p, lon_p, bear)
                if ambiguous:
                    # Gomes & Costa's rule: more than one plausible match, so
                    # leave the estimate unmodified rather than gamble on a
                    # street. Committing to the wrong one is unrecoverable;
                    # declining costs only the correction we skipped.
                    trace.rematch_ambiguous += 1
                elif cand is not None:
                    pos = cand
                    trace.rematch_applied += 1
            track[k + 1] = self.graph.position(pos)

        self.stats["outages"] += 1
        for k in ("junctions", "junction_forced", "rematches",
                  "rematch_applied", "rematch_ambiguous"):
            self.stats[k] += getattr(trace, k)
        self.stats["start_ambiguous"] += int(trace.start_ambiguous)
        self.stats["off_network"] += int(trace.off_network)
        return self._to_harness(track, trace)

    @staticmethod
    def _to_harness(track: np.ndarray, trace: OutageTrace) -> dict:
        """Per-second chords of the tracked path, in the harness's convention.

        The harness accumulates `d_k * [cos psi_k, sin psi_k]`. Emitting each
        second's chord LENGTH and chord BEARING therefore reproduces the
        along-road track exactly. Nothing here integrates a heading: psi is
        read off the geometry the vehicle covered, so a heading error cannot
        propagate into the next second.
        """
        la, lo = track[:, 0], track[:, 1]
        d = geodesic_m(la[:-1], lo[:-1], la[1:], lo[1:])
        psi = bearing_rad(la[:-1], lo[:-1], la[1:], lo[1:])
        # A zero-length second has no defined bearing; hold the previous one so
        # the value is never NaN. It multiplies zero either way.
        psi = np.asarray(psi, dtype=float)
        for i in range(psi.size):
            if d[i] <= 0:
                psi[i] = psi[i - 1] if i else 0.0
        return {"displacements": np.asarray(d, dtype=float),
                "headings": psi, "trace": trace, "track": track}
