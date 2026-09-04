"""Beam-search (bounded Viterbi) along-road tracking.

`mapmatch.predictor.MapMatchedPredictor` decides every junction and every
re-match GREEDILY: `_choose_branch` commits to the single best-matching
branch immediately, and an ambiguous re-match is simply left alone. That is
irrevocable -- a wrong commit at second 8 cannot be undone by evidence that
turns up at second 40, even though the problem statement itself asks for
"Advanced Map-Matching ... (UKF + HMM)".

This keeps every plausible path as a separate weighted hypothesis instead of
collapsing to one at the moment of the fork -- the classic HMM/Viterbi
map-matching idea (Newson & Krumm 2009), adapted to this project's specific
observation model: there is no independent GPS point to score candidates
against during a GNSS outage (that is the whole problem), so the only
"emission" evidence available is how well each branch's turn angle matches
the accumulated gyro at the junction it forks from. A hypothesis that turns
out to have guessed wrong at one junction can still lose out to a rival at a
LATER junction, which greedy matching cannot express at all.

Kept as a separate predictor (`HMMMapMatchedPredictor`) rather than a mode
flag on `MapMatchedPredictor`, so the existing, well-tested greedy tracker is
untouched and the two can be scored side by side (see
`eval.mapmatch_eval.build_predictors`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from eval.harness import OutageSkipped

from .graph import EdgePosition, wrap_pi
from .predictor import MapMatchConfig, MapMatchedPredictor, OutageTrace


@dataclass
class HMMMapMatchConfig(MapMatchConfig):
    """`MapMatchConfig` plus the beam's own tunables."""

    # Hypotheses kept after every prune. Bounds the cost: at most this many
    # survive a junction fork, so a run of junctions costs O(beam_width),
    # not O(branches ** junctions).
    beam_width: int = 5

    # Spread of the turn-angle log-likelihood at a fork, in radians. A branch
    # whose turn matches the observed gyro within this margin is treated as
    # plausible; the ORIGINAL `max_turn_mismatch_rad` (75 deg) is still used
    # as a hard cutoff below, this just shapes the soft weighting between the
    # branches that pass it.
    turn_sigma_rad: float = np.deg2rad(20.0)

    # How many of the top snap candidates seed separate starting hypotheses,
    # instead of the greedy tracker's single best-plus-ambiguity-flag.
    n_start_hypotheses: int = 3


def _copy_trace(tr: OutageTrace) -> OutageTrace:
    """Independent copy of a trace, including its mutable `turns` list."""
    new = replace(tr)
    new.turns = list(tr.turns)
    return new


@dataclass
class _Hyp:
    pos: EdgePosition
    log_prob: float
    track: list = field(default_factory=list)     # (lat, lon) per second mark
    trace: OutageTrace = field(default_factory=OutageTrace)


class HMMMapMatchedPredictor(MapMatchedPredictor):
    """Along-road tracking with a bounded-width Viterbi beam over junctions.

    Reuses every geometric primitive from `MapMatchedPredictor` unchanged
    (`_turn_angle`, `_gyro_sign`, `_match`, `_displacements`, `_to_harness`)
    and overrides only how a junction fork and the outage's start are
    resolved: many weighted hypotheses instead of one.
    """

    name = "hmm_matched"

    # -- soft junction / start matching -------------------------------------

    def _choose_branches_soft(self, arriving: int, node: int,
                              observed_turn: float, trace: OutageTrace):
        """Every plausible outgoing branch, log-weighted by turn-angle match.

        Mirrors `MapMatchedPredictor._choose_branch` exactly for the forced
        and single-branch cases (nothing to be uncertain about there); only
        the multi-branch case differs, returning ALL candidates instead of
        `argmin` alone.
        """
        cand = [int(c) for c in self.graph.out_edges(node)]
        forward = [c for c in cand if (c >> 1) != (arriving >> 1)]
        if not forward:
            if not cand:
                trace.off_network = True
                return []
            trace.junction_forced += 1
            return [(cand[0], 0.0)]
        if len(forward) == 1:
            trace.junction_forced += 1
            return [(forward[0], 0.0)]

        entry = self.graph.entry_bearing(arriving)
        angles = np.array([float(wrap_pi(self.graph.exit_bearing(c) - entry))
                           for c in forward])
        err = np.abs(wrap_pi(angles - observed_turn))
        keep = err <= self.cfg.max_turn_mismatch_rad
        if not keep.any():
            # Same fallback as the greedy tracker: no branch resembles the
            # observed turn, so go straightest rather than assert a turn no
            # evidence supports.
            k = int(np.argmin(np.abs(angles)))
            trace.turns.append((float(observed_turn), float(angles[k])))
            return [(forward[k], 0.0)]

        sigma = self.cfg.turn_sigma_rad if hasattr(self.cfg, "turn_sigma_rad") \
            else np.deg2rad(20.0)
        log_w = -0.5 * (err / sigma) ** 2
        log_w = log_w - log_w[keep].max()      # best surviving branch -> 0
        trace.turns.append((float(observed_turn),
                            float(angles[int(np.argmax(log_w))])))
        return [(int(forward[i]), float(log_w[i]))
                for i in range(len(forward)) if keep[i]]

    def _match_candidates(self, lat: float, lon: float, heading: float | None):
        """Every plausible road at a point, log-weighted by snap distance.

        `MapMatchedPredictor._match` picks one match and raises an
        ambiguity flag the caller must react to by standing still; this
        keeps every candidate alive as a separate hypothesis instead, so a
        bad first guess is not fatal on its own -- it just starts behind in
        the beam and a later junction can still favour the rival that turns
        out to explain the observed turns better.
        """
        cfg = self.cfg
        # Backend-agnostic: `_candidates` is inherited from
        # MapMatchedPredictor and dispatches to the graph or OSRM. The beam
        # therefore runs offline for free, with no OSRM code of its own.
        out = [(pos, -dist / max(cfg.snap_radius_m, 1e-6))
               for pos, dist in self._candidates(lat, lon, heading)
               if pos is not None]
        if not out:
            return []
        n_start = getattr(cfg, "n_start_hypotheses", 3)
        # Sort BEFORE truncating. `out` is built in candidate order and
        # entries are skipped when they fail to locate, so the first n are not
        # the n nearest -- slicing unsorted can seed the beam with the worst
        # candidates and drop the best.
        out.sort(key=lambda pw: pw[1], reverse=True)
        best = out[0][1]
        return [(p, w - best) for p, w in out[:n_start]]

    # -- beam propagation ----------------------------------------------------

    def _advance_hyp(self, pos: EdgePosition, remaining: float, t_start: float,
                     dist: float, sign: float, log_prob: float,
                     trace: OutageTrace, guard: int = 0):
        """Advance ONE hypothesis by `remaining` metres, forking at junctions.

        Same walk as `MapMatchedPredictor._advance`, but recursive instead
        of iterative so a junction can return more than one continuation.
        `dist` is the full second's displacement, held fixed across the
        recursion (as in `_advance`) purely to interpolate the junction
        crossing TIME within the second.
        """
        if guard > 200:                # pathological micro-edge loop
            return [(pos, log_prob)]
        g = self.graph
        edge_len = g.length(pos.edge)
        room = edge_len - pos.offset_m
        if remaining < room:
            return [(EdgePosition(pos.edge, pos.offset_m + remaining), log_prob)]

        rem_after = remaining - max(room, 0.0)
        node = int(g.d_to[pos.edge])
        trace.junctions += 1
        frac = 1.0 - rem_after / dist if dist > 0 else 1.0
        turn = self._turn_angle(t_start + frac, sign)
        branches = self._choose_branches_soft(pos.edge, node, turn, trace)
        if not branches:
            trace.off_network = True
            return [(EdgePosition(pos.edge, edge_len), log_prob)]     # dead end

        out = []
        for edge, log_w in branches:
            out.extend(self._advance_hyp(
                EdgePosition(edge, 0.0), rem_after, t_start, dist, sign,
                log_prob + log_w, trace, guard + 1))
        return out

    @staticmethod
    def _prune(hyps: list, beam_width: int) -> list:
        """Top `beam_width` hypotheses, deduped to the best one per edge.

        Deduping first: two hypotheses that land on the same directed edge
        after a fork represent the same physical claim ("the vehicle is on
        this road"), so only the better-supported one is worth a beam slot.
        """
        if not hyps:
            return hyps
        best_by_edge: dict = {}
        for h in hyps:
            key = h.pos.edge
            if key not in best_by_edge or h.log_prob > best_by_edge[key].log_prob:
                best_by_edge[key] = h
        ranked = sorted(best_by_edge.values(), key=lambda h: h.log_prob,
                        reverse=True)
        return ranked[:beam_width]

    # -- harness interface ----------------------------------------------------

    def predict(self, t0: float, duration: float) -> dict:
        cfg = self.cfg
        n = int(round(duration))
        beam_width = getattr(cfg, "beam_width", 5)

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

        starts = self._match_candidates(lat0, lon0, h0)
        if not starts:
            self.stats["skipped_nomatch"] += 1
            raise OutageSkipped(
                f"no road within {cfg.snap_radius_m:.0f} m of the start fix "
                f"({lat0:.5f}, {lon0:.5f}) — outside the offline extract, or "
                "off the drivable network")

        sign = self._gyro_sign(t0)
        disp = self._displacements(t0, duration)
        if disp.size < n:
            raise OutageSkipped("inner predictor returned too few seconds")

        hyps = []
        for pos, log_w in starts:
            tr = OutageTrace()
            tr.gyro_sign = sign
            tr.start_ambiguous = len(starts) > 1
            hyps.append(_Hyp(pos=pos, log_prob=log_w,
                             track=[self.graph.position(pos)], trace=tr))

        next_rematch = cfg.rematch_every_s
        for k in range(n):
            nxt = []
            d = float(disp[k])
            for h in hyps:
                # Each child gets its OWN trace. `_advance_hyp` mutates the
                # trace it is handed (junction_forced, turns, off_network), so
                # sharing one object across forks lets pruned branches
                # contribute diagnostics to the hypothesis that survives --
                # the reported junction counts and turn list would describe
                # paths that were never taken.
                children = list(self._advance_hyp(h.pos, d, t0 + k, d, sign,
                                                  h.log_prob, h.trace))
                for i, (pos2, lp2) in enumerate(children):
                    child_trace = h.trace if i == len(children) - 1 \
                        else _copy_trace(h.trace)
                    nxt.append(_Hyp(pos=pos2, log_prob=lp2,
                                    track=h.track + [self.graph.position(pos2)],
                                    trace=child_trace))
            hyps = self._prune(nxt, beam_width)

            if (k + 1) >= next_rematch:
                next_rematch += cfg.rematch_every_s
                for h in hyps:
                    h.trace.rematches += 1
                    lat_p, lon_p = self.graph.position(h.pos)
                    bear = self.graph.bearing_at(h.pos)
                    cand, ambiguous = self._match(lat_p, lon_p, bear)
                    if ambiguous:
                        h.trace.rematch_ambiguous += 1
                    elif cand is not None:
                        h.pos = cand
                        h.track[-1] = self.graph.position(cand)
                        h.trace.rematch_applied += 1
                hyps = self._prune(hyps, beam_width)

        best = max(hyps, key=lambda h: h.log_prob)
        trace = best.trace
        self.stats["outages"] += 1
        for kk in ("junctions", "junction_forced", "rematches",
                  "rematch_applied", "rematch_ambiguous"):
            self.stats[kk] += getattr(trace, kk)
        self.stats["start_ambiguous"] += int(trace.start_ambiguous)
        self.stats["off_network"] += int(trace.off_network)
        self.stats["beam_final_hypotheses"] = (
            self.stats.get("beam_final_hypotheses", 0) + len(hyps))

        track = np.asarray(best.track, dtype=float)
        return self._to_harness(track, trace)
