"""Hidden Markov map matching with a real Viterbi decode.

Why this exists
---------------
`HMMMapMatchedPredictor` keeps a beam, but its decision rule provably
degenerates to greedy. At each fork the branch log-weights are renormalised so
the locally-best branch scores exactly 0, and no evidence is ever applied
AFTER a fork. The chain that takes the best option at every junction therefore
keeps `log_prob == 0` -- the maximum attainable -- and can never be overtaken.
Raising the beam width cannot change that. Measured: its drift figures were
byte-identical to the greedy tracker's across every duration.

The missing ingredient is an EMISSION term: evidence that arrives every second
and re-scores hypotheses already alive, so a branch taken on a marginal turn
can still lose later to one that explains the following seconds better.

The observation model, given there is no GPS
--------------------------------------------
Classic map matching (Newson & Krumm 2009) emits on GPS position: the
likelihood of a fix given a candidate road point. During a GNSS outage there
is no fix, so that emission does not exist and the algorithm has to be
re-derived around what the vehicle can still observe:

  emission   the gyro's measured yaw change over the second, against the
             heading change the road geometry actually imposes on a vehicle
             driving that stretch. Available EVERY second, not only at
             junctions, and it is what discriminates "we took the slip road"
             from "we stayed on the carriageway" several seconds later.

  transition route feasibility between consecutive states -- the along-road
             distance travelled must agree with the displacement the model
             predicted, weighted by the model's OWN predicted sigma. This is
             what the log-variance head was for.

Both are proper log-likelihoods that accumulate, so `log_prob` is monotone
non-increasing and genuinely comparable across hypotheses. Viterbi keeps one
backpointer per surviving state and the path is recovered by backtracking from
the best final state -- not by reading the winner's own running track, which
is what makes it a decode rather than a beam of independent walks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from .graph import EdgePosition, wrap_pi
from .hmm_predictor import HMMMapMatchConfig, HMMMapMatchedPredictor
from .predictor import OutageTrace

if TYPE_CHECKING:                       # pragma: no cover
    from .osrm import OSRMClient

LOG_EPS = -60.0            # a hypothesis this unlikely is effectively dead


@dataclass
class ViterbiConfig(HMMMapMatchConfig):
    """Observation-model parameters, in the units of the things they weigh."""

    # Emission: how far the gyro's measured turn may sit from the turn the
    # road imposes before the hypothesis is penalised. 12 deg/s is wide
    # enough to absorb mount-yaw scale error (measured slope 0.72-0.94
    # against GPS heading change) without admitting a wrong road.
    sigma_turn_rad: float = np.deg2rad(12.0)

    # Transition: the model's predicted sigma is used directly, floored so a
    # confident-but-wrong prediction cannot dominate the decode, and scaled
    # because a per-window sigma understates error accumulated over a step.
    min_sigma_m: float = 1.5
    sigma_scale: float = 1.0

    # Road-class prior: a 30 m/s step along a service road is implausible
    # regardless of geometry. Log-penalty applied per second when the implied
    # speed exceeds the class's plausible maximum.
    class_speed_penalty: float = 2.0

    # Viterbi bookkeeping.
    beam_width: int = 12
    offset_quantum_m: float = 8.0   # state dedup granularity along an edge
    emit_every_s: float = 1.0


# Plausible upper speeds by OSM highway class, m/s. Only used as a soft
# prior -- a hypothesis is penalised, never eliminated, since these are
# conventions rather than physics.
CLASS_MAX_MS = {
    "motorway": 40.0, "motorway_link": 30.0,
    "trunk": 36.0, "trunk_link": 25.0,
    "primary": 30.0, "primary_link": 22.0,
    "secondary": 27.0, "secondary_link": 20.0,
    "tertiary": 25.0, "tertiary_link": 18.0,
    "unclassified": 22.0, "residential": 16.0,
    "living_street": 9.0, "service": 12.0,
}


@dataclass
class _State:
    """One Viterbi state: a position on the network at one time step."""
    pos: EdgePosition
    log_prob: float
    back: int = -1                 # index into the previous step's state list
    trace: OutageTrace = field(default_factory=OutageTrace)

    def key(self, quantum: float) -> tuple[int, int]:
        """Dedup key. Two hypotheses on the same edge within a quantum of one
        another are the same state; keeping both wastes beam width on a
        distinction the geometry cannot support."""
        return (int(self.pos.edge), int(self.pos.offset_m // quantum))


class ViterbiMapMatcher(HMMMapMatchedPredictor):
    """Along-road tracking decoded by Viterbi over an explicit HMM."""

    name = "viterbi_matched"

    def __init__(self, inner, session, graph, osrm: "OSRMClient | None" = None,
                 cfg: ViterbiConfig | None = None):
        super().__init__(inner, session, graph, osrm, cfg or ViterbiConfig())

    # -- observation model -------------------------------------------------

    def _road_turn(self, a: EdgePosition, b: EdgePosition) -> float:
        """Heading change the road imposes between two positions, radians."""
        return float(wrap_pi(self.graph.bearing_at(b) - self.graph.bearing_at(a)))

    def _emission_logp(self, a: EdgePosition, b: EdgePosition,
                       observed_turn: float) -> float:
        """log p(gyro turn | this road transition).

        Gaussian on the residual between the turn the gyro measured and the
        turn the road geometry requires. This is the term the previous
        implementation lacked, and the reason hypotheses here can change rank
        after a fork instead of being frozen by their junction score.
        """
        resid = wrap_pi(observed_turn - self._road_turn(a, b))
        z = float(resid) / self.cfg.sigma_turn_rad
        return max(-0.5 * z * z, LOG_EPS)

    def _transition_logp(self, travelled_m: float, predicted_m: float,
                         sigma_m: float) -> float:
        """log p(this much road travelled | the model's displacement).

        Weighted by the model's own predicted sigma, floored so an
        over-confident window cannot dictate the route.
        """
        s = max(float(sigma_m) * self.cfg.sigma_scale, self.cfg.min_sigma_m)
        z = (float(travelled_m) - float(predicted_m)) / s
        return max(-0.5 * z * z, LOG_EPS)

    def _class_logp(self, edge: int, speed_ms: float) -> float:
        """Soft prior: implied speed against the road class's plausible max."""
        cls = self._edge_class(edge)
        cap = CLASS_MAX_MS.get(cls)
        if cap is None or speed_ms <= cap:
            return 0.0
        over = (speed_ms - cap) / max(cap, 1.0)
        return -self.cfg.class_speed_penalty * float(over)

    def _edge_class(self, d_edge: int) -> str:
        g = self.graph
        hw = getattr(g, "highway", None)
        if hw is None:
            return ""
        try:
            return str(hw[int(d_edge) >> 1])
        except (IndexError, ValueError):
            return ""

    # -- successor enumeration ---------------------------------------------

    def _successors(self, pos: EdgePosition, distance_m: float,
                    t_start: float, sign: float, trace: OutageTrace):
        """Every position reachable by driving `distance_m` from `pos`.

        Reuses the recursive walk from the beam implementation, which already
        forks at junctions and honours passability and U-turn exclusion. The
        junction turn-weights it returns are DISCARDED here: turn plausibility
        is scored by the emission term instead, over the whole step, so
        scoring it twice would double-count the same gyro evidence.
        """
        outs = self._advance_hyp(pos, distance_m, t_start, distance_m, sign,
                                 0.0, trace)
        return [p for p, _w in outs]

    # -- decode -------------------------------------------------------------

    def _decode(self, t0: float, n: int, disp: np.ndarray, sigma: np.ndarray,
                sign: float, starts):
        """Viterbi over `n` one-second steps. Returns the decoded path."""
        cfg = self.cfg
        q = cfg.offset_quantum_m

        cur = [_State(pos=p, log_prob=float(w), trace=OutageTrace())
               for p, w in starts]
        for st in cur:
            st.trace.gyro_sign = sign
            st.trace.start_ambiguous = len(starts) > 1
        lattice = [cur]

        for k in range(n):
            d = float(disp[k])
            s_k = float(sigma[k]) if k < len(sigma) else cfg.min_sigma_m
            observed = self._turn_angle(t0 + k + 1.0, sign)
            nxt: dict[tuple[int, int], tuple[float, _State]] = {}

            for i, st in enumerate(cur):
                # Each parent needs its own trace copy: `_advance_hyp` mutates
                # what it is handed, and a shared object would let pruned
                # branches contribute diagnostics to survivors.
                tr = replace(st.trace)
                tr.turns = list(st.trace.turns)
                for pos2 in self._successors(st.pos, d, t0 + k, sign, tr):
                    travelled = self._along_distance(st.pos, pos2, d)
                    lp = (st.log_prob
                          + self._transition_logp(travelled, d, s_k)
                          + self._emission_logp(st.pos, pos2, observed)
                          + self._class_logp(pos2.edge, travelled))
                    cand = _State(pos=pos2, log_prob=lp, back=i, trace=tr)
                    key = cand.key(q)
                    if key not in nxt or lp > nxt[key][0]:
                        nxt[key] = (lp, cand)

            if not nxt:                       # nowhere to go; hold position
                break
            survivors = sorted((v for _, v in nxt.values()),
                               key=lambda s: s.log_prob, reverse=True)
            cur = survivors[:cfg.beam_width]
            # Renormalise so log-probs stay in a usable range over a long
            # outage. A constant shift per step cannot reorder hypotheses,
            # so the decode is unaffected.
            top = cur[0].log_prob
            for st in cur:
                st.log_prob -= top
            lattice.append(cur)

        return self._backtrack(lattice)

    def _along_distance(self, a: EdgePosition, b: EdgePosition,
                        fallback: float) -> float:
        """Road distance actually covered between two positions.

        Exact when both lie on the same edge; otherwise the step crossed at
        least one junction and the commanded distance is the right estimate,
        because `_advance_hyp` consumed exactly that much road getting there.
        """
        if a.edge == b.edge:
            return abs(float(b.offset_m) - float(a.offset_m))
        return float(fallback)

    @staticmethod
    def _backtrack(lattice: list[list[_State]]):
        """Recover the single best path by following backpointers.

        This is what makes it a decode: the reported route is the globally
        best-scoring path through the lattice, which need not be the path any
        one hypothesis was following at the time.
        """
        if not lattice or not lattice[-1]:
            return []
        j = int(np.argmax([s.log_prob for s in lattice[-1]]))
        path = [lattice[-1][j]]
        for step in range(len(lattice) - 1, 0, -1):
            j = lattice[step][j].back
            if j < 0:
                break
            path.append(lattice[step - 1][j])
        return list(reversed(path))

    # -- harness interface --------------------------------------------------

    def predict(self, t0: float, duration: float) -> dict:
        from eval.harness import OutageSkipped

        cfg = self.cfg
        n = int(round(duration))

        acc = float(np.interp(t0, self.t, np.nan_to_num(self.gps_acc,
                                                        nan=np.inf)))
        if not np.isfinite(acc) or acc > cfg.max_gps_accuracy_m:
            self.stats["skipped_gps"] += 1
            raise OutageSkipped(
                f"GPS accuracy {acc:.1f} m at t0 exceeds "
                f"{cfg.max_gps_accuracy_m:.0f} m")

        lat0 = float(np.interp(t0, self.t, self.lat))
        lon0 = float(np.interp(t0, self.t, self.lon))
        h0 = float(np.interp(t0, self.t, self.heading_unwrapped))

        starts = self._match_candidates(lat0, lon0, h0)
        if not starts:
            self.stats["skipped_nomatch"] += 1
            raise OutageSkipped("no road near the start fix")

        sign = self._gyro_sign(t0)
        disp = self._displacements(t0, duration)
        if disp.size < n:
            raise OutageSkipped("inner predictor returned too few seconds")
        sigma = self._sigmas(t0, duration, n)

        path = self._decode(t0, n, disp, sigma, sign, starts)
        if not path:
            raise OutageSkipped("decode produced no path")

        pts = [self.graph.position(st.pos) for st in path]
        while len(pts) < n + 1:                 # decode stopped early
            pts.append(pts[-1])
        track = np.asarray(pts[:n + 1], dtype=float)

        trace = path[-1].trace
        self.stats["outages"] += 1
        for k in ("junctions", "junction_forced", "rematches",
                  "rematch_applied", "rematch_ambiguous"):
            self.stats[k] += getattr(trace, k, 0)
        self.stats["start_ambiguous"] += int(trace.start_ambiguous)
        return self._to_harness(track, trace)

    def _sigmas(self, t0: float, duration: float, n: int) -> np.ndarray:
        """Per-second displacement sigma from the model's log-variance head.

        Falls back to a constant when the inner predictor has no uncertainty
        output (the analytic baselines do not), so the transition term still
        weighs distance consistency, just without per-window confidence.
        """
        inner = self.inner
        getter = getattr(inner, "predict_sigmas", None)
        if callable(getter):
            try:
                s = np.asarray(getter(t0, duration), dtype=float)
                if s.size >= n:
                    return s[:n]
            except Exception:
                pass
        return np.full(n, max(self.cfg.min_sigma_m, 3.0))
