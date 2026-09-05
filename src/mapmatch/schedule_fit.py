"""Scheduled sampling for the HMM map matcher's observation model.

Where scheduled sampling does and does not attach here
-----------------------------------------------------
Scheduled sampling (Bengio et al. 2015) is a training-time remedy for
*exposure bias*: a model trained on ground-truth history but run on its own
history. Two components in this pipeline were checked for that gap and do not
have it:

  * the TCN reads only raw levelled sensor windows -- no predicted quantity
    ever enters its input, at train or at inference, so there is no teacher
    forcing to anneal away;
  * the matcher's 10 s re-match snaps to its OWN tracked position
    (`predictor.py`, `graph.position(pos)`), never to a fix, and the emission
    reads the gyro, which is a live sensor during an outage. The decode is
    already fully free-running.

The gap is in FITTING the observation model. `ViterbiConfig` carries six
hand-set constants -- `sigma_turn_rad`, `min_sigma_m`, `sigma_scale`,
`class_speed_penalty`, and the two bookkeeping knobs. Nothing has ever fitted
them to data. Fitting them is where the exposure gap is real, because the
Viterbi state at step k+1 is conditioned on the state at step k:

  teacher-forced (eps = 1)  score every step from the TRUE road position at
                            step k. Decomposable, low variance, and it gives
                            a usable gradient signal even when the parameters
                            start far from sensible -- but the sigmas it
                            picks never see what a wrong branch costs three
                            seconds later.

  free-running (eps = 0)    score every step from the DECODER'S OWN state.
                            This is deployment exactly, and it is the metric
                            we actually report -- but early in the search it
                            is mostly noise, because a bad parameter set
                            leaves the road in the first two seconds and
                            every subsequent step is scored against nonsense.

So: anneal eps from 1 to 0 across the fitting epochs. That is scheduled
sampling in its literal form, applied to the one place in this system that
has the structure it addresses.

The decode itself is not differentiable -- it is a discrete argmax over a
lattice -- so the parameter search is derivative-free (coordinate descent on
a log scale). Scheduled sampling here shapes the OBJECTIVE, not a gradient.

Nothing in `viterbi.py` is modified: this module drives the existing decoder's
scoring terms step by step so it can inject truth between steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .graph import EdgePosition, geodesic_m
from .predictor import OutageTrace
from .viterbi import ViterbiConfig, ViterbiMapMatcher, _State

# Parameters the search moves, with the range each is allowed to take. The
# bounds are physical, not tuning latitude: a turn sigma below 2 deg cannot
# absorb the measured 0.72-0.94 mount-yaw scale error, and above 60 deg the
# emission stops discriminating roads at all.
BOUNDS = {
    "sigma_turn_rad": (np.deg2rad(2.0), np.deg2rad(60.0)),
    "min_sigma_m": (0.25, 15.0),
    "sigma_scale": (0.25, 6.0),
    "class_speed_penalty": (0.0, 10.0),
}


@dataclass
class ScheduleConfig:
    """The sampling schedule and the rollouts it is measured on."""

    horizon_s: int = 30           # steps per rollout
    n_outages: int = 24           # rollout start times per session
    epochs: int = 6               # eps steps from 1 -> 0
    decay: str = "inverse_sigmoid"      # "inverse_sigmoid" | "linear"
    k: float = 2.5                # inverse-sigmoid sharpness
    snap_radius_m: float = 40.0   # truth snap; wider than the matcher's own
    seed: int = 0

    # Coordinate descent: how many multiplicative trials per parameter per
    # epoch, and the step size, which shrinks as eps falls so the search
    # settles as the objective becomes the deployment one.
    trials: int = 5
    step0: float = 1.6


def teacher_forcing_prob(epoch: int, cfg: ScheduleConfig) -> float:
    """eps for this epoch: 1 at the first, 0 at the last.

    Both schedules are Bengio et al.'s. Inverse-sigmoid is the default
    because it is S-shaped: it holds the teacher near 1 through the early
    epochs, where a free-running rollout carries no usable signal, then hands
    over quickly. Normalised to the endpoints, it crosses linear at the
    midpoint -- it is not uniformly slower, and the two agree there.
    """
    # The LAST epoch is always fully free-running, single-epoch fits
    # included: its objective is the only one comparable to a deployment
    # number, so the schedule must always end there rather than reporting a
    # teacher-forced figure as if it were a drift measurement.
    if epoch >= cfg.epochs - 1:
        return 0.0
    frac = epoch / max(cfg.epochs - 1, 1)
    if cfg.decay == "linear":
        return float(max(0.0, 1.0 - frac))
    # Inverse sigmoid, normalised so eps(0) = 1 and eps(last) = 0 exactly.
    raw = lambda x: 1.0 / (1.0 + math.exp(cfg.k * (2.0 * x - 1.0)))
    lo, hi = raw(1.0), raw(0.0)
    return float((raw(frac) - lo) / (hi - lo))


class ScheduledSamplingFitter:
    """Fits a `ViterbiConfig` against GPS truth with an annealed teacher.

    One fitter per (matcher, session). `matcher` is a live
    `ViterbiMapMatcher`, used for its graph, its sensor grid and its scoring
    terms; its `cfg` is swapped per trial and restored afterwards.
    """

    def __init__(self, matcher: ViterbiMapMatcher,
                 cfg: ScheduleConfig | None = None):
        self.m = matcher
        self.cfg = cfg or ScheduleConfig()
        self._truth: dict[float, EdgePosition | None] = {}
        self._prep: dict[tuple[float, int], tuple | None] = {}

    # -- ground truth ------------------------------------------------------

    def truth_at(self, t: float) -> EdgePosition | None:
        """The true road position at time `t`, from the GPS fix.

        This is the teacher. It is read ONLY by the fitter, and only between
        steps; the decoder itself never sees it, at fit time or after.
        """
        key = round(float(t), 3)
        if key in self._truth:
            return self._truth[key]
        m = self.m
        lat = float(np.interp(t, m.t, m.lat))
        lon = float(np.interp(t, m.t, m.lon))
        hd = float(np.interp(t, m.t, m.heading_unwrapped))
        pos = m.graph.locate(lat, lon, hd, radius_m=self.cfg.snap_radius_m)
        self._truth[key] = pos
        return pos

    def truth_xy(self, t: float) -> tuple[float, float]:
        """True lat/lon at `t` -- the error is measured against the FIX, not
        against its snap, so a bad snap cannot flatter the objective."""
        m = self.m
        return (float(np.interp(t, m.t, m.lat)),
                float(np.interp(t, m.t, m.lon)))

    # -- per-outage inputs, cached ----------------------------------------

    def prep(self, t0: float, n: int):
        """Everything about this outage that the PARAMETERS cannot change.

        The displacements, sigmas, gyro sign and per-second turn angles depend
        only on the sensors and the inner predictor, never on the observation
        model being fitted. Caching them is not an optimisation detail: the
        search evaluates a few thousand rollouts per session, and without this
        every one of them would re-run the TCN over the same window and return
        the same numbers.

        Returns `(disp, sigma, sign, turns, start)`, or None if the outage
        cannot be scored.
        """
        key = (round(float(t0), 3), int(n))
        if key in self._prep:
            return self._prep[key]
        m = self.m
        out = None
        try:
            disp = np.asarray(m._displacements(t0, float(n)), dtype=float)
            if disp.size >= n:
                sign = m._gyro_sign(t0)
                sigma = np.asarray(m._sigmas(t0, float(n), n), dtype=float)
                turns = np.array([m._turn_angle(t0 + k + 1.0, sign)
                                  for k in range(n)], dtype=float)
                start = self.truth_at(t0)
                if start is not None:
                    out = (disp, sigma, sign, turns, start)
        except Exception:
            out = None
        self._prep[key] = out
        return out

    # -- one rollout -------------------------------------------------------

    def rollout(self, t0: float, n: int, eps: float,
                rng: np.random.Generator) -> float | None:
        """Mean per-step position error over `n` seconds, metres.

        At each step, with probability `eps`, the parent state set is replaced
        by the single TRUE road position -- teacher forcing. Otherwise the
        decoder's own surviving beam carries forward, which is deployment.
        The error is always measured against truth either way, so the two
        regimes are on the same scale and the objective is comparable across
        epochs.

        Returns None when the rollout could not be scored at all (no road at
        the start, or the inner predictor had nothing to say here).
        """
        m = self.m
        m_cfg = m.cfg
        prep = self.prep(t0, n)
        if prep is None:
            return None
        disp, sigma, sign, turns, start = prep
        cur = [_State(pos=start, log_prob=0.0, trace=OutageTrace())]
        cur[0].trace.gyro_sign = sign

        q = m_cfg.offset_quantum_m
        errors: list[float] = []
        for k in range(n):
            d = float(disp[k])
            s_k = float(sigma[k]) if k < len(sigma) else m_cfg.min_sigma_m
            observed = turns[k]
            nxt: dict[tuple[int, int], tuple[float, _State]] = {}

            for i, st in enumerate(cur):
                tr = replace(st.trace)
                tr.turns = list(st.trace.turns)
                for pos2 in m._successors(st.pos, d, t0 + k, sign, tr):
                    travelled = m._along_distance(st.pos, pos2, d)
                    lp = (st.log_prob
                          + m._transition_logp(travelled, d, s_k)
                          + m._emission_logp(st.pos, pos2, observed)
                          + m._class_logp(pos2.edge, travelled))
                    cand = _State(pos=pos2, log_prob=lp, back=i, trace=tr)
                    key = cand.key(q)
                    if key not in nxt or lp > nxt[key][0]:
                        nxt[key] = (lp, cand)
            if not nxt:
                break

            survivors = sorted((v for _, v in nxt.values()),
                               key=lambda s: s.log_prob, reverse=True)
            cur = survivors[:m_cfg.beam_width]
            top = cur[0].log_prob
            for st in cur:
                st.log_prob -= top

            # Score the step's best hypothesis against the true fix. Done
            # BEFORE any teacher injection, so the objective always measures
            # what the decoder produced, never what it was handed.
            t_k = t0 + k + 1.0
            lat_p, lon_p = m.graph.position(cur[0].pos)
            lat_t, lon_t = self.truth_xy(t_k)
            errors.append(float(geodesic_m(lat_t, lon_t, lat_p, lon_p)))

            # Scheduled sampling: hand the truth back with probability eps.
            if eps > 0.0 and rng.random() < eps:
                tp = self.truth_at(t_k)
                if tp is not None:
                    forced = _State(pos=tp, log_prob=0.0, trace=cur[0].trace)
                    cur = [forced]

        if not errors:
            return None
        return float(np.mean(errors))

    # -- objective ---------------------------------------------------------

    def objective(self, params: dict, starts: Sequence[float], eps: float,
                  seed: int) -> float:
        """Mean rollout error over `starts` at these parameters.

        `seed` is fixed per evaluation so every candidate in a coordinate
        sweep sees the SAME teacher-forcing draws. Otherwise the search would
        be comparing parameter sets across different curricula and would
        happily accept a worse one that drew an easier schedule.
        """
        m = self.m
        saved = m.cfg
        m.cfg = replace(saved, **params)
        try:
            rng = np.random.default_rng(seed)
            vals = [v for t0 in starts
                    if (v := self.rollout(t0, self.cfg.horizon_s, eps, rng))
                    is not None]
        finally:
            m.cfg = saved
        if not vals:
            return float("inf")
        return float(np.mean(vals))

    # -- the fit -----------------------------------------------------------

    def fit(self, starts: Sequence[float], verbose: bool = True) -> dict:
        """Coordinate descent over BOUNDS, with eps annealed 1 -> 0.

        Returns the history: one row per epoch with eps, the objective at that
        epoch's parameters, and the parameters themselves. The final row is
        eps = 0, i.e. scored purely free-running, which is the only row
        directly comparable to a deployment number.
        """
        params = {k: float(getattr(self.m.cfg, k)) for k in BOUNDS}
        history = []
        for epoch in range(self.cfg.epochs):
            eps = teacher_forcing_prob(epoch, self.cfg)
            seed = self.cfg.seed + epoch
            # The step shrinks with eps: coarse while the teacher still holds
            # the rollout together, fine once the objective is deployment.
            step = 1.0 + (self.cfg.step0 - 1.0) * max(eps, 0.15)
            best = self.objective(params, starts, eps, seed)

            for name, (lo, hi) in BOUNDS.items():
                base = params[name]
                for mult in self._ladder(step, self.cfg.trials):
                    trial = float(np.clip(base * mult, lo, hi))
                    if trial == base:
                        continue
                    cand = dict(params, **{name: trial})
                    score = self.objective(cand, starts, eps, seed)
                    if score < best:
                        best, params = score, cand
            history.append({"epoch": epoch, "eps": eps, "objective": best,
                            **params})
            if verbose:
                print(f"  epoch {epoch}  eps={eps:.2f}  "
                      f"err={best:8.2f} m  "
                      + "  ".join(f"{k}={params[k]:.3f}" for k in BOUNDS),
                      flush=True)
        return {"params": params, "history": history}

    @staticmethod
    def _ladder(step: float, trials: int) -> list[float]:
        """Multiplicative trial points around 1, largest step first."""
        out = []
        for i in range(trials // 2, 0, -1):
            f = step ** i
            out += [f, 1.0 / f]
        return out


def fit_pooled(pairs, cfg: ScheduleConfig, verbose: bool = True) -> dict:
    """Fit ONE parameter set across several sessions at once.

    `pairs` is [(fitter, starts), ...]. The objective is the mean rollout
    error pooled over every session, so a parameter set cannot win by suiting
    one route. This is the entry point the leave-one-session-out driver uses:
    the held-out session simply never appears in `pairs`.
    """
    if not pairs:
        raise ValueError("fit_pooled needs at least one (fitter, starts) pair")
    params = {k: float(getattr(pairs[0][0].m.cfg, k)) for k in BOUNDS}

    def score(ps, eps, seed):
        vals = [f.objective(ps, st, eps, seed) for f, st in pairs]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("inf")

    history = []
    for epoch in range(cfg.epochs):
        eps = teacher_forcing_prob(epoch, cfg)
        seed = cfg.seed + epoch
        step = 1.0 + (cfg.step0 - 1.0) * max(eps, 0.15)
        best = score(params, eps, seed)
        for name, (lo, hi) in BOUNDS.items():
            base = params[name]
            for mult in ScheduledSamplingFitter._ladder(step, cfg.trials):
                trial = float(np.clip(base * mult, lo, hi))
                if trial == base:
                    continue
                cand = dict(params, **{name: trial})
                v = score(cand, eps, seed)
                if v < best:
                    best, params = v, cand
        history.append({"epoch": epoch, "eps": eps, "objective": best,
                        **params})
        if verbose:
            print(f"  epoch {epoch}  eps={eps:.2f}  err={best:8.2f} m  "
                  + "  ".join(f"{k}={params[k]:.3f}" for k in BOUNDS),
                  flush=True)
    return {"params": params, "history": history}


def apply(cfg: ViterbiConfig, params: dict) -> ViterbiConfig:
    """A copy of `cfg` carrying fitted parameters."""
    return replace(cfg, **{k: float(v) for k, v in params.items()
                           if k in ViterbiConfig.__dataclass_fields__})
