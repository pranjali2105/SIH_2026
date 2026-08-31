"""Outage harness: run a predictor across a simulated GNSS outage.

`predictor` is a swappable interface so the same harness serves the classical
baseline and, later, the trained model without either knowing about the other.
A predictor must expose:

    predict(t0: float, duration: float) -> {"displacements": array,
                                            "headings": array}

Both arrays are per second of the outage: how far the vehicle travelled during
that second, and the heading it held. The harness -- not the predictor --
accumulates position, so every predictor is scored on identical arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.loader import Session, _find

from .metrics import geodesic_distance_m, summarise

# GPS heading is meaningless at rest, and the paper's scenarios are all
# in-motion, so outages starting below this are skipped rather than scored.
MIN_START_SPEED_MS = 5.0


class OutageSkipped(RuntimeError):
    """This outage cannot be scored (too slow, too short, no fixes)."""


@dataclass
class OutageResult:
    session: str
    t0: float
    duration: float
    predictor: str
    displacement_errors: np.ndarray = field(default_factory=lambda: np.empty(0))
    position_errors: np.ndarray = field(default_factory=lambda: np.empty(0))
    final_position_error: float = float("nan")
    true_distance_m: float = float("nan")
    start_speed_ms: float = float("nan")

    def metrics(self) -> dict:
        m = summarise(self.displacement_errors, self.position_errors)
        m.update(session=self.session, t0=self.t0, duration=self.duration,
                 predictor=self.predictor,
                 final_position_error_m=self.final_position_error,
                 max_position_error_m=(float(np.nanmax(self.position_errors))
                                       if self.position_errors.size else np.nan),
                 true_distance_m=self.true_distance_m,
                 start_speed_ms=self.start_speed_ms)
        return m


def _gps_series(session: Session):
    lat_c, lon_c = _find(session.df, r"LATITUDE"), _find(session.df, r"LONGITUDE")
    if lat_c is None or lon_c is None or "time_s" not in session.df.columns:
        raise OutageSkipped("session has no usable GPS columns")
    df = session.df
    t = pd.to_numeric(df["time_s"], errors="coerce")
    lat = pd.to_numeric(df[lat_c], errors="coerce")
    lon = pd.to_numeric(df[lon_c], errors="coerce")
    spd = (pd.to_numeric(df["speed_ms"], errors="coerce")
           if "speed_ms" in df.columns else pd.Series(np.nan, index=df.index))
    ok = t.notna() & lat.notna() & lon.notna() & (lat != 0) & (lon != 0)
    if ok.sum() < 20:
        raise OutageSkipped("fewer than 20 usable GPS fixes")
    return (t[ok].to_numpy(float), lat[ok].to_numpy(float),
            lon[ok].to_numpy(float), spd[ok].to_numpy(float))


def run_outage(session: Session, t0: float, duration: float,
               predictor) -> OutageResult:
    """Score one outage.

    Truth comes from GPS at one-second marks; the predictor never sees it.
    Position is accumulated as p_k = p_{k-1} + d_k * [cos psi_k, sin psi_k]
    with psi measured clockwise from north, so cos -> north and sin -> east.
    """
    t, lat, lon, spd = _gps_series(session)
    n = int(round(duration))
    if t0 < t[0] or t0 + duration > t[-1]:
        raise OutageSkipped("outage window falls outside the session")

    start_speed = float(np.interp(t0, t, spd))
    if not np.isfinite(start_speed) or start_speed < MIN_START_SPEED_MS:
        raise OutageSkipped(
            f"start speed {start_speed:.2f} m/s below {MIN_START_SPEED_MS} m/s — "
            "GPS heading is meaningless at rest")

    marks = t0 + np.arange(n + 1, dtype=float)
    true_lat = np.interp(marks, t, lat)
    true_lon = np.interp(marks, t, lon)
    true_step = geodesic_distance_m(true_lat[:-1], true_lon[:-1],
                                    true_lat[1:], true_lon[1:])

    out = predictor.predict(t0, duration)
    disp = np.asarray(out["displacements"], dtype=float)[:n]
    head = np.asarray(out["headings"], dtype=float)[:n]
    if disp.size < n:
        raise OutageSkipped("predictor returned fewer seconds than requested")

    # Predicted track, in metres from the outage start.
    north = np.cumsum(disp * np.cos(head))
    east = np.cumsum(disp * np.sin(head))

    # True track in the same local frame.
    lat0, lon0 = true_lat[0], true_lon[0]
    m_per_deg_lat = geodesic_distance_m(lat0, lon0, lat0 + 1e-4, lon0) / 1e-4
    m_per_deg_lon = geodesic_distance_m(lat0, lon0, lat0, lon0 + 1e-4) / 1e-4
    true_north = (true_lat[1:] - lat0) * m_per_deg_lat
    true_east = (true_lon[1:] - lon0) * m_per_deg_lon

    position_errors = np.sqrt((north - true_north) ** 2 + (east - true_east) ** 2)

    return OutageResult(
        session=f"{session.family}-{session.session}", t0=float(t0),
        duration=float(duration), predictor=getattr(predictor, "name", "unknown"),
        displacement_errors=disp - true_step,
        position_errors=position_errors,
        final_position_error=float(position_errors[-1]),
        true_distance_m=float(np.sum(true_step)),
        start_speed_ms=start_speed)


def iter_outages(session: Session, duration: float, stride: float | None = None):
    """Yield candidate outage start times, non-overlapping by default."""
    t, _, _, _ = _gps_series(session)
    stride = duration if stride is None else stride
    t0 = float(t[0])
    end = float(t[-1]) - duration
    while t0 <= end:
        yield t0
        t0 += stride


def run_session(session: Session, predictor, duration: float = 10.0,
                stride: float | None = None) -> tuple[list[OutageResult], list[str]]:
    """Run every non-overlapping outage in a session. Returns (results, skips).

    If the predictor exposes `predict_many`, all outages are evaluated in one
    batched call. Steps WITHIN an outage may be serial, but outages are
    independent of one another, so batching across them is free and exact.
    """
    starts = list(iter_outages(session, duration, stride))
    if not starts:
        return [], []

    if hasattr(predictor, "predict_many"):
        cached = dict(zip(starts, predictor.predict_many(starts, duration)))

        class _Replay:
            """Serves the precomputed prediction for each t0."""
            name = getattr(predictor, "name", "unknown")

            @staticmethod
            def predict(t0, dur):
                return cached[t0]

        predictor_for_run = _Replay()
    else:
        predictor_for_run = predictor

    results, skips = [], []
    for t0 in starts:
        try:
            results.append(run_outage(session, t0, duration, predictor_for_run))
        except OutageSkipped as exc:
            skips.append(f"t0={t0:.1f}: {exc}")
    return results, skips
