"""Offline mount-yaw (phi) calibration from pre-outage, GNSS-available driving.

`fusion.eskf.NonHolonomicESKF` cannot identify the phone's mount yaw `phi`
from 60 seconds of outage data alone (see `eskf.py`'s module docstring and
`results/findings.md` §8): phi and the gyro bias trade off against each
other, and a filter estimating both together diverges (measured: bias
23.8 rad/s, phi drift 13.7 rad on S-A5). But phi is a property of the MOUNT,
not of the outage -- it does not change second to second, and it IS
observable whenever GPS is available, from the same two relations `eskf.py`
already uses to make the filter's pseudo-measurements:

    a_forward = dv/dt          (from GPS speed)
    a_lateral = v * omega      (from GPS heading rate and speed)

This fits phi ONCE, offline, over the pre-outage history where GPS speed and
heading are both good, then the filter can hold it fixed during the outage
instead of trying to identify it from a signal that cannot support it --
findings.md's own "identified next step, not attempted".

Deployable: everything used here is available before an outage starts;
nothing at or after t0 is read (mirrors `mapmatch.predictor._gyro_sign`,
which makes the same before-t0 guarantee for the same reason).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this speed, dv/dt and v*omega are both dominated by GPS/gyro noise
# rather than genuine vehicle motion.
MIN_SPEED_MPS = 3.0

# Blocks whose forward- or lateral-accel target implies more than this are
# almost certainly a GPS glitch, not real vehicle dynamics -- the same order
# of guard CLAUDE.md documents for the displacement labels themselves (the
# 60 m/s implausible-speed guard in `data.sanity`).
MAX_TARGET_MPS2 = 8.0

# Inflate distrust of the lateral relation with |omega|, mirroring the
# ESKF's own `NHC_OMEGA_INFLATION`: tyre slip and body roll break
# a_lat = v*omega during hard cornering, so those samples should count for
# less in the fit, not more.
OMEGA_INFLATION = 8.0

# An event only carries information about phi if it is bigger than sensor
# noise. Used purely for the observability diagnostics below, not the fit.
EVENT_THRESHOLD_MPS2 = 0.5

BLOCK_S = 1.0
DEFAULT_LOOKBACK_S = 180.0
MIN_FORWARD_EVENTS = 20
MIN_LATERAL_EVENTS = 20


@dataclass
class MountYawEstimate:
    """Result of an offline phi fit, with the diagnostics that say whether it
    should be trusted rather than just a number.
    """

    phi_rad: float
    n_blocks: int
    n_forward_events: int      # blocks where braking/accelerating was the
                                # informative signal (|dv/dt| above noise)
    n_lateral_events: int      # blocks where turning was
    residual_rad: float        # RMS residual (m/s^2) of the fit -- named
                                # for what it constrains, heading, not its units
    confidence: float          # 0..1 diagnostic summary; see `_confidence`

    @property
    def observable(self) -> bool:
        """Whether the fit rests on real evidence, not degenerate data.

        `eskf.py`'s own finding is that the lateral relation alone leaves
        phi and the gyro bias jointly unidentifiable (its `apply_forward`
        option exists for exactly this reason: "one equation for two
        unknowns"). Requiring both kinds of event offline is the same fix,
        applied to a batch fit instead of a sequential filter: a session
        that only ever brakes in a straight line, or only ever corners at
        constant speed, does not determine phi and should say so rather than
        return a number with no support.
        """
        return (self.n_forward_events >= MIN_FORWARD_EVENTS
                and self.n_lateral_events >= MIN_LATERAL_EVENTS)


def _confidence(n_fwd: int, n_lat: int, resid_mps2: float) -> float:
    """A single 0..1 summary for logging/dashboards.

    Not used to gate anything programmatically -- callers that need a hard
    requirement should check `.observable`. This exists so a marginal fit
    shows up as marginal instead of as a bare angle with no context.
    """
    if not np.isfinite(resid_mps2):
        return 0.0
    event_term = min(1.0, n_fwd / 60.0) * min(1.0, n_lat / 60.0)
    fit_term = float(np.exp(-resid_mps2 / 1.0))     # 1 m/s^2 RMS -> ~0.37
    return float(np.clip(event_term * fit_term, 0.0, 1.0) ** 0.5)


def estimate_mount_yaw(gps_t: np.ndarray, speed: np.ndarray,
                       heading_unwrapped: np.ndarray,
                       imu_t: np.ndarray, chan: np.ndarray, t0: float,
                       lookback_s: float = DEFAULT_LOOKBACK_S,
                       block_s: float = BLOCK_S,
                       min_speed_mps: float = MIN_SPEED_MPS,
                       max_target_mps2: float = MAX_TARGET_MPS2,
                       omega_inflation: float = OMEGA_INFLATION
                       ) -> MountYawEstimate | None:
    """Fit phi from the `lookback_s` seconds of history strictly before `t0`.

    `gps_t`, `speed`, `heading_unwrapped` are on the session's native GPS
    grid. `imu_t`, `chan` are the level-frame IMU (columns 0, 1 = the two
    horizontal accelerometer axes, matching `eskf.py`'s `a_h1, a_h2`) on
    its own, denser grid -- deliberately kept separate rather than assuming
    a shared grid, since `mapmatch.predictor` and `train.ModelPredictor`
    each hold IMU and GPS quantities on different grids too.

    Blocks are 1 Hz, not IMU rate: GPS speed and heading update at ~1 Hz and
    are held between fixes, so differencing them faster replays exactly the
    aliasing CLAUDE.md documents ("Never band-limit or difference before
    correlating" / the spike-train warning `mapmatch.predictor._gyro_sign`
    already works around the same way).

    Returns None when there is not enough history to attempt a fit at all.
    Returns an estimate with `.observable == False` when there IS history
    but it does not contain both kinds of event phi needs, rather than
    silently guessing -- callers should check `.observable` before trusting
    `.phi_rad`.
    """
    gps_t = np.asarray(gps_t, dtype=float)
    speed = np.asarray(speed, dtype=float)
    heading_unwrapped = np.asarray(heading_unwrapped, dtype=float)
    imu_t = np.asarray(imu_t, dtype=float)
    chan = np.asarray(chan, dtype=float)

    ok = np.isfinite(gps_t) & np.isfinite(speed) & np.isfinite(heading_unwrapped)
    if not ok.any():
        return None
    sel = ok & (gps_t >= t0 - lookback_s) & (gps_t < t0)
    if sel.sum() < 2 or imu_t.size < 2 or chan.shape[0] < 2:
        return None

    t_lo = float(gps_t[sel].min())
    t_hi = float(gps_t[sel].max())
    marks = np.arange(t_lo, t_hi, block_s)
    if marks.size < 3:
        return None

    a_h1 = np.interp(marks, imu_t, chan[:, 0])
    a_h2 = np.interp(marks, imu_t, chan[:, 1])
    sp = np.interp(marks, gps_t, speed)
    hd = np.interp(marks, gps_t, heading_unwrapped)

    dv = np.diff(sp) / block_s
    dh = np.diff(hd) / block_s              # true yaw rate, from GPS heading
    v_mid = 0.5 * (sp[1:] + sp[:-1])
    x1 = 0.5 * (a_h1[1:] + a_h1[:-1])
    x2 = 0.5 * (a_h2[1:] + a_h2[:-1])

    target_fwd = dv
    target_lat = v_mid * dh

    valid = (np.isfinite(target_fwd) & np.isfinite(target_lat)
            & np.isfinite(x1) & np.isfinite(x2)
            & (v_mid >= min_speed_mps)
            & (np.abs(target_fwd) <= max_target_mps2)
            & (np.abs(target_lat) <= max_target_mps2))
    n = int(valid.sum())
    if n < 10:
        return MountYawEstimate(0.0, n, 0, 0, float("nan"), 0.0)

    tf, tl = target_fwd[valid], target_lat[valid]
    u1, u2 = x1[valid], x2[valid]
    omega = dh[valid]

    # Down-weight cornering samples in proportion to how hard the corner is,
    # for the same physical reason the ESKF's own NHC update does.
    w = 1.0 / (1.0 + omega_inflation * np.abs(omega))

    # Closed-form fit of the rotation angle phi that best aligns raw
    # horizontal accel (u1, u2) with the GPS-derived (target_fwd, target_lat)
    # under the same [[cos,sin],[-sin,cos]] model `eskf.py` uses for
    # `update_forward`/`update_nhc` -- the 2-D analogue of Kabsch /
    # orthogonal Procrustes. Unlike the EKF this needs no initial guess and
    # cannot diverge: it is the exact minimiser of
    # sum_k w_k * |target_k - R(phi) u_k|^2 over the whole lookback window.
    A = float(np.sum(w * (tf * u1 + tl * u2)))
    B = float(np.sum(w * (tf * u2 - tl * u1)))
    phi = float(np.arctan2(B, A))

    c, s = np.cos(phi), np.sin(phi)
    pred_fwd = u1 * c + u2 * s
    pred_lat = -u1 * s + u2 * c
    resid = float(np.sqrt(np.mean((tf - pred_fwd) ** 2 + (tl - pred_lat) ** 2)))

    fwd_events = int(np.sum(np.abs(tf) > EVENT_THRESHOLD_MPS2))
    lat_events = int(np.sum(np.abs(tl) > EVENT_THRESHOLD_MPS2))
    conf = _confidence(fwd_events, lat_events, resid)

    return MountYawEstimate(phi, n, fwd_events, lat_events, resid, conf)


def _unwrap_filled(angles: np.ndarray) -> np.ndarray:
    """Unwrap an angle series, interpolating across missing samples.

    Mirrors `mapmatch.predictor._unwrap_filled` exactly (duplicated rather
    than imported: `fusion` has no other dependency on `mapmatch`, and this
    is eight lines).
    """
    a = np.asarray(angles, dtype=float)
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros_like(a)
    idx = np.arange(a.size)
    filled = np.interp(idx, idx[ok], a[ok])
    return np.unwrap(filled)


def estimate_mount_yaw_for_session(session, t0: float,
                                   lookback_s: float = DEFAULT_LOOKBACK_S,
                                   **kwargs) -> MountYawEstimate | None:
    """Convenience wrapper: assembles the arrays `estimate_mount_yaw` needs
    directly from a `data.loader.Session`.

    Kept separate from `mapmatch.predictor.MapMatchedPredictor.__init__` and
    `train.ModelPredictor.__init__` -- both build a subset of these arrays
    already, but neither exposes GPS speed together with the IMU level-frame
    channels through one object, and duplicating the dozen lines here is
    cheaper than reshaping either.

    NOTE: depends on `data.loader` / `data.windows` / `data.sanity`, which
    this checkout does not currently have (see the repo's `.gitignore`:
    `data/` matches `src/data/` too, so that package appears to have never
    been committed). This function is untested against real sessions for
    that reason; `estimate_mount_yaw` above has no such dependency and is
    tested directly.
    """
    import pandas as pd
    from data.loader import _find
    from data.sanity import gps_cumulative_distance
    from data.windows import GRID_HZ, levelled_channels

    df = session.df
    gps_t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)

    head_c = _find(df, r"^HEADING") or _find(df, r"GPS ORIENTATION")
    heading = (np.radians(pd.to_numeric(df[head_c], errors="coerce")
                          .to_numpy(float))
              if head_c is not None else np.zeros_like(gps_t))
    heading_unwrapped = _unwrap_filled(heading)

    # "GPS SPEED (Kmh)" is mislabelled: corpus-wide on S- files it is
    # already m/s (CLAUDE.md, "Units"). Read it directly rather than assume
    # a derived `speed_ms` column exists.
    speed_c = _find(df, r"GPS SPEED")
    speed = (pd.to_numeric(df[speed_c], errors="coerce").to_numpy(float)
            if speed_c is not None else np.full_like(gps_t, np.nan))

    gps = gps_cumulative_distance(session)
    lo, hi = float(gps[0].min()), float(gps[0].max())
    imu_t = np.arange(lo, hi, 1.0 / GRID_HZ)
    chan = levelled_channels(session, imu_t)
    if chan is None:
        return None

    return estimate_mount_yaw(gps_t, speed, heading_unwrapped, imu_t, chan,
                              t0, lookback_s=lookback_s, **kwargs)


def safe_phi0(session, t0: float, lookback_s: float = DEFAULT_LOOKBACK_S,
             **kwargs) -> float:
    """`estimate_mount_yaw_for_session`, failing closed to 0.0 on any error
    or an unobservable fit.

    The entry point predictor wrappers (`fusion.predictor.ESKFPredictor`,
    `SpeedFusedPredictor`) should call: calibration is a refinement, never a
    hard dependency, so a missing `data` package, too little pre-outage
    history, or a drive that never brakes/turns must fall back to prior
    (phi=0) behaviour rather than raise or return an untrustworthy angle.
    """
    try:
        est = estimate_mount_yaw_for_session(session, t0, lookback_s=lookback_s,
                                             **kwargs)
        if est is not None and est.observable:
            return est.phi_rad
    except Exception:                          # noqa: BLE001
        pass
    return 0.0
