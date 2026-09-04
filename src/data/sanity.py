"""Sanity diagnostics for IO-VNBD: signal alignment, time sync, trajectory checks.

Writes everything under results/sanity/. This pass REPORTS ONLY -- it never
repairs a session or edits the split. Deciding what is a real problem and what
is a session worth dropping happens after reading the summary.

Outputs
-------
results/sanity/signals/<session>.png       speed vs longitudinal accel
results/sanity/xcorr/<session>.png         correlation against lag
results/sanity/trajectory/<session>.png    plotted route
results/sanity/time_offsets.csv            per-session lag, never global
results/sanity/trajectory_checks.csv       pass/fail per invariant
results/sanity/vw12_jpg_comparison.png     the one human eye check
results/sanity/summary.md                  counts, lag distribution, suspects

Run:  python -m data.sanity          (whole corpus)
      python -m data.sanity S-Vw12   (one or more sessions)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in this environment
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.signal import butter, filtfilt

from .loader import (
    DATA_ROOT,
    G_TO_MS2,
    Session,
    _find,
    _norm,
    list_sessions,
    load_session,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "results" / "sanity"
SWEEP_CSV = REPO_ROOT / "results" / "corpus_sweep.csv"

TIME_OFFSETS_CSV = OUT_ROOT / "time_offsets.csv"
DISTANCE_OFFSETS_CSV = OUT_ROOT / "time_offsets_distance.csv"
TRAJ_CHECKS_CSV = OUT_ROOT / "trajectory_checks.csv"
SUMMARY_MD = OUT_ROOT / "summary.md"

R_EARTH = 6371000.0

# Cross-correlation settings.
XCORR_WINDOW_S = 300.0      # the 5-minute stretch
XCORR_MAX_LAG_S = 5.0       # search +/- 5 s
XCORR_GRID_HZ = 10.0        # common grid both signals are resampled onto
# GPS reports at 1 Hz, so GPS-derived acceleration is band-limited to ~0.5 Hz.
# Raw 10 Hz IMU is dominated by road vibration well above that. Both signals
# are low-passed to the same band before correlating, or the comparison is
# between two different bandwidths.
XCORR_LOWPASS_HZ = 0.5

# Suspect thresholds. A weak peak is itself a finding: it means the
# cross-correlation never locked onto real structure.
SUSPECT_LAG_MS = 500.0
SUSPECT_CORR = 0.3

# Declared from domain knowledge, NOT fitted from the data -- a box derived
# from the coordinates it is meant to validate would pass by construction.
# (lat_min, lat_max, lon_min, lon_max)
COUNTRY_BOXES = {
    "UK": (49.8, 59.5, -8.3, 2.1),
    "France": (41.0, 51.6, -5.3, 9.7),
    "Nigeria": (4.0, 14.0, 2.6, 14.8),
}
# Group T is the Renault Megane's France sessions; group I is Lagos. Every
# other group -- Fiesta, Volvo, and the Vf* groups despite their name -- is UK.
GROUP_COUNTRY = {"T": "France", "I": "Nigeria"}
DEFAULT_COUNTRY = "UK"

DISTANCE_TOLERANCE = 0.05   # 5% against the sweep's recorded distance_km
LOOP_CLOSE_M = 300.0        # "returns near its origin" threshold
LOOP_TOLERANCE_M = 150.0    # ...and how close it must then actually get
JUMP_SPEED_MARGIN = 3.0     # a fix may not imply >3x the reported speed
JUMP_FLOOR_MS = 15.0        # ...with this floor, so slow sections aren't noise


def _country_for(group: str) -> str:
    return GROUP_COUNTRY.get(group, DEFAULT_COUNTRY)


# --------------------------------------------------------------------------
# channel extraction
# --------------------------------------------------------------------------

def candidate_accel_channels(s: Session) -> dict[str, pd.Series]:
    """Accelerometer channels the lag search may use.

    V- has one explicit longitudinal channel. For S- the phone's orientation in
    the cradle is unknown and varies between sessions, so all three
    gravity-removed axes are offered and the lag search picks whichever
    correlates best -- selecting an axis at zero lag would be circular, since
    the whole point is that the lag is unknown.
    """
    df = s.df
    if s.family == "V":
        col = _find(df, r"LONGITUDINAL ACCEL")
        if col is None:
            return {}
        return {f"{col} (g->m/s^2)":
                pd.to_numeric(df[col], errors="coerce") * G_TO_MS2}

    axes = [_find(df, r"ACCELEROMETER", rf"\b{a}\b") for a in "XYZ"]
    grav = [_find(df, r"GRAVITY", rf"\b{a}\b") for a in "XYZ"]
    if any(a is None for a in axes):
        return {}
    out = {}
    for name, acol, gcol in zip("XYZ", axes, grav):
        a = pd.to_numeric(df[acol], errors="coerce")
        if gcol is not None:
            a = a - pd.to_numeric(df[gcol], errors="coerce")
        out[f"linear accel {name}"] = a
    return out


def longitudinal_accel(s: Session) -> tuple[pd.Series | None, str]:
    """The along-track acceleration channel, in m/s^2, and a label for it.

    V- files expose an explicit longitudinal channel (in g). S- files do not:
    the phone's orientation in the cradle is unknown and varies by session, so
    there is no fixed "longitudinal axis". Gravity is removed using the
    Gravity X/Y/Z channels, then the axis best correlated with dv/dt is
    selected per session and reported, rather than assumed.
    """
    df = s.df
    if s.family == "V":
        col = _find(df, r"LONGITUDINAL ACCEL")
        if col is None:
            return None, "none"
        return pd.to_numeric(df[col], errors="coerce") * G_TO_MS2, f"{col} (g->m/s^2)"

    axes = [_find(df, r"ACCELEROMETER", rf"\b{a}\b") for a in "XYZ"]
    grav = [_find(df, r"GRAVITY", rf"\b{a}\b") for a in "XYZ"]
    if any(a is None for a in axes):
        return None, "none"

    lin = {}
    for name, acol, gcol in zip("XYZ", axes, grav):
        a = pd.to_numeric(df[acol], errors="coerce")
        if gcol is not None:
            a = a - pd.to_numeric(df[gcol], errors="coerce")
        lin[name] = a

    dv = speed_derivative(s)
    best, best_r = None, 0.0
    if dv is not None:
        for name, a in lin.items():
            ok = a.notna() & dv.notna()
            if ok.sum() < 100:
                continue
            r = float(np.corrcoef(a[ok], dv[ok])[0, 1])
            if np.isfinite(r) and abs(r) > abs(best_r):
                best, best_r = name, r
    if best is None:
        best, best_r = "X", float("nan")
    sign = -1.0 if best_r < 0 else 1.0
    label = f"linear accel {best}{'(-)' if sign < 0 else ''} (r={best_r:+.2f} vs dv/dt)"
    return lin[best] * sign, label


def speed_change_points(s: Session) -> tuple[np.ndarray, np.ndarray] | None:
    """(t, speed) at the rows where the GPS speed actually updates.

    GPS reports at 1 Hz while S- rows tick at 10 Hz, so the speed column is
    HELD between fixes. Differencing every row therefore puts a whole second of
    change onto one 0.1 s step and leaves nine zeros -- the resulting dv/dt is
    a spike train that cross-correlates against nothing. Keeping only the
    update instants recovers the real 1 Hz speed signal.
    """
    if "speed_ms" not in s.df.columns or "time_s" not in s.df.columns:
        return None
    v = pd.to_numeric(s.df["speed_ms"], errors="coerce")
    t = pd.to_numeric(s.df["time_s"], errors="coerce")
    ok = v.notna() & t.notna()
    v, t = v[ok], t[ok]
    if len(v) < 20:
        return None
    changed = v.diff() != 0
    if changed.sum() >= 20:
        changed.iloc[0] = True
        v, t = v[changed], t[changed]
    order = np.argsort(t.to_numpy(), kind="stable")
    return t.to_numpy()[order], v.to_numpy()[order]


def speed_derivative_on_grid(s: Session, grid: np.ndarray) -> np.ndarray | None:
    """d(speed)/dt sampled on `grid`, built from the GPS update instants."""
    cp = speed_change_points(s)
    if cp is None or grid.size < 3:
        return None
    t_cp, v_cp = cp
    uniq, idx = np.unique(t_cp, return_inverse=True)
    if uniq.size < 3:
        return None
    v_u = np.bincount(idx, weights=v_cp) / np.bincount(idx)
    v_grid = np.interp(grid, uniq, v_u)
    return np.gradient(v_grid, grid)


def speed_derivative(s: Session) -> pd.Series | None:
    """Row-aligned d(speed)/dt, interpolated from the GPS update instants.

    Used for choosing the S- longitudinal axis; the cross-correlation uses the
    gridded form above.
    """
    if "time_s" not in s.df.columns:
        return None
    t = pd.to_numeric(s.df["time_s"], errors="coerce")
    ok = t.notna()
    if ok.sum() < 20:
        return None
    dv = speed_derivative_on_grid(s, t[ok].to_numpy())
    if dv is None:
        return None
    out = pd.Series(np.nan, index=s.df.index, dtype=float)
    out.loc[t[ok].index] = dv
    return out.replace([np.inf, -np.inf], np.nan)


def _uniform(t: pd.Series, y: pd.Series, hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample (t, y) onto a uniform grid; duplicate timestamps are averaged."""
    ok = t.notna() & y.notna()
    t, y = t[ok].to_numpy(float), y[ok].to_numpy(float)
    if t.size < 10:
        return np.empty(0), np.empty(0)
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    # Several rows can share a tick (one row per sensor event) -- average them
    # so the grid is single-valued.
    uniq, idx = np.unique(t, return_inverse=True)
    means = np.bincount(idx, weights=y) / np.bincount(idx)
    grid = np.arange(uniq[0], uniq[-1], 1.0 / hz)
    return grid, np.interp(grid, uniq, means)


# --------------------------------------------------------------------------
# 1. per-session signal check
# --------------------------------------------------------------------------

def plot_signals(s: Session, out_dir: Path) -> Path | None:
    """Speed and longitudinal acceleration on a shared time axis.

    Braking should appear as a negative excursion in the accelerometer
    coincident with a falling speed trace.
    """
    accel, label = longitudinal_accel(s)
    if accel is None or "time_s" not in s.df.columns:
        return None
    t = pd.to_numeric(s.df["time_s"], errors="coerce")
    v = pd.to_numeric(s.df.get("speed_ms"), errors="coerce") \
        if "speed_ms" in s.df.columns else None

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(13, 6))
    if v is not None:
        ax1.plot(t, v, lw=0.7, color="#1f77b4")
    ax1.set_ylabel("speed (m/s)")
    ax1.set_title(f"{s.family}-{s.session}  role={s.role}  "
                  f"hz_tick={s.hz_tick:.2f}   [{label}]")
    ax1.grid(alpha=0.3)

    ax2.plot(t, accel, lw=0.5, color="#d62728")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("long. accel (m/s²)")
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{s.family}-{s.session}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 2. time synchronisation -- per session, stored, never global
# --------------------------------------------------------------------------

@dataclass
class LagResult:
    session: str
    role: str
    best_lag_ms: float
    peak_correlation: float
    n_samples_used: int
    lags_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    corrs: np.ndarray = field(default_factory=lambda: np.empty(0))
    note: str = ""
    best_yaw_rad: float = float("nan")
    yaw_thirds: list = field(default_factory=list)
    yaw_drift_rad: float = float("nan")


class CorrelationNormalisationError(RuntimeError):
    """A computed correlation exceeded |1|.

    That is impossible for a correctly normalised correlation, so it means the
    normalisation broke -- in practice, dividing by a std computed over the
    full series while the overlap at that lag is tiny. Clamping to +/-1 would
    hide the bug behind a plausible number, so this raises instead.
    """


def _checked_corr(r: float, context: str) -> float:
    if np.isfinite(r) and abs(r) > 1.0 + 1e-9:
        raise CorrelationNormalisationError(
            f"correlation {r:.4f} outside [-1, 1] in {context}: "
            "normalisation failure, most likely insufficient overlap at this lag")
    return r



def _lowpass(x: np.ndarray, cutoff: float = XCORR_LOWPASS_HZ,
             fs: float = XCORR_GRID_HZ, order: int = 3) -> np.ndarray:
    """Zero-phase low-pass. Zero-phase matters: a causal filter would itself
    introduce a lag and corrupt the very quantity being measured."""
    if x.size < 3 * order + 1:
        return x
    b, a = butter(order, cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def multi_axis_correlation(s: Session) -> tuple[float, float]:
    """Best (R, lag_ms) from a least-squares fit of ALL axes to d(speed)/dt.

    Mounting-invariant: the along-track direction in the phone frame is some
    unknown combination of X/Y/Z, so fitting all three jointly is the fairest
    test of whether the accelerometer carries the vehicle's acceleration at
    all. Reported alongside the single-channel result so a weak peak cannot be
    dismissed as a bad axis choice.
    """
    chans = candidate_accel_channels(s)
    if len(chans) < 2 or "time_s" not in s.df.columns:
        return float("nan"), float("nan")
    t = pd.to_numeric(s.df["time_s"], errors="coerce")
    grids = []
    for ch in chans.values():
        g, a = _uniform(t, ch, XCORR_GRID_HZ)
        grids.append((g, a))
    n = min(g.size for g, _ in grids)
    if n < int(XCORR_GRID_HZ * 30):
        return float("nan"), float("nan")
    gt = grids[0][0][:n]
    gv = speed_derivative_on_grid(s, gt)
    if gv is None:
        return float("nan"), float("nan")
    A = np.column_stack([_lowpass(np.nan_to_num(a[:n])) for _, a in grids])
    B = _lowpass(np.nan_to_num(gv))
    if B.std() < 1e-9:
        return float("nan"), float("nan")
    B = (B - B.mean()) / B.std()
    max_lag = int(XCORR_MAX_LAG_S * XCORR_GRID_HZ)
    best_r, best_lag = 0.0, float("nan")
    for L in range(-max_lag, max_lag + 1):
        if L < 0:
            X, Y = A[-L:], B[:n + L]
        elif L > 0:
            X, Y = A[:n - L], B[L:]
        else:
            X, Y = A, B
        X = X - X.mean(axis=0)
        try:
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = X @ coef
        if pred.std() < 1e-12:
            continue
        r = float(np.corrcoef(pred, Y)[0, 1])
        if np.isfinite(r) and abs(r) > abs(best_r):
            best_r, best_lag = r, L / XCORR_GRID_HZ * 1000.0
    return best_r, best_lag



# ---------------------------------------------------------------------------
# displacement labels (single source, with a standing plausibility guard)
# ---------------------------------------------------------------------------

LABEL_WINDOW_S = 1.0
LABEL_GRID_HZ = 10.0

# Standing guard, not a one-off. Some sessions' GPS positions jump, inflating
# per-window displacement to physically impossible values: S-T1's fix-to-fix
# path length is 76 km against a 22 km integral of its own reported speed,
# giving a mean "displacement" of 104 m/s (375 km/h). 60 m/s (216 km/h) is
# above anything these vehicles do and below the corruption.
MAX_PLAUSIBLE_SPEED_MS = 60.0


def displacement_labels(s: Session, grid_hz: float = LABEL_GRID_HZ,
                        window_s: float = LABEL_WINDOW_S,
                        max_speed_ms: float = MAX_PLAUSIBLE_SPEED_MS):
    """(t, displacement over the next `window_s`, n_rejected) from GPS.

    Built from cumulative geodesic distance -- an integral of position, never
    a differentiated speed. Windows implying a speed above `max_speed_ms` are
    dropped as corrupt rather than trusted.
    """
    gps = gps_cumulative_distance(s)
    if gps is None:
        return None
    t, cum = gps
    lo, hi = float(t.min()), float(t.max())
    if not np.isfinite(lo) or hi - lo < 10 * window_s:
        return None
    grid = np.arange(lo, hi, 1.0 / grid_hz)
    c = np.interp(grid, t, cum)
    w = int(window_s * grid_hz)
    if c.size <= w:
        return None
    tt, yy = grid[:-w], c[w:] - c[:-w]

    implied = yy / window_s
    keep = np.isfinite(implied) & (implied >= 0) & (implied <= max_speed_ms)
    n_rejected = int((~keep).sum())
    if keep.sum() < 50:
        return None
    return tt[keep], yy[keep], n_rejected



# ---------------------------------------------------------------------------
# distance-domain synchronisation (supersedes the acceleration-domain method)
# ---------------------------------------------------------------------------

# Geodesic distance. pyproj's Geod.inv is vectorised and agrees with the
# `vincenty` package to under a millimetre (verified in tests); the pure-Python
# Vincenty implementation is scalar-only and far too slow for the corpus.
_GEOD = Geod(ellps="WGS84")

DISTANCE_MAX_LAG_S = 5.0
DISTANCE_GRID_HZ = 10.0
GROSS_LAG_MS = 2000.0      # only gross misalignment is worth reviewing
MIN_DISTANCE_SAMPLES = 300


def gps_cumulative_distance(s: Session) -> tuple[np.ndarray, np.ndarray] | None:
    """(t, cumulative metres) from GPS fixes, geodesic.

    Differenced across FIX UPDATES only: the fix is held between 1 Hz updates
    while rows tick faster, so differencing every row would attribute a full
    second of travel to one sub-tick step.
    """
    lat_c, lon_c = _find(s.df, r"LATITUDE"), _find(s.df, r"LONGITUDE")
    if lat_c is None or lon_c is None or "time_s" not in s.df.columns:
        return None
    lat = pd.to_numeric(s.df[lat_c], errors="coerce")
    lon = pd.to_numeric(s.df[lon_c], errors="coerce")
    t = pd.to_numeric(s.df["time_s"], errors="coerce")
    ok = lat.notna() & lon.notna() & t.notna() & (lat != 0) & (lon != 0)
    lat, lon, t = lat[ok], lon[ok], t[ok]
    if len(lat) < 20:
        return None
    moved = (lat.diff() != 0) | (lon.diff() != 0)
    moved.iloc[0] = True
    lat, lon, t = lat[moved], lon[moved], t[moved]
    if len(lat) < 10:
        return None
    la, lo = lat.to_numpy(float), lon.to_numpy(float)
    _, _, step = _GEOD.inv(lo[:-1], la[:-1], lo[1:], la[1:])
    cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(step))])
    return t.to_numpy(float), cum


def imu_cumulative_motion(s: Session, grid: np.ndarray) -> np.ndarray | None:
    """A monotone 'distance-like' curve from the IMU.

    The IMU cannot give absolute distance without double integration, which
    drifts without bound. What it can give is a monotone curve with the same
    STOP/GO SHAPE as cumulative distance: the running integral of horizontal
    linear-acceleration magnitude, which is ~0 while parked and accumulates
    while driving. Both curves are normalised to [0, 1] before matching, so
    only that shared shape is compared -- never the absolute scale.
    """
    if s.family == "V":
        col = _find(s.df, r"LONGITUDINAL ACCEL")
        lat_col = _find(s.df, r"LATERAL ACCEL")
        if col is None:
            return None
        t = pd.to_numeric(s.df["time_s"], errors="coerce")
        a1 = _sensor_grid(t, pd.to_numeric(s.df[col], errors="coerce") * G_TO_MS2, grid)
        if lat_col is not None:
            a2 = _sensor_grid(t, pd.to_numeric(s.df[lat_col], errors="coerce") * G_TO_MS2,
                              grid)
            mag = np.sqrt(np.nan_to_num(a1) ** 2 + np.nan_to_num(a2) ** 2)
        else:
            mag = np.abs(np.nan_to_num(a1))
    else:
        comp = level_frame_components(s, grid)
        if comp is None:
            return None
        a1, a2 = comp
        mag = np.sqrt(np.nan_to_num(a1) ** 2 + np.nan_to_num(a2) ** 2)
    if grid.size < 3:
        return None
    return np.concatenate([[0.0], np.cumsum(mag[1:] * np.diff(grid))])


def _normalise_curve(y: np.ndarray) -> np.ndarray:
    y = y - y[0]
    span = y[-1]
    return y / span if np.isfinite(span) and abs(span) > 1e-12 else y


def estimate_offset_by_distance(s: Session) -> dict:
    """Align IMU to GPS in the INTEGRAL domain.

    Differentiating 1 Hz GPS speed to make a 10 Hz acceleration reference
    high-passes a quantised signal and amplifies its noise; even a perfect ECU
    accelerometer only reaches r = 0.12-0.58 against that reference, versus
    0.93 against clean ECU speed. Matching cumulative curves instead involves
    no differentiation at all, so the 1 Hz GPS is used at the resolution it
    actually has.

    Returns the shift minimising the mean absolute difference between the two
    normalised cumulative curves.
    """
    name = f"{s.family}-{s.session}"
    out = {"session": name, "role": s.role, "group": s.group,
           "best_lag_ms": np.nan, "residual": np.nan, "n_samples_used": 0,
           "method": "distance", "note": ""}
    if "time_s" not in s.df.columns:
        out["note"] = "no time column"
        return out

    gps = gps_cumulative_distance(s)
    if gps is None:
        out["note"] = "no usable GPS fixes"
        return out
    t_gps, cum_gps = gps
    lo, hi = float(t_gps.min()), float(t_gps.max())
    if not np.isfinite(lo) or hi - lo < 60:
        out["note"] = "under 60 s of GPS coverage"
        return out
    grid = np.arange(lo, hi, 1.0 / DISTANCE_GRID_HZ)
    if grid.size < MIN_DISTANCE_SAMPLES:
        out["note"] = "grid too short"
        return out

    g_curve = _normalise_curve(np.interp(grid, t_gps, cum_gps))
    imu = imu_cumulative_motion(s, grid)
    if imu is None:
        out["note"] = "no usable IMU channels"
        return out
    i_curve = _normalise_curve(imu)
    if not np.isfinite(i_curve).all() or i_curve[-1] == 0:
        out["note"] = "flat IMU curve"
        return out

    n = grid.size
    max_lag = int(DISTANCE_MAX_LAG_S * DISTANCE_GRID_HZ)
    best_res, best_L = np.inf, 0
    for L in range(-max_lag, max_lag + 1):
        if L < 0:
            x, y = i_curve[-L:], g_curve[:n + L]
        elif L > 0:
            x, y = i_curve[:n - L], g_curve[L:]
        else:
            x, y = i_curve, g_curve
        if x.size < MIN_DISTANCE_SAMPLES:
            continue
        # Both curves were normalised ONCE over their full extent, before the
        # loop. Re-normalising each overlap here would let a shift win by
        # truncating the curve instead of by aligning it, which drives the
        # argmin to the search boundary.
        res = float(np.mean(np.abs(x - y)))
        if res < best_res:
            best_res, best_L = res, L

    out.update(best_lag_ms=best_L / DISTANCE_GRID_HZ * 1000.0,
               residual=best_res if np.isfinite(best_res) else np.nan,
               n_samples_used=int(n))
    return out



# ---------------------------------------------------------------------------
# levelled-frame reference reconstruction (step 2)
# ---------------------------------------------------------------------------

YAW_STEPS = 72              # 5-degree grid over [0, 2*pi)
MIN_LEVEL_SAMPLES = 300


def _sensor_grid(t: pd.Series, y: pd.Series, grid: np.ndarray) -> np.ndarray:
    """Interpolate one channel onto `grid` using ONLY its own valid samples.

    AndroSensor writes one row per sensor event, so a given column is blank or
    stale in most rows of a tick. Treating the row grid as uniform per column
    is the trap that produced the spike-train artefact on GPS speed.
    """
    ok = t.notna() & y.notna()
    if ok.sum() < 10:
        return np.full(grid.size, np.nan)
    tt = t[ok].to_numpy(float)
    yy = y[ok].to_numpy(float)
    order = np.argsort(tt, kind="stable")
    tt, yy = tt[order], yy[order]
    uniq, idx = np.unique(tt, return_inverse=True)
    means = np.bincount(idx, weights=yy) / np.bincount(idx)
    return np.interp(grid, uniq, means, left=np.nan, right=np.nan)


def level_frame_components(s: Session, grid: np.ndarray):
    """Horizontal linear-acceleration components in the levelled frame.

    Gravity is ~9.8 m/s^2 against vehicle accelerations of 1-2 m/s^2 -- roughly
    5x the signal -- so it is removed with the Gravity X/Y/Z channels before
    anything is correlated. The median gravity direction then defines "down",
    and the two axes perpendicular to it span the horizontal plane. After this
    the only unknown left in the mounting is a single yaw offset.

    Returns (a1, a2) on `grid`, or None.
    """
    axes = [_find(s.df, r"ACCELEROMETER", rf"\b{a}\b") for a in "XYZ"]
    grav = [_find(s.df, r"GRAVITY", rf"\b{a}\b") for a in "XYZ"]
    if any(a is None for a in axes) or any(g is None for g in grav):
        return None
    t = pd.to_numeric(s.df["time_s"], errors="coerce")

    A = np.column_stack([_sensor_grid(t, pd.to_numeric(s.df[c], errors="coerce"), grid)
                         for c in axes])
    G = np.column_stack([_sensor_grid(t, pd.to_numeric(s.df[c], errors="coerce"), grid)
                         for c in grav])
    if not np.isfinite(A).any() or not np.isfinite(G).any():
        return None

    lin = A - G                       # gravity removed
    g_med = np.nanmedian(G, axis=0)
    norm = np.linalg.norm(g_med)
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    down = g_med / norm

    # Any two axes perpendicular to `down` span the horizontal plane; which
    # pair is chosen only shifts the origin of yaw, which the search absorbs.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ref @ down)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - (ref @ down) * down
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(down, e1)

    return lin @ e1, lin @ e2


def _corr_over_yaw(x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
                   yaws: np.ndarray) -> np.ndarray:
    """Correlation of (cos t * x1 + sin t * x2) against y, for every yaw t.

    Evaluated from the 2x2 Gram matrix and the two cross-products rather than
    rebuilding the projection per angle, so the yaw grid costs O(n) not
    O(n * len(yaws)).
    """
    c1, c2 = float(x1 @ y), float(x2 @ y)
    g11, g22, g12 = float(x1 @ x1), float(x2 @ x2), float(x1 @ x2)
    c, sn = np.cos(yaws), np.sin(yaws)
    num = c * c1 + sn * c2
    den = np.sqrt(np.maximum(c * c * g11 + sn * sn * g22 + 2 * c * sn * g12, 1e-12))
    return num / den / max(np.sqrt(float(y @ y)), 1e-12)


def estimate_lag_levelled(s: Session) -> LagResult:
    """Joint (yaw, lag) search against GPS-derived acceleration.

    The phone's forward direction in the levelled frame is unknown, and so is
    the clock offset. Estimating either alone biases the other, so both are
    searched together and the joint peak is taken.
    """
    name = f"{s.family}-{s.session}"
    empty = LagResult(name, s.role, np.nan, np.nan, 0)
    if "time_s" not in s.df.columns:
        empty.note = "no time column"
        return empty

    t = pd.to_numeric(s.df["time_s"], errors="coerce")
    if t.notna().sum() < MIN_LEVEL_SAMPLES:
        empty.note = "too few timestamps"
        return empty
    span_lo, span_hi = float(t.min()), float(t.max())
    grid = np.arange(span_lo, span_hi, 1.0 / XCORR_GRID_HZ)
    if grid.size < MIN_LEVEL_SAMPLES:
        empty.note = "under 30 s of usable data"
        return empty

    if s.family == "V":
        col = _find(s.df, r"LONGITUDINAL ACCEL")
        if col is None:
            empty.note = "no longitudinal channel"
            return empty
        a1 = _sensor_grid(t, pd.to_numeric(s.df[col], errors="coerce") * G_TO_MS2, grid)
        a2 = np.zeros_like(a1)            # already along-track; no yaw freedom
        yaws = np.array([0.0])
    else:
        comp = level_frame_components(s, grid)
        if comp is None:
            empty.note = "missing accelerometer or gravity channels"
            return empty
        a1, a2 = comp
        yaws = np.arange(YAW_STEPS) * (2 * np.pi / YAW_STEPS)

    gv = speed_derivative_on_grid(s, grid)
    if gv is None:
        empty.note = "could not derive speed on the grid"
        return empty

    ok = np.isfinite(a1) & np.isfinite(a2) & np.isfinite(gv)
    if ok.sum() < MIN_LEVEL_SAMPLES:
        empty.note = f"only {int(ok.sum())} aligned samples"
        return empty
    a1 = _lowpass(np.where(ok, a1, 0.0))
    a2 = _lowpass(np.where(ok, a2, 0.0))
    y = _lowpass(np.where(ok, gv, 0.0))
    for arr in (a1, a2, y):
        arr -= arr.mean()
    if y.std() < 1e-9 or (a1.std() < 1e-9 and a2.std() < 1e-9):
        empty.note = "flat signal"
        return empty

    n = a1.size
    max_lag = int(XCORR_MAX_LAG_S * XCORR_GRID_HZ)
    lags = np.arange(-max_lag, max_lag + 1)
    best = (0.0, 0, 0.0)                       # (r, lag, yaw)
    curve = np.full(lags.size, np.nan)
    for i, L in enumerate(lags):
        if L < 0:
            x1, x2, yy = a1[-L:], a2[-L:], y[:n + L]
        elif L > 0:
            x1, x2, yy = a1[:n - L], a2[:n - L], y[L:]
        else:
            x1, x2, yy = a1, a2, y
        if x1.size < MIN_LEVEL_SAMPLES:
            continue
        rs = _corr_over_yaw(x1, x2, yy, yaws)
        k = int(np.argmax(np.abs(rs)))
        _checked_corr(float(rs[k]), f"levelled yaw/lag search at lag {L}")
        curve[i] = rs[k]
        if abs(rs[k]) > abs(best[0]):
            best = (float(rs[k]), int(L), float(yaws[k]))

    r, lag, yaw = best
    res = LagResult(name, s.role, lag / XCORR_GRID_HZ * 1000.0, r, int(n),
                    lags / XCORR_GRID_HZ * 1000.0, curve,
                    note="levelled (yaw,lag) search")
    res.best_yaw_rad = yaw
    res.yaw_thirds, res.yaw_drift_rad = _yaw_stability(a1, a2, y, lag, yaws)
    return res


def _yaw_stability(a1: np.ndarray, a2: np.ndarray, y: np.ndarray,
                   lag: int, yaws: np.ndarray) -> tuple[list[float], float]:
    """Re-estimate yaw on each third of the session at the fixed best lag.

    A stable mount should give a roughly constant yaw. Large drift means the
    phone moved mid-session, which is a finding in its own right rather than
    something to average away.
    """
    if yaws.size <= 1:
        return [], float("nan")
    n = a1.size
    if lag < 0:
        x1, x2, yy = a1[-lag:], a2[-lag:], y[:n + lag]
    elif lag > 0:
        x1, x2, yy = a1[:n - lag], a2[:n - lag], y[lag:]
    else:
        x1, x2, yy = a1, a2, y
    m = x1.size // 3
    if m < MIN_LEVEL_SAMPLES:
        return [], float("nan")
    out = []
    for k in range(3):
        sl = slice(k * m, (k + 1) * m)
        rs = _corr_over_yaw(x1[sl] - x1[sl].mean(), x2[sl] - x2[sl].mean(),
                            yy[sl] - yy[sl].mean(), yaws)
        out.append(float(yaws[int(np.argmax(np.abs(rs)))]))
    # Circular spread: the angles wrap, so a plain std would call 0.05 and
    # 6.23 rad far apart when they are 0.1 rad apart.
    vec = np.mean(np.exp(1j * np.array(out)))
    drift = float(np.sqrt(max(-2.0 * np.log(max(abs(vec), 1e-12)), 0.0)))
    return out, drift



def estimate_lag(s: Session) -> LagResult:
    """Cross-correlate d(GPS speed)/dt against the IMU longitudinal channel.

    Four vehicles and three phones recorded over months have no reason to share
    a mounting or clock offset, so this is computed and stored per session and
    never collapsed to a project-wide constant.
    """
    name = f"{s.family}-{s.session}"
    empty = LagResult(name, s.role, np.nan, np.nan, 0)

    channels = candidate_accel_channels(s)
    if not channels or "time_s" not in s.df.columns:
        empty.note = "missing speed or accelerometer channel"
        return empty

    t = pd.to_numeric(s.df["time_s"], errors="coerce")
    best_res: LagResult | None = None
    for label, accel in channels.items():
        gt, ga = _uniform(t, accel, XCORR_GRID_HZ)
        if gt.size < int(XCORR_GRID_HZ * 30):
            empty.note = "under 30 s of usable data"
            continue
        gv = speed_derivative_on_grid(s, gt)
        if gv is None:
            empty.note = "could not derive speed on the grid"
            continue
        res = _xcorr_one(name, s.role, ga, gv, label)
        if res is None:
            continue
        if best_res is None or abs(res.peak_correlation) > abs(best_res.peak_correlation):
            best_res = res
    if best_res is None:
        return empty
    return best_res


def _xcorr_one(name: str, role: str, ga: np.ndarray, gv: np.ndarray,
               label: str) -> LagResult | None:
    """Cross-correlate one accelerometer channel against d(speed)/dt."""
    # Pick the busiest 5-minute stretch: a stationary window has no structure
    # to correlate, which would produce a meaningless peak.
    win = int(XCORR_WINDOW_S * XCORR_GRID_HZ)
    if ga.size > win:
        energy = pd.Series(np.abs(gv)).rolling(win, min_periods=win).sum()
        end = int(energy.idxmax()) if energy.notna().any() else win - 1
        sl = slice(max(0, end - win + 1), end + 1)
        ga, gv = ga[sl], gv[sl]

    # The two channels are gridded independently and can differ by a sample.
    m = min(ga.size, gv.size)
    ga, gv = ga[:m], gv[:m]
    ga = _lowpass(np.nan_to_num(ga))
    gv = _lowpass(np.nan_to_num(gv))
    a = ga - ga.mean()
    b = gv - gv.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    a, b = a / a.std(), b / b.std()

    max_lag = int(XCORR_MAX_LAG_S * XCORR_GRID_HZ)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.empty(lags.size)
    n = a.size
    for i, L in enumerate(lags):
        if L < 0:
            x, y = a[-L:], b[:n + L]
        elif L > 0:
            x, y = a[:n - L], b[L:]
        else:
            x, y = a, b
        corrs[i] = np.dot(x, y) / max(x.size, 1)

    k = int(np.argmax(np.abs(corrs)))
    return LagResult(name, role,
                     float(lags[k] / XCORR_GRID_HZ * 1000.0),
                     float(corrs[k]), int(n),
                     lags / XCORR_GRID_HZ * 1000.0, corrs, note=label)


def plot_xcorr(res: LagResult, out_dir: Path) -> Path | None:
    if res.lags_ms.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(res.lags_ms, res.corrs, lw=1.0)
    ax.axvline(res.best_lag_ms, color="#d62728", ls="--", lw=1,
               label=f"best {res.best_lag_ms:+.0f} ms  r={res.peak_correlation:+.2f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lag (ms)   [positive = IMU leads GPS]")
    ax.set_ylabel("normalised correlation")
    ax.set_title(f"{res.session}  ({res.role})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{res.session}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def is_suspect(lag_ms: float, corr: float) -> bool:
    if not np.isfinite(lag_ms) or not np.isfinite(corr):
        return True
    return abs(lag_ms) > SUSPECT_LAG_MS or abs(corr) < SUSPECT_CORR


def apply_time_offset(df: pd.DataFrame, session: str,
                      offsets_csv: Path | str = DISTANCE_OFFSETS_CSV) -> pd.DataFrame:
    """Shift a session's IMU clock onto its GPS clock, per session.

    Reads the measured lag for `session` from time_offsets.csv. Raises if the
    session has no stored offset -- there is deliberately no global fallback
    constant to silently apply instead.
    """
    path = Path(offsets_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m data.sanity` first.")
    table = pd.read_csv(path, comment="#")
    row = table[table["session"].str.lower() == session.lower()]
    if row.empty:
        raise KeyError(f"no stored time offset for {session!r}; "
                       "offsets are per session and are never defaulted")
    lag_ms = float(row.iloc[0]["best_lag_ms"])
    if not np.isfinite(lag_ms):
        raise ValueError(f"{session!r} has no usable lag "
                         f"({row.iloc[0].get('note', '')})")
    out = df.copy()
    if "time_s" not in out.columns:
        raise KeyError("frame has no time_s column; load it via load_session()")
    out["time_s_aligned"] = out["time_s"] - lag_ms / 1000.0
    return out


# --------------------------------------------------------------------------
# 3. trajectory invariants
# --------------------------------------------------------------------------

def _latlon(s: Session) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    lat_c, lon_c = _find(s.df, r"LATITUDE"), _find(s.df, r"LONGITUDE")
    if lat_c is None or lon_c is None:
        return None, None
    lat = pd.to_numeric(s.df[lat_c], errors="coerce")
    lon = pd.to_numeric(s.df[lon_c], errors="coerce")
    ok = lat.notna() & lon.notna() & (lat != 0) & (lon != 0)
    if ok.sum() < 10:
        return None, None
    return lat[ok], lon[ok]


def _step_metres(lat: pd.Series, lon: pd.Series) -> np.ndarray:
    dlat = np.radians(lat.diff()) * R_EARTH
    dlon = np.radians(lon.diff()) * R_EARTH * np.cos(np.radians(lat))
    return np.sqrt(dlat**2 + dlon**2).to_numpy()


def trajectory_checks(s: Session, sweep_distance_km: float | None) -> dict:
    """Four invariants a coordinate or projection bug would break."""
    name = f"{s.family}-{s.session}"
    row = {"session": name, "role": s.role, "group": s.group,
           "country_expected": _country_for(s.group)}
    lat, lon = _latlon(s)
    if lat is None:
        row.update(bbox_pass=False, distance_pass=False, jump_pass=False,
                   loop_pass=None, note="no usable coordinates")
        return row

    # -- (a) start and end inside the declared country box -------------------
    box = COUNTRY_BOXES[row["country_expected"]]
    lat_lo, lat_hi, lon_lo, lon_hi = box

    def inside(la: float, lo: float) -> bool:
        return lat_lo <= la <= lat_hi and lon_lo <= lo <= lon_hi

    start_ok = inside(float(lat.iloc[0]), float(lon.iloc[0]))
    end_ok = inside(float(lat.iloc[-1]), float(lon.iloc[-1]))
    row["bbox_pass"] = bool(start_ok and end_ok)
    row["start_lat"], row["start_lon"] = float(lat.iloc[0]), float(lon.iloc[0])
    row["end_lat"], row["end_lon"] = float(lat.iloc[-1]), float(lon.iloc[-1])

    # -- (b) path length agrees with the sweep -------------------------------
    steps = _step_metres(lat, lon)
    path_km = float(np.nansum(steps) / 1000.0)
    row["path_km"] = path_km
    row["sweep_distance_km"] = sweep_distance_km
    if sweep_distance_km and np.isfinite(sweep_distance_km) and sweep_distance_km > 0:
        rel = abs(path_km - sweep_distance_km) / sweep_distance_km
        row["distance_rel_err"] = rel
        row["distance_pass"] = bool(rel <= DISTANCE_TOLERANCE)
    else:
        row["distance_rel_err"] = np.nan
        row["distance_pass"] = None

    # -- (c) no fix jump beyond what the reported speed permits --------------
    # Measured across FIX UPDATES, not rows. GPS is held between 1 Hz updates
    # while rows tick at 10 Hz, so a row-wise comparison charges a whole
    # second of displacement to a 0.1 s interval and flags every session.
    if "speed_ms" in s.df.columns and "time_s" in s.df.columns:
        moved = (lat.diff() != 0) | (lon.diff() != 0)
        moved.iloc[0] = True
        lat_u, lon_u = lat[moved], lon[moved]
        steps_u = _step_metres(lat_u, lon_u)
        t = pd.to_numeric(s.df["time_s"], errors="coerce").reindex(lat_u.index)
        v = pd.to_numeric(s.df["speed_ms"], errors="coerce").reindex(lat_u.index)
        dt = t.diff().to_numpy()
        steps = steps_u
        allowed = np.maximum(np.nan_to_num(v.to_numpy()) * JUMP_SPEED_MARGIN,
                             JUMP_FLOOR_MS) * np.nan_to_num(dt)
        bad = np.isfinite(steps) & (allowed > 0) & (steps > allowed)
        row["n_jumps"] = int(bad.sum())
        row["max_jump_m"] = float(np.nanmax(steps)) if steps.size else np.nan
        row["jump_pass"] = bool(bad.sum() == 0)
    else:
        row["n_jumps"], row["max_jump_m"], row["jump_pass"] = 0, np.nan, None

    # -- (d) loop closure, only where the route actually returns -------------
    d_start = np.sqrt(
        (np.radians(lat - lat.iloc[0]) * R_EARTH) ** 2
        + (np.radians(lon - lon.iloc[0]) * R_EARTH
           * np.cos(np.radians(lat))) ** 2)
    end_gap = float(d_start.iloc[-1])
    row["end_to_start_m"] = end_gap
    # Only meaningful if the route comes back near its origin at the end.
    if end_gap <= LOOP_CLOSE_M:
        row["loop_pass"] = bool(end_gap <= LOOP_TOLERANCE_M)
    else:
        row["loop_pass"] = None      # not a loop; invariant does not apply
    row["note"] = ""
    return row


def plot_trajectory(s: Session, out_dir: Path) -> Path | None:
    lat, lon = _latlon(s)
    if lat is None:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(lon, lat, lw=0.6, color="#1f77b4")
    ax.scatter([lon.iloc[0]], [lat.iloc[0]], s=40, c="green", zorder=3, label="start")
    ax.scatter([lon.iloc[-1]], [lat.iloc[-1]], s=40, c="red", zorder=3, label="end")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(f"{s.family}-{s.session}  ({s.role})")
    ax.set_aspect(1.0 / max(np.cos(np.radians(float(lat.mean()))), 1e-6))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{s.family}-{s.session}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 4. the one human eye check: V-Vw12 against its JPG route map
# --------------------------------------------------------------------------

def vw12_jpg_comparison(data_root: Path = DATA_ROOT) -> Path | None:
    """Side-by-side of the plotted V-Vw12 trajectory and the supplied JPG.

    JPGs exist only for train-role and stage0_only sessions, so this is the one
    place a real map can be eyeballed. V-Vw12 is the Stage 0 known-answer
    session, which makes it the right one to check.
    """
    jpgs = [p for p in data_root.rglob("*.JPG") if p.stem.lower() == "v-vw12"]
    if not jpgs:
        return None
    s = load_session("V-Vw12", data_root)
    lat, lon = _latlon(s)
    if lat is None:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
    ax1.plot(lon, lat, lw=0.9, color="#1f77b4")
    ax1.scatter([lon.iloc[0]], [lat.iloc[0]], s=45, c="green", zorder=3)
    ax1.scatter([lon.iloc[-1]], [lat.iloc[-1]], s=45, c="red", zorder=3)
    ax1.set_aspect(1.0 / max(np.cos(np.radians(float(lat.mean()))), 1e-6))
    ax1.set_title("V-Vw12 plotted from coordinates")
    ax1.set_xlabel("longitude"); ax1.set_ylabel("latitude")
    ax1.grid(alpha=0.3)
    ax2.imshow(mpimg.imread(jpgs[0]))
    ax2.set_title(f"supplied route map ({jpgs[0].name})")
    ax2.axis("off")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "vw12_jpg_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _sweep_distances() -> dict[str, float]:
    if not SWEEP_CSV.exists():
        return {}
    d = pd.read_csv(SWEEP_CSV)
    return {f"{r.family}-{r.session}": r.distance_km for r in d.itertuples()}


def _pass_counts(df: pd.DataFrame, col: str) -> tuple[int, int, int]:
    passed = int((df[col] == True).sum())    # noqa: E712 - None must not count
    failed = int((df[col] == False).sum())   # noqa: E712
    na = int(len(df) - passed - failed)
    return passed, failed, na


def write_summary(lags: pd.DataFrame, checks: pd.DataFrame,
                  jpg_path: Path | None) -> None:
    lines: list[str] = []
    w = lines.append
    w("# Sanity check summary\n")
    w(f"Sessions processed: **{len(checks)}**. Report only — nothing was "
      "repaired and the split was not changed.\n")

    w("\n## Trajectory invariants\n")
    w("| Invariant | pass | fail | n/a |")
    w("|---|---|---|---|")
    for col, label in (("bbox_pass", "start/end inside country box"),
                       ("distance_pass", "path length vs sweep (±5%)"),
                       ("jump_pass", "no fix jump beyond reported speed"),
                       ("loop_pass", "loop closes (where a loop)")):
        p, f, n = _pass_counts(checks, col)
        w(f"| {label} | {p} | {f} | {n} |")
    w("\n`n/a` means the invariant does not apply (e.g. a route that never "
      "returns near its origin) or the input was missing.\n")

    for col, label in (("bbox_pass", "outside the expected country box"),
                       ("distance_pass", "path length disagreeing with the sweep"),
                       ("jump_pass", "containing an implausible fix jump"),
                       ("loop_pass", "failing loop closure")):
        bad = checks[checks[col] == False]   # noqa: E712
        if bad.empty:
            continue
        w(f"\n### Sessions {label} ({len(bad)})\n")
        cols = ["session", "role", "group"]
        extra = {"bbox_pass": ["start_lat", "start_lon", "end_lat", "end_lon"],
                 "distance_pass": ["path_km", "sweep_distance_km", "distance_rel_err"],
                 "jump_pass": ["n_jumps", "max_jump_m"],
                 "loop_pass": ["end_to_start_m"]}[col]
        w("```")
        w(bad[cols + extra].to_string(index=False))
        w("```")

    w("\n## Time synchronisation\n")
    ok = lags[lags["peak_correlation"].notna()]
    w(f"Lag estimated for **{len(ok)}/{len(lags)}** sessions. "
      "Offsets are stored per session in `time_offsets.csv`; there is no "
      "global constant.\n")
    if not ok.empty:
        w("\n### Lag distribution by group\n")
        w("| group | n | median lag (ms) | p05 | p95 | median peak r |")
        w("|---|---|---|---|---|---|")
        for g, gs in ok.groupby("group"):
            w(f"| {g} | {len(gs)} | {gs.best_lag_ms.median():.0f} | "
              f"{gs.best_lag_ms.quantile(.05):.0f} | "
              f"{gs.best_lag_ms.quantile(.95):.0f} | "
              f"{gs.peak_correlation.median():.2f} |")
        w("\n### Lag distribution by role\n")
        w("| role | n | median lag (ms) | median peak r |")
        w("|---|---|---|---|")
        for r, rs in ok.groupby("role"):
            w(f"| {r} | {len(rs)} | {rs.best_lag_ms.median():.0f} | "
              f"{rs.peak_correlation.median():.2f} |")

    susp = lags[lags["suspect"] == True]      # noqa: E712
    w(f"\n## Suspect sessions ({len(susp)})\n")
    w(f"Flagged when `|best_lag_ms| > {SUSPECT_LAG_MS:.0f}` or "
      f"`|peak_correlation| < {SUSPECT_CORR}`. A weak peak means the "
      "cross-correlation never locked onto real structure — that is a finding "
      "about the session, not a tuning problem.\n")
    if susp.empty:
        w("None.\n")
    else:
        w("```")
        w(susp[["session", "role", "group", "best_lag_ms", "peak_correlation",
                "multi_axis_R", "n_samples_used", "channel"]].to_string(index=False))
        w("```")

    w("\n### Single-channel vs multi-axis correlation\n")
    w("`multi_axis_R` is a least-squares fit of ALL accelerometer axes to "
      "d(speed)/dt at the best lag — mounting-invariant, so a weak single-axis "
      "peak cannot simply be blamed on picking the wrong axis.\n")
    if "multi_axis_R" in lags.columns:
        m = lags.dropna(subset=["multi_axis_R"])
        if not m.empty:
            w("\n| family | n | median peak r (single) | median R (multi-axis) |")
            w("|---|---|---|---|")
            for fam, fs in m.groupby(m["session"].str[0]):
                w(f"| {fam}- | {len(fs)} | {fs.peak_correlation.abs().median():.2f} "
                  f"| {fs.multi_axis_R.abs().median():.2f} |")

    w("\n## Eye check\n")
    w(f"`{jpg_path.name}` — V-Vw12 plotted trajectory beside its supplied route "
      "map.\n" if jpg_path else "V-Vw12 JPG not found; no eye check produced.\n")
    w("\nJPGs exist only for train-role and `stage0_only` sessions — none for "
      "groups T, A or I — so this is the only session where the plotted route "
      "can be compared against a real map.\n")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    sessions = list_sessions()
    if argv:
        wanted = {a.lower() for a in argv}
        sessions = sessions[sessions.apply(
            lambda r: f"{r.family}-{r.session}".lower() in wanted, axis=1)]
        if sessions.empty:
            print(f"no sessions matching {argv}", file=sys.stderr)
            return 1

    distances = _sweep_distances()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lag_rows, check_rows, dist_rows = [], [], []

    for i, r in enumerate(sessions.itertuples(), 1):
        name = f"{r.family}-{r.session}"
        print(f"\r  [{i}/{len(sessions)}] {name:<20}", end="", file=sys.stderr)
        try:
            # check_rate=False: this pass diagnoses, it does not gate.
            s = load_session(name, check_rate=False)
        except Exception as exc:
            lag_rows.append({"session": name, "role": r.role, "group": r.group,
                             "best_lag_ms": np.nan, "peak_correlation": np.nan,
                             "n_samples_used": 0, "suspect": True,
                             "note": f"load failed: {type(exc).__name__}"})
            check_rows.append({"session": name, "role": r.role, "group": r.group,
                               "bbox_pass": False, "distance_pass": False,
                               "jump_pass": False, "loop_pass": None,
                               "note": f"load failed: {type(exc).__name__}"})
            dist_rows.append({"session": name, "role": r.role, "group": r.group,
                              "best_lag_ms": np.nan, "residual": np.nan,
                              "n_samples_used": 0, "method": "distance",
                              "note": f"load failed: {type(exc).__name__}"})
            continue

        plot_signals(s, OUT_ROOT / "signals")
        res = estimate_lag_levelled(s)
        plot_xcorr(res, OUT_ROOT / "xcorr")
        multi_r, multi_lag = multi_axis_correlation(s)
        lag_rows.append({
            "session": res.session, "role": res.role, "group": s.group,
            "best_lag_ms": res.best_lag_ms,
            "peak_correlation": res.peak_correlation,
            "n_samples_used": res.n_samples_used,
            "suspect": is_suspect(res.best_lag_ms, res.peak_correlation),
            "multi_axis_R": multi_r,
            "multi_axis_lag_ms": multi_lag,
            "best_yaw_rad": res.best_yaw_rad,
            "yaw_drift_rad": res.yaw_drift_rad,
            "yaw_thirds": ";".join(f"{v:.2f}" for v in res.yaw_thirds),
            "channel": res.note,
            "note": "",
        })
        dist_rows.append(estimate_offset_by_distance(s))
        check_rows.append(trajectory_checks(s, distances.get(name)))
        plot_trajectory(s, OUT_ROOT / "trajectory")
    print(file=sys.stderr)

    lags = pd.DataFrame(lag_rows)
    checks = pd.DataFrame(check_rows)
    dist = pd.DataFrame(dist_rows)

    # The acceleration-domain result is retained for the record but is
    # reference-limited and superseded; the header says so in the file itself,
    # so a reader who opens the CSV without the summary cannot miss it.
    lags["method"] = "accel_correlation"
    with open(TIME_OFFSETS_CSV, "w", encoding="utf-8") as fh:
        fh.write("# SUPERSEDED — reference-limited. Lags here come from "
                 "correlating IMU acceleration against d(GPS speed)/dt.\n")
        fh.write("# Differentiating 1 Hz quantised GPS speed caps the achievable "
                 "correlation: even a perfect ECU\n")
        fh.write("# accelerometer reaches only r = 0.12-0.58 against this "
                 "reference, versus 0.93 against ECU speed.\n")
        fh.write("# For S- files |r| is ~0.29, so the lag argmax is noise. Use "
                 "time_offsets_distance.csv instead.\n")
        lags.to_csv(fh, index=False)
    with open(DISTANCE_OFFSETS_CSV, "w", encoding="utf-8") as fh:
        fh.write("# GROSS MISALIGNMENT ONLY — sub-second output is UNRESOLVED.\n")
        fh.write("# The objective matches cumulative distance curves, which are "
                 "monotone and very smooth, so a\n")
        fh.write("# shift of a few seconds barely perturbs a curve spanning "
                 "thousands. The objective is near-flat\n")
        fh.write("# under small shifts and its argmin is noise at that scale. "
                 "An argmin sitting AT the search\n")
        fh.write(f"# boundary (+/-{DISTANCE_MAX_LAG_S*1000:.0f} ms) means "
                 "NON-CONVERGENCE, not a large measured offset.\n")
        fh.write(f"# Only |best_lag_ms| > {GROSS_LAG_MS:.0f} is worth review, "
                 "and only where the argmin is interior.\n")
        dist.to_csv(fh, index=False)
    checks.to_csv(TRAJ_CHECKS_CSV, index=False)
    jpg = vw12_jpg_comparison()
    write_summary(lags, checks, jpg)

    # Exclude boundary argmins: those are non-convergence, not misalignment.
    bound = DISTANCE_MAX_LAG_S * 1000.0
    gross = (dist[(dist["best_lag_ms"].abs() > GROSS_LAG_MS)
                  & (dist["best_lag_ms"].abs() < bound - 1e-9)]
             if not dist.empty else dist)
    if not gross.empty:
        print(f"\ngross misalignment (|lag| > {GROSS_LAG_MS:.0f} ms), "
              f"{len(gross)} session(s) for individual review:")
        print(gross[["session", "role", "group", "best_lag_ms", "residual"]]
              .to_string(index=False))
    print(f"\nwrote {TIME_OFFSETS_CSV.relative_to(REPO_ROOT)}, "
          f"{DISTANCE_OFFSETS_CSV.relative_to(REPO_ROOT)}, "
          f"{TRAJ_CHECKS_CSV.relative_to(REPO_ROOT)}, "
          f"{SUMMARY_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
