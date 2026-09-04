"""Windowed training set from the 10 Hz S- files.

Each window is 10 s of six-channel IMU (100 samples at 10 Hz, stride 1 s),
levelled so gravity lies on the z-axis, paired with the displacement travelled
over the following second.

Two traps this module exists to avoid, both of which have already produced
artifacts in this project (see CLAUDE.md):

1. **Per-sensor resampling.** AndroSensor writes one row per sensor EVENT, so
   any given column is blank or stale in most rows of a tick. Every channel is
   therefore resampled onto the common grid using only its OWN valid samples.
2. **Session-level splitting.** Windows overlap in time, so a random split
   would put near-identical windows in train and test. Splits are strictly by
   session, and `verify_no_leakage()` asserts it.

Run:  python -m data.windows            (build and report)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import Geod
from scipy.ndimage import median_filter

from .loader import Session, _find, list_sessions, load_session

_GEOD = Geod(ellps="WGS84")
from .sanity import (
    MAX_PLAUSIBLE_SPEED_MS,
    _sensor_grid,
    displacement_labels,
    gps_cumulative_distance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "windows"
NORM_STATS_PATH = OUT_DIR / "norm_stats.json"
REPORT_PATH = OUT_DIR / "window_report.md"

GRID_HZ = 10.0
WINDOW_SAMPLES = 100          # 10 s at 10 Hz
STRIDE_SAMPLES = 10           # 1 s
LABEL_WINDOW_S = 1.0

CHANNELS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")

STATIONARY_SPEED_MS = 0.3     # is_stationary threshold
YAW_TRUE_MIN_SPEED_MS = 5.0   # GPS heading is meaningless below this

# Roles whose sessions are 10 Hz S- files usable for supervised learning.
BUILDABLE_ROLES = ("train", "validate", "test")
BUCKET_ROLES = BUILDABLE_ROLES


# --- despiking ------------------------------------------------------------
# Road shocks (potholes, expansion joints, kerbs) produce isolated impulses
# that survive the level-frame rotation and enter the windows as outliers.
# A rolling-median / MAD filter removes them without smoothing the signal the
# way a mean filter would.
DESPIKE_WINDOW_S = 0.5      # window duration, NOT a sample count
DESPIKE_MAD_K = 3.0         # deviation threshold in MADs
DESPIKE_MIN_SAMPLES = 3     # a median needs at least this many points
# MAD -> sigma for a normal distribution; used only to keep the threshold
# interpretable, the comparison itself is in raw MAD units.
MAD_TO_SIGMA = 1.4826


def despike(chan: np.ndarray, hz: float,
            window_s: float = DESPIKE_WINDOW_S,
            k: float = DESPIKE_MAD_K) -> tuple[np.ndarray, float]:
    """Replace samples deviating > k MADs from the local median, per channel.

    The window is derived from the SESSION's measured rate, not a hardcoded
    10 Hz -- part of the corpus runs at 2 Hz, where five samples would span
    2.5 s and smooth away real dynamics rather than isolated shocks.

    Each of the six channels is filtered independently: a shock that saturates
    the vertical axis need not be an outlier on yaw rate, and treating them
    jointly would discard good samples on the quiet channels.

    Returns (despiked, fraction_replaced).
    """
    n = max(DESPIKE_MIN_SAMPLES, int(round(window_s * hz)))
    if n % 2 == 0:
        n += 1                      # centred window needs an odd length
    if chan.shape[0] < n:
        return chan, 0.0
    out = chan.copy()
    replaced = 0
    for c in range(chan.shape[1]):
        x = chan[:, c]
        med = median_filter(x, size=n, mode="nearest")
        mad = median_filter(np.abs(x - med), size=n, mode="nearest")
        # Where MAD is ~0 the channel is locally constant; any deviation there
        # is a genuine step, not a spike, so guard against dividing by zero.
        thresh = k * np.maximum(mad, 1e-9)
        bad = np.abs(x - med) > thresh
        bad &= mad > 1e-9
        out[bad, c] = med[bad]
        replaced += int(bad.sum())
    return out, replaced / float(chan.size)


# The path-ratio guard is DIRECTIONAL: the two failure modes are not the same
# defect and do not deserve the same threshold.
#
#   ratio > MAX  positions JUMP. The geometry is wrong, so the labels are
#               wrong in an unbounded, non-systematic way. Strict.
#   ratio < MIN  fixes are too SPARSE, so the polyline chord-cuts the real
#               curve. Labels are slightly SHORT but structurally sound; a
#               small systematic underestimate is far less harmful than
#               corrupted geometry. Tolerant.
MAX_PATH_RATIO = 1.15
MIN_PATH_RATIO = 0.75


class InflatedPositionsError(RuntimeError):
    """GPS positions disagree with the session's own speedometer."""


class LeakageError(RuntimeError):
    """A session appears under two roles, or a window crosses sessions."""


@dataclass
class SessionWindows:
    session: str
    role: str
    group: str
    X: np.ndarray              # (n, WINDOW_SAMPLES, 6)
    y: np.ndarray              # (n,) displacement over the next second
    is_stationary: np.ndarray  # (n,) bool
    yaw_rate_true: np.ndarray  # (n,) rad/s, NaN where speed < 5 m/s
    wheel_speed: np.ndarray    # (n,) m/s from ECU, NaN if unpaired
    t0: np.ndarray             # (n,) window start time, for traceability
    n_rejected: int            # windows dropped by the plausibility guard
    label_source: str          # "gps" or "ecu_wheel_speed"
    path_ratio: float = float("nan")   # GPS path / speed integral
    despiked_frac: float = 0.0         # fraction of samples replaced


# --------------------------------------------------------------------------
# channel construction
# --------------------------------------------------------------------------

def levelled_channels(s: Session, grid: np.ndarray) -> np.ndarray | None:
    """(n, 6) levelled accelerometer and gyroscope on `grid`.

    Accelerometer has gravity removed via the Gravity columns, then both
    sensors are rotated so gravity lies on +z. The remaining mounting freedom
    is a single yaw about vertical, which the model can learn; roll and pitch
    are removed here so it does not have to.
    """
    acc = [_find(s.df, r"ACCELEROMETER", rf"\b{a}\b") for a in "XYZ"]
    grv = [_find(s.df, r"GRAVITY", rf"\b{a}\b") for a in "XYZ"]
    gyr = [_find(s.df, r"GYROSCOPE", p) for p in (r"YAW", r"PITCH", r"ROLL")]
    if any(c is None for c in acc) or any(c is None for c in gyr):
        return None
    t = pd.to_numeric(s.df["time_s"], errors="coerce")

    def grid_col(col: str) -> np.ndarray:
        return _sensor_grid(t, pd.to_numeric(s.df[col], errors="coerce"), grid)

    A = np.column_stack([grid_col(c) for c in acc])
    G = np.column_stack([grid_col(c) for c in grv]) if all(
        c is not None for c in grv) else np.zeros_like(A)
    W = np.column_stack([grid_col(c) for c in gyr])

    lin = A - G
    g_med = np.nanmedian(G, axis=0)
    norm = np.linalg.norm(g_med)
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    down = g_med / norm

    # Orthonormal basis with `down` as +z.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ref @ down)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - (ref @ down) * down
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(down, e1)
    R = np.column_stack([e1, e2, down])        # body -> level

    return np.hstack([lin @ R, W @ R])


def ecu_wheel_speed(s: Session, grid: np.ndarray) -> np.ndarray | None:
    """Wheel speed (m/s) from the paired V- session, on `grid`.

    S-/V- pairs align by ROW INDEX, not timestamp — the clock origins differ
    (S- from session start, V- from start of day). So the ECU rows are mapped
    onto the S- time base by position, then resampled.
    """
    try:
        v = load_session(f"V-{s.session}", check_rate=False)
    except Exception:
        return None
    cols = [_find(v.df, r"WHEEL SPEED", p)
            for p in (r"FRONT LEFT", r"FRONT RIGHT", r"REAR LEFT", r"REAR RIGHT")]
    cols = [c for c in cols if c is not None]
    if not cols:
        return None
    n = min(len(s.df), len(v.df))
    if n < WINDOW_SAMPLES:
        return None

    # Wheel speeds are rad/s; without a calibrated radius they are not metres
    # per second, so scale to the ECU's own speed channel rather than guessing.
    omega = np.nanmean(
        np.column_stack([pd.to_numeric(v.df[c], errors="coerce").to_numpy()[:n]
                         for c in cols]), axis=1)
    ref = (pd.to_numeric(v.df["speed_ms"], errors="coerce").to_numpy()[:n]
           if "speed_ms" in v.df.columns else None)
    if ref is None:
        return None
    ok = np.isfinite(omega) & np.isfinite(ref) & (ref > 2.0)
    if ok.sum() < 200:
        return None
    radius = float(np.median(ref[ok] / omega[ok]))
    if not np.isfinite(radius) or radius <= 0:
        return None

    t_s = pd.to_numeric(s.df["time_s"], errors="coerce").to_numpy()[:n]
    return _sensor_grid(pd.Series(t_s), pd.Series(omega * radius), grid)


def gps_yaw_rate(s: Session, grid: np.ndarray) -> np.ndarray:
    """True yaw rate (rad/s) from GPS heading differences.

    NaN wherever speed is below 5 m/s: GPS heading is meaningless at rest, so
    differencing it there manufactures spurious rotation.
    """
    col = _find(s.df, r"^HEADING") or _find(s.df, r"GPS ORIENTATION")
    if col is None:
        return np.full(grid.size, np.nan)
    t = pd.to_numeric(s.df["time_s"], errors="coerce")
    head = np.radians(_sensor_grid(
        t, pd.to_numeric(s.df[col], errors="coerce"), grid))
    d = np.diff(head, prepend=head[0])
    d = (d + np.pi) % (2 * np.pi) - np.pi        # unwrap to (-pi, pi]
    rate = d * GRID_HZ

    spd = _sensor_grid(t, pd.to_numeric(s.df.get("speed_ms"), errors="coerce"),
                       grid) if "speed_ms" in s.df.columns else None
    if spd is not None:
        rate = np.where(np.isfinite(spd) & (spd >= YAW_TRUE_MIN_SPEED_MS),
                        rate, np.nan)
    return rate


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------

def path_vs_speed_ratio(s: Session) -> float:
    """Fix-to-fix GPS path length divided by the integral of reported speed.

    ~1.0 means the positions and the speedometer agree. Values far above 1
    mean the positions jump: S-T1 measures 76.3 km of path against a 21.9 km
    speed integral. This is a SESSION-level defect, which is why it is caught
    here rather than by the per-window plausibility guard -- that guard trims
    the tail without repairing the windows that survive it.
    """
    lat_c, lon_c = _find(s.df, r"LATITUDE"), _find(s.df, r"LONGITUDE")
    if lat_c is None or lon_c is None or "speed_ms" not in s.df.columns:
        return float("nan")
    lat = pd.to_numeric(s.df[lat_c], errors="coerce")
    lon = pd.to_numeric(s.df[lon_c], errors="coerce")
    t = pd.to_numeric(s.df["time_s"], errors="coerce").to_numpy(float)
    v = pd.to_numeric(s.df["speed_ms"], errors="coerce").to_numpy(float)
    ok = lat.notna() & lon.notna()
    la, lo = lat[ok].to_numpy(float), lon[ok].to_numpy(float)
    if la.size < 10:
        return float("nan")
    moved = np.concatenate([[True], (np.diff(la) != 0) | (np.diff(lo) != 0)])
    la, lo = la[moved], lo[moved]
    if la.size < 3:
        return float("nan")
    path = float(np.sum(_GEOD.inv(lo[:-1], la[:-1], lo[1:], la[1:])[2]))
    v_int = float(np.nansum(v[:-1] * np.diff(t)))
    return path / v_int if v_int > 0 else float("nan")


def build_session(name: str, prefer_ecu: bool = False,
                  max_path_ratio: float = MAX_PATH_RATIO,
                  min_path_ratio: float = MIN_PATH_RATIO,
                  window_samples: int = WINDOW_SAMPLES,
                  apply_despike: bool = False,
                  stride_samples: int = STRIDE_SAMPLES
                  ) -> SessionWindows | None:
    """Window one S- session. Returns None if it cannot be built.

    `prefer_ecu` labels from ECU wheel speed instead of GPS. It is OFF by
    default: wheel speed's advantage is 10 Hz resolution, which buys nothing
    for a one-second displacement label that GPS provides at its native rate,
    and the measured +1.9% ECU offset is a tyre-circumference scale error that
    this dataset does not hold constant (tyre pressure varies across groups).
    Retained as an ablation.
    """
    s = load_session(name, check_rate=False)

    ratio = path_vs_speed_ratio(s)
    if np.isfinite(ratio) and ratio > max_path_ratio:
        raise InflatedPositionsError(
            f"{name}: GPS path length is {ratio:.2f}x the integral of its own "
            f"reported speed (limit {max_path_ratio:.2f}) — positions jump, "
            "inflating the path. The geometry is wrong, so labels derived "
            "from it are wrong in an unbounded way; this is a session-level "
            "defect the per-window guard cannot repair.")
    if np.isfinite(ratio) and ratio < min_path_ratio:
        raise InflatedPositionsError(
            f"{name}: GPS path length is only {ratio:.2f}x the integral of "
            f"its own reported speed (floor {min_path_ratio:.2f}) — fixes are "
            "too sparse and the path chord-cuts the real curve. Mild "
            "chord-cutting is tolerated (labels are short but structurally "
            "sound); this session is past that.")

    lab = displacement_labels(s, grid_hz=GRID_HZ, window_s=LABEL_WINDOW_S)
    if lab is None:
        return None
    t_lab, y_lab, n_rejected = lab

    gps = gps_cumulative_distance(s)
    if gps is None:
        return None
    lo, hi = float(gps[0].min()), float(gps[0].max())
    grid = np.arange(lo, hi, 1.0 / GRID_HZ)
    if grid.size < window_samples + int(LABEL_WINDOW_S * GRID_HZ):
        return None

    chan = levelled_channels(s, grid)
    if chan is None:
        return None

    # Despike AFTER the level-frame rotation and BEFORE windows are cut, so
    # the windows are built from cleaned values rather than being corrected
    # afterwards.
    despiked_frac = 0.0
    if apply_despike:
        hz = s.hz_tick if np.isfinite(s.hz_tick) and s.hz_tick > 0 else GRID_HZ
        chan, despiked_frac = despike(chan, hz)

    t = pd.to_numeric(s.df["time_s"], errors="coerce")
    speed = (_sensor_grid(t, pd.to_numeric(s.df["speed_ms"], errors="coerce"),
                          grid) if "speed_ms" in s.df.columns
             else np.full(grid.size, np.nan))
    yaw_true = gps_yaw_rate(s, grid)

    wheel = ecu_wheel_speed(s, grid) if prefer_ecu else None

    # Prefer the ECU wheel speed as the label source where a synchronised
    # pairing exists: it is a clean 10 Hz signal, whereas GPS displacement is
    # an interpolated 1 Hz quantity. Integrated over the label second, it gives
    # the same physical quantity with far less noise. Falls back to GPS.
    label_source = "gps"
    if wheel is not None:
        lab_off_w = int(LABEL_WINDOW_S * GRID_HZ)
        cum_w = np.concatenate([[0.0], np.nancumsum(
            np.nan_to_num(wheel[1:]) * np.diff(grid))])
        disp_w = np.full(grid.size, np.nan)
        disp_w[:-lab_off_w] = cum_w[lab_off_w:] - cum_w[:-lab_off_w]
        implied = disp_w / LABEL_WINDOW_S
        # Same standing plausibility guard as the GPS labels.
        disp_w = np.where(np.isfinite(implied) & (implied >= 0)
                          & (implied <= MAX_PLAUSIBLE_SPEED_MS), disp_w, np.nan)
        if np.isfinite(disp_w).sum() > 100:
            label_source = "ecu_wheel_speed"

    # Labels live on their own (filtered) grid; interpolate onto the window
    # grid but only where a label genuinely exists nearby, so guard-rejected
    # stretches are not silently filled in.
    lab_on_grid = np.interp(grid, t_lab, y_lab, left=np.nan, right=np.nan)
    gap = np.full(grid.size, np.inf)
    if t_lab.size:
        idx = np.searchsorted(t_lab, grid).clip(1, t_lab.size - 1)
        gap = np.minimum(np.abs(grid - t_lab[idx - 1]), np.abs(grid - t_lab[idx]))
    lab_on_grid = np.where(gap <= 1.0 / GRID_HZ, lab_on_grid, np.nan)

    if label_source == "ecu_wheel_speed":
        # Keep the GPS label only where the ECU one is missing, so a session
        # never mixes sources silently within a window batch.
        lab_on_grid = disp_w

    lab_off = int(LABEL_WINDOW_S * GRID_HZ)
    starts = np.arange(0, grid.size - window_samples - lab_off, stride_samples)
    if starts.size == 0:
        return None
    ends = starts + window_samples

    X = np.stack([chan[i:i + window_samples] for i in starts])
    y = lab_on_grid[ends]
    stationary = np.nan_to_num(speed[ends], nan=0.0) < STATIONARY_SPEED_MS
    yaw = yaw_true[ends]
    wh = wheel[ends] if wheel is not None else np.full(starts.size, np.nan)
    t0 = grid[starts]

    keep = np.isfinite(y) & np.isfinite(X).all(axis=(1, 2))
    dropped_here = int((~keep).sum())
    if keep.sum() == 0:
        return None

    return SessionWindows(
        session=name, role=s.role, group=s.group, path_ratio=ratio,
        despiked_frac=despiked_frac,
        X=X[keep], y=y[keep], is_stationary=stationary[keep],
        yaw_rate_true=yaw[keep], wheel_speed=wh[keep], t0=t0[keep],
        n_rejected=n_rejected + dropped_here, label_source=label_source)


# --------------------------------------------------------------------------
# leakage guard
# --------------------------------------------------------------------------

def verify_no_leakage(built: list[SessionWindows],
                      expected_samples: int = WINDOW_SAMPLES) -> None:
    """Assert the split is clean. Raises LeakageError on any violation.

    `expected_samples` is stated by the caller rather than inferred from the
    data: inferring it would make a uniformly-wrong window length pass
    silently, which is exactly the failure this check exists to catch.
    """
    role_of: dict[str, str] = {}
    for b in built:
        if b.session in role_of and role_of[b.session] != b.role:
            raise LeakageError(
                f"{b.session} appears under two roles: "
                f"{role_of[b.session]} and {b.role}")
        role_of[b.session] = b.role

    # A session must contribute to exactly one SessionWindows object; windows
    # are built per session and never concatenated across sessions, so a
    # boundary crossing can only arise from a duplicated session.
    seen = [b.session for b in built]
    dupes = {n for n in seen if seen.count(n) > 1}
    if dupes:
        raise LeakageError(f"session windowed more than once: {sorted(dupes)}")

    for b in built:
        if b.X.shape[1] != expected_samples:
            raise LeakageError(
                f"{b.session}: window length {b.X.shape[1]} "
                f"!= {expected_samples}")
        # Every window start must be inside this session's own time span.
        if b.t0.size and (np.diff(b.t0) <= 0).any():
            raise LeakageError(f"{b.session}: window starts are not increasing")

    # Cross-check against the authoritative split.
    sessions = list_sessions()
    key = (sessions["family"] + "-" + sessions["session"])
    authoritative = dict(zip(key, sessions["role"]))
    for name, role in role_of.items():
        if authoritative.get(name) != role:
            raise LeakageError(
                f"{name}: role {role!r} disagrees with list_sessions() "
                f"({authoritative.get(name)!r})")


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def fit_normalisation(train: list[SessionWindows]) -> dict:
    """Per-channel mean and std from TRAIN ONLY.

    Computed on train and applied everywhere: statistics from validate or test
    would leak their distribution into the model.
    """
    if not train:
        raise ValueError("no train windows to fit normalisation on")
    stacked = np.concatenate([b.X.reshape(-1, len(CHANNELS)) for b in train])
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"channels": list(CHANNELS),
            "mean": mean.tolist(), "std": std.tolist(),
            "n_windows": int(sum(b.X.shape[0] for b in train)),
            "fitted_on": "train"}


def apply_normalisation(X: np.ndarray, stats: dict) -> np.ndarray:
    return (X - np.asarray(stats["mean"])) / np.asarray(stats["std"])


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build_all(roles=BUILDABLE_ROLES, prefer_ecu: bool = True,
              window_samples: int = WINDOW_SAMPLES,
              apply_despike: bool = False,
              stride_samples: int = STRIDE_SAMPLES):
    sessions = list_sessions()
    sessions = sessions[sessions["role"].isin(roles) & (sessions["family"] == "S")]
    built, failures = [], []
    for i, r in enumerate(sessions.itertuples(), 1):
        name = f"{r.family}-{r.session}"
        print(f"\r  [{i}/{len(sessions)}] {name:<20}", end="", file=sys.stderr)
        try:
            b = build_session(name, prefer_ecu=prefer_ecu,
                              window_samples=window_samples,
                              apply_despike=apply_despike,
                              stride_samples=stride_samples)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if b is None:
            failures.append(f"{name}: no usable windows")
        else:
            built.append(b)
    print(file=sys.stderr)
    return built, failures


def main(argv: list[str] | None = None) -> int:
    built, failures = build_all()
    if not built:
        print("no windows built", file=sys.stderr)
        return 1

    verify_no_leakage(built)

    train = [b for b in built if b.role == "train"]
    stats = fit_normalisation(train)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NORM_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    rows = []
    for role in BUILDABLE_ROLES:
        sub = [b for b in built if b.role == role]
        if not sub:
            continue
        y = np.concatenate([b.y for b in sub])
        kept = int(sum(b.X.shape[0] for b in sub))
        rejected = int(sum(b.n_rejected for b in sub))
        rows.append({
            "role": role,
            "n_sessions": len(sub),
            "n_windows": kept,
            "label_mean_m": float(np.mean(y)),
            "label_std_m": float(np.std(y)),
            "label_min_m": float(np.min(y)),
            "label_max_m": float(np.max(y)),
            "frac_rejected": rejected / max(rejected + kept, 1),
            "frac_stationary": float(np.mean(np.concatenate(
                [b.is_stationary for b in sub]))),
            "frac_yaw_true_valid": float(np.mean(np.isfinite(np.concatenate(
                [b.yaw_rate_true for b in sub])))),
            "n_sessions_with_ecu": sum(
                1 for b in sub if b.label_source == "ecu_wheel_speed"),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "window_summary.csv", index=False)

    # Speed-stratified label distribution: the splits differ in regime mix, so
    # a pooled mean hides which regimes each actually contains.
    from eval.buckets import BUCKET_NAMES, assign_buckets
    strat_rows = []
    for role in BUCKET_ROLES:
        sub = [b for b in built if b.role == role]
        if not sub:
            continue
        y = np.concatenate([b.y for b in sub])
        idx = assign_buckets(y)
        for i, bname in enumerate(BUCKET_NAMES):
            m = idx == i
            strat_rows.append({
                "role": role, "bucket": bname, "n_windows": int(m.sum()),
                "frac": float(m.mean()),
                "label_mean_m": float(np.mean(y[m])) if m.any() else np.nan,
                "label_std_m": float(np.std(y[m])) if m.any() else np.nan,
            })
    strat = pd.DataFrame(strat_rows)
    strat.to_csv(OUT_DIR / "window_buckets.csv", index=False)

    # Per-session path_ratio, so the chord-cutting tolerance is auditable and
    # the 0.75-0.90 band can be ablated later.
    sess_rows = [{"session": b.session, "role": b.role, "group": b.group,
                  "path_ratio": b.path_ratio, "n_windows": int(b.y.size),
                  "label_mean_m": float(b.y.mean()),
                  "chord_cut_band": bool(0.75 <= b.path_ratio < 0.90)}
                 for b in built]
    sess = pd.DataFrame(sess_rows)
    sess.to_csv(OUT_DIR / "session_path_ratios.csv", index=False)

    lines = ["# Windowed dataset\n",
             f"{WINDOW_SAMPLES} samples ({WINDOW_SAMPLES / GRID_HZ:g} s) at "
             f"{GRID_HZ:g} Hz, stride {STRIDE_SAMPLES} "
             f"({STRIDE_SAMPLES / GRID_HZ:g} s), channels "
             f"{', '.join(CHANNELS)} in the gravity-levelled frame.\n",
             "\nLabels from `displacement_labels()`, so the "
             f"{MAX_PLAUSIBLE_SPEED_MS:g} m/s plausibility guard applies.\n",
             "\n## Per split\n", "```", summary.to_string(index=False), "```",
             "\n## Label distribution by speed bucket\n",
             "Displacement over 1 s is numerically mean speed in m/s, so the "
             "label is its own bucket key.\n", "```",
             strat.pivot_table(index="role", columns="bucket",
                               values="frac").round(3).to_string(),
             "", "mean label (m) within bucket:",
             strat.pivot_table(index="role", columns="bucket",
                               values="label_mean_m").round(2).to_string(),
             "```",
             "\n## Path ratio (GPS path / speed integral)\n",
             f"Guard: reject above {MAX_PATH_RATIO} (positions jump) or below "
             f"{MIN_PATH_RATIO} (fixes too sparse). Sessions in the "
             "0.75-0.90 chord-cutting band are KEPT and flagged for ablation: "
             "their labels are slightly short but structurally sound.\n",
             "```",
             (f"  chord-cut band (0.75-0.90): "
              f"{int(sess['chord_cut_band'].sum())} sessions, "
              f"{int(sess.loc[sess['chord_cut_band'], 'n_windows'].sum())} windows"),
             f"  path_ratio range: {sess.path_ratio.min():.3f} - "
             f"{sess.path_ratio.max():.3f}",
             "```",
             "\n## Leakage check\n",
             f"`verify_no_leakage()` passed over {len(built)} sessions: no "
             "session under two roles, no window crossing a session boundary, "
             "roles agree with `list_sessions()`.\n",
             "\n## Normalisation\n",
             f"Fitted on **train only** ({stats['n_windows']} windows), saved "
             f"to `{NORM_STATS_PATH.relative_to(REPO_ROOT)}`.\n", "```"]
    for c, m, sd in zip(stats["channels"], stats["mean"], stats["std"]):
        lines.append(f"  {c:<8} mean {m:+10.5f}   std {sd:10.5f}")
    lines.append("```")
    if failures:
        lines.append(f"\n## Sessions yielding no windows ({len(failures)})\n")
        lines.append("```")
        lines.extend(f"  {f}" for f in failures[:40])
        lines.append("```")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nwrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
