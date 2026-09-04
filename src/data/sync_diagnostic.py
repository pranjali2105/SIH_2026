"""Decisive diagnostic: phone accelerometer vs ECU accelerometer.

The S- time-sync failure measured against GPS could mean either (a) the phone
signal does not carry recoverable vehicle motion, or (b) the GPS-side reference
was badly constructed. Correlating the phone against the ECU settles it without
GPS in the loop: both are inertial sensors on the same vehicle at the same
instant, so a high correlation exonerates the phone data and indicts the
reference; a low one changes the project.

Both sides use ACCELERATION MAGNITUDE after gravity removal, which is invariant
to sensor orientation -- so an unknown phone mounting cannot suppress the
correlation, and the test measures signal content rather than alignment.

Run:  python -m data.sync_diagnostic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .loader import G_TO_MS2, Session, _find, list_sessions, load_session

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "sanity" / "phone_vs_ecu_sync.csv"

GRID_HZ = 10.0
MAX_LAG_ROWS = 100    # +/-10 s at 10 Hz, on the row grid
MIN_SAMPLES = 500
LOWPASS_HZ = 2.0      # both are inertial at 10 Hz; keep more band than the
                      # 0.5 Hz used against GPS, but drop road vibration


def _lowpass(x: np.ndarray, cutoff: float = LOWPASS_HZ,
             fs: float = GRID_HZ, order: int = 3) -> np.ndarray:
    if x.size < 3 * order + 1:
        return x
    b, a = butter(order, cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def _per_sensor_grid(t: pd.Series, y: pd.Series, grid: np.ndarray) -> np.ndarray:
    """Interpolate a channel onto `grid` using only ITS OWN valid samples.

    AndroSensor writes one row per sensor event, so a given column is blank or
    stale in most rows of a tick. Treating the row grid as uniform for every
    column is the same trap that produced the spike-train artefact on speed.
    """
    ok = t.notna() & y.notna()
    if ok.sum() < 10:
        return np.full(grid.size, np.nan)
    tt, yy = t[ok].to_numpy(float), y[ok].to_numpy(float)
    order = np.argsort(tt, kind="stable")
    tt, yy = tt[order], yy[order]
    uniq, idx = np.unique(tt, return_inverse=True)
    means = np.bincount(idx, weights=yy) / np.bincount(idx)
    return np.interp(grid, uniq, means, left=np.nan, right=np.nan)


def phone_accel_magnitude(s: Session) -> tuple[pd.Series, pd.Series] | None:
    """(time, |accel - gravity|) for an S- session.

    Gravity is ~9.8 m/s^2 against vehicle accelerations of ~1-2 m/s^2, so it is
    roughly 5x the signal and must be removed before anything is correlated.
    """
    axes = [_find(s.df, r"ACCELEROMETER", rf"\b{a}\b") for a in "XYZ"]
    grav = [_find(s.df, r"GRAVITY", rf"\b{a}\b") for a in "XYZ"]
    if any(a is None for a in axes) or any(g is None for g in grav):
        return None
    lin = []
    for acol, gcol in zip(axes, grav):
        lin.append(pd.to_numeric(s.df[acol], errors="coerce")
                   - pd.to_numeric(s.df[gcol], errors="coerce"))
    mag = np.sqrt(sum(c**2 for c in lin))
    return pd.to_numeric(s.df["time_s"], errors="coerce"), mag


def ecu_accel_magnitude(s: Session) -> tuple[pd.Series, pd.Series] | None:
    """(time, |a_horizontal|) for a V- session, converted from g."""
    lon = _find(s.df, r"LONGITUDINAL ACCEL")
    lat = _find(s.df, r"LATERAL ACCEL")
    if lon is None or lat is None:
        return None
    a_lon = pd.to_numeric(s.df[lon], errors="coerce") * G_TO_MS2
    a_lat = pd.to_numeric(s.df[lat], errors="coerce") * G_TO_MS2
    return pd.to_numeric(s.df["time_s"], errors="coerce"), np.sqrt(a_lon**2 + a_lat**2)


def xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, float, int]:
    """(peak_r, lag_samples, n) over lags where both signals are valid."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 300:
        return float("nan"), float("nan"), int(ok.sum())
    a = np.where(ok, a, 0.0)
    b = np.where(ok, b, 0.0)
    a = _lowpass(a - a[ok].mean())
    b = _lowpass(b - b[ok].mean())
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan"), float("nan"), int(ok.sum())
    a, b = a / a.std(), b / b.std()
    n = a.size
    best_r, best_L = 0.0, 0
    for L in range(-max_lag, max_lag + 1):
        if L < 0:
            x, y = a[-L:], b[:n + L]
        elif L > 0:
            x, y = a[:n - L], b[L:]
        else:
            x, y = a, b
        if x.size < 300:
            continue
        r = float(np.dot(x, y) / x.size)
        if abs(r) > abs(best_r):
            best_r, best_L = r, L
    return best_r, best_L, int(ok.sum())


def compare_pair(session: str) -> dict:
    """Cross-correlate the S- and V- recordings of one session.

    Compared on the ROW grid, not on timestamps. Every Synchronised-tree pair
    has identical row counts (e.g. S-M and V-M are both 105974 rows), i.e. the
    authors aligned the two families by row index. The clocks themselves have
    different origins -- S- counts from session start, V- from start of day --
    so a timestamp-based lag search chases an offset of minutes and pins itself
    to the search boundary.
    """
    row: dict = {"session": session}
    try:
        sp = load_session(f"S-{session}", check_rate=False)
        sv = load_session(f"V-{session}", check_rate=False)
    except Exception as exc:
        row["note"] = f"load failed: {type(exc).__name__}"
        return row

    p = phone_accel_magnitude(sp)
    v = ecu_accel_magnitude(sv)
    if p is None or v is None:
        row["note"] = "missing accelerometer channels"
        return row

    A, B = p[1].to_numpy(float), v[1].to_numpy(float)
    row["s_rows"], row["v_rows"] = len(A), len(B)
    row["row_counts_equal"] = len(A) == len(B)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    ok = np.isfinite(A) & np.isfinite(B)
    if ok.sum() < MIN_SAMPLES:
        row["note"] = f"only {int(ok.sum())} usable rows"
        return row

    A = _lowpass(np.where(ok, A, 0.0))
    B = _lowpass(np.where(ok, B, 0.0))
    if A.std() < 1e-9 or B.std() < 1e-9:
        row["note"] = "flat signal"
        return row
    A = (A - A.mean()) / A.std()
    B = (B - B.mean()) / B.std()

    row["r_at_row0"] = float(np.dot(A, B) / n)
    best_r, best_L = 0.0, 0
    for L in range(-MAX_LAG_ROWS, MAX_LAG_ROWS + 1):
        if L < 0:
            x, y = A[-L:], B[:n + L]
        elif L > 0:
            x, y = A[:n - L], B[L:]
        else:
            x, y = A, B
        if x.size < MIN_SAMPLES:
            continue
        r = float(np.dot(x, y) / x.size)
        if abs(r) > abs(best_r):
            best_r, best_L = r, L
    row.update(peak_r=best_r, best_lag_rows=best_L,
               best_lag_ms=best_L / GRID_HZ * 1000.0,
               n_samples=int(ok.sum()), note="")
    row["speed_correlation"] = _speed_correlation(sp, sv)
    return row


def _speed_correlation(sp: Session, sv: Session) -> float:
    """Correlate the S- GPS speed against the V- ECU speed, on the row grid.

    A direct measure of S- GPS quality for this session: the ECU speed is a
    clean 10 Hz wheel-derived signal, so how well the phone's GPS speed tracks
    it is a per-session quality score. Both are already in m/s (the loader
    applies the family-wise unit override).
    """
    if "speed_ms" not in sp.df.columns or "speed_ms" not in sv.df.columns:
        return float("nan")
    a = pd.to_numeric(sp.df["speed_ms"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(sv.df["speed_ms"], errors="coerce").to_numpy(float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_SAMPLES:
        return float("nan")
    a = np.where(ok, a, 0.0)
    b = np.where(ok, b, 0.0)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    # Allow a small row offset: a zero-lag correlation would penalise a
    # session for being misaligned rather than for having noisy GPS, and
    # alignment is exactly what we have established we cannot pin down.
    best = 0.0
    for L in range(-MAX_LAG_ROWS, MAX_LAG_ROWS + 1):
        if L < 0:
            x, y = a[-L:], b[:n + L]
        elif L > 0:
            x, y = a[:n - L], b[L:]
        else:
            x, y = a, b
        if x.size < MIN_SAMPLES:
            continue
        r = float(np.dot(x, y) / x.size)
        if abs(r) > abs(best):
            best = r
    return best


def paired_sessions() -> list[str]:
    """Sessions recorded by BOTH families, Fiesta groups first."""
    d = list_sessions()
    d["key"] = d["session"].str.lower()
    s = set(d.loc[d["family"] == "S", "key"])
    v = set(d.loc[d["family"] == "V", "key"])
    both = s & v
    lookup = {r.key: r.session for r in d.itertuples() if r.family == "S"}
    order = {"M": 0, "S": 1, "Y": 2, "Vfa": 3, "Vw": 4, "Vta": 5, "Vtb": 6}
    groups = {r.key: r.group for r in d.itertuples() if r.family == "S"}
    return sorted((lookup[k] for k in both),
                  key=lambda x: (order.get(groups[x.lower()], 9), x.lower()))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    names = argv or paired_sessions()
    rows = []
    for i, name in enumerate(names, 1):
        print(f"\r  [{i}/{len(names)}] {name:<16}", end="", file=sys.stderr)
        rows.append(compare_pair(name))
    print(file=sys.stderr)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    ok = df[df["peak_r"].notna()] if "peak_r" in df.columns else df.iloc[:0]
    print("\nPhone |accel-gravity| vs ECU |accel|  (no GPS involved)\n")
    cols = [c for c in ("session", "r_at_row0", "peak_r", "speed_correlation",
                        "best_lag_rows", "row_counts_equal", "n_samples", "note")
            if c in df.columns]
    print(df[cols].to_string(index=False))
    if not ok.empty:
        print(f"\npairs compared: {len(ok)}")
        print(f"median |peak_r| : {ok['peak_r'].abs().median():.3f}")
        print(f"max    |peak_r| : {ok['peak_r'].abs().max():.3f}")
        print(f"pairs with |r| >= 0.5 : {int((ok['peak_r'].abs() >= 0.5).sum())}/{len(ok)}")
    print(f"\nwrote {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
