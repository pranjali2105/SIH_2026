"""Empirically verify IO-VNBD file schemas and unit conventions.

CLAUDE.md records the unit conventions as *assumed*. This script settles them
from the data itself: it prints headers, dtypes, head rows, row counts and
numeric summaries for four representative files, then runs explicit unit checks
with a stated verdict for each, and finally diffs the column sets across the
three S- files.

Findings go to stdout and to results/schema_report.md.

Run:  python src/inspect_schema.py
"""

from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
REPORT_PATH = REPO_ROOT / "results" / "schema_report.md"

# The files are ISO-8859-1, not UTF-8: headers contain "m/s²" and "µT" as
# latin-1 bytes. Reading as UTF-8 raises or mojibakes.
ENCODING = "latin-1"

SYNC = "Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
UNSYNC = "Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/S-Dataset"


@dataclass
class Target:
    key: str
    path: str
    note: str
    kind: str  # "S" or "V"


TARGETS = [
    Target("S-M", f"{SYNC}/S-M.csv", "train — Ford Fiesta / Huawei P20 Pro", "S"),
    Target("V-M", f"{SYNC}/V-M.csv", "train — ECU pair for the same session", "V"),
    Target("S-T1", f"{UNSYNC}/S-T1.csv", "validate — Renault Megane / Moto G7 Power", "S"),
    Target("S-A1", f"{UNSYNC}/S-A1.csv", "test — Volvo XC70 / Blackberry Priv", "S"),
]


# --------------------------------------------------------------------------
# output plumbing: everything printed is also captured for the markdown report
# --------------------------------------------------------------------------

class Tee:
    def __init__(self) -> None:
        self.buf = io.StringIO()

    def __call__(self, *parts: object) -> None:
        line = " ".join(str(p) for p in parts)
        print(line)
        self.buf.write(line + "\n")


out = Tee()


def norm(col: str) -> str:
    """Canonical column key: strip, collapse whitespace, uppercase.

    Headers are inconsistently spaced across files (leading spaces, a trailing
    space on 'ACCELEROMETER X (m/s2) '), so raw string equality under-reports
    which columns two files share.
    """
    return re.sub(r"\s+", " ", col.strip()).upper()


def find_col(df: pd.DataFrame, *patterns: str) -> str | None:
    """First column whose normalised name matches every regex in `patterns`."""
    for col in df.columns:
        key = norm(col)
        if all(re.search(p, key) for p in patterns):
            return col
    return None


def numeric(df: pd.DataFrame, col: str | None) -> pd.Series | None:
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s if not s.empty else None


# --------------------------------------------------------------------------
# verdict helper
# --------------------------------------------------------------------------

@dataclass
class Verdicts:
    rows: list[tuple[str, str, str, str]] = field(default_factory=list)

    def add(self, file_key: str, check: str, evidence: str, verdict: str) -> None:
        self.rows.append((file_key, check, evidence, verdict))
        out(f"  {check:<14} {verdict}")
        out(f"  {'':<14} evidence: {evidence}")


verdicts = Verdicts()


# --------------------------------------------------------------------------
# per-file description
# --------------------------------------------------------------------------

def describe(df: pd.DataFrame, t: Target) -> None:
    out(f"\n{'=' * 78}\n{t.key}  ({t.note})\n  {t.path}\n{'=' * 78}")
    out(f"\nrows: {len(df)}    columns: {len(df.columns)}")

    out("\n-- columns (index, exact header, dtype) --")
    for i, col in enumerate(df.columns):
        out(f"  [{i:2d}] {col!r}  ->  {df[col].dtype}")

    out("\n-- first 3 rows --")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        out(df.head(3).to_string())

    out("\n-- numeric summary (min / median / max) --")
    for col in df.columns:
        s = numeric(df, col)
        if s is None:
            out(f"  {norm(col):<46} (non-numeric / empty)")
        else:
            out(f"  {norm(col):<46} {s.min():>14.4f} {s.median():>14.4f} {s.max():>14.4f}")


# --------------------------------------------------------------------------
# unit checks
# --------------------------------------------------------------------------

def check_latitude(df: pd.DataFrame, t: Target) -> None:
    col = find_col(df, r"LATITUDE")
    s = numeric(df, col)
    if s is None:
        verdicts.add(t.key, "latitude", "no latitude column", "N/A")
        return
    med = s.median()
    ev = f"col {col!r}: min {s.min():.4f}, median {med:.4f}, max {s.max():.4f}"
    if 45 <= abs(med) <= 60:
        v = "DEGREES (45-60 band; UK ~52, France ~46). NOT arc-minutes."
    elif 2700 <= abs(med) <= 3600:
        v = "ARC-MINUTES (~3000-3300). Divide by 60."
    else:
        v = f"UNRECOGNISED magnitude ({med:.3f}) — inspect manually."
    verdicts.add(t.key, "latitude", ev, v)


def check_longitude(df: pd.DataFrame, t: Target) -> None:
    col = find_col(df, r"LONGITUDE")
    s = numeric(df, col)
    if s is None:
        verdicts.add(t.key, "longitude", "no longitude column", "N/A")
        return
    med = s.median()
    ev = f"col {col!r}: min {s.min():.4f}, median {med:.4f}, max {s.max():.4f}"
    mag = abs(med)
    if mag <= 10:
        scale = "DEGREES"
    elif 60 <= mag <= 600:
        scale = "ARC-MINUTES (divide by 60)"
    else:
        scale = f"UNRECOGNISED magnitude ({mag:.3f})"
    sign = "NEGATIVE (west)" if med < 0 else "POSITIVE (east)"
    # UK longitude is near zero and genuinely slightly negative, so its sign is
    # weak evidence. France (S-T1, ~ -0.5 to +5) is the discriminating case.
    verdicts.add(t.key, "longitude", ev, f"{scale}, sign {sign}")


def check_acceleration(df: pd.DataFrame, t: Target) -> None:
    if t.kind == "S":
        cols = [find_col(df, r"ACCELEROMETER", rf"\b{ax}\b") for ax in "XYZ"]
        if any(c is None for c in cols):
            verdicts.add(t.key, "acceleration", "accelerometer axes not found", "N/A")
            return
        arr = np.column_stack([pd.to_numeric(df[c], errors="coerce") for c in cols])
        mag = np.linalg.norm(arr, axis=1)
        mag = mag[np.isfinite(mag)]
        med = float(np.median(mag))
        ev = f"|accel| over XYZ: median {med:.4f} (min {mag.min():.4f}, max {mag.max():.4f})"
        if 8.5 <= med <= 11.0:
            v = "m/s^2 (vector magnitude ~9.8 = 1 g)."
        elif 0.85 <= med <= 1.15:
            v = "g (vector magnitude ~1). Multiply by 9.80665."
        else:
            v = f"UNRECOGNISED magnitude ({med:.3f})."
        verdicts.add(t.key, "acceleration", ev, v)
        return

    # V- files expose longitudinal/lateral acceleration only; there is no
    # vertical channel, so the 1 g gravity vector is absent and the magnitude
    # test does not apply. Judge by dynamic range instead.
    lon = numeric(df, find_col(df, r"LONGITUDINAL ACCEL"))
    lat = numeric(df, find_col(df, r"LATERAL ACCEL"))
    if lon is None or lat is None:
        verdicts.add(t.key, "acceleration", "no longitudinal/lateral accel columns", "N/A")
        return
    peak = max(lon.abs().max(), lat.abs().max())
    ev = (f"longitudinal range [{lon.min():.4f}, {lon.max():.4f}], "
          f"lateral range [{lat.min():.4f}, {lat.max():.4f}], peak |a| {peak:.4f}")
    if peak <= 2.0:
        v = "g (peak well under 2; road vehicles rarely exceed ~1 g). Multiply by 9.80665."
    elif peak <= 20.0:
        v = "m/s^2 (peak in the 2-20 band)."
    else:
        v = f"UNRECOGNISED magnitude (peak {peak:.3f})."
    verdicts.add(t.key, "acceleration", ev, v)


def check_gyroscope(df: pd.DataFrame, t: Target) -> None:
    col = find_col(df, r"GYROSCOPE", r"YAW") or find_col(df, r"YAW RATE")
    s = numeric(df, col)
    if s is None:
        verdicts.add(t.key, "gyroscope", "no yaw-rate column", "N/A")
        return
    p99 = float(s.abs().quantile(0.99))
    peak = float(s.abs().max())
    ev = f"col {col!r}: |yaw| p99 {p99:.4f}, max {peak:.4f}"
    # Roundabout peak is ~0.5 rad/s == ~30 deg/s; the two scales differ by 57x.
    if p99 <= 3.0:
        v = "rad/s (p99 <= 3; a roundabout peaks near 0.5 rad/s)."
    elif p99 >= 10.0:
        v = "deg/s (p99 >= 10; a roundabout peaks near 30 deg/s). Multiply by pi/180."
    else:
        v = f"AMBIGUOUS (p99 {p99:.3f} falls between the two bands)."
    verdicts.add(t.key, "gyroscope", ev, v)


def gps_derived_speed_ratio(df: pd.DataFrame, reported: pd.Series) -> float | None:
    """Median ratio of reported speed to speed derived from GPS fixes (m/s).

    Unit-independent discriminator: ~1.0 means the column is m/s, ~3.6 km/h.

    The GPS fix updates at 1 Hz while S- rows tick at 10 Hz, so lat/lon are
    HELD between updates. Differencing every row would put a whole second of
    displacement across a 0.1 s step and inflate the derived speed tenfold, so
    we difference only across rows where the fix actually changes, and divide
    by the elapsed time between those changes.

    Equirectangular distance -- accurate to far under a percent at these steps.
    """
    lat_c, lon_c = find_col(df, r"LATITUDE"), find_col(df, r"LONGITUDE")
    t_c = find_col(df, r"TIME SINCE START")
    if lat_c is None or lon_c is None or t_c is None:
        return None
    lat = pd.to_numeric(df[lat_c], errors="coerce")
    lon = pd.to_numeric(df[lon_c], errors="coerce")
    tim = pd.to_numeric(df[t_c], errors="coerce")
    ok = lat.notna() & lon.notna() & tim.notna()
    lat, lon, tim, rep = lat[ok], lon[ok], tim[ok], reported.reindex(lat[ok].index)
    if len(lat) < 50:
        return None

    # Keep only rows where the fix moved (a new GPS sample).
    changed = (lat.diff() != 0) | (lon.diff() != 0)
    changed.iloc[0] = True
    lat, lon, tim, rep = lat[changed], lon[changed], tim[changed], rep[changed]
    if len(lat) < 50:
        return None

    step = tim.diff()
    med_step = float(step[step > 0].median())
    dt_scale = 1000.0 if med_step > 10 else 1.0   # ms vs s

    R = 6371000.0
    dlat = np.radians(lat.diff()) * R
    dlon = np.radians(lon.diff()) * R * np.cos(np.radians(lat))
    dist = np.sqrt(dlat**2 + dlon**2)
    dt = step / dt_scale

    good = (dt > 0) & np.isfinite(dist)
    derived = dist[good] / dt[good]
    rep = rep[good]
    # Compare only while clearly moving: at rest, GPS jitter dominates and the
    # ratio is meaningless.
    moving = (derived > 3.0) & (rep > 0)
    if moving.sum() < 30:
        return None
    return float((rep[moving] / derived[moving]).median())


def trip_mean_speed_ratio(df: pd.DataFrame, reported: pd.Series) -> float | None:
    """Coarser fallback: mean reported speed / (total GPS path length / duration).

    Robust where fixes are sparse (S-A1 updates once per ~250 rows), because it
    never divides a long displacement by a short step. Coarse, so it is only
    used when the per-fix ratio cannot be computed.
    """
    lat_c, lon_c = find_col(df, r"LATITUDE"), find_col(df, r"LONGITUDE")
    t_c = find_col(df, r"TIME SINCE START")
    if lat_c is None or lon_c is None or t_c is None:
        return None
    lat = pd.to_numeric(df[lat_c], errors="coerce")
    lon = pd.to_numeric(df[lon_c], errors="coerce")
    tim = pd.to_numeric(df[t_c], errors="coerce")
    step = tim.diff()
    med = float(step[step > 0].median()) if (step > 0).any() else 0.0
    if med <= 0:
        return None
    dur = (tim.max() - tim.min()) / (1000.0 if med > 10 else 1.0)
    R = 6371000.0
    dl = np.radians(lat.diff()) * R
    dn = np.radians(lon.diff()) * R * np.cos(np.radians(lat))
    dist = float(np.nansum(np.sqrt(dl**2 + dn**2)))
    if dur <= 0 or dist <= 0:
        return None
    mean_rep = float(reported.mean())
    return mean_rep / (dist / dur) if mean_rep > 0 else None


def time_regularity(df: pd.DataFrame) -> str:
    """Describe how well-behaved the time column is."""
    t_c = find_col(df, r"TIME SINCE START")
    t = pd.to_numeric(df[t_c], errors="coerce") if t_c else None
    if t is None:
        return "no time column"
    dups = int(t.duplicated().sum())
    return (f"monotonic={t.is_monotonic_increasing}, duplicate timestamps={dups}"
            f" of {len(t)} rows")



def check_speed(df: pd.DataFrame, t: Target) -> None:
    col = (find_col(df, r"GPS SPEED") or find_col(df, r"INDICATED VEHICLE SPEED")
           or find_col(df, r"^VELOCITY"))
    s = numeric(df, col)
    if s is None:
        verdicts.add(t.key, "speed", "no speed column", "N/A")
        return
    p99, mx = float(s.quantile(0.99)), float(s.max())
    ev = f"col {col!r}: min {s.min():.3f}, median {s.median():.3f}, p99 {p99:.3f}, max {mx:.3f}"

    # A magnitude threshold alone cannot settle this: a session that never
    # leaves town tops out near 30 in EITHER unit. Decide it instead by
    # comparing the reported speed against ground speed derived from
    # consecutive GPS fixes, which is unit-independent.
    ratio = gps_derived_speed_ratio(df, s)
    method = "per-fix"
    if ratio is None:
        ratio, method = trip_mean_speed_ratio(df, s), "trip-integral"
    if ratio is not None:
        ev += f" [{method}]"
        ev += f"; reported/GPS-derived(m/s) ratio = {ratio:.3f}"
        if 0.8 <= ratio <= 1.25:
            v = "m/s (matches GPS-derived ground speed 1:1)."
        elif 3.0 <= ratio <= 4.3:
            v = "km/h (reads ~3.6x GPS-derived m/s). Divide by 3.6."
        else:
            v = (f"UNRESOLVED (ratio {ratio:.3f} matches neither 1.0 nor 3.6). "
                 f"Time base: {time_regularity(df)}.")
        if mx <= 45.0 and 3.0 <= ratio <= 4.3:
            v += " NOTE: max is only {:.1f}, so a magnitude test alone would have".format(mx)
            v += " misread this as m/s — the session never reaches motorway speed."
    elif mx >= 60.0:
        v = "km/h (max >= 60; motorway ~100 km/h), on magnitude alone. Divide by 3.6."
    else:
        v = f"AMBIGUOUS (max {mx:.3f}, no GPS cross-check available)."
    verdicts.add(t.key, "speed", ev, v)


def check_time(df: pd.DataFrame, t: Target) -> None:
    col = find_col(df, r"TIME SINCE START")
    s = numeric(df, col)
    if s is None:
        verdicts.add(t.key, "time", "no time column", "N/A")
        return
    d = s.diff().dropna()
    d = d[d > 0]
    if d.empty:
        verdicts.add(t.key, "time", f"col {col!r}: no positive deltas", "INDETERMINATE")
        return
    med = float(d.median())
    ev = (f"col {col!r}: start {s.iloc[0]:.3f}, end {s.iloc[-1]:.3f}, "
          f"median positive delta {med:.4f} (p05 {d.quantile(.05):.4f}, p95 {d.quantile(.95):.4f})")
    # Decide the unit first, then report the rate the data actually has --
    # do not assume 10 Hz, since these files do not all share a rate.
    unit, secs = ("MILLISECONDS", med / 1000.0) if med > 10 else ("SECONDS", med)
    hz = 1.0 / secs if secs > 0 else float("nan")
    ev += f"; implied rate {hz:.2f} Hz; {time_regularity(df)}"
    v = f"{unit} since start; median delta {med:g} -> {hz:.2f} Hz"
    if abs(hz - 10.0) < 0.5:
        v += " == the documented 10 Hz IMU rate. Confirmed."
    else:
        v += f" -- NOT 10 Hz. This file is {hz:.2f} Hz; resampling required."
    verdicts.add(t.key, "time", ev, v)


def run_checks(df: pd.DataFrame, t: Target) -> None:
    out(f"\n-- unit checks: {t.key} --")
    check_latitude(df, t)
    check_longitude(df, t)
    check_acceleration(df, t)
    check_gyroscope(df, t)
    check_speed(df, t)
    check_time(df, t)


# --------------------------------------------------------------------------
# schema comparison across the S- files
# --------------------------------------------------------------------------

def compare_schemas(frames: dict[str, pd.DataFrame]) -> bool:
    s_keys = [t.key for t in TARGETS if t.kind == "S" and t.key in frames]
    out(f"\n{'=' * 78}\nS- SCHEMA COMPARISON (set difference across splits)\n{'=' * 78}")

    sets = {k: {norm(c) for c in frames[k].columns} for k in s_keys}
    union = sorted(set().union(*sets.values()))

    width = max(len(c) for c in union) + 2
    header = "  " + "column".ljust(width) + "".join(k.ljust(8) for k in s_keys)
    out("\n" + header)
    out("  " + "-" * (width + 8 * len(s_keys)))
    for col in union:
        marks = "".join(("yes" if col in sets[k] else "NO").ljust(8) for k in s_keys)
        flag = "" if all(col in sets[k] for k in s_keys) else "   <-- differs"
        out("  " + col.ljust(width) + marks + flag)

    out("\n-- pairwise set differences --")
    for a in s_keys:
        for b in s_keys:
            if a >= b:
                continue
            only_a, only_b = sorted(sets[a] - sets[b]), sorted(sets[b] - sets[a])
            out(f"\n  {a} vs {b}:")
            out(f"    only in {a} ({len(only_a)}): {only_a if only_a else '-'}")
            out(f"    only in {b} ({len(only_b)}): {only_b if only_b else '-'}")

    common = set.intersection(*sets.values())
    out(f"\n  columns common to all {len(s_keys)} S- files: {len(common)}")

    train_only = sets.get("S-M", set()) - sets.get("S-T1", set())
    schema_differs = bool(train_only)
    if schema_differs:
        out("\n" + "!" * 78)
        out("!! SPLIT SCHEMA MISMATCH — the loader MUST handle this.")
        out("!!")
        out(f"!! Train (S-M) has {len(train_only)} columns that validate (S-T1) lacks:")
        for c in sorted(train_only):
            out(f"!!   - {c}")
        out("!!")
        out("!! This matches the paper's note that Driver F's France sessions lack")
        out("!! 3-axis orientation and magnetic field data. Any model trained on")
        out("!! these features cannot be evaluated on the validation split. Restrict")
        out("!! the feature set to the intersection, or the validation step will")
        out("!! fail — or, worse, silently impute and report an optimistic score.")
        out("!" * 78)
    else:
        out("\n  No train/validate schema mismatch detected.")
    return schema_differs


# --------------------------------------------------------------------------

def main() -> int:
    if not DATA_ROOT.exists():
        print(f"error: {DATA_ROOT} not found — run scripts/fetch_data.sh first.", file=sys.stderr)
        return 1

    out("# IO-VNBD schema and unit report")
    out(f"\nGenerated by `src/inspect_schema.py` from `{DATA_ROOT.relative_to(REPO_ROOT)}`.")
    out(f"Files read as {ENCODING} (they are not UTF-8).")

    frames: dict[str, pd.DataFrame] = {}
    missing: list[Target] = []
    for t in TARGETS:
        path = DATA_ROOT / t.path
        if not path.exists():
            missing.append(t)
            continue
        # skipinitialspace: headers and values carry inconsistent leading spaces.
        frames[t.key] = pd.read_csv(path, encoding=ENCODING, skipinitialspace=True,
                                    low_memory=False)

    for t in missing:
        out(f"\n!! MISSING: {t.path}")

    for t in TARGETS:
        if t.key not in frames:
            continue
        describe(frames[t.key], t)
        run_checks(frames[t.key], t)

    schema_differs = compare_schemas(frames) if len(frames) > 1 else False

    out(f"\n{'=' * 78}\nVERDICT SUMMARY\n{'=' * 78}\n")
    out(f"  {'file':<7} {'check':<14} verdict")
    out("  " + "-" * 74)
    for key, check, _ev, verdict in verdicts.rows:
        out(f"  {key:<7} {check:<14} {verdict}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("```\n" + out.buf.getvalue() + "\n```\n", encoding="utf-8")
    print(f"\nwrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 2 if schema_differs else 0


if __name__ == "__main__":
    raise SystemExit(main())
