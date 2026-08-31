"""Diagnostic sweep of the whole IO-VNBD corpus.

Purely diagnostic: writes results/corpus_sweep.csv (one row per session file)
plus results/corpus_sweep.md, and changes nothing else. The rate distribution
it produces is what the split decision will be based on.

Deduplication
-------------
The same session appears in up to five places. Reading them all would
double-count hours and distance, so exactly one copy of each (family, session)
is read, by this precedence:

    1. Synchronised   / Categorised
    2. Synchronised   / Uncategorised
    3. Unsynchronised / Categorised
    4. Unsynchronised / Uncategorised

"Prefer Synchronised where the session exists there, else Unsynchronised",
with Categorised winning inside a tree since that is the organised form. The
chosen tree and subtree are recorded on every row.

Session keys are matched case-insensitively: the corpus contains both
`V-Vta10.csv` and `V-vta10.csv` for the same session.

Run:  python src/sweep_corpus.py
"""

from __future__ import annotations

import hashlib
import io
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
CSV_OUT = REPO_ROOT / "results" / "corpus_sweep.csv"
MD_OUT = REPO_ROOT / "results" / "corpus_sweep.md"

ENCODING = "iso-8859-1"

SYNC_TREE = "Synchronised V abd S datasets"
UNSYNC_TREE = "Unsynchronised V and S Dataset"

# Longest-first so Vta/Vtb/Vfa/Vfb win over any shorter prefix.
KNOWN_GROUPS = ["Vta", "Vtb", "Vfa", "Vfb", "Vw", "St", "S", "M", "Y", "T", "A", "I"]

# A file is re-timed from its DATE column when this share of its rows share a
# timestamp with another row -- the TIME SINCE START granularity is then too
# coarse to trust as a rate.
DUP_FRAC_THRESHOLD = 0.05

R_EARTH = 6371000.0


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
    return re.sub(r"\s+", " ", col.strip()).upper()


def find_col(df: pd.DataFrame, *patterns: str) -> str | None:
    for col in df.columns:
        key = norm(col)
        if all(re.search(p, key) for p in patterns):
            return col
    return None


def to_num(df: pd.DataFrame, col: str | None) -> pd.Series | None:
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    return s if s.notna().any() else None


# --------------------------------------------------------------------------
# file discovery + dedup
# --------------------------------------------------------------------------

@dataclass
class FileRef:
    path: Path
    family: str      # S or V
    session: str     # e.g. Vw12
    group: str       # e.g. Vw
    tree: str        # Synchronised | Unsynchronised
    subtree: str     # Categorised | Uncategorised
    priority: int


def parse_group(session: str) -> str:
    low = session.lower()
    for g in KNOWN_GROUPS:
        if low.startswith(g.lower()):
            return g
    return "?"


def discover() -> tuple[list[FileRef], int]:
    refs: list[FileRef] = []
    for path in DATA_ROOT.rglob("*.csv"):
        if ".git" in path.parts:
            continue
        stem = path.stem
        m = re.match(r"^([SV])-(.+)$", stem, flags=re.IGNORECASE)
        if not m:
            continue
        family, session = m.group(1).upper(), m.group(2)
        rel = path.relative_to(DATA_ROOT).as_posix()
        tree = "Synchronised" if rel.startswith(SYNC_TREE) else "Unsynchronised"
        subtree = "Categorised" if "Categorised" in rel and "Uncategorised" not in rel \
            else "Uncategorised"
        priority = {("Synchronised", "Categorised"): 0,
                    ("Synchronised", "Uncategorised"): 1,
                    ("Unsynchronised", "Categorised"): 2,
                    ("Unsynchronised", "Uncategorised"): 3}[(tree, subtree)]
        refs.append(FileRef(path, family, session, parse_group(session),
                            tree, subtree, priority))

    total = len(refs)
    best: dict[tuple[str, str], FileRef] = {}
    for r in refs:
        key = (r.family, r.session.lower())
        if key not in best or r.priority < best[key].priority:
            best[key] = r
    chosen = sorted(best.values(), key=lambda r: (r.group, r.family, r.session.lower()))
    return chosen, total


# --------------------------------------------------------------------------
# per-file metrics
# --------------------------------------------------------------------------

def time_seconds(df: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    """TIME SINCE START normalised to seconds, plus the column name."""
    col = find_col(df, r"TIME SINCE START")
    s = to_num(df, col)
    if s is None:
        return None, col
    # Scale from the HEADER, not from the delta magnitude. S- headers say
    # "TIME SINCE START (ms)" and V- says "(seconds)"; that is authoritative.
    # A magnitude heuristic misreads S-T1, whose 1 ms quantisation makes the
    # median delta 1 and looks like seconds -- its span (738388 ms = 738.4 s)
    # matches its DATE column exactly, confirming milliseconds.
    key = norm(col)
    if "(MS)" in key or "MILLISEC" in key:
        scale = 1000.0
    elif "SECOND" in key:
        scale = 1.0
    else:
        d = s.diff()
        d = d[d > 0]
        scale = 1000.0 if (not d.empty and float(d.median()) > 10) else 1.0
    return s / scale, col


def date_seconds(df: pd.DataFrame) -> pd.Series | None:
    """Seconds derived from the DATE column.

    Format is 'YYYY-MM-DD HH:MM:SS:mmm' -- the milliseconds are separated by a
    colon, not a dot, so the last colon is swapped before parsing.
    """
    col = find_col(df, r"^DATE")
    if col is None:
        return None
    raw = df[col].astype(str).str.strip()
    fixed = raw.str.replace(r"^(.*\d{2}:\d{2}:\d{2}):(\d+)$", r"\1.\2", regex=True)
    try:
        ts = pd.to_datetime(fixed, format="mixed", errors="coerce")
    except Exception:
        return None
    # Some files contain out-of-range or malformed dates; differencing against
    # them overflows int64. Keep only plausible collection dates.
    ts = ts.where((ts >= pd.Timestamp("2015-01-01")) & (ts <= pd.Timestamp("2025-01-01")))
    if ts.notna().sum() < 10:
        return None
    try:
        return (ts - ts.min()).dt.total_seconds()
    except (OverflowError, OutOfBoundsDatetime):
        return None


def hz_from(t: pd.Series | None) -> tuple[float, float, float, float]:
    """(dt_median_ms, dt_p05_ms, dt_p95_ms, effective_hz) from a seconds series."""
    if t is None:
        return (np.nan,) * 4
    d = t.diff()
    d = d[d > 0]
    if d.empty:
        return (np.nan,) * 4
    med = float(d.median())
    return (med * 1000, float(d.quantile(0.05)) * 1000,
            float(d.quantile(0.95)) * 1000, 1.0 / med if med > 0 else np.nan)


def tick_hz(t: pd.Series | None) -> tuple[float, float, bool]:
    """(hz_tick, dt_mode_ms, is_bimodal) — the sensor tick rate.

    AndroSensor writes one row per sensor EVENT, so several rows can share a
    tick and be separated by ~1 ms. The delta distribution is then bimodal:
    a spike near 1 ms (intra-tick) and one at the true period (e.g. 100 ms).
    A plain median lands in the intra-tick spike and reports a nonsense rate
    (S-T1 reads as 1000 Hz that way). Estimating from deltas at or above 20 ms
    ignores the intra-tick rows and recovers the real tick.
    """
    if t is None:
        return np.nan, np.nan, False
    d = (t.diff() * 1000.0)
    d = d[d > 0]
    if d.empty:
        return np.nan, np.nan, False
    mode = float(d.round().mode().iloc[0]) if not d.mode().empty else np.nan
    ticks = d[d >= 20.0]
    small_frac = float((d < 20.0).mean())
    bimodal = small_frac > 0.2 and len(ticks) >= 20
    if len(ticks) < 20:
        return np.nan, mode, bimodal
    med = float(ticks.median())
    return (1000.0 / med if med > 0 else np.nan), mode, bimodal



def path_length_km(lat: pd.Series | None, lon: pd.Series | None) -> float:
    if lat is None or lon is None:
        return np.nan
    dlat = np.radians(lat.diff()) * R_EARTH
    dlon = np.radians(lon.diff()) * R_EARTH * np.cos(np.radians(lat))
    return float(np.nansum(np.sqrt(dlat**2 + dlon**2)) / 1000.0)


def speed_ratio(df: pd.DataFrame, t: pd.Series | None) -> float:
    """GPS-derived ground speed (m/s) divided into the reported speed column.

    ~1.0 => the column is m/s; ~3.6 => km/h.

    GPS fixes are HELD between updates (1 Hz fix under a 10 Hz row rate), so
    displacement is differenced only across rows where the fix actually moves,
    over the elapsed time between those moves.
    """
    lat, lon = to_num(df, find_col(df, r"LATITUDE")), to_num(df, find_col(df, r"LONGITUDE"))
    rep = to_num(df, find_col(df, r"GPS SPEED") or find_col(df, r"INDICATED VEHICLE SPEED")
                 or find_col(df, r"^VELOCITY"))
    if lat is None or lon is None or rep is None or t is None:
        return np.nan
    ok = lat.notna() & lon.notna() & t.notna() & rep.notna()
    lat, lon, t, rep = lat[ok], lon[ok], t[ok], rep[ok]
    if len(lat) < 50:
        return np.nan
    moved = (lat.diff() != 0) | (lon.diff() != 0)
    if moved.sum() < 30:
        return np.nan
    lat, lon, t, rep = lat[moved], lon[moved], t[moved], rep[moved]
    dlat = np.radians(lat.diff()) * R_EARTH
    dlon = np.radians(lon.diff()) * R_EARTH * np.cos(np.radians(lat))
    dist = np.sqrt(dlat**2 + dlon**2)
    dt = t.diff()
    good = (dt > 0) & np.isfinite(dist)
    derived = dist[good] / dt[good]
    rep = rep[good]
    driving = (derived > 3.0) & (rep > 0)
    if driving.sum() < 30:
        return np.nan
    return float((rep[driving] / derived[driving]).median())


def measure(ref: FileRef) -> dict:
    row: dict = {
        "filename": ref.path.name,
        "session": ref.session,
        "group": ref.group,
        "family": ref.family,
        "tree": ref.tree,
        "subtree": ref.subtree,
        "relpath": ref.path.relative_to(DATA_ROOT).as_posix(),
    }
    try:
        df = pd.read_csv(ref.path, encoding=ENCODING, skipinitialspace=True,
                         low_memory=False)
    except Exception as exc:  # keep the sweep going; record the failure
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    cols = sorted(norm(c) for c in df.columns)
    row["n_rows"] = len(df)
    row["n_columns"] = len(df.columns)
    row["columns_hash"] = hashlib.sha1("|".join(cols).encode()).hexdigest()[:12]

    t, t_col = time_seconds(df)
    dt_med, dt_p05, dt_p95, hz = hz_from(t)
    hzt, dt_mode, bimodal = tick_hz(t)
    row.update(dt_median_ms=dt_med, dt_p05_ms=dt_p05, dt_p95_ms=dt_p95,
               effective_hz=hz, hz_tick=hzt, dt_mode_ms=dt_mode,
               bimodal_dt=bool(bimodal))
    row["duration_s"] = float(t.max() - t.min()) if t is not None else np.nan

    raw_t = to_num(df, t_col)
    row["dup_timestamp_frac"] = (
        float(raw_t.duplicated(keep=False).mean()) if raw_t is not None else np.nan)

    # DATE-derived rate/duration. Computed for every file (cheap, and it is the
    # only independent check on the primary timestamp), but only *reported*
    # side by side for files whose timestamps are heavily duplicated.
    d = date_seconds(df)
    row["hz_from_date"] = hz_from(d)[3]
    row["duration_s_date"] = float(d.max() - d.min()) if d is not None else np.nan

    # Rows per second of wall clock. Where this greatly exceeds the tick rate,
    # the file writes several rows per timestamp (AndroSensor emits one row per
    # sensor event), which is NOT the same thing as a higher sampling rate.
    span = row["duration_s_date"] if np.isfinite(row.get("duration_s_date", np.nan)) \
        else row["duration_s"]
    row["rows_per_s"] = (len(df) - 1) / span if span and np.isfinite(span) and span > 0 \
        else np.nan

    lat, lon = to_num(df, find_col(df, r"LATITUDE")), to_num(df, find_col(df, r"LONGITUDE"))
    row["distance_km"] = path_length_km(lat, lon)
    for name, s in (("lat", lat), ("lon", lon)):
        row[f"{name}_min"] = float(s.min()) if s is not None else np.nan
        row[f"{name}_max"] = float(s.max()) if s is not None else np.nan

    if ref.family == "S":
        axes = [find_col(df, r"ACCELEROMETER", rf"\b{a}\b") for a in "XYZ"]
        if all(a is not None for a in axes):
            arr = np.column_stack([pd.to_numeric(df[a], errors="coerce") for a in axes])
            mag = np.linalg.norm(arr, axis=1)
            mag = mag[np.isfinite(mag)]
            row["accel_mag_median"] = float(np.median(mag)) if mag.size else np.nan
        else:
            row["accel_mag_median"] = np.nan
    else:
        # V- files have no vertical channel, so there is no 1 g gravity vector
        # to measure; use the planar longitudinal/lateral magnitude instead.
        lo = to_num(df, find_col(df, r"LONGITUDINAL ACCEL"))
        la = to_num(df, find_col(df, r"LATERAL ACCEL"))
        if lo is not None and la is not None:
            row["accel_mag_median"] = float(np.nanmedian(np.sqrt(lo**2 + la**2)))
        else:
            row["accel_mag_median"] = np.nan

    gyro = to_num(df, find_col(df, r"GYROSCOPE", r"YAW")) 
    if gyro is None:
        gyro = to_num(df, find_col(df, r"YAW RATE"))
    row["gyro_abs_p99"] = float(gyro.abs().quantile(0.99)) if gyro is not None else np.nan

    row["speed_ratio"] = speed_ratio(df, t)
    row["error"] = ""
    return row


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------

def hz_bucket(hz: float) -> str:
    if not np.isfinite(hz):
        return "unknown"
    for target in (100, 50, 20, 10, 5, 2, 1):
        if abs(hz - target) / target < 0.15:
            return f"{target} Hz"
    return f"{hz:.2f} Hz (odd)"


def summary_rate(df: pd.DataFrame) -> None:
    out(f"\n{'=' * 90}\n1. RATE DISTRIBUTION BY GROUP  (files / hours at each effective_hz)\n{'=' * 90}")
    out("\nThis is the table the split decision rests on.\n")
    d = df[df["error"] == ""].copy()
    # Bucket on hz_tick for bimodal files: their raw median delta sits in the
    # intra-tick spike and would report a fictitious rate.
    rate = d["effective_hz"].copy()
    use_tick = d["bimodal_dt"].fillna(False) & d["hz_tick"].notna()
    rate[use_tick] = d.loc[use_tick, "hz_tick"]
    d["bucket"] = rate.map(hz_bucket)
    if use_tick.any():
        out(f"  ({int(use_tick.sum())} files with bimodal timestamps are bucketed by "
            f"hz_tick, not by median delta.)")
    # Prefer the DATE-derived duration where the primary timestamp is
    # duplicate-heavy, so broken time columns cannot inflate the hour totals.
    dur = d["duration_s"].copy()
    swap = (d["dup_timestamp_frac"] > DUP_FRAC_THRESHOLD) & d["duration_s_date"].notna()
    dur[swap] = d.loc[swap, "duration_s_date"]
    d["hours"] = dur / 3600.0
    if swap.any():
        out(f"  ({int(swap.sum())} duplicate-heavy files use their DATE-derived duration.)")

    for family in ("S", "V"):
        sub = d[d["family"] == family]
        if sub.empty:
            continue
        out(f"\n-- {family}- files --")
        buckets = sorted(sub["bucket"].unique(),
                         key=lambda b: -(float(b.split()[0]) if b[0].isdigit() else -1))
        head = f"  {'group':<7}" + "".join(b.rjust(18) for b in buckets) + f"{'total h':>10}"
        out(head)
        out("  " + "-" * (len(head) - 2))
        for g in sorted(sub["group"].unique()):
            gs = sub[sub["group"] == g]
            cells = ""
            for b in buckets:
                bs = gs[gs["bucket"] == b]
                cells += (f"{len(bs)}f / {bs['hours'].sum():.1f}h".rjust(18)
                          if len(bs) else "-".rjust(18))
            out(f"  {g:<7}{cells}{gs['hours'].sum():>10.1f}")
        tot = ""
        for b in buckets:
            bs = sub[sub["bucket"] == b]
            tot += f"{len(bs)}f / {bs['hours'].sum():.1f}h".rjust(18)
        out(f"  {'ALL':<7}{tot}{sub['hours'].sum():>10.1f}")

    mixed = [g for g, gs in d.groupby("group") if gs["bucket"].nunique() > 1]
    if mixed:
        out(f"\n  !! groups spanning more than one rate: {', '.join(sorted(mixed))}")

    redated = d[(d["dup_timestamp_frac"] > DUP_FRAC_THRESHOLD) & d["hz_from_date"].notna()]
    if not redated.empty:
        out(f"\n-- files re-timed from DATE (dup_timestamp_frac > {DUP_FRAC_THRESHOLD}) --")
        out(f"  {'file':<16}{'dup_frac':>10}{'hz_from_TIME':>14}{'hz_from_DATE':>14}"
            f"{'rows_per_s':>12}{'hz_tick':>10}{'dur_DATE_s':>12}")
        out("  " + "-" * 88)
        for _, r in redated.sort_values("filename").iterrows():
            out(f"  {r['filename']:<16}{r['dup_timestamp_frac']:>10.3f}"
                f"{r['effective_hz']:>14.2f}{r['hz_from_date']:>14.2f}"
                f"{r['rows_per_s']:>12.2f}{r['hz_tick']:>10.2f}"
                f"{r['duration_s_date']:>12.1f}")
        out("\n  rows_per_s >> hz means several rows share each tick (one row per")
        out("  sensor event), not a faster sampling rate.")


def summary_schemas(df: pd.DataFrame) -> None:
    out(f"\n{'=' * 90}\n2. DISTINCT COLUMN SETS\n{'=' * 90}")
    d = df[df["error"] == ""]
    # Compare within family: an S- schema differenced against the V- schema is
    # just the S/V difference and says nothing useful.
    for family in ("S", "V"):
        fam = d[d["family"] == family]
        if fam.empty:
            continue
        hashes = list(fam.groupby("columns_hash"))
        largest = max(hashes, key=lambda kv: len(COLUMN_SETS.get(kv[0], ())))[0]
        out(f"\n-- {family}- files: {len(hashes)} distinct column set(s) --")
        for h, gs in sorted(hashes, key=lambda kv: -len(COLUMN_SETS.get(kv[0], ()))):
            cols = COLUMN_SETS.get(h, set())
            out(f"\n  hash {h}  ({len(cols)} cols, {len(gs)} files)")
            out(f"    groups: {', '.join(sorted(gs['group'].unique()))}")
            if h == largest:
                out(f"    (largest {family}- schema — reference)")
                continue
            missing = sorted(COLUMN_SETS[largest] - cols)
            extra = sorted(cols - COLUMN_SETS[largest])
            out(f"    missing ({len(missing)}): {missing if missing else '-'}")
            out(f"    extra   ({len(extra)}): {extra if extra else '-'}")
        if len(hashes) > 1:
            common = set.intersection(*(COLUMN_SETS.get(h, set()) for h, _ in hashes))
            out(f"\n  columns common to ALL {family}- files: {len(common)}")


def summary_units(df: pd.DataFrame) -> None:
    out(f"\n{'=' * 90}\n3. UNIT VERDICTS BY GROUP\n{'=' * 90}")
    d = df[df["error"] == ""]

    def modal_ratio(v: pd.Series) -> float:
        v = v.dropna()
        return float(v.median()) if len(v) else np.nan

    fam_modal = {f: modal_ratio(g["speed_ratio"]) for f, g in d.groupby("family")}
    fam_accel = {f: float(g["accel_mag_median"].dropna().median()) for f, g in d.groupby("family")}
    out("\n  family-level majority:")
    for f in sorted(fam_modal):
        out(f"    {f}-: speed_ratio {fam_modal[f]:.3f}   accel_mag_median {fam_accel[f]:.3f}")

    out(f"\n  {'group':<7}{'fam':<5}{'n':>4}{'speed_ratio':>13}{'unit':>8}"
        f"{'accel_med':>11}{'unit':>10}   flag")
    out("  " + "-" * 86)
    for (g, f), gs in sorted(d.groupby(["group", "family"])):
        sr = modal_ratio(gs["speed_ratio"])
        am = float(gs["accel_mag_median"].dropna().median()) if gs["accel_mag_median"].notna().any() else np.nan
        s_unit = "m/s" if 0.8 <= sr <= 1.25 else "km/h" if 3.0 <= sr <= 4.3 else "?"
        if f == "S":
            a_unit = "m/s^2" if 8.5 <= am <= 11 else "g" if 0.85 <= am <= 1.15 else "?"
        else:
            a_unit = "g" if am <= 2 else "m/s^2" if am <= 20 else "?"
        flags = []
        fm = fam_modal.get(f, np.nan)
        if np.isfinite(sr) and np.isfinite(fm) and abs(sr - fm) / fm > 0.25:
            flags.append(f"speed disagrees with {f}- majority ({fm:.2f})")
        if s_unit == "?" and np.isfinite(sr):
            flags.append("speed unit unresolved")
        if a_unit == "?" and np.isfinite(am):
            flags.append("accel unit unresolved")
        if not np.isfinite(sr):
            flags.append("no speed cross-check")
        srs = f"{sr:.3f}" if np.isfinite(sr) else "n/a"
        ams = f"{am:.3f}" if np.isfinite(am) else "n/a"
        out(f"  {g:<7}{f:<5}{len(gs):>4}{srs:>13}{s_unit:>8}{ams:>11}{a_unit:>10}   "
            f"{'; '.join(flags)}")


COLUMN_SETS: dict[str, set[str]] = {}


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"error: {DATA_ROOT} not found — run scripts/fetch_data.sh first.",
              file=sys.stderr)
        return 1

    refs, total = discover()
    out("# IO-VNBD corpus sweep (diagnostic)")
    out(f"\nDiscovered {total} CSVs; {len(refs)} unique (family, session) pairs after "
        f"dedup.\nRead with encoding={ENCODING!r}.")
    counts = Counter((r.tree, r.subtree) for r in refs)
    out("\nChosen copies by tree:")
    for (tree, sub), n in sorted(counts.items()):
        out(f"  {tree:<16}/ {sub:<14} {n:>4}")

    rows = []
    for i, ref in enumerate(refs, 1):
        print(f"\r  reading {i}/{len(refs)}: {ref.path.name:<24}", end="", file=sys.stderr)
        row = measure(ref)
        rows.append(row)
        try:
            df_cols = pd.read_csv(ref.path, encoding=ENCODING, skipinitialspace=True,
                                  nrows=0)
            COLUMN_SETS[row.get("columns_hash", "")] = {norm(c) for c in df_cols.columns}
        except Exception:
            pass
    print(file=sys.stderr)

    df = pd.DataFrame(rows)
    order = ["filename", "session", "group", "family", "tree", "subtree",
             "n_rows", "duration_s", "distance_km",
             "dt_median_ms", "dt_p05_ms", "dt_p95_ms", "effective_hz", "hz_tick", "dt_mode_ms", "bimodal_dt", "hz_from_date",
             "duration_s_date", "rows_per_s", "dup_timestamp_frac", "accel_mag_median", "gyro_abs_p99", "speed_ratio",
             "n_columns", "columns_hash",
             "lat_min", "lat_max", "lon_min", "lon_max", "relpath", "error"]
    df = df.reindex(columns=[c for c in order if c in df.columns])
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)

    bad = df[df["error"] != ""]
    if not bad.empty:
        out(f"\n!! {len(bad)} files failed to read:")
        for _, r in bad.iterrows():
            out(f"   {r['filename']}: {r['error']}")

    summary_rate(df)
    summary_schemas(df)
    summary_units(df)

    MD_OUT.write_text("```\n" + out.buf.getvalue() + "\n```\n", encoding="utf-8")
    print(f"\nwrote {CSV_OUT.relative_to(REPO_ROOT)} and {MD_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
