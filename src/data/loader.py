"""IO-VNBD loader: session discovery, role assignment, and unit-corrected reads.

Every non-obvious behaviour here exists because the corpus was measured, not
assumed -- see CLAUDE.md and results/corpus_sweep.csv.

Key invariants enforced at load time:

* Exactly one copy of each (family, session) is read, preferring the
  Synchronised tree; session names match case-insensitively.
* Every session receives exactly one role, and every 10 Hz S- file is covered.
* `^UNNAMED` columns (trailing-comma artefacts) are dropped.
* Speed is unit-corrected by FAMILY, overriding the header: S- values are
  already m/s despite reading "(Kmh)"; V- values are genuine km/h.
* The sampling rate is measured per session as `hz_tick` and checked against
  the rate the split claims. A session tagged 10 Hz that measures outside
  8-12 Hz raises rather than loading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw"

# Files are ISO-8859-1 with mixed internal encodings ("m/s²" latin-1, "µT"
# UTF-8 on the same header line). UTF-8 raises.
ENCODING = "iso-8859-1"

SYNC_TREE = "Synchronised V abd S datasets"

# Longest-first so Vta/Vtb/Vfa/Vfb win over shorter prefixes.
KNOWN_GROUPS = ["Vta", "Vtb", "Vfa", "Vfb", "Vw", "St", "S", "M", "Y", "T", "A", "I"]

ROLES = ("train", "validate", "test", "robustness_2hz", "ood_spotcheck",
         "stage0_only", "excluded")

TRAIN_GROUPS = frozenset({"S", "M", "Y", "Vta", "Vtb", "Vw", "Vfa"})
TEST_SESSIONS = frozenset({"a5", "a6", "a7", "a8"})
ROBUSTNESS_2HZ_SESSIONS = frozenset({"a1", "a2", "a3", "a9", "a10", "a11", "a12", "a13"})
# Excluded sessions and why. Reasons are recorded here rather than in a
# comment so they survive into anything that reports on the split.
EXCLUSION_REASONS = {
    "a4": "no_usable_timestamps",        # and a stray UNNAMED: 24 column
    "y1": "poor_gps_label_quality",      # 7.1% label instability AND the
                                         # corpus-worst 0.11 correlation
                                         # against ECU speed -- two
                                         # independent measures agreeing
    # Fix-to-fix path length disagrees with the integral of the session's own
    # reported speed by 2.1-3.5x: the GPS positions are inflated, so every
    # displacement label derived from them is too. The 60 m/s window guard
    # trims the tail (69-94% of their windows) but does not repair what
    # survives -- their post-guard label means are 30-42 m against 4-27 m for
    # the clean T sessions.
    "t1": "inflated_gps_positions",      # path/speed-integral 3.48
    "t4": "inflated_gps_positions",      # 2.06
    "t5": "inflated_gps_positions",      # 2.19
    "t6": "inflated_gps_positions",      # 3.37
}
EXCLUDED_SESSIONS = frozenset(EXCLUSION_REASONS)

# Rate each role is expected to hold, and the tolerance band a session must
# measure within. A rate assumption that silently drifts is exactly the failure
# Stage 0 cannot catch, so this is enforced, not documented.
ROLE_EXPECTED_HZ = {
    "train": 10.0, "validate": 10.0, "test": 10.0,
    "ood_spotcheck": 10.0, "stage0_only": 10.0,
    "robustness_2hz": 2.0,
    "excluded": None,
}
HZ_TOLERANCE = {10.0: (8.0, 12.0), 2.0: (1.6, 2.4)}

# Deltas below this are intra-tick: AndroSensor writes one row per sensor
# event, so several rows share a tick ~1 ms apart. Including them drags the
# median into the intra-tick spike and reports ~1000 Hz.
MIN_TICK_MS = 20.0

KMH_TO_MS = 1.0 / 3.6

# V- acceleration channels are in g. (Migrated from the deleted src/iovnbd/;
# the lat/lon helpers that lived beside this one applied the refuted
# arc-minute conversion and were removed rather than moved.)
G_TO_MS2 = 9.80665


def v_accel_ms2(accel_g):
    """Convert a V- acceleration channel from g to m/s^2."""
    return accel_g * G_TO_MS2

_TREE_PRIORITY = {
    ("Synchronised", "Categorised"): 0,
    ("Synchronised", "Uncategorised"): 1,
    ("Unsynchronised", "Categorised"): 2,
    ("Unsynchronised", "Uncategorised"): 3,
}


class RateMismatchError(RuntimeError):
    """A session's measured rate contradicts the rate its role claims."""


class SplitCoverageError(RuntimeError):
    """A session received no role, or a 10 Hz S- file was left uncovered."""


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------

def _norm(col: str) -> str:
    return re.sub(r"\s+", " ", str(col).strip()).upper()


def parse_group(session: str) -> str:
    low = session.lower()
    for g in KNOWN_GROUPS:
        if low.startswith(g.lower()):
            return g
    return "?"


def exclusion_reason(family: str, session: str) -> str | None:
    """Why a session is excluded, or None if it is not."""
    if family.upper() == "V":
        return None
    return EXCLUSION_REASONS.get(session.lower())


def session_role(family: str, session: str) -> str:
    """The single role for one (family, session) pair.

    `stage0_only` is family-wide: every V- file gets it, including V- files
    whose group is a train group. V- data feeds the Stage 0 INS check only.
    """
    if family.upper() == "V":
        return "stage0_only"
    key = session.lower()
    if key in EXCLUDED_SESSIONS:
        return "excluded"
    if key in TEST_SESSIONS:
        return "test"
    if key in ROBUSTNESS_2HZ_SESSIONS:
        return "robustness_2hz"
    group = parse_group(session)
    if group == "T":
        return "validate"
    if group == "I":
        return "ood_spotcheck"
    if group in TRAIN_GROUPS:
        return "train"
    raise SplitCoverageError(
        f"no role for {family}-{session} (group {group!r}); "
        "the split in CLAUDE.md and loader.py must cover every session")


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionRef:
    path: Path
    family: str
    session: str
    group: str
    tree: str
    subtree: str
    role: str


def _discover(data_root: Path) -> list[SessionRef]:
    """One SessionRef per (family, session), Synchronised tree preferred.

    Sessions are keyed case-insensitively: both V-Vta10.csv and V-vta10.csv
    exist for the same session and must not be read twice.
    """
    best: dict[tuple[str, str], tuple[int, SessionRef]] = {}
    for path in sorted(data_root.rglob("*.csv")):
        if ".git" in path.parts:
            continue
        m = re.match(r"^([SV])-(.+)$", path.stem, flags=re.IGNORECASE)
        if not m:
            continue
        family, session = m.group(1).upper(), m.group(2)
        rel = path.relative_to(data_root).as_posix()
        tree = "Synchronised" if rel.startswith(SYNC_TREE) else "Unsynchronised"
        subtree = ("Categorised" if "Categorised" in rel and "Uncategorised" not in rel
                   else "Uncategorised")
        priority = _TREE_PRIORITY[(tree, subtree)]
        key = (family, session.lower())
        if key in best and best[key][0] <= priority:
            continue
        best[key] = (priority, SessionRef(path, family, session,
                                          parse_group(session), tree, subtree,
                                          session_role(family, session)))
    return sorted((ref for _, ref in best.values()),
                  key=lambda r: (r.role, r.family, r.session.lower()))


def list_sessions(data_root: Path | str = DATA_ROOT,
                  role: str | None = None) -> pd.DataFrame:
    """All unique sessions with their assigned role.

    Asserts that every session gets exactly one role and that no 10 Hz S- file
    is left uncovered -- if the split and the corpus drift apart, this raises
    instead of quietly training on the wrong data.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found — run scripts/fetch_data.sh first.")
    refs = _discover(root)
    df = pd.DataFrame([{
        "family": r.family, "session": r.session, "group": r.group,
        "role": r.role, "tree": r.tree, "subtree": r.subtree,
        "exclusion_reason": exclusion_reason(r.family, r.session),
        "path": str(r.path), "filename": r.path.name,
    } for r in refs])

    if df.empty:
        raise SplitCoverageError(f"no session CSVs found under {root}")

    dupes = df.duplicated(subset=["family", "session"], keep=False)
    if dupes.any():
        raise SplitCoverageError(
            f"session read twice: {sorted(df.loc[dupes, 'filename'])}")

    bad = sorted(set(df["role"]) - set(ROLES))
    if bad:
        raise SplitCoverageError(f"unknown role(s): {bad}")

    # Every S- file the split claims is 10 Hz must have exactly one role. The
    # roles are mutually exclusive by construction (session_role returns one
    # string), so this checks coverage: no S- session falls through.
    s_files = df[df["family"] == "S"]
    unroled = s_files[~s_files["role"].isin(ROLES)]
    if not unroled.empty:
        raise SplitCoverageError(f"S- sessions without a role: "
                                 f"{sorted(unroled['filename'])}")

    df = _attach_label_stability(df)

    if role is not None:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        df = df[df["role"] == role].reset_index(drop=True)
    return df


LABEL_QUALITY_CSV = REPO_ROOT / "results" / "sanity" / "label_quality.csv"


def _attach_label_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Join per-session GPS label quality, when it has been computed.

    `label_stability_pct` is the mean absolute change in the 1 s displacement
    label under a +/-500 ms window shift, as a percentage of mean displacement
    (see src/data/sync_sensitivity.py). It measures GPS label quality, not
    clock offset, and is the only such measure available for groups T, A and
    I, which have no ECU data. Absent until the sweep has been run; the column
    is then NaN rather than missing, so callers can rely on it existing.
    """
    df = df.copy()
    df["label_stability_pct"] = np.nan
    if not LABEL_QUALITY_CSV.exists():
        return df
    q = pd.read_csv(LABEL_QUALITY_CSV, comment="#")
    if "label_stability_pct" not in q.columns:
        return df
    q["key"] = q["session"].astype(str).str.lower()
    key = (df["family"] + "-" + df["session"]).str.lower()
    mapping = dict(zip(q["key"], q["label_stability_pct"]))
    df["label_stability_pct"] = key.map(mapping)
    return df


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def _find(df: pd.DataFrame, *patterns: str) -> str | None:
    for col in df.columns:
        key = _norm(col)
        if all(re.search(p, key) for p in patterns):
            return col
    return None


def _time_seconds(df: pd.DataFrame) -> pd.Series | None:
    """TIME SINCE START normalised to seconds, scaled from the HEADER.

    Scaling from the delta magnitude misreads S-T1, whose 1 ms quantisation
    makes the median delta 1 and looks like seconds; its span (738388 ms) in
    fact matches its DATE span of 738.4 s exactly.
    """
    col = _find(df, r"TIME SINCE START")
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() < 2:
        return None
    key = _norm(col)
    if "(MS)" in key or "MILLISEC" in key:
        return s / 1000.0
    if "SECOND" in key:
        return s
    d = s.diff()
    d = d[d > 0]
    return s / (1000.0 if (not d.empty and float(d.median()) > 10) else 1.0)


def compute_hz_tick(t_seconds: pd.Series | None) -> float:
    """Sensor tick rate: 1 / median(delta) over deltas >= MIN_TICK_MS."""
    if t_seconds is None:
        return float("nan")
    d = t_seconds.diff() * 1000.0
    d = d[d >= MIN_TICK_MS]
    if len(d) < 20:
        return float("nan")
    med = float(d.median())
    return 1000.0 / med if med > 0 else float("nan")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

@dataclass
class Session:
    """One loaded session: the frame plus what was measured about it."""
    family: str
    session: str
    group: str
    role: str
    path: Path
    df: pd.DataFrame
    hz_tick: float
    n_rows: int
    speed_col: str | None
    dropped_columns: list[str]

    def __repr__(self) -> str:  # keep test output readable
        return (f"Session({self.family}-{self.session}, role={self.role}, "
                f"hz_tick={self.hz_tick:.2f}, rows={self.n_rows})")


def _resolve(name: str, data_root: Path) -> SessionRef:
    """Find one session by name, case-insensitively.

    Accepts 'S-Vw12', 'S-Vw12.csv', or 'Vw12' when unambiguous.
    """
    stem = name[:-4] if name.lower().endswith(".csv") else name
    m = re.match(r"^([SV])-(.+)$", stem, flags=re.IGNORECASE)
    refs = _discover(data_root)
    if m:
        family, session = m.group(1).upper(), m.group(2).lower()
        hits = [r for r in refs if r.family == family and r.session.lower() == session]
    else:
        hits = [r for r in refs if r.session.lower() == stem.lower()]
    if not hits:
        raise FileNotFoundError(f"no session matching {name!r} under {data_root}")
    if len(hits) > 1:
        raise ValueError(f"{name!r} is ambiguous: "
                         f"{sorted(f'{r.family}-{r.session}' for r in hits)}")
    return hits[0]


def load_session(name: str, data_root: Path | str = DATA_ROOT,
                 check_rate: bool = True) -> Session:
    """Load one session with units corrected and its rate verified.

    Raises RateMismatchError if the measured rate contradicts the role's
    expected rate (unless check_rate is False, for diagnostics only).
    """
    root = Path(data_root)
    ref = _resolve(name, root)
    df = pd.read_csv(ref.path, encoding=ENCODING, skipinitialspace=True,
                     low_memory=False)

    # Trailing commas produce phantom columns (S-A4 has UNNAMED: 24).
    dropped = [c for c in df.columns if _norm(c).startswith("UNNAMED")]
    if dropped:
        df = df.drop(columns=dropped)

    t = _time_seconds(df)
    hz = compute_hz_tick(t)
    if t is not None:
        df["time_s"] = t.to_numpy()

    # Speed unit override, by FAMILY not by header. S- reads "GPS SPEED (Kmh)"
    # but the values are m/s (corpus-wide ratio 1.00-1.02 against GPS-derived
    # ground speed). V- speed is genuinely km/h (3.60-3.63).
    speed_col = (_find(df, r"GPS SPEED") or _find(df, r"INDICATED VEHICLE SPEED")
                 or _find(df, r"^VELOCITY"))
    if speed_col is not None:
        vals = pd.to_numeric(df[speed_col], errors="coerce")
        df["speed_ms"] = vals if ref.family == "S" else vals * KMH_TO_MS

    if check_rate:
        expected = ROLE_EXPECTED_HZ.get(ref.role)
        if expected is not None:
            lo, hi = HZ_TOLERANCE[expected]
            if not np.isfinite(hz) or not (lo <= hz <= hi):
                measured = f"{hz:.3f}" if np.isfinite(hz) else "undeterminable"
                raise RateMismatchError(
                    f"{ref.family}-{ref.session} has role {ref.role!r}, which "
                    f"expects {expected:g} Hz (accepting {lo:g}-{hi:g}), but "
                    f"measured hz_tick = {measured}. Either the file changed or "
                    f"the split in CLAUDE.md is stale — do not train on it "
                    f"until this is resolved.")

    return Session(family=ref.family, session=ref.session, group=ref.group,
                   role=ref.role, path=ref.path, df=df, hz_tick=hz,
                   n_rows=len(df), speed_col=speed_col, dropped_columns=dropped)
