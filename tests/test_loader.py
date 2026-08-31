"""Loader tests against the real corpus.

These read the actual dataset (data/raw), so they are skipped when it is
absent. The point is to catch the corpus and the split drifting apart, which
no synthetic fixture can do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# No sys.path manipulation: the package is installed (pip install -e .), and
# these tests must exercise the installed package, not a path shim.
from data.loader import (
    ROLES,
    RateMismatchError,
    compute_hz_tick,
    list_sessions,
    load_session,
    session_role,
)

DATA_ROOT = REPO_ROOT / "data" / "raw"

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="dataset absent — run scripts/fetch_data.sh")


# One session per role, with the rate each must measure. These are the
# assertions that fail loudly if the corpus changes underneath the split.
ROLE_EXEMPLARS = [
    ("train", "S-Vw12", 10.0),
    ("validate", "S-T2", 10.0),      # 18-col schema; 1000 Hz under a naive median
    ("test", "S-A5", 10.0),
    ("robustness_2hz", "S-A1", 2.0),
]


@pytest.fixture(scope="module")
def sessions():
    return list_sessions()


# -- split integrity -------------------------------------------------------

def test_expected_session_count(sessions):
    assert len(sessions) == 187


def test_every_session_has_exactly_one_known_role(sessions):
    assert set(sessions["role"]) <= set(ROLES)
    assert sessions["role"].notna().all()
    # One row per (family, session) means one role per session.
    assert not sessions.duplicated(subset=["family", "session"]).any()


def test_role_counts_match_the_documented_split(sessions):
    assert sessions.groupby("role").size().to_dict() == {
        "train": 71, "validate": 7, "test": 4, "robustness_2hz": 8,
        "ood_spotcheck": 1, "stage0_only": 90, "excluded": 6,
    }


def test_stage0_only_is_family_wide(sessions):
    """Every V- file is stage0_only, including V- files in train groups."""
    v = sessions[sessions["family"] == "V"]
    assert (v["role"] == "stage0_only").all()
    assert set(sessions.loc[sessions["role"] == "stage0_only", "family"]) == {"V"}
    # V- files exist for train groups; they must NOT be labelled train.
    assert "Vw" in set(v["group"])


def test_group_a_splits_by_rate_not_by_prefix(sessions):
    a = sessions[(sessions["family"] == "S") & (sessions["group"] == "A")]
    assert set(a["role"]) == {"test", "robustness_2hz", "excluded"}


def test_session_matching_is_case_insensitive(sessions):
    """V-Vta10.csv and V-vta10.csv are one session, not two."""
    vta10 = sessions[(sessions["family"] == "V")
                     & (sessions["session"].str.lower() == "vta10")]
    assert len(vta10) == 1


def test_synchronised_tree_is_preferred(sessions):
    """A session present in both trees is read from Synchronised."""
    both = sessions[sessions["session"].str.lower() == "vta10"]
    assert (both["tree"] == "Synchronised").all()


def test_list_sessions_filters_by_role(sessions):
    train = list_sessions(role="train")
    assert len(train) == 71
    assert set(train["role"]) == {"train"}


def test_unknown_role_rejected():
    with pytest.raises(ValueError, match="unknown role"):
        list_sessions(role="nonexistent")


def test_session_role_rejects_uncovered_group():
    from data.loader import SplitCoverageError
    with pytest.raises(SplitCoverageError):
        session_role("S", "Zz99")


# -- per-role loading ------------------------------------------------------

@pytest.mark.parametrize("role,name,expected_hz", ROLE_EXEMPLARS,
                         ids=[r for r, _, _ in ROLE_EXEMPLARS])
def test_exemplar_loads_with_expected_rate(role, name, expected_hz):
    s = load_session(name)
    assert s.role == role
    assert s.hz_tick == pytest.approx(expected_hz, abs=0.5), (
        f"{name} measured {s.hz_tick:.3f} Hz, expected ~{expected_hz}")
    assert s.n_rows > 0
    assert "time_s" in s.df.columns


@pytest.mark.parametrize("role,name,expected_hz", ROLE_EXEMPLARS,
                         ids=[r for r, _, _ in ROLE_EXEMPLARS])
def test_exemplar_has_no_unnamed_columns(role, name, expected_hz):
    s = load_session(name)
    assert not [c for c in s.df.columns if str(c).upper().startswith("UNNAMED")]


def test_s_t1_rate_survives_the_bimodal_timestamp_trap():
    """S-T1's median delta is 1 ms (intra-tick rows); its real tick is 10 Hz.

    S-T1 is now `excluded` for inflated GPS positions, but its RATE was always
    fine — the two defects are independent and this pins the distinction.
    """
    s = load_session("S-T1", check_rate=False)
    assert s.hz_tick == pytest.approx(10.0, abs=0.5)
    assert s.role == "excluded"


# -- units -----------------------------------------------------------------

def test_smartphone_speed_is_left_in_ms_despite_the_kmh_header():
    s = load_session("S-Vw12")
    assert s.speed_col is not None and "KMH" in s.speed_col.upper()
    raw = s.df[s.speed_col].astype(float)
    assert (s.df["speed_ms"] - raw).abs().max() < 1e-9


def test_vehicle_speed_is_converted_from_kmh():
    s = load_session("V-Vw12")
    assert s.role == "stage0_only"
    raw = s.df[s.speed_col].astype(float)
    assert (s.df["speed_ms"] - raw / 3.6).abs().max() < 1e-9
    assert s.df["speed_ms"].max() < raw.max()


def test_smartphone_accelerometer_is_ms2():
    import numpy as np
    s = load_session("S-Vw12")
    axes = [c for c in s.df.columns if c.upper().startswith("ACCELEROMETER")]
    assert len(axes) == 3
    mag = np.linalg.norm(s.df[axes].astype(float).to_numpy(), axis=1)
    assert 8.5 < float(np.nanmedian(mag)) < 11.0


# -- rate guard ------------------------------------------------------------

def test_rate_mismatch_raises(monkeypatch):
    """A 10 Hz-tagged session that measures 2 Hz must refuse to load."""
    import data.loader as loader
    monkeypatch.setitem(loader.ROLE_EXPECTED_HZ, "robustness_2hz", 10.0)
    with pytest.raises(RateMismatchError, match="expects 10 Hz"):
        load_session("S-A1")


def test_excluded_session_has_no_rate_expectation():
    """S-A4 is broken; loading it must not raise on rate."""
    s = load_session("S-A4")
    assert s.role == "excluded"
    assert s.dropped_columns, "S-A4 should have a stray UNNAMED column"


def test_compute_hz_tick_ignores_intra_tick_deltas():
    import pandas as pd
    # 100 ms ticks with two extra rows 1 and 2 ms after each tick, mimicking
    # AndroSensor's one-row-per-sensor-event writing. The boundary gap is
    # 98 ms (not 100), so the honest answer is ~10.2 Hz -- the point is that
    # the intra-tick 1 ms deltas do not drag it to ~1000 Hz.
    t = []
    for i in range(200):
        base = i * 0.1
        t += [base, base + 0.001, base + 0.002]
    hz = compute_hz_tick(pd.Series(t))
    assert hz == pytest.approx(10.0, abs=0.5)
    assert hz < 50, "intra-tick deltas leaked into the rate estimate"


# -- single source of truth ------------------------------------------------

EXPECTED_ROLE_COUNTS = {
    "train": 71, "validate": 7, "test": 4, "robustness_2hz": 8,
    "ood_spotcheck": 1, "stage0_only": 90, "excluded": 6,
}


def test_split_has_exactly_one_source_of_truth(sessions):
    """list_sessions() is the only place the split is defined.

    Guards against a second, drifting definition reappearing -- the failure
    mode that src/iovnbd/splits.py represented before it was deleted.
    """
    # 1. the corpus is fully enumerated
    assert len(sessions) == 187

    # 2. role counts match the documented split exactly
    assert sessions.groupby("role").size().to_dict() == EXPECTED_ROLE_COUNTS
    assert sum(EXPECTED_ROLE_COUNTS.values()) == 187

    # 3. nothing is unassigned
    assert sessions["role"].notna().all()
    assert (sessions["role"] != "").all()
    assert set(sessions["role"]) == set(EXPECTED_ROLE_COUNTS)

    # 4. no session appears under two roles
    per_session = sessions.groupby(["family", "session"])["role"].nunique()
    assert (per_session == 1).all(), (
        f"sessions with multiple roles: "
        f"{sorted(per_session[per_session > 1].index)}")
    assert len(per_session) == 187

    # 5. the union of roles reconstructs the corpus with no overlap
    total = sum(len(list_sessions(role=r)) for r in EXPECTED_ROLE_COUNTS)
    assert total == 187


def test_old_split_module_is_gone():
    """The stale import path must fail loudly, not resolve to a shim."""
    import importlib
    for name in ("iovnbd.splits", "data.splits", "splits"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_no_module_level_rate_constant():
    """Rate is a property of a session, never of the project."""
    import data.loader as loader
    assert not hasattr(loader, "IMU_HZ")
    assert not hasattr(loader, "NYQUIST_HZ")
    assert not hasattr(loader, "GPS_HZ")


def test_excluded_sessions_carry_a_reason(sessions):
    """Exclusions must say why, so a future reader can re-litigate them."""
    excl = sessions[sessions["role"] == "excluded"]
    assert set(excl["session"].str.lower()) == {"a4", "y1", "t1", "t4", "t5", "t6"}
    assert excl["exclusion_reason"].notna().all()
    reasons = dict(zip(excl["session"].str.lower(), excl["exclusion_reason"]))
    assert reasons["a4"] == "no_usable_timestamps"
    assert reasons["y1"] == "poor_gps_label_quality"


def test_only_excluded_sessions_have_a_reason(sessions):
    kept = sessions[sessions["role"] != "excluded"]
    assert kept["exclusion_reason"].isna().all()


def test_group_y_has_no_trainable_smartphone_session(sessions):
    """S-Y1 was the only S- file in group Y; excluding it empties the group."""
    y = sessions[(sessions["group"] == "Y") & (sessions["family"] == "S")]
    assert set(y["role"]) == {"excluded"}
