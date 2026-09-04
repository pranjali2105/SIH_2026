"""Does time-sync error actually matter for the labels we train on?

The sync estimates are noisy, so the useful question is not "what is the
offset" but "how much would getting it wrong cost us". This injects known
offsets and measures the resulting change in the displacement labels.

The expectation is that sub-500 ms offsets barely register: the label
integrates GPS displacement over a full second, and the input window spans
ten, so a fraction-of-a-second shift moves both by a small part of their own
support. If that holds, sync is documented as non-critical with evidence
rather than assumed to be.

Run:  python -m data.sync_sensitivity
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .loader import list_sessions, load_session
from .sanity import MAX_PLAUSIBLE_SPEED_MS, displacement_labels

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "sanity" / "sync_sensitivity.csv"
OUT_MD = REPO_ROOT / "results" / "sanity" / "sync_sensitivity.md"

OFFSETS_MS = (0, 100, -100, 250, -250, 500, -500, 1000, -1000)
LABEL_WINDOW_S = 1.0        # displacement per second -- the prediction target
INPUT_WINDOW_S = 10.0       # IMU context the model sees
GRID_HZ = 10.0
DEFAULT_N_SESSIONS = 6


def sensitivity_for_session(name: str) -> list[dict]:
    """Label change induced by each injected offset, for one session."""
    s = load_session(name, check_rate=False)
    lab = displacement_labels(s)
    if lab is None:
        return [{"session": name, "note": "no usable GPS labels"}]
    t, y, _rejected = lab
    mean_disp = float(np.mean(y))
    if not np.isfinite(mean_disp) or mean_disp <= 0:
        return [{"session": name, "note": "zero mean displacement"}]

    rows = []
    for off_ms in OFFSETS_MS:
        # An offset does not change the label VALUES; it changes which input
        # window is paired with which label. So the cost is the difference
        # between the label a window should get and the one it would get.
        shift = int(round(off_ms / 1000.0 * GRID_HZ))
        if shift == 0:
            shifted = y
        elif shift > 0:
            shifted = np.concatenate([y[shift:], np.full(shift, np.nan)])
        else:
            shifted = np.concatenate([np.full(-shift, np.nan), y[:shift]])
        ok = np.isfinite(shifted)
        if ok.sum() < 100:
            continue
        mad = float(np.mean(np.abs(shifted[ok] - y[ok])))
        rows.append({
            "session": name,
            "role": s.role,
            "group": s.group,
            "offset_ms": off_ms,
            "mean_displacement_m": mean_disp,
            "mean_abs_label_diff_m": mad,
            "pct_of_mean_displacement": 100.0 * mad / mean_disp,
            "n_labels": int(ok.sum()),
            "note": "",
        })
    return rows


# ---------------------------------------------------------------------------
# label quality (corpus-wide)
# ---------------------------------------------------------------------------

LABEL_QUALITY_CSV = REPO_ROOT / "results" / "sanity" / "label_quality.csv"
STABILITY_OFFSET_MS = 500.0

# label_stability_pct is a RATIO, so it explodes when the denominator is small.
# A session that barely moves scores badly for being slow, not for having bad
# GPS: V-Vw1 averages 0.02 m per second and scores 35%. Below this threshold
# the score is not interpretable and the session is excluded from the suspect
# list rather than judged by it.
LOW_DENOMINATOR_M = 2.0

# Cut points are proposed from the observed distribution in main(), not fixed
# here; these are only the labels attached to whatever cuts are chosen.
BAND_NAMES = ("clean", "usable", "suspect")


def label_stability(s) -> dict:
    """Mean absolute label change at +/-500 ms, as % of mean displacement.

    Originally built to measure sync sensitivity, this actually measures GPS
    LABEL QUALITY: a session whose displacement labels swing wildly when the
    window moves half a second has noisy GPS positions, regardless of any
    clock offset. That makes it usable on groups T, A and I, which have no ECU
    data and therefore no other way to assess label quality.
    """
    name = f"{s.family}-{s.session}"
    out = {"session": name, "role": s.role, "group": s.group,
           "label_stability_pct": np.nan, "mean_displacement_m": np.nan,
           "n_windows": 0, "n_windows_rejected": 0,
           "pct_windows_rejected": np.nan, "low_denominator": False,
           "note": ""}
    lab = displacement_labels(s)
    if lab is None:
        out["note"] = "no usable GPS labels"
        return out
    _, y, rejected = lab
    out["n_windows_rejected"] = rejected
    mean_disp = float(np.mean(y))
    if not np.isfinite(mean_disp) or mean_disp <= 0:
        out["note"] = "zero mean displacement"
        return out

    shift = int(round(STABILITY_OFFSET_MS / 1000.0 * GRID_HZ))
    diffs = []
    for sgn in (+1, -1):
        k = sgn * shift
        if k > 0:
            a, b = y[k:], y[:-k]
        else:
            a, b = y[:k], y[-k:]
        if a.size < 100:
            continue
        diffs.append(np.mean(np.abs(a - b)))
    if not diffs:
        out["note"] = "too few windows"
        return out
    mad = float(np.mean(diffs))
    out.update(label_stability_pct=100.0 * mad / mean_disp,
               mean_displacement_m=mean_disp, n_windows=int(y.size),
               n_windows_rejected=rejected,
               pct_windows_rejected=100.0 * rejected / max(rejected + y.size, 1),
               low_denominator=bool(mean_disp < LOW_DENOMINATOR_M))
    return out


def corpus_label_quality() -> pd.DataFrame:
    sessions = list_sessions()
    rows = []
    for i, r in enumerate(sessions.itertuples(), 1):
        name = f"{r.family}-{r.session}"
        print(f"\r  [{i}/{len(sessions)}] {name:<20}", end="", file=sys.stderr)
        try:
            rows.append(label_stability(load_session(name, check_rate=False)))
        except Exception as exc:
            rows.append({"session": name, "role": r.role, "group": r.group,
                         "label_stability_pct": np.nan,
                         "mean_displacement_m": np.nan, "n_windows": 0,
                         "low_denominator": False,
                         "note": f"load failed: {type(exc).__name__}"})
    print(file=sys.stderr)
    df = pd.DataFrame(rows)
    LABEL_QUALITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABEL_QUALITY_CSV, index=False)
    return df


def validate_against_ecu(quality: pd.DataFrame) -> tuple[float, float, int]:
    """Spearman rho between label instability and S-/V- speed correlation.

    If these rank-correlate on the Fiesta pairs -- where ECU ground truth
    exists -- then label_stability is a validated GPS-quality proxy and can be
    trusted on T, A and I, which have no V- files at all.
    """
    pair_csv = REPO_ROOT / "results" / "sanity" / "phone_vs_ecu_sync.csv"
    if not pair_csv.exists():
        return float("nan"), float("nan"), 0
    pairs = pd.read_csv(pair_csv)
    if "speed_correlation" not in pairs.columns:
        return float("nan"), float("nan"), 0
    q = quality.copy()
    q["key"] = q["session"].str.replace(r"^S-", "", regex=True).str.lower()
    pairs["key"] = pairs["session"].astype(str).str.lower()
    m = q[q["session"].str.startswith("S-")].merge(pairs, on="key",
                                                   suffixes=("", "_pair"))
    m = m.dropna(subset=["label_stability_pct", "speed_correlation"])
    if len(m) < 5:
        return float("nan"), float("nan"), len(m)
    rho, pval = spearmanr(m["label_stability_pct"], m["speed_correlation"])
    return float(rho), float(pval), len(m)


def propose_cut_points(v: pd.Series) -> tuple[float, float]:
    """Cut points from the observed distribution, not round numbers.

    Uses quantiles of the actual spread so the bands reflect this corpus
    rather than an assumption about it.
    """
    clean = float(np.percentile(v, 60))
    suspect = float(np.percentile(v, 90))
    return clean, suspect


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--offsets-only":
        argv = argv[1:]
        return _offsets_report(argv)

    quality = corpus_label_quality()
    ok = quality.dropna(subset=["label_stability_pct"]).copy()
    if "low_denominator" not in ok.columns:
        ok["low_denominator"] = False
    ok["low_denominator"] = ok["low_denominator"].fillna(False).astype(bool)
    # Cut points are derived from interpretable scores only; including
    # low-denominator sessions would drag the quantiles toward slow sessions.
    scored = ok[~ok["low_denominator"]]
    v = scored["label_stability_pct"]

    lines: list[str] = []
    w = lines.append
    w("# Label quality (GPS), corpus-wide\n")
    w("`label_stability_pct` = mean absolute change in the 1 s displacement "
      f"label when the window moves +/-{STABILITY_OFFSET_MS:.0f} ms, as a "
      "percentage of mean displacement.\n")
    w("\nBuilt as a sync-sensitivity test, it turns out to measure GPS label "
      "quality: a session whose labels swing when the window shifts half a "
      "second has noisy positions, independent of any clock offset. That makes "
      "it usable on groups T, A and I, which have no ECU data.\n")
    w(f"\nScored: **{len(ok)}/{len(quality)}** sessions "
      f"({int(ok['low_denominator'].sum())} of them flagged "
      f"`low_denominator`, mean displacement < {LOW_DENOMINATOR_M:g} m, and "
      "excluded from the suspect lists and from the cut-point quantiles).\n")

    rho, pval, n_pairs = validate_against_ecu(quality)
    w("\n## Validation against ECU ground truth\n")
    if np.isfinite(rho):
        w(f"Spearman rho between `label_stability_pct` and the S-/V- speed "
          f"correlation, over **{n_pairs}** Fiesta pairs: "
          f"**rho = {rho:+.3f}** (p = {pval:.2g}).\n")
        w("\nA strong negative rho is the expected direction: worse GPS "
          "(lower speed correlation) should mean less stable labels (higher "
          "percentage).\n")
    else:
        w("Not computable — the pair diagnostic has not been run.\n")

    w("\n## Distribution\n")
    w("```")
    edges = [0, 1, 2, 3, 5, 10, 20, 50, 100, np.inf]
    counts, _ = np.histogram(v, bins=edges)
    for lo_e, hi_e, c in zip(edges[:-1], edges[1:], counts):
        hi_s = "inf" if not np.isfinite(hi_e) else f"{hi_e:g}"
        bar = "#" * int(round(40 * c / max(counts.max(), 1)))
        w(f"  {lo_e:>5g} - {hi_s:>5}%  {c:>4}  {bar}")
    w("```")
    w(f"\nquantiles: p50 {v.median():.2f}%, p60 {np.percentile(v, 60):.2f}%, "
      f"p75 {np.percentile(v, 75):.2f}%, p90 {np.percentile(v, 90):.2f}%, "
      f"p95 {np.percentile(v, 95):.2f}%, max {v.max():.2f}%\n")

    clean_cut, suspect_cut = propose_cut_points(v)
    w("\n## Proposed cut points\n")
    w("Derived from this corpus's own distribution (p60 and p90) rather than "
      "chosen as round numbers:\n")
    w(f"\n- **clean**: < {clean_cut:.2f}%")
    w(f"- **usable (down-weight)**: {clean_cut:.2f}% - {suspect_cut:.2f}%")
    w(f"- **suspect**: > {suspect_cut:.2f}%\n")

    def band(x: float) -> str:
        return ("clean" if x < clean_cut
                else "usable" if x <= suspect_cut else "suspect")

    ok["band"] = ok["label_stability_pct"].map(band)
    # A low-denominator session is slow, not bad: it is never suspect.
    ok.loc[ok["low_denominator"], "band"] = "low_denominator"
    w("\n### Band counts by role\n")
    w("```")
    w(ok.pivot_table(index="role", columns="band", values="session",
                     aggfunc="count", fill_value=0).to_string())
    w("```")

    w("\n## What each application would exclude (NOTHING IS DROPPED YET)\n")
    for role, action in (("train", "down-weight or exclude"),
                         ("validate", "exclude, so early stopping is not "
                                      "driven by label noise"),
                         ("test", "exclude entirely — unreliable ground truth "
                                  "corrupts the headline number")):
        sub = ok[ok["role"] == role]
        bad = sub[sub["band"] == "suspect"]   # low_denominator can never be suspect
        w(f"\n### {role} — {action}\n")
        w(f"{len(sub)} sessions scored; **{len(bad)} suspect**.\n")
        if not bad.empty:
            w("```")
            w(bad[["session", "group", "label_stability_pct",
                   "mean_displacement_m", "n_windows"]]
              .sort_values("label_stability_pct", ascending=False)
              .to_string(index=False))
            w("```")

    w("\n### Worst 15 interpretable sessions\n")
    w("```")
    w(scored.nlargest(15, "label_stability_pct")[
        ["session", "role", "group", "label_stability_pct",
         "mean_displacement_m"]].to_string(index=False))
    w("```")

    low = ok[ok["low_denominator"]]
    w(f"\n### Excluded as low-denominator ({len(low)})\n")
    w("Slow sessions, not bad GPS: the ratio's denominator is too small for "
      "the percentage to mean anything.\n")
    if not low.empty:
        w("```")
        w(low.sort_values("label_stability_pct", ascending=False)[
            ["session", "role", "group", "label_stability_pct",
             "mean_displacement_m"]].to_string(index=False))
        w("```")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    (OUT_MD.parent / "label_quality.md").write_text("\n".join(lines),
                                                    encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {LABEL_QUALITY_CSV.relative_to(REPO_ROOT)} and "
          f"{(OUT_MD.parent / 'label_quality.md').relative_to(REPO_ROOT)}")
    return 0


def _offsets_report(names: list[str]) -> int:
    """The original per-offset sensitivity table (0, +/-100 ... +/-1000 ms)."""
    if not names:
        train = list_sessions(role="train")
        train = train[train["family"] == "S"]
        names = (train.sort_values("session").drop_duplicates("group")
                 .head(DEFAULT_N_SESSIONS)
                 .apply(lambda r: f"{r.family}-{r.session}", axis=1).tolist())
    rows: list[dict] = []
    for name in names:
        print(f"  {name}", file=sys.stderr)
        try:
            rows.extend(sensitivity_for_session(name))
        except Exception as exc:
            rows.append({"session": name, "note": f"{type(exc).__name__}: {exc}"})
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    ok = df[df.get("pct_of_mean_displacement", pd.Series(dtype=float)).notna()]
    if not ok.empty:
        piv = ok.pivot_table(index=["session", "group"], columns="offset_ms",
                             values="pct_of_mean_displacement").round(2)
        print(piv.to_string())
    print(f"\nwrote {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
