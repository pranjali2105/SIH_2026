"""Does dropping the suspect validation sessions change what validation says?

The validate split has 11 sessions, 4 of them flagged suspect on GPS label
quality. Dropping them costs data; keeping them risks early stopping being
driven by label noise. Rather than guess, this measures whether the two
candidate splits RANK predictors the same way -- which is the only thing a
validation set has to do reliably.

Three deliberately arbitrary predictors, none trained on anything meaningful:

  mean        - predict the training-set mean displacement, always
  previous    - predict the previous second's displacement (persistence)
  accel_var   - a single-feature least-squares fit on accelerometer variance

If both splits rank these identically, the suspect sessions are not distorting
the signal and can be kept for the extra data. If the ranking flips, the
validation set is being driven by label noise and must be trimmed.

Run:  python -m data.val_split_check
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import list_sessions, load_session
from .sanity import (MAX_PLAUSIBLE_SPEED_MS, gps_cumulative_distance,
                     level_frame_components)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "sanity" / "val_split_check.csv"
OUT_MD = REPO_ROOT / "results" / "sanity" / "val_split_check.md"

GRID_HZ = 10.0
LABEL_WINDOW_S = 1.0        # displacement per second: the target
INPUT_WINDOW_S = 10.0       # IMU context
N_TRAIN_FIT = 8             # train sessions used to fit the trivial models
SUSPECT_CUT = 5.21          # from label_quality.md (p90 of interpretable scores)


def session_windows(name: str):
    """(accel_variance, previous_displacement, label) per window."""
    s = load_session(name, check_rate=False)
    gps = gps_cumulative_distance(s)
    if gps is None:
        return None
    t, cum = gps
    lo, hi = float(t.min()), float(t.max())
    if not np.isfinite(lo) or hi - lo < 5 * INPUT_WINDOW_S:
        return None
    grid = np.arange(lo, hi, 1.0 / GRID_HZ)
    c = np.interp(grid, t, cum)

    w_lab = int(LABEL_WINDOW_S * GRID_HZ)
    w_in = int(INPUT_WINDOW_S * GRID_HZ)
    if c.size <= w_in + w_lab:
        return None

    comp = level_frame_components(s, grid)
    if comp is None:
        return None
    a1, a2 = comp
    mag = np.sqrt(np.nan_to_num(a1) ** 2 + np.nan_to_num(a2) ** 2)

    # Windows step by one second so samples are not near-duplicates.
    step = int(LABEL_WINDOW_S * GRID_HZ)
    starts = np.arange(0, c.size - w_in - w_lab, step)
    if starts.size < 50:
        return None

    var = np.array([np.var(mag[i:i + w_in]) for i in starts])
    end = starts + w_in
    label = c[end + w_lab] - c[end]              # next second's displacement
    prev = c[end] - c[end - w_lab]               # previous second's
    # Same standing plausibility guard as label construction: a window whose
    # implied speed is impossible is corrupt GPS, not a hard sample.
    ok = (np.isfinite(var) & np.isfinite(label) & np.isfinite(prev)
          & (label >= 0) & (label / LABEL_WINDOW_S <= MAX_PLAUSIBLE_SPEED_MS)
          & (prev >= 0) & (prev / LABEL_WINDOW_S <= MAX_PLAUSIBLE_SPEED_MS))
    if ok.sum() < 50:
        return None
    return var[ok], prev[ok], label[ok]


def gather(names: list[str]) -> dict[str, tuple]:
    out = {}
    for n in names:
        print(f"  {n}", file=sys.stderr)
        try:
            r = session_windows(n)
        except Exception as exc:
            print(f"    skipped: {type(exc).__name__}", file=sys.stderr)
            continue
        if r is not None:
            out[n] = r
    return out


def main(argv: list[str] | None = None) -> int:
    sessions = list_sessions()

    train = sessions[(sessions["role"] == "train") & (sessions["family"] == "S")]
    train_names = (train.sort_values("session")
                   .head(N_TRAIN_FIT)
                   .apply(lambda r: f"{r.family}-{r.session}", axis=1).tolist())
    val = sessions[sessions["role"] == "validate"]
    val_names = val.apply(lambda r: f"{r.family}-{r.session}", axis=1).tolist()

    print("fitting on train:", file=sys.stderr)
    tr = gather(train_names)
    if not tr:
        print("no usable train sessions", file=sys.stderr)
        return 1
    tr_var = np.concatenate([v[0] for v in tr.values()])
    tr_lab = np.concatenate([v[2] for v in tr.values()])

    # The three predictors, all fitted (or not) on train only.
    train_mean = float(np.mean(tr_lab))
    slope, intercept = np.polyfit(tr_var, tr_lab, 1)

    print("evaluating on validate:", file=sys.stderr)
    va = gather(val_names)

    rows = []
    for name, (var, prev, lab) in va.items():
        stab = val.loc[(val["family"] + "-" + val["session"]) == name,
                       "label_stability_pct"]
        stab = float(stab.iloc[0]) if len(stab) else np.nan
        rows.append({
            "session": name,
            "label_stability_pct": stab,
            "suspect": bool(np.isfinite(stab) and stab > SUSPECT_CUT),
            "n_windows": int(lab.size),
            "mae_mean": float(np.mean(np.abs(lab - train_mean))),
            "mae_previous": float(np.mean(np.abs(lab - prev))),
            "mae_accel_var": float(np.mean(np.abs(lab - (slope * var + intercept)))),
        })
    df = pd.DataFrame(rows).sort_values("session")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    preds = ["mae_mean", "mae_previous", "mae_accel_var"]

    def pooled(sub: pd.DataFrame) -> dict[str, float]:
        # Weight by window count: a session-mean of MAEs would let a
        # 50-window session outvote a 5000-window one.
        w = sub["n_windows"].to_numpy(float)
        return {p: float(np.sum(sub[p].to_numpy(float) * w) / w.sum()) for p in preds}

    all_11 = pooled(df)
    clean = pooled(df[~df["suspect"]])
    rank_all = sorted(preds, key=lambda p: all_11[p])
    rank_clean = sorted(preds, key=lambda p: clean[p])
    stable = rank_all == rank_clean

    lines: list[str] = []
    w = lines.append
    w("# Validation split check\n")
    w("Do the suspect validation sessions change what validation *says*? "
      "Three arbitrary, untrained predictors are ranked on both candidate "
      "splits; if the ranking holds, the suspect sessions are not distorting "
      "the signal.\n")
    w(f"\nSuspect cut: `label_stability_pct` > {SUSPECT_CUT}%. "
      f"Sessions evaluated: {len(df)} of {len(val_names)}; "
      f"suspect: {int(df['suspect'].sum())}.\n")

    w("\n## Pooled MAE (metres), window-weighted\n")
    w(f"| predictor | all {len(df)} | clean only ({int((~df['suspect']).sum())}) | delta |")
    w("|---|---|---|---|")
    for p in preds:
        w(f"| `{p.replace('mae_', '')}` | {all_11[p]:.3f} | {clean[p]:.3f} | "
          f"{clean[p] - all_11[p]:+.3f} |")

    w("\n## Ranking (best first)\n")
    w(f"- all {len(df)}:      {' < '.join(p.replace('mae_', '') for p in rank_all)}")
    w(f"- clean only: {' < '.join(p.replace('mae_', '') for p in rank_clean)}")
    w(f"\n**Ranking is {'STABLE' if stable else 'NOT stable'}.**\n")
    if stable:
        if int(df["suspect"].sum()) == 0:
            w(f"\nNo suspect sessions remain among the {len(df)}: the two "
              "splits are identical, so the ranking is trivially stable. The "
              "check is retained as a regression guard — if a future change "
              "readmits a noisy session, this will catch it.\n")
        else:
            w(f"\nKeep all {len(df)} sessions: the suspect sessions inflate "
              "the absolute MAE but do not change which predictor wins, which "
              "is the only property early stopping depends on.\n")
    else:
        w("\nDrop to the clean subset: the suspect sessions change which "
          "predictor validation prefers, so early stopping would be driven by "
          "label noise.\n")

    n_clean = int((~df["suspect"]).sum())
    if n_clean < 6:
        w(f"\n**Note:** only {n_clean} clean sessions remain, which is thin. "
          "Consider promoting one or two clean 10 Hz Fiesta groups from train "
          "(50 sessions available) into validate. Not done here.\n")

    w("\n## Per session\n")
    w("```")
    w(df.to_string(index=False))
    w("```")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_CSV.relative_to(REPO_ROOT)} and "
          f"{OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
