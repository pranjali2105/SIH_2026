"""Untrained baselines, scored on the same metric as the model.

The point is that the first trained model has something to beat other than
zero. Two predictors, neither of which looks at the IMU at all:

  mean       predict the TRAIN mean displacement for every window
  previous   predict the previous second's displacement (persistence)

Persistence is the one that matters. Displacement is near-continuous second to
second, so it is a genuinely strong baseline -- a model that cannot beat it has
learned nothing from the IMU that the label's own autocorrelation does not
already supply.

Scored with `eval.buckets`: bucket-mean MAE is the number of record, pooled MAE
is reported alongside.

Run:  python -m eval.baselines
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.windows import build_all

from .buckets import BUCKET_NAMES, bucketed_mae, early_stopping_metric, pooled_mae

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "baselines.csv"
OUT_MD = REPO_ROOT / "results" / "baselines.md"


def persistence_predictions(y: np.ndarray, t0: np.ndarray,
                            stride_s: float = 1.0,
                            tol: float = 1e-6) -> np.ndarray:
    """Previous window's label, within one session.

    The first window of a session has no predecessor, and neither does a window
    following a gap; both get NaN rather than a borrowed value from elsewhere.
    """
    prev = np.full(y.shape, np.nan)
    if y.size < 2:
        return prev
    contiguous = np.abs(np.diff(t0) - stride_s) < tol
    prev[1:][contiguous] = y[:-1][contiguous]
    return prev


def evaluate(built, train_mean: float) -> pd.DataFrame:
    rows = []
    for role in ("validate", "test"):
        sub = [b for b in built if b.role == role]
        if not sub:
            continue
        y = np.concatenate([b.y for b in sub])

        preds = {"mean": np.full(y.shape, train_mean),
                 "previous": np.concatenate(
                     [persistence_predictions(b.y, b.t0) for b in sub])}

        for name, p in preds.items():
            ok = np.isfinite(p)
            yy, pp = y[ok], p[ok]
            buckets = bucketed_mae(yy, pp)
            row = {"role": role, "predictor": name,
                   "n_windows": int(ok.sum()),
                   "n_dropped_no_predecessor": int((~ok).sum()),
                   "pooled_mae_m": pooled_mae(yy, pp),
                   "bucket_mean_mae_m": early_stopping_metric(yy, pp)}
            for _, b in buckets.iterrows():
                row[f"mae_{b.bucket}"] = b.mae
                row[f"n_{b.bucket}"] = int(b.n_windows)
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    built, failures = build_all()
    if not built:
        print("no windows built", file=sys.stderr)
        return 1

    train = [b for b in built if b.role == "train"]
    train_mean = float(np.mean(np.concatenate([b.y for b in train])))

    df = evaluate(built, train_mean)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    lines = ["# Untrained baselines\n",
             "Targets the first trained model must beat. Neither predictor "
             "looks at the IMU.\n",
             f"\n- `mean`: predict the train mean, {train_mean:.3f} m, always.",
             "- `previous`: predict the previous second's displacement. "
             "Displacement is near-continuous, so this is a strong baseline; "
             "a model that cannot beat it has learned nothing the label's own "
             "autocorrelation does not already give.\n",
             "\n**Bucket-mean MAE is the number of record.** Pooled MAE is "
             "shown alongside and is dominated by each split's regime mix.\n",
             "\n## Results\n",
             "| split | predictor | bucket-mean MAE | pooled MAE | "
             + " | ".join(f"MAE {b}" for b in BUCKET_NAMES) + " |",
             "|---|---|---|---|" + "---|" * len(BUCKET_NAMES)]
    for _, r in df.iterrows():
        cells = " | ".join(
            f"{r[f'mae_{b}']:.3f}" if np.isfinite(r[f"mae_{b}"]) else "—"
            for b in BUCKET_NAMES)
        lines.append(f"| {r.role} | `{r.predictor}` | **{r.bucket_mean_mae_m:.3f}** "
                     f"| {r.pooled_mae_m:.3f} | {cells} |")

    lines.append("\n## Window counts per bucket\n")
    lines.append("```")
    for _, r in df.iterrows():
        counts = "  ".join(f"{b}:{int(r[f'n_{b}'])}" for b in BUCKET_NAMES)
        lines.append(f"  {r.role:<9}{r.predictor:<10}n={int(r.n_windows):<7}"
                     f"dropped={int(r.n_dropped_no_predecessor):<6}{counts}")
    lines.append("```")
    lines.append("\n`dropped` counts windows with no in-session predecessor "
                 "(first of a session, or following a gap); they are excluded "
                 "rather than given a borrowed value.\n")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
