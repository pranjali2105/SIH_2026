"""Speed-stratified error reporting.

Pooled MAE is dominated by whichever speed regime a split happens to contain,
and the splits differ: train averages 13.5 m of displacement per second,
validate 20.0, test 19.7. A model that improves only on the regime a split is
made of would look better than it is, and early stopping driven by pooled MAE
would follow that regime mix rather than the model.

So every error report is stratified by speed, and the EARLY-STOPPING METRIC is
the unweighted mean across occupied buckets -- each regime counts once,
regardless of how many windows it contributes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Displacement over one second is numerically equal to mean speed in m/s, so
# the label doubles as the bucket key.
BUCKET_EDGES = (0.0, 5.0, 15.0, 25.0, np.inf)
BUCKET_NAMES = ("0-5", "5-15", "15-25", ">25")

MIN_BUCKET_WINDOWS = 30     # below this a bucket's MAE is too noisy to average


def assign_buckets(speeds) -> np.ndarray:
    """Bucket index per sample, from speed (or equivalently 1 s displacement)."""
    s = np.asarray(speeds, dtype=float)
    return np.clip(np.digitize(s, BUCKET_EDGES[1:-1], right=False),
                   0, len(BUCKET_NAMES) - 1)


def bucketed_mae(y_true, y_pred, speeds=None,
                 min_windows: int = MIN_BUCKET_WINDOWS) -> pd.DataFrame:
    """Per-bucket MAE, count, and share of the data.

    `speeds` defaults to the true label, which for a one-second window IS the
    mean speed -- bucketing on truth rather than prediction, so a bad model
    cannot relabel its own hard cases into an easier bucket.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    speeds = y_true if speeds is None else np.asarray(speeds, dtype=float)
    idx = assign_buckets(speeds)
    err = np.abs(y_pred - y_true)

    rows = []
    for i, name in enumerate(BUCKET_NAMES):
        m = idx == i
        n = int(m.sum())
        rows.append({
            "bucket": name,
            "n_windows": n,
            "frac": n / max(y_true.size, 1),
            "mae": float(np.mean(err[m])) if n else np.nan,
            "label_mean": float(np.mean(y_true[m])) if n else np.nan,
            "counted": bool(n >= min_windows),
        })
    return pd.DataFrame(rows)


def early_stopping_metric(y_true, y_pred, speeds=None,
                          min_windows: int = MIN_BUCKET_WINDOWS) -> float:
    """Unweighted mean MAE across occupied buckets.

    This -- not pooled MAE -- is the quantity to stop on. Pooled MAE would let
    the split's regime mix drive the stopping point; this weights each speed
    regime equally, so improving only on the dominant regime does not look
    like improvement overall.

    Buckets with fewer than `min_windows` samples are skipped rather than
    allowed to swing the mean on a handful of points.
    """
    df = bucketed_mae(y_true, y_pred, speeds, min_windows)
    used = df[df["counted"] & df["mae"].notna()]
    if used.empty:
        return float("nan")
    return float(used["mae"].mean())


def pooled_mae(y_true, y_pred) -> float:
    """Plain MAE. Reported alongside, never used for stopping."""
    return float(np.mean(np.abs(np.asarray(y_pred, dtype=float)
                                - np.asarray(y_true, dtype=float))))


def report(y_true, y_pred, speeds=None, label: str = "") -> str:
    df = bucketed_mae(y_true, y_pred, speeds)
    lines = [f"{label}  pooled MAE {pooled_mae(y_true, y_pred):.4f} m   "
             f"bucket-mean MAE {early_stopping_metric(y_true, y_pred, speeds):.4f} m"
             "  <- stopping metric"]
    lines.append(f"  {'bucket':<8}{'n':>9}{'frac':>8}{'MAE':>10}{'label mean':>12}")
    for _, r in df.iterrows():
        flag = "" if r.counted else "   (too few, excluded from bucket-mean)"
        mae = f"{r.mae:.4f}" if np.isfinite(r.mae) else "—"
        lm = f"{r.label_mean:.3f}" if np.isfinite(r.label_mean) else "—"
        lines.append(f"  {r.bucket:<8}{int(r.n_windows):>9}{r.frac:>8.3f}"
                     f"{mae:>10}{lm:>12}{flag}")
    return "\n".join(lines)


def inverse_frequency_weights(labels, min_windows: int = MIN_BUCKET_WINDOWS
                              ) -> tuple[np.ndarray, dict]:
    """Per-sample weights inversely proportional to bucket frequency.

    Train is 16/43/27/14% across the four buckets while validate is
    6/10/31/53%, so unweighted training lets the 5-15 m regime dominate and the
    model shrinks its high-speed predictions toward the training mean. These
    weights make each speed regime contribute equally, matching the
    bucket-mean metric the model is actually judged on.

    Normalised to mean 1 so the loss scale is unchanged.
    """
    y = np.asarray(labels, dtype=float)
    idx = assign_buckets(y)
    counts = np.bincount(idx, minlength=len(BUCKET_NAMES)).astype(float)
    # An empty or near-empty bucket would otherwise get an enormous weight.
    counts[counts < min_windows] = np.inf
    per_bucket = np.where(np.isfinite(counts), 1.0 / counts, 0.0)
    w = per_bucket[idx]
    if w.sum() <= 0:
        return np.ones_like(y), {}
    w = w / w.mean()
    table = {BUCKET_NAMES[i]: float(per_bucket[i] / (w.mean() or 1.0))
             for i in range(len(BUCKET_NAMES))}
    detail = {BUCKET_NAMES[i]: {"n": int(counts[i]) if np.isfinite(counts[i]) else 0,
                                "weight": float(w[idx == i][0]) if (idx == i).any() else 0.0}
              for i in range(len(BUCKET_NAMES))}
    return w, detail
