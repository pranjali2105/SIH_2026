"""Hand-crafted-feature baselines: the bar a 3.85 M-parameter CNN must clear.

Predict-the-mean (7.17 m windowed MAE) is a floor, not a competitor. The real
question is whether a deep network earns its complexity over 24 cheap summary
statistics, so this is a required ablation rather than a curiosity.

Two fits on the same features:
  ridge    linear, closed form
  xgboost  gradient-boosted trees -- the strongest form of "this is really
           tabular data"

FEATURES (24 = 4 statistics x 6 channels), all per window, per channel:
  mean      the window's average -- a residual bias after levelling
  std       variation about that mean
  absmean   mean |value| -- energy irrespective of sign
  absdiff   mean |first difference| -- high-frequency texture, i.e. vibration

The `absdiff`/`std` statistics on the VERTICAL accelerometer axis dominate,
which is the mechanism claim this module also tests: road vibration scales with
speed, so the probe is reading a vibration-texture channel rather than
integrating acceleration. `mechanism_check()` refits on the 2 Hz sessions,
where a 1 Hz Nyquist limit removes most of that channel.

Run:  python -m baseline.feature_probe
"""

from __future__ import annotations

import sys
from pathlib import Path

# xgboost MUST be imported before torch. Both ship their own OpenMP runtime,
# and on macOS loading torch's first makes a later xgboost fit die silently --
# a hard crash with exit code 0 and no traceback, at any n_jobs. Importing
# xgboost first is the only workaround that works (KMP_DUPLICATE_LIB_OK does
# not). Keep this import at module top and above any torch import.
import xgboost  # noqa: F401  (import order is load-bearing)

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "feature_probe.csv"
OUT_MD = REPO_ROOT / "results" / "feature_probe.md"

STAT_NAMES = ("mean", "std", "absmean", "absdiff")
CHANNEL_NAMES = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
FEATURE_NAMES = [f"{s}_{c}" for s in STAT_NAMES for c in CHANNEL_NAMES]
RIDGE_ALPHA = 1.0


def window_features(X: np.ndarray) -> np.ndarray:
    """(n, 100, 6) -> (n, 24). Order matches FEATURE_NAMES."""
    return np.concatenate([
        X.mean(axis=1),
        X.std(axis=1),
        np.abs(X).mean(axis=1),
        np.abs(np.diff(X, axis=1)).mean(axis=1),
    ], axis=1)


class RidgeProbe:
    """Closed-form ridge on standardised features."""

    name = "ridge_probe"

    def fit(self, F: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA):
        self.mu_, self.sd_ = F.mean(0), F.std(0) + 1e-9
        A = np.hstack([(F - self.mu_) / self.sd_, np.ones((len(F), 1))])
        self.w_ = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        A = np.hstack([(F - self.mu_) / self.sd_, np.ones((len(F), 1))])
        # Displacement cannot be negative; a linear fit can be.
        return np.maximum(A @ self.w_, 0.0)


class XGBProbe:
    name = "xgb_probe"

    def fit(self, F: np.ndarray, y: np.ndarray):
        from xgboost import XGBRegressor
        self.m_ = XGBRegressor(n_estimators=400, max_depth=6,
                               learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, reg_lambda=1.0,
                               n_jobs=4, tree_method="hist")
        self.m_.fit(F, y)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        return np.maximum(self.m_.predict(F), 0.0)


class ProbeHarnessPredictor:
    """Adapts a fitted probe to the outage harness predictor interface.

    Autoregressive in the same sense as the model: it reads the IMU window
    ending at each second and never sees GPS after t0. Heading is held at its
    t0 value, since the probe has no yaw output -- so its drift isolates the
    displacement contribution.
    """

    def __init__(self, probe, session, name: str):
        from data.sanity import gps_cumulative_distance
        from data.windows import GRID_HZ, WINDOW_SAMPLES, levelled_channels
        from data.loader import _find

        self.probe, self.name = probe, name
        self.hz, self.win = GRID_HZ, WINDOW_SAMPLES
        gps = gps_cumulative_distance(session)
        lo, hi = float(gps[0].min()), float(gps[0].max())
        self.grid = np.arange(lo, hi, 1.0 / GRID_HZ)
        self.chan = levelled_channels(session, self.grid)
        col = _find(session.df, r"^HEADING") or _find(session.df, r"GPS ORIENTATION")
        t = pd.to_numeric(session.df["time_s"], errors="coerce").to_numpy(float)
        self.heading = (np.radians(np.interp(
            self.grid, t,
            np.nan_to_num(pd.to_numeric(session.df[col], errors="coerce")
                          .to_numpy(float)))) if col is not None
            else np.zeros_like(self.grid))

    def predict(self, t0: float, duration: float) -> dict:
        n = int(round(duration))
        h = float(np.interp(t0, self.grid, self.heading))
        wins, valid = [], []
        for k in range(n):
            end = int(np.searchsorted(self.grid, t0 + k))
            start = end - self.win
            if self.chan is None or start < 0 or end > self.chan.shape[0]:
                valid.append(False)
                wins.append(np.zeros((self.win, 6), dtype=np.float32))
            else:
                valid.append(True)
                wins.append(self.chan[start:end])
        F = window_features(np.stack(wins))
        d = self.probe.predict(F)
        d = np.where(np.asarray(valid), d, 0.0)
        return {"displacements": d, "headings": np.full(n, h)}


# --------------------------------------------------------------------------

def _collect(built, role: str):
    sub = [b for b in built if b.role == role]
    if not sub:
        return None, None
    return (window_features(np.concatenate([b.X for b in sub])),
            np.concatenate([b.y for b in sub]))


def _session_split_eval(built, rng, frac: float = 0.7):
    """Fit on 70% of SESSIONS, score on the rest.

    Session-level, not random. Windows overlap by 9 of their 10 seconds, so a
    random split puts near-duplicates on both sides and inflates every score --
    the same leakage the main pipeline guards against.
    """
    built = [b for b in built if b.y.size >= 100]
    if len(built) < 4:
        return None
    order = rng.permutation(len(built))
    k = max(2, int(len(built) * frac))
    tr = [built[i] for i in order[:k]]
    te = [built[i] for i in order[k:]]
    Ftr = window_features(np.concatenate([b.X for b in tr]))
    ytr = np.concatenate([b.y for b in tr])
    Fte = window_features(np.concatenate([b.X for b in te]))
    yte = np.concatenate([b.y for b in te])
    mean_mae = float(np.mean(np.abs(yte - ytr.mean())))
    out = {"n_sessions_fit": len(tr), "n_sessions_eval": len(te),
           "n_windows_eval": int(yte.size), "mean_mae_m": mean_mae}
    for name, probe in (("ridge", RidgeProbe().fit(Ftr, ytr)),
                        ("xgb", XGBProbe().fit(Ftr, ytr))):
        mae = float(np.mean(np.abs(probe.predict(Fte) - yte)))
        out[f"{name}_mae_m"] = mae
        out[f"{name}_improvement_pct"] = 100.0 * (1 - mae / mean_mae)
    return out


def mechanism_check(seed: int = 0) -> pd.DataFrame:
    """Does the probe depend on high-frequency vibration texture?

    The probe's strongest features are vertical-acceleration variation, which
    should be road vibration scaling with speed. At 2 Hz the Nyquist limit is
    1 Hz, so most of that channel is gone. Comparing the probe's advantage over
    predict-the-mean at the two rates tests the mechanism instead of asserting
    it. Both sides use a SESSION-level split so the comparison is like-for-like
    and uncontaminated by window overlap.
    """
    from data.windows import build_all

    rng = np.random.default_rng(seed)
    rows = []
    for label, roles in (("10 Hz (train)", ("train",)),
                         ("2 Hz (robustness)", ("robustness_2hz",))):
        built, _ = build_all(roles=roles)
        res = _session_split_eval(built, rng)
        if res is None:
            rows.append({"regime": label, "note": "too few sessions"})
            continue
        rows.append({"regime": label, **res})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    from data.windows import build_all
    from eval.buckets import BUCKET_NAMES, bucketed_mae, early_stopping_metric, pooled_mae

    built, _ = build_all()
    Ftr, ytr = _collect(built, "train")
    if Ftr is None:
        print("no train windows", file=sys.stderr)
        return 1

    probes = {"ridge_probe": RidgeProbe().fit(Ftr, ytr),
              "xgb_probe": XGBProbe().fit(Ftr, ytr)}
    train_mean = float(ytr.mean())

    rows = []
    for role in ("validate", "test"):
        F, y = _collect(built, role)
        if F is None:
            continue
        preds = {"mean": np.full(y.shape, train_mean)}
        preds.update({n: p.predict(F) for n, p in probes.items()})
        for name, p in preds.items():
            b = bucketed_mae(y, p)
            row = {"role": role, "predictor": name, "n_windows": int(y.size),
                   "pooled_mae_m": pooled_mae(y, p),
                   "bucket_mean_mae_m": early_stopping_metric(y, p)}
            for _, r in b.iterrows():
                row[f"mae_{r.bucket}"] = r.mae
            rows.append(row)
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Feature importance from the ridge weights, for the mechanism claim.
    ridge = probes["ridge_probe"]
    imp = sorted(zip(FEATURE_NAMES, np.abs(ridge.w_[:-1])),
                 key=lambda kv: -kv[1])[:8]

    mech = mechanism_check()

    lines = ["# Hand-crafted feature probes\n",
             "The ablation that decides whether a 3.85 M-parameter CNN earns "
             "its complexity. 24 features = 4 statistics x 6 channels.\n",
             "\n## Windowed MAE\n",
             "| split | predictor | bucket-mean | pooled | "
             + " | ".join(BUCKET_NAMES) + " |",
             "|---|---|---|---|" + "---|" * len(BUCKET_NAMES)]
    for _, r in df.iterrows():
        cells = " | ".join(f"{r[f'mae_{b}']:.3f}" if np.isfinite(r[f"mae_{b}"])
                           else "—" for b in BUCKET_NAMES)
        lines.append(f"| {r.role} | `{r.predictor}` | "
                     f"**{r.bucket_mean_mae_m:.3f}** | {r.pooled_mae_m:.3f} "
                     f"| {cells} |")

    lines.append("\n## Top ridge weights (standardised features)\n")
    lines.append("```")
    for name, w in imp:
        lines.append(f"  {name:<14}{w:8.3f}")
    lines.append("```")

    lines.append("\n## Mechanism check: 2 Hz sessions\n")
    lines.append("At 2 Hz the Nyquist limit is 1 Hz, so the high-frequency "
                 "vertical-acceleration texture the probe leans on at 10 Hz is "
                 "largely gone. If the probe's advantage over predict-the-mean "
                 "collapses, the vibration mechanism is demonstrated.\n")
    if mech.empty:
        lines.append("No 2 Hz windows could be built.\n")
    else:
        lines.append("```")
        lines.append(mech.round(3).to_string(index=False))
        lines.append("```")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
