"""Final scoring: model vs deployable baselines, with mechanism diagnostics.

Reports CRSE and CAE together at 10/30/60 s. The pair discriminates:

  |CAE| ~= CRSE   errors share a sign -> systematic bias compounding linearly
  |CAE| <<  CRSE   errors scatter -> the amplification is elsewhere, most
                   likely heading integration rotating the track away from truth

and the drift-vs-duration curve localises it: early divergence from the
baselines implicates displacement, late divergence implicates heading.

Also scores the chosen checkpoint on BOTH the 90-outage subset used for early
stopping and the full outage set, since the subset is what selection saw.

Run:  python -m eval.final_scoring [--checkpoint results/training/best_ma3.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import order is load-bearing: baseline.feature_probe imports xgboost, which
# must precede torch or the xgboost fit crashes silently on macOS. See the note
# in src/baseline/feature_probe.py.
import baseline.feature_probe  # noqa: F401  (import order is load-bearing)

import numpy as np
import pandas as pd
import torch

from baseline.constant_velocity import ConstantVelocityDR
from baseline.ins_dr import INSDeadReckoning
from data.loader import list_sessions, load_session
from eval.buckets import BUCKET_NAMES, assign_buckets
from eval.harness import OutageSkipped, iter_outages, run_outage
from model.resnet1d import ResNet1D

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "results" / "training"
OUT_CSV = REPO_ROOT / "results" / "final_scoring.csv"
OUT_MD = REPO_ROOT / "results" / "final_scoring.md"

DURATIONS = (10.0, 30.0, 60.0)
CALIBRATOR = None
ROLES = ("validate", "test")


def load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ResNet1D().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


class GroundTruthHeadingPredictor:
    """The model's displacement with TRUE heading substituted.

    The diagnostic that separates the two failure modes: if drift collapses
    when true heading is supplied, the displacement head is fine and heading
    integration is eating the gain. If it barely moves, the problem is the
    displacement head.
    """

    name = "model_true_heading"

    def __init__(self, inner, session):
        from data.loader import _find
        self.inner = inner
        t = pd.to_numeric(session.df["time_s"], errors="coerce").to_numpy(float)
        col = _find(session.df, r"^HEADING") or _find(session.df, r"GPS ORIENTATION")
        self.t = t
        self.head = (np.radians(np.nan_to_num(
            pd.to_numeric(session.df[col], errors="coerce").to_numpy(float)))
            if col is not None else np.zeros_like(t))

    def _swap(self, t0: float, out: dict) -> dict:
        n = len(out["displacements"])
        marks = t0 + np.arange(n, dtype=float)
        return {"displacements": out["displacements"],
                "headings": np.interp(marks, self.t, self.head)}

    def predict(self, t0: float, duration: float) -> dict:
        return self._swap(t0, self.inner.predict(t0, duration))

    def predict_many(self, t0s, duration: float) -> list[dict]:
        outs = self.inner.predict_many(t0s, duration)
        return [self._swap(t0, o) for t0, o in zip(t0s, outs)]


def score(session, predictor, duration, starts=None) -> list[dict]:
    rows = []
    starts = list(iter_outages(session, duration)) if starts is None else starts
    for t0 in starts:
        try:
            r = run_outage(session, t0, duration, predictor)
        except OutageSkipped:
            continue
        m = r.metrics()
        # Bucket by the true mean speed over the outage.
        speed = m["true_distance_m"] / duration
        rows.append({"t0": t0, "crse_m": m["crse"], "cae_m": m["cae"],
                     "abs_cae_m": abs(m["cae"]), "aeps_m": m["aeps"],
                     "true_distance_m": m["true_distance_m"],
                     "mean_speed_ms": speed,
                     "bucket": BUCKET_NAMES[int(assign_buckets([speed])[0])]})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                    else "cpu")
    args = ap.parse_args(argv)

    from baseline.feature_probe import (ProbeHarnessPredictor, RidgeProbe,
                                        XGBProbe, window_features)
    from data.windows import build_all

    # Fit the tree probe BEFORE any torch MPS context exists. xgboost's thread
    # pool and the Metal backend conflict: initialising MPS first makes a
    # subsequent xgboost fit die silently (exit 0, no traceback). Ordering is
    # the whole fix; nothing else about either library needs changing.
    built, _ = build_all(roles=("train",))
    F = window_features(np.concatenate([b.X for b in built]))
    y = np.concatenate([b.y for b in built])
    print("fitting probes (before MPS init)", file=sys.stderr)
    probes = {"ridge_probe": RidgeProbe().fit(F, y),
              "xgb_probe": XGBProbe().fit(F, y)}
    del F, y, built

    global CALIBRATOR
    cal_path = REPO_ROOT / "results" / "training" / "calibration.json"
    if cal_path.exists():
        from model.calibration import PiecewiseCalibrator
        CALIBRATOR = PiecewiseCalibrator.load(cal_path)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    model = ckpt = None
    if ckpt_path and ckpt_path.exists():
        model, ckpt = load_model(ckpt_path, args.device)
        print(f"loaded {ckpt_path.name} (epoch {ckpt.get('epoch')}, "
              f"drift {ckpt.get('val_drift_60s_m')})", file=sys.stderr)
    from train import ModelPredictor

    sessions = list_sessions()
    sessions = sessions[sessions["role"].isin(ROLES) & (sessions["family"] == "S")]

    rows = []
    for r in sessions.itertuples():
        name = f"{r.family}-{r.session}"
        print(f"  {name}", file=sys.stderr)
        try:
            s = load_session(name, check_rate=False)
        except Exception:
            continue

        makers = {"ins_dr": lambda: INSDeadReckoning(s),
                  "constant_velocity_dr": lambda: ConstantVelocityDR(s)}
        makers.update({n: (lambda p=p, n=n: ProbeHarnessPredictor(p, s, n))
                       for n, p in probes.items()})
        if model is not None:
            base = ModelPredictor(model, s, ckpt["norm_stats"], args.device,
                                  zupt=False)
            zup = ModelPredictor(model, s, ckpt["norm_stats"], args.device,
                                 zupt=True)
            makers["model"] = lambda p=base: p
            makers["model_zupt"] = lambda p=zup: p
            makers["model_true_heading"] = lambda p=base: \
                GroundTruthHeadingPredictor(p, s)
            makers["model_zupt_true_heading"] = lambda p=zup: \
                GroundTruthHeadingPredictor(p, s)
            final = ModelPredictor(model, s, ckpt["norm_stats"], args.device,
                                   zupt=True, heading_source="gyro",
                                   conjunction_zupt=True)
            makers["model_final"] = lambda p=final: p
            makers["model_final_true_heading"] = lambda p=final: \
                GroundTruthHeadingPredictor(p, s)
            if CALIBRATOR is not None:
                cal = ModelPredictor(model, s, ckpt["norm_stats"], args.device,
                                     zupt=True, calibration=CALIBRATOR)
                makers["model_zupt_cal"] = lambda p=cal: p
                makers["model_zupt_cal_true_heading"] = lambda p=cal: \
                    GroundTruthHeadingPredictor(p, s)

        for pname, make in makers.items():
            try:
                pred = make()
            except Exception as exc:
                print(f"    {pname}: {exc}", file=sys.stderr)
                continue
            for dur in DURATIONS:
                for row in score(s, pred, dur):
                    rows.append({"role": r.role, "session": name,
                                 "predictor": pname, "duration_s": dur, **row})

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    agg = (df.groupby(["role", "predictor", "duration_s"])
           .agg(n=("crse_m", "size"), crse_mean=("crse_m", "mean"),
                crse_median=("crse_m", "median"),
                cae_mean=("cae_m", "mean"), abs_cae_mean=("abs_cae_m", "mean"))
           .reset_index())
    agg["abs_cae_over_crse"] = agg["abs_cae_mean"] / agg["crse_mean"]

    lines = ["# Final scoring\n",
             "CRSE is the norm of the accumulated 2-D position error; CAE is "
             "the signed sum of per-second displacement errors.\n",
             "\n**|CAE| / CRSE near 1** means the per-second errors share a "
             "sign — systematic bias compounding linearly. **Near 0** means "
             "they scatter and the amplification comes from elsewhere, most "
             "likely heading rotating the track away from truth.\n",
             "\n## Mean CRSE by duration\n",
             "| split | predictor | 10 s | 30 s | 60 s |", "|---|---|---|---|---|"]
    for role in ROLES:
        for pname in agg[agg.role == role]["predictor"].unique():
            cells = []
            for dur in DURATIONS:
                m = agg[(agg.role == role) & (agg.predictor == pname)
                        & (agg.duration_s == dur)]
                cells.append(f"{float(m.crse_mean.iloc[0]):.1f}" if len(m) else "—")
            lines.append(f"| {role} | `{pname}` | " + " | ".join(cells) + " |")

    lines.append("\n## CAE vs CRSE at 60 s\n")
    lines.append("| split | predictor | CRSE | mean CAE | mean \\|CAE\\| | "
                 "\\|CAE\\|/CRSE | reading |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, a in agg[agg.duration_s == 60.0].iterrows():
        ratio = a.abs_cae_over_crse
        reading = ("systematic bias" if ratio > 0.7 else
                   "scatter / heading" if ratio < 0.3 else "mixed")
        lines.append(f"| {a.role} | `{a.predictor}` | {a.crse_mean:.1f} | "
                     f"{a.cae_mean:+.1f} | {a.abs_cae_mean:.1f} | {ratio:.2f} "
                     f"| {reading} |")

    for dur in DURATIONS:
        lines.append(f"\n## {dur:.0f} s drift (CRSE) by speed bucket\n")
        sub = df[df.duration_s == dur]
        piv = sub.pivot_table(index=["role", "predictor"], columns="bucket",
                              values="crse_m", aggfunc="mean")
        lines.append("```")
        lines.append(piv.round(1).to_string())
        lines.append("```")

    lines.append("\n## 60 s CAE (SIGNED) by speed bucket\n")
    lines.append("A consistent sign within a bucket is a calibration problem "
                 "in that regime specifically — narrower and more fixable than "
                 "a general failure. Scatter about zero is not.\n")
    sub = df[df.duration_s == 60.0]
    piv = sub.pivot_table(index=["role", "predictor"], columns="bucket",
                          values="cae_m", aggfunc="mean")
    lines.append("```")
    lines.append(piv.round(1).to_string())
    lines.append("```")
    lines.append("\nfraction of outages with NEGATIVE CAE (under-prediction):\n")
    sub = sub.assign(neg=(sub["cae_m"] < 0).astype(float))
    piv2 = sub.pivot_table(index=["role", "predictor"], columns="bucket",
                           values="neg", aggfunc="mean")
    lines.append("```")
    lines.append(piv2.round(2).to_string())
    lines.append("```")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
