"""XGBoost displacement probe vs the TCN, through the outage harness.

Both use the SAME heading source (raw levelled gyro), so the comparison
isolates the displacement model. The probe is the one already fitted in the
earlier ablation work -- no retraining, same 24 hand features -- but refitted
on 30-sample windows so its input matches the TCN's.

Reports signed CAE and |CAE|/CRSE alongside drift: if the probe's per-second
errors scatter where the TCN's are systematic, |CAE|/CRSE separates them even
where mean drift does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import baseline.feature_probe as fp   # noqa: F401  xgboost before torch
import numpy as np
import pandas as pd
import torch

from baseline.constant_velocity import ConstantVelocityDR
from baseline.feature_probe import (ProbeHarnessPredictor, RidgeProbe, XGBProbe,
                                    window_features)
from data.loader import list_sessions, load_session
from data.windows import build_all
from eval.buckets import BUCKET_NAMES, assign_buckets
from eval.final_scoring import GroundTruthHeadingPredictor, load_model
from eval.harness import run_session

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_MD = REPO_ROOT / "results" / "probe_vs_tcn.md"
DURATIONS = (10.0, 30.0, 60.0)
WINDOW = 30
TCN_CKPT = REPO_ROOT / "results" / "run_tcn_nhc" / "best.pt"


def main(argv=None) -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("fitting probes on train (30-sample windows)", file=sys.stderr)
    built, _ = build_all(roles=("train",), window_samples=WINDOW)
    F = window_features(np.concatenate([b.X for b in built]))
    y = np.concatenate([b.y for b in built])
    probes = {"xgb_probe": XGBProbe().fit(F, y),
              "ridge_probe": RidgeProbe().fit(F, y)}
    del F, y, built

    model, ckpt = load_model(TCN_CKPT, device)
    from train import ModelPredictor

    sess = list_sessions()
    sess = sess[sess.role.isin(("validate", "test")) & (sess.family == "S")]

    rows = []
    for r in sess.itertuples():
        name = f"{r.family}-{r.session}"
        print(f"  {name}", file=sys.stderr)
        try:
            s = load_session(name, check_rate=False)
        except Exception:
            continue

        tcn = ModelPredictor(model, s, ckpt["norm_stats"], device, zupt=True,
                             heading_source="gyro", conjunction_zupt=True,
                             window_samples=WINDOW)
        makers = {
            "constant_velocity_dr": lambda: ConstantVelocityDR(s),
            "tcn_nhc": lambda p=tcn: p,
            "tcn_nhc_oracle": lambda p=tcn: GroundTruthHeadingPredictor(p, s),
        }
        for pname, probe in probes.items():
            makers[pname] = (lambda p=probe, n=pname: ProbeHarnessPredictor(
                p, s, n, heading_source="gyro", window_samples=WINDOW))
        makers["xgb_probe_oracle"] = lambda: GroundTruthHeadingPredictor(
            ProbeHarnessPredictor(probes["xgb_probe"], s, "xgb",
                                  heading_source="gyro",
                                  window_samples=WINDOW), s)

        for pname, make in makers.items():
            try:
                pred = make()
            except Exception as exc:
                print(f"    {pname}: {exc}", file=sys.stderr)
                continue
            for dur in DURATIONS:
                res, _ = run_session(s, pred, dur)
                for o in res:
                    m = o.metrics()
                    spd = m["true_distance_m"] / dur
                    rows.append({"role": r.role, "session": name,
                                 "predictor": pname, "duration_s": dur,
                                 "crse": m["crse"], "cae": m["cae"],
                                 "bucket": BUCKET_NAMES[
                                     int(assign_buckets([spd])[0])]})
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "probe_vs_tcn.csv", index=False)

    agg = (df.groupby(["role", "predictor", "duration_s"])
           .agg(n=("crse", "size"), crse=("crse", "mean"),
                cae=("cae", "mean"),
                abs_cae=("cae", lambda v: float(np.mean(np.abs(v)))))
           .reset_index())
    agg["abs_cae_over_crse"] = agg.abs_cae / agg.crse

    lines = ["# XGBoost probe vs TCN through the outage harness\n",
             "Both use raw levelled gyro for heading, so this compares the "
             "displacement model only.\n",
             "\n## Mean drift (CRSE, m)\n",
             "| split | predictor | 10 s | 30 s | 60 s |", "|---|---|---|---|---|"]
    for role in ("test", "validate"):
        for p in agg[agg.role == role].predictor.unique():
            cells = []
            for d in DURATIONS:
                m = agg[(agg.role == role) & (agg.predictor == p)
                        & (agg.duration_s == d)]
                cells.append(f"{float(m.crse.iloc[0]):.1f}" if len(m) else "—")
            lines.append(f"| {role} | `{p}` | " + " | ".join(cells) + " |")

    lines.append("\n## Error character at 60 s\n")
    lines.append("| split | predictor | CRSE | mean CAE | \\|CAE\\|/CRSE | reading |")
    lines.append("|---|---|---|---|---|---|")
    for _, a in agg[agg.duration_s == 60.0].iterrows():
        rd = ("systematic" if a.abs_cae_over_crse > 0.7
              else "scattered" if a.abs_cae_over_crse < 0.3 else "mixed")
        lines.append(f"| {a.role} | `{a.predictor}` | {a.crse:.1f} | "
                     f"{a.cae:+.1f} | {a.abs_cae_over_crse:.2f} | {rd} |")

    sub = df[df.duration_s == 60.0]
    lines.append("\n## 60 s drift by speed bucket\n```")
    lines.append(sub.pivot_table(index=["role", "predictor"], columns="bucket",
                                 values="crse", aggfunc="mean").round(1).to_string())
    lines.append("```\n\n## 60 s signed CAE by bucket\n```")
    lines.append(sub.pivot_table(index=["role", "predictor"], columns="bucket",
                                 values="cae", aggfunc="mean").round(1).to_string())
    lines.append("```\n\n## fraction of outages with negative CAE\n```")
    lines.append(sub.assign(neg=(sub.cae < 0).astype(float))
                 .pivot_table(index=["role", "predictor"], columns="bucket",
                              values="neg", aggfunc="mean").round(2).to_string())
    lines.append("```")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
