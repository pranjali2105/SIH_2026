"""Compare trained checkpoints on the pre-registered success criteria.

Reports, per checkpoint: bucket-mean MAE on both splits, per-bucket signed CAE
with fraction-negative (the primary signal -- validate's >25 bucket moving off
-953.6), and 60 s drift with and without oracle heading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import baseline.feature_probe  # noqa: F401  (xgboost before torch; see feature_probe)
import numpy as np
import pandas as pd
import torch

from data.loader import list_sessions, load_session
from data.windows import build_all
from eval.buckets import BUCKET_NAMES, assign_buckets, early_stopping_metric, pooled_mae
from eval.final_scoring import GroundTruthHeadingPredictor, load_model
from eval.harness import run_session

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_MD = REPO_ROOT / "results" / "run_comparison.md"

RUNS = {
    "resnet_unweighted": "results/training/best.pt",
    "resnet_w_base": "results/run_w_base/best.pt",
    "resnet_w_narrow": "results/run_w_narrow/best.pt",
    "resnet_w_short_3s": "results/run_w_short/best.pt",
    "tcn_physics_3s": "results/run_tcn/best.pt",
    "tcn_no_accz": "results/run_tcn_noz/best.pt",
    "tcn_despike": "results/run_tcn_despike/best.pt",
    "tcn_nhc": "results/run_tcn_nhc/best.pt",
}


def windowed_metrics(model, ckpt, device):
    """Bucket-mean and pooled MAE on both splits."""
    cfg = ckpt.get("config", {})
    win = int(cfg.get("window_samples", 100))
    drop = tuple(cfg.get("drop_channels", ()))
    dscale = float(cfg.get("drop_scale", 0.0))
    stats = ckpt["norm_stats"]
    mean = np.asarray(stats["mean"], np.float32)
    sd = np.asarray(stats["std"], np.float32)
    out = {}
    for role in ("validate", "test"):
        built, _ = build_all(roles=(role,), window_samples=win,
                             apply_despike=bool(cfg.get("despike", False)))
        X = np.concatenate([b.X for b in built])
        y = np.concatenate([b.y for b in built])
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                blk = (X[i:i + 2048] - mean) / sd
                for c in drop:
                    blk[:, :, c] *= dscale
                t = torch.from_numpy(np.ascontiguousarray(
                    blk.transpose(0, 2, 1).astype(np.float32))).to(device)
                preds.append(model(t)["mu"].float().cpu().numpy())
        p = np.concatenate(preds)
        out[role] = {"bucket_mean_mae": early_stopping_metric(y, p),
                     "pooled_mae": pooled_mae(y, p)}
    return out


def drift_metrics(model, ckpt, device):
    """60 s CRSE/CAE per bucket, with and without oracle heading."""
    from train import ModelPredictor
    win = int(ckpt.get("config", {}).get("window_samples", 100))
    sess = list_sessions()
    sess = sess[sess.role.isin(("validate", "test")) & (sess.family == "S")]
    rows = []
    for r in sess.itertuples():
        name = f"{r.family}-{r.session}"
        try:
            s = load_session(name, check_rate=False)
            cfg = ckpt.get("config", {})
            base = ModelPredictor(model, s, ckpt["norm_stats"], device,
                                  zupt=True, heading_source="gyro",
                                  conjunction_zupt=True, window_samples=win,
                                  drop_channels=tuple(cfg.get("drop_channels", ())),
                                  drop_scale=float(cfg.get("drop_scale", 0.0)),
                                  despike_input=bool(cfg.get("despike", False)))
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        for tag, pred in (("model", base),
                          ("oracle_heading", GroundTruthHeadingPredictor(base, s))):
            res, _ = run_session(s, pred, 60.0)
            for o in res:
                m = o.metrics()
                spd = m["true_distance_m"] / 60.0
                rows.append({"role": r.role, "variant": tag,
                             "crse": m["crse"], "cae": m["cae"],
                             "bucket": BUCKET_NAMES[int(assign_buckets([spd])[0])]})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lines = ["# Run comparison\n",
             "Primary success criterion, stated in advance: validate's `>25` "
             "bucket CAE moving substantially off **-953.6** (fraction "
             "negative 1.00).\n"]
    summary = []

    for name, rel in RUNS.items():
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"missing {rel}", file=sys.stderr)
            continue
        print(f"scoring {name}", file=sys.stderr)
        model, ckpt = load_model(path, device)
        wm = windowed_metrics(model, ckpt, device)
        df = drift_metrics(model, ckpt, device)

        d = df[(df.role == "validate") & (df.variant == "model")
               & (df.bucket == ">25")]
        cae25 = float(d.cae.mean()) if len(d) else np.nan
        neg25 = float((d.cae < 0).mean()) if len(d) else np.nan

        row = {"run": name,
               "params": sum(p.numel() for p in model.parameters()),
               "window": int(ckpt.get("config", {}).get("window_samples", 100)),
               "val_bucket_mae": wm["validate"]["bucket_mean_mae"],
               "test_bucket_mae": wm["test"]["bucket_mean_mae"],
               "val_drift60": float(df[(df.role == "validate")
                                       & (df.variant == "model")].crse.mean()),
               "test_drift60": float(df[(df.role == "test")
                                        & (df.variant == "model")].crse.mean()),
               "test_drift60_oracle": float(df[(df.role == "test")
                                               & (df.variant == "oracle_heading")].crse.mean()),
               "val_gt25_cae": cae25, "val_gt25_frac_neg": neg25}
        summary.append(row)

        lines.append(f"\n## {name}\n")
        lines.append("```")
        piv = df[df.variant == "model"].pivot_table(
            index=["role"], columns="bucket", values="cae", aggfunc="mean")
        lines.append("60 s CAE (signed) by bucket, model:")
        lines.append(piv.round(1).to_string())
        negp = df[df.variant == "model"].assign(n=(df.cae < 0).astype(float)) \
            .pivot_table(index=["role"], columns="bucket", values="n", aggfunc="mean")
        lines.append("\nfraction negative:")
        lines.append(negp.round(2).to_string())
        lines.append("```")

    s = pd.DataFrame(summary)
    lines.insert(2, "\n## Summary\n")
    lines.insert(3, "```")
    lines.insert(4, s.round(3).to_string(index=False))
    lines.insert(5, "```")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    s.to_csv(REPO_ROOT / "results" / "run_comparison.csv", index=False)
    print(s.round(2).to_string(index=False))
    print(f"\nwrote {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
