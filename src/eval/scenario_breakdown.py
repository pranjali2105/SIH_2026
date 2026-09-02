"""Per-scenario MAE, for the NHC falsification test.

The corpus carries no per-session surface or manoeuvre labels (the categorised
folders are named by session ID only), so scenarios are defined from window
statistics rather than metadata:

    cornering  |gyro_z| high   -- the vehicle is turning
    straight   |gyro_z| low, speed moderate
    motorway   |gyro_z| low, speed high

The NHC penalty's predicted failure mode is specific: it should teach the
model to suppress speed whenever the yaw rate is nonzero. That shows up as
WORSE displacement error in `cornering` while `straight`/`motorway` are
comparatively unaffected -- which aggregate MAE would hide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import baseline.feature_probe  # noqa: F401  (xgboost before torch)
import numpy as np
import pandas as pd
import torch

from data.windows import build_all
from eval.final_scoring import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results" / "scenario_breakdown.md"

# |gyro_z| thresholds in rad/s, from the corpus distribution.
CORNER_HI = 0.05
STRAIGHT_LO = 0.02
MOTORWAY_SPEED = 25.0

RUNS = {
    "tcn_physics_3s": "results/run_tcn/best.pt",
    "tcn_nhc": "results/run_tcn_nhc/best.pt",
}


def scenario_of(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    gz = np.abs(X[:, :, 5]).mean(axis=1)
    out = np.full(len(y), "other", dtype=object)
    out[gz >= CORNER_HI] = "cornering"
    straight = gz <= STRAIGHT_LO
    out[straight & (y < MOTORWAY_SPEED)] = "straight"
    out[straight & (y >= MOTORWAY_SPEED)] = "motorway"
    return out


def predict(model, ckpt, X, device):
    cfg = ckpt.get("config", {})
    mean = np.asarray(ckpt["norm_stats"]["mean"], np.float32)
    sd = np.asarray(ckpt["norm_stats"]["std"], np.float32)
    drop = tuple(cfg.get("drop_channels", ()))
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            blk = (X[i:i + 2048] - mean) / sd
            for c in drop:
                blk[:, :, c] *= float(cfg.get("drop_scale", 0.0))
            t = torch.from_numpy(np.ascontiguousarray(
                blk.transpose(0, 2, 1).astype(np.float32))).to(device)
            out.append(model(t)["mu"].float().cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for role in ("validate", "test"):
        built, _ = build_all(roles=(role,), window_samples=30)
        X = np.concatenate([b.X for b in built])
        y = np.concatenate([b.y for b in built])
        scen = scenario_of(X, y)
        for name, rel in RUNS.items():
            path = REPO_ROOT / rel
            if not path.exists():
                continue
            model, ckpt = load_model(path, device)
            p = predict(model, ckpt, X, device)
            for s in ("cornering", "straight", "motorway"):
                m = scen == s
                if m.sum() < 30:
                    continue
                rows.append({"role": role, "run": name, "scenario": s,
                             "n": int(m.sum()),
                             "mae": float(np.abs(p[m] - y[m]).mean()),
                             "bias": float(np.mean(p[m] - y[m])),
                             "label_mean": float(y[m].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "scenario_breakdown.csv", index=False)

    lines = ["# Per-scenario MAE (NHC falsification test)\n",
             "Scenarios defined from window statistics; the corpus has no "
             "manoeuvre labels.\n",
             f"\n- `cornering`: mean |gyro_z| >= {CORNER_HI}",
             f"- `straight`: mean |gyro_z| <= {STRAIGHT_LO}, label < {MOTORWAY_SPEED} m",
             f"- `motorway`: mean |gyro_z| <= {STRAIGHT_LO}, label >= {MOTORWAY_SPEED} m\n",
             "\n## MAE (m)\n", "```",
             df.pivot_table(index=["role", "scenario"], columns="run",
                            values="mae").round(3).to_string(), "```",
             "\n## Signed bias (predicted - true, m)\n", "```",
             df.pivot_table(index=["role", "scenario"], columns="run",
                            values="bias").round(3).to_string(), "```",
             "\n## Window counts\n", "```",
             df.pivot_table(index=["role", "scenario"], columns="run",
                            values="n").to_string(), "```"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
