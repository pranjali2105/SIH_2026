"""Deployable baselines through the outage harness, at several durations.

Reports INS dead reckoning and constant-velocity DR side by side on validate
and test at 10, 30 and 60 s. These are the numbers a trained model has to beat;
the windowed persistence figure is not a competitor (see CLAUDE.md).

Run:  python -m eval.harness_baselines
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.constant_velocity import ConstantVelocityDR
from baseline.ins_dr import INSDeadReckoning
from data.loader import list_sessions, load_session

from .harness import run_session

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "harness_baselines.csv"
OUT_MD = REPO_ROOT / "results" / "harness_baselines.md"

DURATIONS = (10.0, 30.0, 60.0)
ROLES = ("validate", "test")

PREDICTORS = {
    "ins_dr": INSDeadReckoning,
    "constant_velocity_dr": ConstantVelocityDR,
}


def _fit_probes():
    """Fit the hand-feature probes on train, for harness evaluation."""
    from baseline.feature_probe import RidgeProbe, XGBProbe, window_features
    from data.windows import build_all
    import numpy as _np

    built, _ = build_all(roles=("train",))
    F = window_features(_np.concatenate([b.X for b in built]))
    y = _np.concatenate([b.y for b in built])
    return {"ridge_probe": RidgeProbe().fit(F, y),
            "xgb_probe": XGBProbe().fit(F, y)}


def main(argv: list[str] | None = None) -> int:
    sessions = list_sessions()
    sessions = sessions[sessions["role"].isin(ROLES) & (sessions["family"] == "S")]

    from baseline.feature_probe import ProbeHarnessPredictor
    probes = _fit_probes()

    rows = []
    for r in sessions.itertuples():
        name = f"{r.family}-{r.session}"
        print(f"  {name}", file=sys.stderr)
        try:
            s = load_session(name, check_rate=False)
        except Exception as exc:
            print(f"    load failed: {exc}", file=sys.stderr)
            continue
        builders = {n: (lambda c=c: c(s)) for n, c in PREDICTORS.items()}
        builders.update({n: (lambda p=p, n=n: ProbeHarnessPredictor(p, s, n))
                         for n, p in probes.items()})
        for pname, make in builders.items():
            try:
                pred = make()
            except Exception as exc:
                print(f"    {pname}: {exc}", file=sys.stderr)
                continue
            for dur in DURATIONS:
                results, _skips = run_session(s, pred, dur)
                for res in results:
                    m = res.metrics()
                    rows.append({"role": r.role, "session": name,
                                 "predictor": pname, "duration_s": dur,
                                 "crse_m": m["crse"],
                                 "final_pos_err_m": m["final_position_error_m"],
                                 "aeps_m": m["aeps"],
                                 "true_distance_m": m["true_distance_m"]})
    if not rows:
        print("no outages scored", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    agg = (df.groupby(["role", "predictor", "duration_s"])
           .agg(n=("crse_m", "size"), mean_drift_m=("crse_m", "mean"),
                median_drift_m=("crse_m", "median"),
                p90_drift_m=("crse_m", lambda v: float(np.percentile(v, 90))),
                max_drift_m=("crse_m", "max"))
           .reset_index())

    lines = ["# Deployable baselines through the outage harness\n",
             "Drift = CRSE, the norm of the accumulated 2-D position error at "
             "the end of the outage.\n",
             "\n`constant_velocity_dr` holds the speed and heading observed at "
             "t0 and extrapolates — it is windowed persistence fed through the "
             "harness, where it can only see its own previous output. It reads "
             "no IMU at all, so a model's margin over it is exactly the part "
             "that comes from the IMU rather than from knowing the speed "
             "regime.\n",
             "\n## Mean drift (m)\n",
             "| split | duration | INS DR | constant-velocity DR | ridge probe "
             "| xgb probe | n |",
             "|---|---|---|---|---|---|---|"]
    for role in ROLES:
        for dur in DURATIONS:
            sub = agg[(agg.role == role) & (agg.duration_s == dur)]
            if sub.empty:
                continue
            def get(p, col="mean_drift_m"):
                row = sub[sub.predictor == p]
                return float(row[col].iloc[0]) if len(row) else float("nan")
            n = int(sub["n"].max())
            lines.append(f"| {role} | {dur:.0f} s | {get('ins_dr'):.2f} | "
                         f"{get('constant_velocity_dr'):.2f} | "
                         f"{get('ridge_probe'):.2f} | {get('xgb_probe'):.2f} "
                         f"| {n} |")

    lines.append("\n## Full statistics\n")
    lines.append("```")
    lines.append(agg.round(2).to_string(index=False))
    lines.append("```")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
