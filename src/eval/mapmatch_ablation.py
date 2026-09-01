"""Ablations for the along-road tracker.

Three claims are taken on trust from the brief. Each is cheap to check here,
and two of them are load-bearing:

1. **Re-matching every 10 s beats re-matching every step.** Gomes & Costa
   swept this; whether it transfers depends on how far their tracked position
   could wander from the network, and ours cannot wander at all.
2. **A 3-5 s gyro window is the right span for a junction turn.**
3. **Skipping outages whose start fix is poor helps.** It also removes
   outages from the score, so it has to be reported as a coverage cost, not
   only as an accuracy gain.

Run:  python -m eval.mapmatch_ablation [--inner constant_velocity_dr]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.constant_velocity import ConstantVelocityDR
from data.loader import list_sessions, load_session
from eval.harness import OutageSkipped, iter_outages, run_outage
from mapmatch.graph import RoadGraph
from mapmatch.osrm import DEFAULT_DATASET, OSRMClient
from mapmatch.predictor import MapMatchConfig, MapMatchedPredictor

REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = REPO_ROOT / "data" / "osm"
GRAPH_CACHE = MAP_DIR / "road_graph.npz"
OUT_CSV = REPO_ROOT / "results" / "mapmatch_ablation.csv"
OUT_MD = REPO_ROOT / "results" / "mapmatch_ablation.md"

DURATION = 60.0
NEVER = 1e9          # a cadence longer than any outage = never re-match


def run_variant(sessions, graph, osrm, cfg, stride: float):
    """Mean/median drift over the test outages for one configuration."""
    drifts, skipped, scored = [], 0, 0
    stats = {}
    for s in sessions:
        pred = MapMatchedPredictor(ConstantVelocityDR(s), s, graph, osrm, cfg)
        for t0 in iter_outages(s, DURATION, stride):
            try:
                r = run_outage(s, t0, DURATION, pred)
            except OutageSkipped:
                skipped += 1
                continue
            scored += 1
            drifts.append(r.metrics()["crse"])
        for k, v in pred.stats.items():
            stats[k] = stats.get(k, 0) + v
    if not drifts:
        return None
    d = np.asarray(drifts, dtype=float)
    return {"n": int(d.size), "mean_m": float(d.mean()),
            "median_m": float(np.median(d)),
            "p90_m": float(np.percentile(d, 90)),
            "pct_under_100m": 100.0 * float(np.mean(d < 100.0)),
            "scored": scored, "skipped": skipped,
            "junctions": stats.get("junctions", 0),
            "rematch_applied": stats.get("rematch_applied", 0),
            "rematch_ambiguous": stats.get("rematch_ambiguous", 0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--stride", type=float, default=120.0,
                    help="outage spacing; wider than the outage to keep the "
                         "sweep cheap while covering the whole session")
    args = ap.parse_args(argv)

    if not GRAPH_CACHE.exists():
        print(f"{GRAPH_CACHE} missing — run `python -m mapmatch.build_map`",
              file=sys.stderr)
        return 1
    graph = RoadGraph.load(GRAPH_CACHE)

    meta = list_sessions()
    meta = meta[(meta.role == "test") & (meta.family == "S")]
    sessions = [load_session(f"{r.family}-{r.session}", check_rate=False)
                for r in meta.itertuples()]

    variants: list[tuple[str, str, MapMatchConfig]] = []
    for c in (1.0, 5.0, 10.0, 20.0, NEVER):
        label = "never" if c == NEVER else f"{c:g} s"
        variants.append(("re-match cadence", label,
                         MapMatchConfig(rematch_every_s=c)))
    for w in (2.0, 3.0, 4.0, 5.0, 8.0):
        variants.append(("gyro turn window", f"{w:g} s",
                         MapMatchConfig(turn_window_s=w)))
    for lead in (0.0, 1.0, 2.0):
        variants.append(("gyro window lead", f"+{lead:g} s",
                         MapMatchConfig(turn_lead_s=lead)))
    for a in (10.0, 20.0, 50.0, 1e9):
        label = "no gate" if a > 1e8 else f"{a:g} m"
        variants.append(("start-fix GPS gate", label,
                         MapMatchConfig(max_gps_accuracy_m=a)))

    rows = []
    with OSRMClient(Path(args.dataset)) as osrm:
        for group, label, cfg in variants:
            print(f"  {group}: {label}", file=sys.stderr)
            res = run_variant(sessions, graph, osrm, cfg, args.stride)
            if res is None:
                continue
            rows.append({"group": group, "setting": label, **res})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    lines = [f"# Along-road tracker ablations\n",
             f"Test split, {DURATION:.0f} s outages spaced {args.stride:.0f} s "
             "apart, constant-velocity displacement throughout so that only "
             "the map-side setting varies. Drift is CRSE.\n",
             "\nThe displacement source is deliberately the WEAK one: it reads "
             "no IMU, so any difference between rows is the map machinery and "
             "not the model.\n"]
    for group in df.group.unique():
        sub = df[df.group == group]
        lines.append(f"\n## {group}\n")
        lines.append("| setting | n | mean | median | p90 | % under 100 m | "
                     "skipped | junctions | re-match applied | ambiguous |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sub.itertuples():
            lines.append(
                f"| {r.setting} | {r.n} | {r.mean_m:.1f} | {r.median_m:.1f} | "
                f"{r.p90_m:.1f} | {r.pct_under_100m:.0f}% | {r.skipped} | "
                f"{r.junctions} | {r.rematch_applied} | "
                f"{r.rematch_ambiguous} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
