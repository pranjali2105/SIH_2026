"""Score along-road map matching against the 100 m / 1 km / 60 s benchmark.

Runs the map-matched tracker on the `test` role (S-A5..S-A8, the four Volvo
sessions the offline extract was cropped to) beside the deployable baselines,
so the map's contribution is separated from the displacement model's.

The comparison that matters is `X` against `map_matched(X)` for the same X.
Along-road tracking removes heading error entirely and leaves distance error
untouched, so the difference between a row and its mapped counterpart is
exactly the part of the drift that was heading.

Run:  python -m eval.mapmatch_eval [--checkpoint results/training/best_ma3.pt]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.constant_velocity import ConstantVelocityDR
from baseline.ins_dr import INSDeadReckoning
from data.loader import list_sessions, load_session
from eval.harness import iter_outages, run_session
from mapmatch.graph import RoadGraph
from mapmatch.hmm_predictor import HMMMapMatchConfig, HMMMapMatchedPredictor
from mapmatch.osrm import DEFAULT_DATASET, OSRMClient
from mapmatch.predictor import MapMatchConfig, MapMatchedPredictor

REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = REPO_ROOT / "data" / "osm"
GRAPH_CACHE = MAP_DIR / "road_graph.npz"
EXTENT_PBF = MAP_DIR / "extent.osm.pbf"
OUT_CSV = REPO_ROOT / "results" / "mapmatch.csv"
OUT_MD = REPO_ROOT / "results" / "mapmatch.md"

DURATIONS = (10.0, 30.0, 60.0)
ROLE = "test"

# The problem statement's benchmark: under 100 m drift over a 1 km, 60 s
# outage. 60 s at 13-20 m/s is 800-1200 m, so that band is the "1 km outage".
BENCH_DURATION = 60.0
BENCH_MIN_M, BENCH_MAX_M = 800.0, 1200.0
BENCH_TARGET_M = 100.0


def load_model(path: Path, device: str):
    import torch
    from model.resnet1d import ResNet1D
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ResNet1D().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def score_session(session, predictors, durations, rows, skips):
    """Run every predictor over every outage of one session."""
    for pname, pred in predictors.items():
        for dur in durations:
            starts = list(iter_outages(session, dur))
            if hasattr(pred, "prime"):
                try:
                    pred.prime(starts, dur)
                except Exception as exc:            # noqa: BLE001
                    print(f"    prime failed {pname}: {exc}", file=sys.stderr)
            results, sk = run_session(session, pred, dur)
            skips.append({"session": f"{session.family}-{session.session}",
                          "predictor": pname, "duration_s": dur,
                          "n_scored": len(results), "n_skipped": len(sk),
                          "n_candidates": len(starts)})
            for r in results:
                m = r.metrics()
                rows.append({"session": m["session"], "predictor": pname,
                             "duration_s": dur, "t0": m["t0"],
                             "crse_m": m["crse"], "cae_m": m["cae"],
                             "abs_cae_m": abs(m["cae"]), "aeps_m": m["aeps"],
                             "final_pos_err_m": m["final_position_error_m"],
                             "max_pos_err_m": m["max_position_error_m"],
                             "true_distance_m": m["true_distance_m"]})


def build_predictors(session, graph, osrm, cfg, model, ckpt, device,
                     hmm_cfg=None):
    """Baselines, and each wrapped in the along-road tracker.

    Every inner predictor is wrapped TWICE: once through the existing
    greedy tracker (`map_matched_X`) and once through the beam-search
    tracker (`hmm_matched_X`, see `mapmatch.hmm_predictor`), so a single run
    of this script puts them side by side on the same outages.
    """
    inner = {"constant_velocity_dr": ConstantVelocityDR(session),
             "ins_dr": INSDeadReckoning(session)}
    if model is not None:
        from train import ModelPredictor
        inner["model"] = ModelPredictor(model, session, ckpt["norm_stats"],
                                        device, zupt=True,
                                        heading_source="gyro",
                                        conjunction_zupt=True)
        # Only the model has a `.chan`/uncertainty worth fusing against the
        # accelerometer -- the baselines read no IMU at all (or, for INS DR,
        # already integrate it directly), so there is nothing for
        # `SpeedFusedPredictor` to add there.
        from fusion.predictor import SpeedFusedPredictor
        inner["model_speed_fused"] = SpeedFusedPredictor(inner["model"], session)
    preds = dict(inner)
    for name, p in inner.items():
        mm = MapMatchedPredictor(p, session, graph, osrm, cfg)
        mm.name = f"map_matched_{name}"
        preds[mm.name] = mm

        hmm = HMMMapMatchedPredictor(p, session, graph, osrm,
                                     hmm_cfg or HMMMapMatchConfig(**vars(cfg)))
        hmm.name = f"hmm_matched_{name}"
        preds[hmm.name] = hmm
    return preds


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["predictor", "duration_s"])
            .agg(n=("crse_m", "size"),
                 mean_drift_m=("crse_m", "mean"),
                 median_drift_m=("crse_m", "median"),
                 p90_drift_m=("crse_m", lambda v: float(np.percentile(v, 90))),
                 max_drift_m=("crse_m", "max"),
                 mean_cae_m=("cae_m", "mean"),
                 mean_abs_cae_m=("abs_cae_m", "mean"))
            .reset_index())


def benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """The 1 km / 60 s subset, and the share meeting the 100 m target."""
    sub = df[(df.duration_s == BENCH_DURATION)
             & (df.true_distance_m >= BENCH_MIN_M)
             & (df.true_distance_m <= BENCH_MAX_M)]
    if sub.empty:
        return sub
    return (sub.groupby("predictor")
            .agg(n=("crse_m", "size"),
                 mean_drift_m=("crse_m", "mean"),
                 median_drift_m=("crse_m", "median"),
                 p90_drift_m=("crse_m", lambda v: float(np.percentile(v, 90))),
                 pct_under_100m=("crse_m",
                                 lambda v: 100.0 * float(np.mean(
                                     np.asarray(v) < BENCH_TARGET_M))))
            .reset_index()
            .sort_values("median_drift_m"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "results"
                                                / "training" / "best_ma3.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--rematch-every-s", type=float, default=10.0)
    ap.add_argument("--turn-window-s", type=float, default=4.0)
    ap.add_argument("--max-gps-accuracy-m", type=float, default=20.0)
    ap.add_argument("--durations", type=float, nargs="*", default=list(DURATIONS))
    args = ap.parse_args(argv)

    if GRAPH_CACHE.exists():
        graph = RoadGraph.load(GRAPH_CACHE)
    elif EXTENT_PBF.exists():
        graph = RoadGraph.build(EXTENT_PBF, GRAPH_CACHE)
    else:
        print(f"neither {GRAPH_CACHE.name} nor {EXTENT_PBF.name} present — "
              "run `python -m mapmatch.build_map`", file=sys.stderr)
        return 1

    model = ckpt = None
    ck = Path(args.checkpoint)
    if ck.exists():
        model, ckpt = load_model(ck, args.device)
        print(f"loaded {ck.name}", file=sys.stderr)
    else:
        print(f"no checkpoint at {ck}; scoring baselines only", file=sys.stderr)

    cfg = MapMatchConfig(max_gps_accuracy_m=args.max_gps_accuracy_m,
                         rematch_every_s=args.rematch_every_s,
                         turn_window_s=args.turn_window_s)

    sessions = list_sessions()
    sessions = sessions[(sessions.role == ROLE) & (sessions.family == "S")]

    rows, skips = [], []
    tracker: dict = defaultdict(dict)
    with OSRMClient(Path(args.dataset)) as osrm:
        for r in sessions.itertuples():
            name = f"{r.family}-{r.session}"
            print(f"  {name}", file=sys.stderr)
            s = load_session(name, check_rate=False)
            preds = build_predictors(s, graph, osrm, cfg, model, ckpt,
                                     args.device)
            score_session(s, preds, args.durations, rows, skips)
            for pname, p in preds.items():
                if hasattr(p, "stats"):
                    for k, v in p.stats.items():
                        tracker[pname][k] = tracker[pname].get(k, 0) + v

    if not rows:
        print("no outages scored", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    agg = summarise(df)
    bench = benchmark(df)
    sk = pd.DataFrame(skips).groupby("predictor")[
        ["n_scored", "n_skipped", "n_candidates"]].sum().reset_index()

    lines = [
        "# Offline map matching and along-road tracking\n",
        "Drift is CRSE: the norm of the accumulated 2-D position error at the "
        "end of the outage. Split `test` — S-A5..S-A8, the four Volvo "
        "sessions the offline OSRM extract was cropped to.\n",
        "\nEvery `map_matched_X` row uses exactly the same per-second "
        "displacements as row `X`; the only difference is that position is a "
        "scalar along a road polyline instead of an integrated heading. The "
        "gap between the two is therefore the heading component of the drift.\n",
        f"\n## Benchmark: under {BENCH_TARGET_M:.0f} m over a 1 km "
        f"({BENCH_MIN_M:.0f}-{BENCH_MAX_M:.0f} m), {BENCH_DURATION:.0f} s "
        "outage\n",
    ]
    if bench.empty:
        lines.append("_no outages in the 1 km band_\n")
    else:
        lines.append("| predictor | n | mean | median | p90 | % under 100 m |")
        lines.append("|---|---|---|---|---|---|")
        for b in bench.itertuples():
            lines.append(f"| `{b.predictor}` | {b.n} | {b.mean_drift_m:.1f} | "
                         f"{b.median_drift_m:.1f} | {b.p90_drift_m:.1f} | "
                         f"{b.pct_under_100m:.0f}% |")

    lines.append("\n## Mean drift (m) by duration\n")
    lines.append("| predictor | 10 s | 30 s | 60 s |")
    lines.append("|---|---|---|---|")
    for p in sorted(agg.predictor.unique()):
        cells = []
        for d in args.durations:
            row = agg[(agg.predictor == p) & (agg.duration_s == d)]
            cells.append(f"{row.mean_drift_m.iloc[0]:.1f}" if len(row) else "—")
        lines.append(f"| `{p}` | " + " | ".join(cells) + " |")

    lines.append("\n## Median drift (m) by duration\n")
    lines.append("| predictor | 10 s | 30 s | 60 s |")
    lines.append("|---|---|---|---|")
    for p in sorted(agg.predictor.unique()):
        cells = []
        for d in args.durations:
            row = agg[(agg.predictor == p) & (agg.duration_s == d)]
            cells.append(f"{row.median_drift_m.iloc[0]:.1f}" if len(row) else "—")
        lines.append(f"| `{p}` | " + " | ".join(cells) + " |")

    lines.append("\n## Outages skipped\n")
    lines.append("An outage is skipped when GPS accuracy at t0 exceeds "
                 f"{args.max_gps_accuracy_m:.0f} m, or when no road lies within "
                 "the snap radius of the start fix. The harness skips slow "
                 "starts (<5 m/s) for every predictor alike.\n")
    lines.append("| predictor | scored | skipped | candidates |")
    lines.append("|---|---|---|---|")
    for r in sk.itertuples():
        lines.append(f"| `{r.predictor}` | {r.n_scored} | {r.n_skipped} | "
                     f"{r.n_candidates} |")

    if tracker:
        lines.append("\n## What the tracker did\n")
        lines.append("Counts pooled over every outage and duration. "
                     "`forced` junctions had only one branch to take, so the "
                     "gyro was not consulted; `ambiguous` re-matches are the "
                     "ones where more than one road was plausible and the "
                     "estimate was therefore left unmodified.\n")
        lines.append("| predictor | outages | junctions | forced | re-matches "
                     "| applied | ambiguous | skipped (GPS) | skipped (no road) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for pname in sorted(tracker):
            t = tracker[pname]
            lines.append(
                f"| `{pname}` | {t.get('outages', 0)} | "
                f"{t.get('junctions', 0)} | {t.get('junction_forced', 0)} | "
                f"{t.get('rematches', 0)} | {t.get('rematch_applied', 0)} | "
                f"{t.get('rematch_ambiguous', 0)} | {t.get('skipped_gps', 0)} | "
                f"{t.get('skipped_nomatch', 0)} |")

    lines.append("\n## Full statistics\n```")
    lines.append(agg.round(2).to_string(index=False))
    lines.append("```")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_CSV.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
