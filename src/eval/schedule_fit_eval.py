"""Fit the HMM observation model by scheduled sampling, then score it.

Why leave-one-session-out
-------------------------
The offline OSM extract was cropped to the four `test` sessions (S-A5..S-A8),
so those are the only sessions with road coverage: fitting on `train` is not
merely undesirable here, it is impossible, because there are no roads under
those routes in `road_graph.npz`.

Fitting on the test split and reporting on it would be leakage. So the fit is
LEAVE-ONE-SESSION-OUT: for each held-out session the parameters are fitted on
the other three and the held-out session is scored with parameters that never
saw it. Every scored outage is therefore out-of-sample, and the pooled number
covers the whole test split rather than half of it.

Run:
  python -m eval.schedule_fit_eval --checkpoint results/run_tcn_nhc/best.pt \
      --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.constant_velocity import ConstantVelocityDR
from data.loader import list_sessions, load_session
from eval.harness import iter_outages, run_session
from eval.mapmatch_eval import GRAPH_CACHE, load_model, summarise
from mapmatch.graph import RoadGraph
from mapmatch.predictor import MapMatchConfig
from mapmatch.schedule_fit import (ScheduleConfig, ScheduledSamplingFitter,
                                   apply, fit_pooled)
from mapmatch.viterbi import ViterbiConfig, ViterbiMapMatcher

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "schedule_fit.csv"
OUT_MD = REPO_ROOT / "results" / "schedule_fit.md"
OUT_JSON = REPO_ROOT / "results" / "schedule_fit_params.json"
DURATIONS = (10.0, 30.0, 60.0)


def _matcher(inner, session, graph, cfg: ViterbiConfig) -> ViterbiMapMatcher:
    return ViterbiMapMatcher(inner, session, graph, None, cfg)


def _starts(session, horizon_s: int, n: int) -> list[float]:
    """Fitting outages: evenly spread over the session, not the first n.

    Taking the first n would fit the parameters to one stretch of road, and
    the sessions are long enough that the early minutes are not representative
    of the whole route.
    """
    all_t = list(iter_outages(session, float(horizon_s)))
    if len(all_t) <= n:
        return all_t
    idx = np.linspace(0, len(all_t) - 1, n).round().astype(int)
    return [all_t[i] for i in sorted(set(idx.tolist()))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",
                    default=str(REPO_ROOT / "results" / "run_tcn_nhc" / "best.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--inner", default="model",
                    choices=("model", "constant_velocity_dr"),
                    help="which displacement source the matcher wraps")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--horizon-s", type=int, default=30)
    ap.add_argument("--n-outages", type=int, default=20)
    ap.add_argument("--decay", default="inverse_sigmoid",
                    choices=("inverse_sigmoid", "linear"))
    ap.add_argument("--durations", type=float, nargs="*", default=list(DURATIONS))
    args = ap.parse_args(argv)

    if not GRAPH_CACHE.exists():
        print(f"no {GRAPH_CACHE}; run `python -m mapmatch.build_map`",
              file=sys.stderr)
        return 1
    graph = RoadGraph.load(GRAPH_CACHE)

    model = ckpt = None
    if args.inner == "model":
        ck = Path(args.checkpoint)
        if not ck.exists():
            print(f"no checkpoint at {ck}", file=sys.stderr)
            return 1
        model, ckpt = load_model(ck, args.device)

    base = ViterbiConfig(**vars(MapMatchConfig(backend="graph")))
    sched = ScheduleConfig(epochs=args.epochs, horizon_s=args.horizon_s,
                           n_outages=args.n_outages, decay=args.decay)

    sess = list_sessions()
    sess = sess[(sess.role == "test") & (sess.family == "S")]
    names = [f"{r.family}-{r.session}" for r in sess.itertuples()]
    print(f"test sessions: {names}", file=sys.stderr)

    loaded = {n: load_session(n, check_rate=False) for n in names}

    def inner_for(s):
        if args.inner == "constant_velocity_dr":
            return ConstantVelocityDR(s)
        from train import ModelPredictor
        return ModelPredictor(model, s, ckpt["norm_stats"], args.device,
                              zupt=True, heading_source="gyro",
                              conjunction_zupt=True)

    # One fitter per session, reused across folds -- the caches (truth snaps,
    # inner-model displacements) are what make the search affordable, and they
    # are fold-independent because they hold nothing parameter-dependent.
    fitters, starts = {}, {}
    for n in names:
        m = _matcher(inner_for(loaded[n]), loaded[n], graph, base)
        fitters[n] = ScheduledSamplingFitter(m, sched)
        starts[n] = _starts(loaded[n], args.horizon_s, args.n_outages)

    rows, fold_params, histories = [], {}, {}
    for held in names:
        train_pairs = [(fitters[n], starts[n]) for n in names if n != held]
        print(f"\nfold: hold out {held}, fit on "
              f"{[n for n in names if n != held]}", file=sys.stderr)
        out = fit_pooled(train_pairs, sched, verbose=True)
        fold_params[held] = out["params"]
        histories[held] = out["history"]

        fitted = apply(base, out["params"])
        s = loaded[held]
        for tag, cfg in (("viterbi_base", base), ("viterbi_fitted", fitted)):
            p = _matcher(inner_for(s), s, graph, cfg)
            p.name = f"{tag}_{args.inner}"
            for d in args.durations:
                starts_d = list(iter_outages(s, d))
                if hasattr(p, "prime"):
                    try:
                        p.prime(starts_d, d)
                    except Exception as exc:            # noqa: BLE001
                        print(f"    prime failed: {exc}", file=sys.stderr)
                res, _sk = run_session(s, p, duration=d)
                for r in res:
                    m = r.metrics()
                    rows.append({"predictor": p.name, "session": held,
                                 "duration_s": d, "t0": m["t0"],
                                 "crse_m": m["crse"], "cae_m": m["cae"],
                                 "abs_cae_m": abs(m["cae"])})

    if not rows:
        print("no outages scored", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(
        {"folds": fold_params, "history": histories,
         "schedule": vars(sched), "inner": args.inner}, indent=2, default=float))

    agg = summarise(df)
    lines = [
        "# Scheduled sampling for the HMM observation model\n",
        f"Inner predictor: `{args.inner}`. Schedule: {args.decay}, "
        f"{args.epochs} epochs, eps 1 -> 0, {args.horizon_s} s fitting "
        f"rollouts.\n",
        "\nLeave-one-session-out: each session is scored with parameters "
        "fitted on the other three, so every outage below is out-of-sample.\n",
        "\n## Mean drift (m) by duration\n",
        "| predictor | " + " | ".join(f"{d:.0f} s" for d in args.durations) + " |",
        "|---" * (len(args.durations) + 1) + "|",
    ]
    for p in sorted(df.predictor.unique()):
        cells = []
        for d in args.durations:
            row = agg[(agg.predictor == p) & (agg.duration_s == d)]
            cells.append(f"{row.mean_drift_m.iloc[0]:.1f}" if len(row) else "—")
        lines.append(f"| `{p}` | " + " | ".join(cells) + " |")

    lines.append("\n## Fitted parameters per fold\n")
    keys = sorted(next(iter(fold_params.values())))
    lines.append("| held out | " + " | ".join(f"`{k}`" for k in keys) + " |")
    lines.append("|---" * (len(keys) + 1) + "|")
    for held, ps in fold_params.items():
        lines.append(f"| {held} | "
                     + " | ".join(f"{ps[k]:.3f}" for k in keys) + " |")
    lines.append(f"\nHand-set defaults: "
                 + ", ".join(f"`{k}`={getattr(base, k):.3f}" for k in keys)
                 + "\n")

    lines.append("\n## Curriculum\n")
    lines.append("The objective at eps>0 is teacher-forced and is NOT a "
                 "deployment number; only the eps=0 row is comparable to the "
                 "drift table above.\n")
    lines.append("| held out | " + " | ".join(
        f"e{h['epoch']} (eps {h['eps']:.2f})" for h in next(iter(histories.values()))
    ) + " |")
    lines.append("|---" * (args.epochs + 1) + "|")
    for held, h in histories.items():
        lines.append(f"| {held} | "
                     + " | ".join(f"{r['objective']:.1f}" for r in h) + " |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_MD} and {OUT_CSV}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
