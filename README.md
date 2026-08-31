# GNSS-Denied Vehicle Navigation

Predicting per-second vehicle displacement from smartphone IMU data, so a
vehicle can keep navigating through GNSS outages.

Dataset: [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) (not committed — see
`scripts/fetch_data.sh`).

## Headline result

60-second outage, test split (4 sessions, 417 outages), mean position drift:

| predictor | drift | reads IMU? |
|---|---|---|
| constant-velocity DR | **285.4 m** | no |
| classical INS DR | 310.5 m | yes |
| our model (CNN + raw-gyro heading) | 648.6 m | yes |
| our model + oracle heading | **294.8 m** | yes |

**The learned model does not beat a baseline that reads no IMU.** The
oracle-heading row shows why: with heading supplied it reaches parity, so the
displacement estimate is sound and the deficit is heading integration. At
motorway speed with oracle heading it is 2.4x better than the baseline
(119.5 m vs 283.6 m).

Full write-up, including several negative results:
**[`results/findings.md`](results/findings.md)**.

## Getting started

```bash
pip install -e .
bash scripts/fetch_data.sh      # ~1.8 GB via Git LFS
pytest                          # 86 tests
```

The dataset needs Git LFS installed *before* cloning, or every CSV arrives as
a ~130-byte pointer. `fetch_data.sh` checks this and verifies three files by
exact byte count.

## Pipeline

| stage | command | output |
|---|---|---|
| corpus survey | `python -m sweep_corpus` | `results/corpus_sweep.*` |
| sanity + sync | `python -m data.sanity` | `results/sanity/` |
| label quality | `python -m data.sync_sensitivity` | `results/sanity/label_quality.*` |
| windowed dataset | `python -m data.windows` | `results/windows/` |
| Stage 0 known-answer | `python -m eval.stage0_validation` | `results/stage0_validation.*` |
| baselines | `python -m eval.harness_baselines` | `results/harness_baselines.*` |
| feature probes | `python -m baseline.feature_probe` | `results/feature_probe.*` |
| training | `python -m train` | `results/training/` |
| final scoring | `python -m eval.final_scoring` | `results/final_scoring.*` |
| figure | `python -m eval.make_figure` | `results/figures/` |

## Layout

```
src/data/      loading, splits, sanity checks, windowing
src/baseline/  INS dead reckoning, constant-velocity DR, feature probes
src/model/     1-D ResNet, multi-task losses, calibration
src/fusion/    non-holonomic ESKF (attempted; see findings §8)
src/eval/      outage harness, metrics, speed-bucket reporting
tests/         86 tests
```

## Notes for anyone picking this up

`CLAUDE.md` holds the empirically verified dataset facts. Several contradict
the published description and the authors' own preprocessing code — the `V-`
coordinates are already decimal degrees, `GPS SPEED (Kmh)` in `S-` files is
actually m/s, and sampling is not uniformly 10 Hz. Read it before touching the
loader.

Two implementation traps that cost real time and are documented in-place:

- **`xgboost` must be imported before `torch`** on macOS, or the xgboost fit
  dies as a hard crash with exit code 0 and no traceback.
- **Never band-limit or difference before correlating** without checking the
  raw result first; every such step in this project produced an artifact.
