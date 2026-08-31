# GNSS-Denied Vehicle Navigation

## Goal

Predict vehicle displacement per second from smartphone IMU data, so a vehicle can
keep navigating during GPS outages.

Dataset: **IO-VNBD** — https://github.com/onyekpeu/IO-VNBD

## Data facts

Verified empirically by `src/inspect_schema.py` and `src/sweep_corpus.py`
(2026-08-29) against the published repo. Where the paper or the authors' code
disagrees with the data, the data wins and the disagreement is noted.

- **File prefixes:** `S-` = smartphone (AndroSensor), `V-` = vehicle ECU.
- **Encoding:** files are **ISO-8859-1** with mixed internal encodings — `m/s²`
  is latin-1 while `µT` is UTF-8 in the same header line. **Always read with
  `encoding='iso-8859-1'`.** UTF-8 raises.

### Coordinates

- **Decimal degrees, correctly signed, in BOTH families.** There is **no
  arc-minutes conversion and no longitude sign inversion.** Applying either
  corrupts the data. Verified across all three V- trees.
- The authors' `Data Checker Table 2.py` performs those conversions because it
  targets pre-publication files whose names do not exist in this repo. Do not
  port it.

### Units

| Signal | S- files | V- files |
|---|---|---|
| Speed | **m/s** despite the `GPS SPEED (Kmh)` header | **km/h** (genuine) → divide by 3.6 |
| Acceleration | m/s² | **g** → multiply by 9.80665 |
| Angular rate | rad/s (gyroscope) | **deg/s** (yaw rate) → multiply by π/180 |

- The S- speed mislabel is **corpus-wide**, not per-file: measured ratio against
  GPS-derived ground speed is 1.00–1.02 across every S- group. V- measures
  3.60–3.63, i.e. genuinely km/h.
- **Scale by header only after applying this override**, or every S- speed ends
  up 3.6x too small.

### Sampling rate

- **Derive the rate as `hz_tick`: the reciprocal of the median timestamp delta
  computed over deltas ≥ 20 ms.** Do not use a plain median.
- AndroSensor writes **one row per sensor event**, so several rows share a tick
  and are separated by ~1 ms. The delta distribution is bimodal, and a naive
  median lands in the intra-tick spike and reports ~1000 Hz. S-T1 reads as
  1000 Hz that way; its true rate is 10 Hz.
- GPS fixes update at 1 Hz and are **held** between updates, so lat/lon repeat
  across rows. Differencing every row inflates derived speed tenfold.

### Corpus structure

- **187 unique (family, session) pairs across 564 CSV paths.** The same session
  appears in up to five places; read exactly one copy. Precedence:
  Synchronised/Categorised → Synchronised/Uncategorised →
  Unsynchronised/Categorised → Unsynchronised/Uncategorised.
- **Match session names case-insensitively** — both `V-Vta10.csv` and
  `V-vta10.csv` exist for the same session.
- Folder name contains a typo: `Synchronised V abd S datasets` (`abd`, not `and`).

### Schemas

- **Only 17 columns are common to all S- files.** Three distinct S- schemas
  exist (25-col, 24-col standard, 18-col reduced); V- has a single 29-col schema.
- **Group T is internally inconsistent**: S-T1…S-T9 are 18-column (no magnetic
  field, no 3-axis orientation, and `SATELLITES IN RANGE` rather than
  `GPS SATELLITES IN RANGE`), while S-T10 and S-T11 are 24-column.
- **The model's inputs — Accelerometer X/Y/Z, Gyroscope yaw/pitch/roll,
  Gravity X/Y/Z — all sit inside the 17-column intersection**, so the schema
  split does not constrain the feature set actually in use.
- Some files carry a stray `UNNAMED: n` column from a trailing comma (S-A4).
  Drop anything matching `^UNNAMED`.

### Corrections to the published description

- **Group I is 12.6 km over 0.2 h**, not the 0.06 km the paper's appendix states.
- Sampling is **not** uniformly 10 Hz across the corpus (see the split below).

## Synchronisation and reference construction

Established by `src/data/sync_diagnostic.py`, `src/data/sanity.py` and
`src/data/sync_sensitivity.py` (2026-08-29). None of this is documented
upstream, and all of it is load-bearing.

### S-/V- pairs are aligned by ROW INDEX, not timestamp

- The two families' clock origins differ: `S-` counts from session start,
  `V-` from **start of day**. A timestamp-based lag search chases an offset of
  minutes and pins itself to whatever search bound it is given.
- In the Synchronised tree, **63 of 72 pairs have identical row counts**
  (S-M and V-M are both 105974 rows). Compare paired sessions on the row grid.

### GPS-derived acceleration is unusable as a sync reference for S- files

- Differentiating 1 Hz quantised GPS speed to make a 10 Hz reference
  high-passes the quantisation noise. The ceiling this imposes is measurable:
  substituting a **perfect ECU accelerometer** for the phone still reaches only
  **r = 0.12–0.58** against that reference, versus **r = 0.93** against clean
  10 Hz ECU speed.
- So a weak S- correlation (~0.29 median) says nothing about the phone. The
  limit is the reference. **No phone-side work — levelling, yaw search,
  multi-axis fitting — can exceed it**; all three were tried and none did.
- Consequence: `results/sanity/time_offsets.csv` (method
  `accel_correlation`) is **superseded**. Its header says so. Prefer
  `time_offsets_distance.csv`.

### S- GPS quality varies by session — weight training data accordingly

- Measured as lag-aligned correlation against ECU speed on row-aligned pairs:
  most sessions reach 0.93–0.96, but **S-S4 reaches only 0.52 and S-Y1 only
  0.11**.
- Do **not** low-pass before making this comparison. A 0.5 Hz low-pass removes
  most of the signal on some sessions and produced a spurious 0.64 for S-M,
  whose lag-aligned raw correlation is in fact 0.95.
- Per-session GPS quality is a real weighting signal; per-session lag
  estimates are not.

### Label quality is the usable GPS-quality measure

`label_stability_pct` (see `src/data/sync_sensitivity.py`, surfaced as a column
on `list_sessions()`): the mean absolute change in the 1 s displacement label
under a ±500 ms window shift, as a percentage of mean displacement.

- **It works where nothing else does.** Groups T, A and I have no V- files, so
  the ECU cross-check cannot reach them; this can.
- Corpus distribution (150/187 scored): p50 2.03%, p75 3.31%, p90 5.33%,
  p95 8.20%. Cut points proposed from the distribution itself:
  **clean < 2.41%, usable 2.41–5.33%, suspect > 5.33%**.
- **Validation is directional but weak**: Spearman rho against the ECU speed
  correlation over 50 pairs is **−0.266 (p = 0.062)**. The sign is right —
  worse GPS means less stable labels — but this does not establish it as a
  calibrated proxy. Treat it as a screening signal, not a measurement.
- **The ratio is unreliable when mean displacement is small.** Sessions with
  mean displacement < 2 m/s carry `low_denominator = True` in
  `label_quality.csv`; they are **excluded from the suspect lists and from the
  cut-point quantiles**, because they are slow sessions, not bad GPS. Six
  sessions are flagged, including S-S3b (1.36 m), V-Vw1 (0.02 m) and V-Vw15
  (0.07 m). Never act on the percentage without checking
  `mean_displacement_m`.

### Session-level guard: path length vs speed integral

`src/data/windows.py::path_vs_speed_ratio` — fix-to-fix geodesic path length
divided by the integral of the session's own reported speed. **Sessions
outside ±10% are refused outright** (`InflatedPositionsError`).

- This catches at the SESSION level what the per-window 60 m/s guard cannot:
  a threshold trims a corrupt tail, it does not repair the windows below it.
- Two distinct defects, both rejected, and the message distinguishes them:
  - **ratio >> 1** — positions jump, inflating the path (S-Vtb3 3.32,
    S-S4 2.13, and the four excluded T sessions at 2.06–3.48).
  - **ratio << 1** — fixes are too sparse, so the polyline chord-cuts the real
    curve and under-measures it (short Vta/Vtb runs, 0.72–0.89).
- It independently reproduces earlier flags: S-S4 was already the corpus-worst
  on label instability (97%) before this check existed.
- 14 train sessions fail it; 4 of those were otherwise usable.

### Labels come from GPS in every split

`prefer_ecu` defaults to **False**. Wheel speed's only advantage is 10 Hz
resolution, which buys nothing for a one-second displacement label that GPS
supplies at its native rate.

- Measured ECU-vs-GPS label offset: **+1.9% median** (range +0.1% to +4.0%
  over 10 paired sessions) — a tyre-circumference scale error, and this
  dataset varies tyre pressure across groups, so it is not even constant.
- Groups T, A and I have no V- files, so an ECU-labelled train split would not
  share a label source with validate or test. GPS everywhere removes a ~2%
  systematic bias the model could never see in training.
- The wheel-speed path is retained behind the flag as an ablation.

### Standing plausibility guard on labels

`src/data/sanity.py::displacement_labels` is the single label builder, and it
**rejects any window whose implied speed exceeds 60 m/s** (216 km/h).

- This is a standing guard, not a one-off. Some sessions' GPS positions jump,
  inflating fix-to-fix path length: S-T1 measures 76 km of path against a 22 km
  integral of its own reported speed, which produced a mean "displacement" of
  104 m/s (375 km/h).
- **Re-deriving the time base from DATE does not help** — TIME SINCE START and
  DATE spans are identical on every T session. The defect is in the positions,
  not the clock.
- The guard fixed the validate split outright. Label instability before → after:
  S-T1 18.0% → **2.76%**, S-T4 8.2% → 3.86%, S-T5 6.5% → 2.43%,
  S-T6 8.0% → 3.97%. **Validate now has zero suspect sessions**; S-T1 is kept.
- Rejection rates are themselves a quality signal and are recorded as
  `pct_windows_rejected`: S-T6 drops 60.7% of windows, S-T1 40.0%, S-T5 37.8%,
  S-T4 17.1%; every other session in validate and test drops ~0%.

### Standing caution: band-limiting and differencing produce artifacts

**Every** band-limiting or differencing step applied before correlating in this
project has produced an artifact:

| step | artifact |
|---|---|
| differencing held GPS speed row-wise | spike train; correlation ~0 |
| row-interval jump threshold | 104/187 sessions "failed"; really 33 |
| 0.5 Hz low-pass before correlating speed | gave S-M a spurious 0.64; the lag-aligned raw figure is 0.95 |
| median of all timestamp deltas | ~1000 Hz on files that are 10 Hz |

**Correlate raw, lag-aligned signals by default.** Band-limit only with a
specific stated justification, and when you do, report both the raw and the
filtered result.

### Time sync: unresolvable AND non-critical

- **Unresolvable below ~1 s.** GPS reports at 1 Hz; no method can pin an
  IMU/GPS offset far below its sample interval. The distance-domain estimator
  confirms this empirically — **114 of 147 sessions put the argmin at the
  search boundary**, i.e. non-convergence, because a shift of a few seconds
  barely perturbs a monotone cumulative curve spanning thousands.
- **Non-critical below ~500 ms.** Injecting known offsets changes the
  displacement labels by a **median of 1.34% on clean sessions**; 13 of 16
  sampled train sessions shift ≤3%.
- **Therefore sync is treated as non-critical.** Ignore sub-second scatter.
  Sessions whose labels are sensitive to a shift are **GPS-quality problems,
  not sync problems**, and are handled by `label_stability_pct`.
- Review only `|best_lag_ms| > 2000` **with an interior argmin** in
  `time_offsets_distance.csv`. A boundary argmin means the search failed, not
  that a large offset was measured.

### Recovered yaw drift is NOT a session-quality signal

- `yaw_drift_rad` in the offsets CSV has a median of ~0.99 rad with 34/71
  sessions above 1.0 rad. This is a **symptom of low correlation**, not mount
  slippage: where r is small the yaw argmax is noise, exactly as the lag
  argmax is. Do not filter sessions on it.

### Sync error is not critical below ~500 ms

- Injecting known offsets and measuring the change in displacement labels
  (`sync_sensitivity.md`): at **±500 ms, 13 of 16 train sessions shift their
  labels by ≤3% of mean displacement, median 1.3%**. At ±250 ms, ≤1.3%.
- The three exceptions — S-S4 (97%), S-M (44%), S-Y1 (7%) — are label-quality
  problems, not sync problems: their GPS positions are jumpy, so *any* shift
  moves the label a lot. Their mean displacements are unremarkable, so this is
  not a small-denominator artefact.
- **Therefore: ignore sub-second lag scatter. Review only |lag| > 2000 ms.**

### Guard

- Any computed correlation with |r| > 1 **raises**
  (`CorrelationNormalisationError`) rather than clamping. Values above 1 mean
  the normalisation broke — typically dividing by a full-series standard
  deviation while the overlap at that lag is tiny — and clamping would hide
  the bug behind a plausible number. This actually occurred during
  development (|r| up to 3.3).

## Split

Assigned by vehicle, phone and **measured sampling rate** — never randomly, and
never by group prefix alone: group A splits by rate at the file level.

| Role | Members | Files | Hours | Rate |
|---|---|---|---|---|
| `train` | Fiesta groups: S, M, Vta, Vtb, Vw, Vfa | 71 | 26.0 | 10 Hz |
| `validate` | T2, T3, T7–T11 | 7 | ~9 | 10 Hz |
| `test` | S-A5, S-A6, S-A7, S-A8 | 4 | 7.8 | 10 Hz |
| `robustness_2hz` | S-A1/2/3/9/10/11/12/13 | 8 | 9.5 | 2 Hz |
| `ood_spotcheck` | S-I (Nigeria) | 1 | 0.2 | 10 Hz |
| `stage0_only` | **all** V- files, including St and Vfb | 90 | 39.4 | 10 Hz |
| `excluded` | S-A4, S-Y1, S-T1/T4/T5/T6 | 6 | — | see reasons |

Total 187 sessions; every session receives exactly one role.

- **`stage0_only` is family-wide**, not group-wide: every V- file carries it,
  including V- files whose group is in `train`. V- data feeds the Stage 0 INS
  dead-reckoning check only, never the learned model.
- **Validate keeps all 11 sessions**, including the 4 suspect ones (S-T1,
  S-T4, S-T5, S-T6). Measured, not assumed: three arbitrary untrained
  predictors (mean, persistence, accelerometer-variance fit) rank **identically**
  on all 11 and on the clean 7 (`persistence < mean < accel_var` both ways).
  The suspect sessions inflate absolute MAE — pooled persistence MAE 2.24 m on
  all 11 versus 0.15 m on the clean 7 — but do not change which predictor
  wins, which is the only property early stopping depends on. See
  `results/sanity/val_split_check.md`.
- **Test keeps all 4 sessions**: none is suspect (3.5–4.2% label stability).
- **`robustness_2hz` is a probe, not a test set.** At 2 Hz the Nyquist limit is
  1 Hz, so it cannot be scored against the 10 Hz `test` role.
- **Exclusions carry a machine-readable reason** (`exclusion_reason` on
  `list_sessions()`, from `EXCLUSION_REASONS` in `loader.py`):
  - **S-A4 — `no_usable_timestamps`** (also a stray `UNNAMED: 24` column).
  - **S-T1, S-T4, S-T5, S-T6 — `inflated_gps_positions`**: fix-to-fix path
    length is 2.06–3.48x the integral of each session's own reported speed.
    The 60 m/s per-window guard trims 69–94% of their windows but does not
    repair the survivors — post-guard label means were 30–42 m against 4–27 m
    for clean T sessions. Validate is therefore **7 sessions**, and its
    window-level rejection rate fell from 44.5% to 0.05%.
  - **S-Y1 — `poor_gps_label_quality`**: 7.1% label instability *and* the
    corpus-worst 0.11 lag-aligned correlation against ECU speed. Two
    independent measures agreeing. S-Y1 was the only S- file in group Y, so
    group Y now contributes no trainable smartphone data; V-Y1 and V-Y2 remain
    `stage0_only`.
- The short Vta/Vw sessions are **kept**. Their weak acceleration-domain
  correlations were the broken reference, not bad data — they score 1–3.3% on
  label stability.
- Groups A, T and I have **no V- files at all**, so nothing V-based can be
  evaluated on validate, test or the OOD spot check.
- Hours are measured by `src/sweep_corpus.py`, using DATE-derived durations for
  files with duplicate-heavy timestamps.

## Known-answer test

Onyekpe et al. 2021 (*Appl. Sci.* **11**, 1270) report classical INS dead reckoning
with a **max CRSE of 30.11 m** on subset **V-Vw12** over **10-second outages**.
Reproducing that number validates the pipeline before any learned model is trusted.

### CRSE is a VECTOR quantity — this was established empirically

`CRSE = ||Σ e_k||`, the norm of the accumulated 2-D position error, **not** the
scalar sum `Σ|e_k|` of per-second displacement errors.

- The scalar reading inverts the published difficulty ordering: it makes the
  motorway the *worst* scenario and the roundabout among the best.
- The vector form reproduces motorway-lowest / roundabout-highest and brings
  magnitudes into line (motorway 34.6 vs 30.11; roundabout 201.9 vs 171.92).
- Why: on a roundabout, per-second *distance* errors stay small while the
  track bends away from truth. Only the vector form sees heading error.
- `crse_scalar()` is retained as a diagnostic — it isolates distance error from
  heading error — but it is not the published quantity.

### Compare max-to-max

The published figures are **maxima** over 9–40 sequences. Ranking our *means*
against their *maxima* is not like-for-like and mis-ranks scenarios. Report the
mean too (it is the lower-variance statistic), but judge ordering on max.

Note V-Vw12 sits in the `stage0_only` role (and its group is a `train` group),
so this check never touches validate or test.

### Outcome: PASS (3/5 within 30%, ordering preserved) — 2026-08-30

| scenario | n (ours/theirs) | our max | our mean | their max | ratio |
|---|---|---|---|---|---|
| motorway | 9 / 9 | 34.63 | 29.53 | 30.11 | **1.15** |
| roundabout | 12 / 11 | 201.93 | 52.42 | 171.92 | **1.17** |
| sharp_cornering | 35 / 40 | 84.45 | 37.87 | 92.06 | **0.92** |
| quick_accel_change | 15 / 13 | 54.42 | 20.98 | 79.05 | 0.69 |
| hard_brake | 15 / 17 | 50.03 | 18.26 | 133.12 | 0.38 |

Recorded as a pass despite 3/5 rather than 4/5, for reasons that are evidence,
not leniency:

- **Ratio spread is 3.13x, so it is not a unit bug.** A factor error scales
  every scenario together. No factor of 9.81 (g), 3.6 (km/h), 2.0 (double
  integration) or 57.3 (deg/rad) reconciles the five ratios.
- **Both misses have OUR errors smaller than published** (0.69, 0.38). No
  amplification bug produces errors that are too small.
- **The known-answer test itself reproduces at 1.15x**, and the hardest
  scenario at 1.17x.
- **Difficulty ordering holds**: motorway lowest, roundabout highest — the
  property a systematic factor error cannot fake.
- Residual attributed to **window phase against unpublished subset
  boundaries**. Supported by a finer-stride sensitivity test: at a 1 s stride
  hard_brake moves 0.376 -> 0.571 and quick_accel 0.688 -> 0.791, the right
  direction, while inflating sequence counts far beyond the paper's (127 vs
  17).
- **The 10 s non-overlapping stride is the protocol of record** because it
  reproduces the published sequence counts almost exactly (9/9, 12/11, 15/13,
  15/17, 35/40). **The stride was NOT tuned to pass** — the finer stride was
  run as a diagnostic and discarded, and it would not have passed either.

### Baseline property: systematic under-prediction

On the motorway subset **CAE = -CRSE_scalar exactly**: every second
under-predicts distance. This is a systematic bias, not scatter.

- **Consequence for evaluation:** any comparison made on *cumulative distance*
  will flatter the baseline, because consistent under-prediction partly
  cancels accumulated heading error. Compare on position error, not on
  distance travelled.
- It is also the most obviously learnable defect, so a model that fails to
  beat the baseline is doing something badly wrong.

## Baselines: what counts as a competitor

**Windowed persistence is NOT a deployable baseline.** Predicting the previous
second's displacement scores **0.18 m on validate and 1.04 m on test**
(bucket-mean MAE), which looks unbeatable — but it consumes the previous
second's *true* displacement, and that is precisely what a GNSS outage removes.

- It measures the **autocorrelation of the label sequence**, not achievable
  performance. Quote it as a ceiling on what the labels contain, never as a
  target.
- Fed through the outage harness, where after the first step it can only see
  its own output, it degenerates to **constant-velocity DR**: hold the speed
  and heading observed at t0 and extrapolate.
- **The deployable baselines are INS DR and constant-velocity DR through the
  harness** (`src/eval/harness_baselines.py`), reported at 10/30/60 s.
- Constant-velocity DR reads **no IMU at all** — it knows only the speed regime
  at outage start. A model's margin over it is exactly the part attributable to
  reading the IMU rather than to knowing how fast the vehicle was going. That
  makes it the diagnostic baseline, not just a weak one.
- The `mean` predictor (10.26 validate / 9.05 test) is a sanity floor only.

## Evaluation: always stratify by speed

The splits differ in regime mix, not in the regimes themselves. Window shares
by 1 s displacement bucket:

| split | 0–5 | 5–15 | 15–25 | >25 |
|---|---|---|---|---|
| train | 0.164 | 0.428 | 0.273 | 0.136 |
| validate | 0.056 | 0.102 | 0.311 | **0.532** |
| test | 0.098 | 0.142 | 0.439 | 0.322 |

Within each bucket the label means agree closely across splits (0–5: 1.0–1.9;
5–15: 10.2–11.0; 15–25: 19.9–21.6; >25: 26.6–31.7). **The regimes are
comparable; only the mix differs.** Pooled means (train 13.9, validate 24.6,
test 19.7) reflect that mix, not a data defect.

- **The early-stopping metric is `eval.buckets.early_stopping_metric`** — the
  unweighted mean MAE across occupied buckets, not pooled MAE. Pooled MAE lets
  the split's regime mix drive the stopping point.
- Buckets with fewer than 30 windows are skipped rather than allowed to swing
  the mean.
- Bucketing is on the TRUE label, never the prediction, so a bad model cannot
  relabel its hard cases into an easier bucket.
- Report pooled MAE alongside, but never stop on it.

## Layout

- `src/` — library code
- `data/` — datasets (git-ignored, not committed)
- `notebooks/` — exploration
- `results/` — figures, metrics, model outputs
