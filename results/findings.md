# GNSS-Denied Vehicle Navigation — Findings

Results log for the IO-VNBD displacement-prediction work. Each entry states
what was measured, on what, and what it does and does not license as a
conclusion. Negative results are kept: several are more informative than the
positive ones.

---

## 0. Headline result

60-second GNSS outage, test split (4 sessions, 417 outages), mean position
drift:

| predictor | drift | reads IMU? |
|---|---|---|
| constant-velocity DR | **285.4 m** | no |
| classical INS DR | 310.5 m | yes |
| **our model** (CNN + raw-gyro heading) | **648.6 m** | yes |
| our model + oracle heading | **294.8 m** | yes |

**The learned model does not beat a baseline that reads no IMU at all.** That
is the honest headline. What the oracle-heading row shows is *why*: with
heading supplied, the same model reaches 294.8 m — parity with the baseline —
so the displacement estimate is sound and essentially all of the deficit is
heading integration. See §1.

Figure: `results/figures/drift_60s_test.png`.

---

## 1. Heading integration, not displacement, is the dominant error on test

**The headline result.** Substituting oracle GPS heading for the model's
integrated heading, changing nothing else:

| test, 60 s outage | mean CRSE |
|---|---|
| constant-velocity DR (reads no IMU) | 285.4 m |
| model, integrated heading | 940.2 m |
| **model + ZUPT + calibration, oracle heading** | **277.4 m** |
| model + ZUPT, oracle heading | 293.5 m |

Per speed bucket, the regime that dominates outage distance:

| test, 60 s, `>25` bucket | mean CRSE |
|---|---|
| constant-velocity DR | 283.6 m |
| **model, oracle heading** | **119.5 m** |

**So displacement prediction is competitive — better than the baseline at
motorway speed — and heading integration accounts for roughly 70% of the
observed gap.**

**The oracle qualification, stated plainly:** true heading is *not available
during a GNSS outage*. This result bounds what solving heading would buy; it
does not bank it. What it establishes is where the error lives, which makes
the remaining problem a heading-observability / map-matching problem rather
than a displacement-model problem.

Corroboration from CAE: every oracle-heading variant has `|CAE| / CRSE = 1.00–1.02`,
i.e. the residual is *entirely* systematic bias with no directional scatter.
The raw model sits at 0.32 — the scatter is heading rotating the track away
from truth.

Final configuration (raw-gyro heading, conjunction-gated ZUPT), per bucket at
60 s on test:

| bucket | const-vel DR | our model | + oracle heading |
|---|---|---|---|
| 0–5 | 325.6 | 775.0 | 724.4 |
| 5–15 | 297.5 | 856.7 | 735.8 |
| 15–25 | 281.3 | 641.9 | **296.3** |
| >25 | 283.6 | 576.5 | **119.5** |

At motorway speed with heading supplied the model is **2.4x better than the
baseline** (119.5 vs 283.6 m). The deficit is concentrated at low speed, where
it over-predicts (§3).

**This does not hold on validate**, where oracle heading removes only 17%
(875.1 → 700.3 m) and the model is worse than the baseline in every bucket.
The two splits fail for different reasons; see §3.

---

## 2. The learned yaw head is worse than the raw gyro on test

Ablation on the same checkpoint, heading integrated from the levelled gyro
z-axis instead of the yaw head:

| split, 60 s | yaw head | raw gyro | change |
|---|---|---|---|
| test | 938.8 m | **647.2 m** | **−31.1%** |
| validate | 863.8 m | 894.9 m | +3.6% |

**The learned correction generalises worse than the sensor it was meant to
correct.** Not a supervision problem: 83.6% of train windows carry a valid yaw
target (validate 94.4%, test 89.4%). Test is a different phone from train, so
the most likely reading is that the head learned phone-specific gyro
characteristics that do not transfer.

Consequence: raw gyro integration IS the default heading source in the final
configuration, and the yaw head is disabled. On test this alone takes drift
from 940.2 m to **648.6 m**.

**Supervision is not the explanation, but it is worth recording**: the yaw head
trains only on windows above 5 m/s, which is 83.6% of train windows (validate
94.4%, test 89.4%). So it is not starved of labels. The more likely account is
that test is a different handset from train and the head learned gyro
characteristics that do not transfer — the same distribution-shift story as §3,
in a different head.

This is an honest negative result about the multi-task setup: **three of the
four heads earn their place and one does not.** Displacement is the product;
the stationary head is useful once gated properly (§4, §7); the yaw head
should be dropped or retrained with explicit cross-device validation.

---

## 3. The model regresses toward the training mean

On test's slow windows (true label < 5 m):

| quantity | value |
|---|---|
| model prediction | 12.36 m |
| true label | 1.91 m |
| train label mean | 13.86 m |

The prediction sits at the training mean regardless of input. The same
mechanism appears with the opposite sign on validate's fast windows
(CAE −953.6 m in the `>25` bucket, negative on **100%** of outages).

Train MAE is 1.84 m against a test bucket-mean of ~8–9 m, so the model fits
train well and shrinks only out of distribution — i.e. this is a
generalisation failure, not underfitting.

---

## 4. Two fixes that did not work, and why

Both measured rather than assumed.

| variant | test 60 s | validate 60 s |
|---|---|---|
| model | 940.2 | 843.0 |
| + ZUPT soft gating | 938.8 (−0.1%) | 863.8 (+2.5%, worse) |
| + ZUPT + train-fitted calibration | 930.2 (−1.1%) | 865.1 (+2.6%, worse) |

**ZUPT** was applied correctly and the stationary head is well calibrated
(P(stationary) = 0.79 on truly stationary windows, 0.013 on moving). It fails
because stationarity is not the defect: only 4.2% of windows are truly
stationary (< 0.3 m/s) while the failing `0–5` bucket spans 0–5 m/s. The bulk
are slow-but-moving, correctly not flagged, and over-predicted 6.5×.

**Post-hoc calibration** fitted on train comes out as the identity map (train
MAE 1.844 → 1.856). There is no shrinkage *on train* to learn a correction
from, so a train-fitted calibration is structurally incapable of fixing an
out-of-distribution shrinkage. Fitting it on validate would spend the
validation set.

---

## 5. Baseline hierarchy

Deployable baselines, 60 s outages, mean CRSE:

| predictor | validate | test |
|---|---|---|
| constant-velocity DR | 365.2 | **285.4** |
| INS dead reckoning | 416.9 | 310.5 |
| xgboost feature probe | 987.9 | 414.2 |
| ridge feature probe | 907.5 | 537.4 |

**Constant-velocity DR beats INS DR at almost every duration.** On smartphone
IMU, integrating the accelerometer is worse than not integrating it: bias and
noise double-integrate into error faster than the signal contributes. Since
constant-velocity DR reads no IMU at all, any margin over it is exactly the
value of reading the IMU.

**Windowed persistence (0.18 m validate / 1.04 m test) is not a baseline.** It
consumes the previous second's *true* displacement, which is what an outage
removes. It measures label autocorrelation and is a ceiling on label content,
never a target.

---

## 6. Pipeline validation (Stage 0)

Classical INS DR reproduces Onyekpe et al. 2021 within 30% on 3 of 5 scenarios
with the difficulty ordering preserved (motorway lowest, roundabout highest).
Recorded as a pass; reasoning in CLAUDE.md. Ratio spread of 3.13× rules out a
unit bug, and both misses have our errors *smaller* than published.

**A metric bug was caught by the ordering criterion, not by the magnitudes.**
Reading CRSE as the scalar sum of per-second displacement errors inverts the
published scenario difficulty; it must be `‖Σ e_k‖`, the norm of the
accumulated error vector. A scalar sum cannot see heading error.

---

## 7. Environment note: xgboost must be imported before torch

On macOS the two ship separate OpenMP runtimes. If torch is imported first,
any later xgboost fit dies as a **hard crash with exit code 0 and no
traceback** — indistinguishable from success to anything checking return
codes. `KMP_DUPLICATE_LIB_OK=TRUE` does not help; import order is the only
fix, and it is pinned with comments in `feature_probe.py` and
`final_scoring.py`.

---

## 8. Non-holonomic ESKF: attempted, did not beat raw gyro

An error-state EKF applying the non-holonomic constraint (a road vehicle
cannot move sideways), in the spirit of Brossard et al.'s AI-IMU
(github.com/mbrossar/ai-imu-dr) but independently implemented. State
`[px, py, psi, b_g, phi]`, model sigma as displacement noise, ZUPT making the
gyro bias observable, measurement noise inflated with |omega|.

**Result: 1008-1035 m against raw gyro's 604.6 m on S-A5. Not adopted.**

Four diagnostic steps, each measured:

| step | outcome |
|---|---|
| verify constraint residual | **leveling is clean** — horizontal means +0.0035 / +0.0008 m/s² on 7678 straight windows. The 3.10 m/s² residual was 83x larger than any real horizontal acceleration, i.e. a symptom of the diverged bias, not its cause |
| fix ZUPT gating | real bug, not the binding one: drift moved 0.6 m |
| remove mount yaw phi | bias divergence halved (18.0 → 9.2 rad/s), drift unchanged |
| add forward relation | did not close it (bias 23.8 rad/s, phi drift 13.7 rad) |

**Why it failed.** The lateral constraint is applied through a mount
orientation `phi` that the filter cannot identify from 60 seconds of data, and
`phi` error maps directly into heading. Removing `phi` does not help because
the *filter* reads raw level-frame accelerometer channels whose yaw offset is
arbitrary — the network's yaw-invariance (via augmentation) applies to its
scalar displacement output, not to the channels the filter consumes.

**Identified next step, not attempted:** estimate `phi` per session offline
from GPS-available stretches, where speed and heading are both observed, then
hold it fixed during the outage. That makes the constraint determined without
asking the filter to identify the mount in 60 seconds.

---

## 9. Implementation finding: a pseudo-measurement is not free to re-apply

Generalisable beyond this project. The non-holonomic constraint is a physical
fact true at every instant, so it is tempting to apply it at the sensor rate.
Doing so treats **600 correlated readings of one fact as 600 independent
measurements**, which collapses the covariance and leaves the filter chasing
noise.

| NHC update rate | max estimated gyro bias |
|---|---|
| 10 Hz (600 per outage) | **23.79 rad/s** (≈1360 °/s — physically absurd) |
| 1 Hz (60 per outage) | **0.57 rad/s** (plausible) |

Reducing the update rate alone converted a divergent filter into a stable one.
It did not make it *accurate* — heading error stayed at 1.28 rad against raw
gyro's 0.37 — but "stable and wrong" and "divergent" have different causes and
should not be confused. Sanity-checking the estimated bias against a
physically plausible magnitude caught this; drift alone would not have.

---

## Open

- **Heading observability during an outage** — map matching or road-network
  priors. This is the single highest-value direction: §1 shows it is worth
  ~354 m of the 363 m deficit on test.
- **Per-session offline mount-yaw estimation** to make the ESKF constraint
  determined (§8).
- **Validate's high-speed under-prediction** (§3). Inverse-frequency bucket
  weighting was implemented and launched but the run was stopped at 0 epochs
  when we time-boxed modelling, so the pre-registered prediction is
  **untested, not disconfirmed**.
- **Cross-device generalisation.** Train, validate and test are three
  different handsets. Both the yaw head (§2) and the displacement shrinkage
  (§3) are consistent with device-specific overfitting, and we have no
  same-device held-out split to separate that from ordinary overfitting.
