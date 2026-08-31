# Windowed dataset

100 samples (10 s) at 10 Hz, stride 10 (1 s), channels acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z in the gravity-levelled frame.


Labels from `displacement_labels()`, so the 60 m/s plausibility guard applies.


## Per split

```
    role  n_sessions  n_windows  label_mean_m  label_std_m  label_min_m  label_max_m  frac_rejected  frac_stationary  frac_yaw_true_valid  n_sessions_with_ecu
   train          47      83238     13.845651     8.598916     0.000000    36.410694       0.001056         0.055744             0.835628                   47
validate           7      38117     24.632842     9.455053     0.027200    48.226740       0.000524         0.033240             0.944303                    0
    test           4      27937     19.668059     8.089815     0.006265    47.872048       0.000000         0.042309             0.893546                    0
```

## Label distribution by speed bucket

Displacement over 1 s is numerically mean speed in m/s, so the label is its own bucket key.

```
bucket      0-5  15-25   5-15    >25
role                                
test      0.098  0.439  0.142  0.322
train     0.164  0.273  0.428  0.135
validate  0.056  0.311  0.102  0.532

mean label (m) within bucket:
bucket     0-5  15-25   5-15    >25
role                               
test      1.88  21.62  10.24  26.56
train     1.62  19.85  10.20  28.09
validate  1.01  21.17  11.03  31.74
```

## Path ratio (GPS path / speed integral)

Guard: reject above 1.15 (positions jump) or below 0.75 (fixes too sparse). Sessions in the 0.75-0.90 chord-cutting band are KEPT and flagged for ablation: their labels are slightly short but structurally sound.

```
  chord-cut band (0.75-0.90): 1 sessions, 133 windows
  path_ratio range: 0.882 - 1.118
```

## Leakage check

`verify_no_leakage()` passed over 58 sessions: no session under two roles, no window crossing a session boundary, roles agree with `list_sessions()`.


## Normalisation

Fitted on **train only** (83238 windows), saved to `results/windows/norm_stats.json`.

```
  acc_x    mean   +0.01426   std    1.88703
  acc_y    mean   -0.05846   std    1.68322
  acc_z    mean   +0.05386   std    0.82766
  gyr_x    mean   -0.00333   std    0.11757
  gyr_y    mean   -0.00508   std    0.25560
  gyr_z    mean   -0.00007   std    0.15586
```

## Sessions yielding no windows (24)

```
  S-S4: InflatedPositionsError: S-S4: GPS path length is 2.13x the integral of its own reported speed (limit 1.15) — positions jump, inflating the path. The geometry is wrong, so labels derived from it are wrong in an unbounded way; this is a session-level defect the per-window guard cannot repair.
  S-Vta11: InflatedPositionsError: S-Vta11: GPS path length is only 0.74x the integral of its own reported speed (floor 0.75) — fixes are too sparse and the path chord-cuts the real curve. Mild chord-cutting is tolerated (labels are short but structurally sound); this session is past that.
  S-Vta12: no usable windows
  S-Vta13: no usable windows
  S-Vta19: no usable windows
  S-Vta25: no usable windows
  S-Vta3: no usable windows
  S-Vta5: no usable windows
  S-Vta9: no usable windows
  S-Vtb10: no usable windows
  S-Vtb11: no usable windows
  S-Vtb12: InflatedPositionsError: S-Vtb12: GPS path length is only 0.72x the integral of its own reported speed (floor 0.75) — fixes are too sparse and the path chord-cuts the real curve. Mild chord-cutting is tolerated (labels are short but structurally sound); this session is past that.
  S-Vtb3: InflatedPositionsError: S-Vtb3: GPS path length is 3.32x the integral of its own reported speed (limit 1.15) — positions jump, inflating the path. The geometry is wrong, so labels derived from it are wrong in an unbounded way; this is a session-level defect the per-window guard cannot repair.
  S-Vtb4: no usable windows
  S-Vtb6: no usable windows
  S-Vtb7: no usable windows
  S-Vtb8: no usable windows
  S-Vtb9: no usable windows
  S-Vw1: no usable windows
  S-Vw10: no usable windows
  S-Vw13: no usable windows
  S-Vw15: no usable windows
  S-Vw17: no usable windows
  S-Vw9: no usable windows
```