# Offline map matching and along-road tracking

Drift is CRSE: the norm of the accumulated 2-D position error at the end of the outage. Split `test` — S-A5..S-A8, the four Volvo sessions the offline OSRM extract was cropped to.


Every `map_matched_X` row uses exactly the same per-second displacements as row `X`; the only difference is that position is a scalar along a road polyline instead of an integrated heading. The gap between the two is therefore the heading component of the drift.


## Benchmark: under 100 m over a 1 km (800-1200 m), 60 s outage

| predictor | n | mean | median | p90 | % under 100 m |
|---|---|---|---|---|---|
| `map_matched_constant_velocity_dr` | 58 | 252.1 | 168.3 | 494.0 | 31% |
| `map_matched_ins_dr` | 58 | 264.6 | 202.7 | 600.0 | 29% |
| `ins_dr` | 60 | 380.4 | 292.1 | 790.1 | 8% |
| `constant_velocity_dr` | 60 | 387.3 | 316.0 | 868.3 | 12% |
| `map_matched_model` | 58 | 436.6 | 350.3 | 736.6 | 9% |
| `model` | 60 | 753.5 | 562.3 | 1460.3 | 0% |

## Mean drift (m) by duration

| predictor | 10 s | 30 s | 60 s |
|---|---|---|---|
| `constant_velocity_dr` | 18.1 | 98.8 | 285.4 |
| `ins_dr` | 28.9 | 120.6 | 310.5 |
| `map_matched_constant_velocity_dr` | 12.2 | 46.8 | 122.0 |
| `map_matched_ins_dr` | 25.2 | 84.4 | 189.3 |
| `map_matched_model` | 52.8 | 143.3 | 291.5 |
| `model` | 63.4 | 242.5 | 602.1 |

## Median drift (m) by duration

| predictor | 10 s | 30 s | 60 s |
|---|---|---|---|
| `constant_velocity_dr` | 13.5 | 75.1 | 225.2 |
| `ins_dr` | 26.9 | 101.8 | 254.3 |
| `map_matched_constant_velocity_dr` | 6.6 | 19.4 | 46.1 |
| `map_matched_ins_dr` | 24.1 | 74.1 | 152.9 |
| `map_matched_model` | 42.4 | 103.7 | 191.8 |
| `model` | 53.1 | 207.0 | 537.4 |

## Outages skipped

An outage is skipped when GPS accuracy at t0 exceeds 20 m, or when no road lies within the snap radius of the start fix. The harness skips slow starts (<5 m/s) for every predictor alike.

| predictor | scored | skipped | candidates |
|---|---|---|---|
| `constant_velocity_dr` | 3752 | 457 | 4209 |
| `ins_dr` | 3752 | 457 | 4209 |
| `map_matched_constant_velocity_dr` | 3731 | 478 | 4209 |
| `map_matched_ins_dr` | 3731 | 478 | 4209 |
| `map_matched_model` | 3731 | 478 | 4209 |
| `model` | 3752 | 457 | 4209 |

## What the tracker did

Counts pooled over every outage and duration. `forced` junctions had only one branch to take, so the gyro was not consulted; `ambiguous` re-matches are the ones where more than one road was plausible and the estimate was therefore left unmodified.

| predictor | outages | junctions | forced | re-matches | applied | ambiguous | skipped (GPS) | skipped (no road) |
|---|---|---|---|---|---|---|---|---|
| `map_matched_constant_velocity_dr` | 3731 | 3802 | 2338 | 7463 | 6762 | 701 | 21 | 0 |
| `map_matched_ins_dr` | 3731 | 3423 | 2131 | 7463 | 6772 | 691 | 21 | 0 |
| `map_matched_model` | 3731 | 5203 | 2982 | 7463 | 6811 | 652 | 21 | 0 |

## Full statistics
```
                       predictor  duration_s    n  mean_drift_m  median_drift_m  p90_drift_m  max_drift_m  mean_cae_m  mean_abs_cae_m
            constant_velocity_dr        10.0 2499         18.11           13.50        38.11       128.25       -0.19           10.02
            constant_velocity_dr        30.0  836         98.76           75.09       213.32       791.21        1.59           37.55
            constant_velocity_dr        60.0  417        285.40          225.17       611.59      1070.75        7.35          100.85
                          ins_dr        10.0 2499         28.87           26.86        46.67       136.64      -21.67           24.13
                          ins_dr        30.0  836        120.61          101.77       219.54       793.77      -60.80           77.64
                          ins_dr        60.0  417        310.53          254.28       610.64      1059.09     -113.22          170.34
map_matched_constant_velocity_dr        10.0 2486         12.19            6.59        27.57       315.17       -0.11            9.92
map_matched_constant_velocity_dr        30.0  831         46.81           19.43       106.29      1241.69        1.69           37.34
map_matched_constant_velocity_dr        60.0  414        121.98           46.06       324.24      2392.80        7.62          100.68
              map_matched_ins_dr        10.0 2486         25.24           24.07        39.31       307.46      -21.61           24.06
              map_matched_ins_dr        30.0  831         84.35           74.10       130.88      1174.29      -60.73           77.52
              map_matched_ins_dr        60.0  414        189.29          152.95       310.91      2231.73     -113.09          170.16
               map_matched_model        10.0 2486         52.78           42.41       107.96       321.62       18.78           51.79
               map_matched_model        30.0  831        143.34          103.73       321.61      1284.64       57.67          138.14
               map_matched_model        60.0  414        291.53          191.81       650.90      2556.60      118.00          261.92
                           model        10.0 2499         63.40           53.10       119.60       305.24       18.96           51.89
                           model        30.0  836        242.53          207.05       456.23      1022.17       58.71          138.87
                           model        60.0  417        602.08          537.36      1118.63      2256.36      118.91          263.41
```