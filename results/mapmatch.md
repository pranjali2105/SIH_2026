# Offline map matching and along-road tracking

Drift is CRSE: the norm of the accumulated 2-D position error at the end of the outage. Split `test` — S-A5..S-A8, the four Volvo sessions the offline OSRM extract was cropped to.


Every `map_matched_X` row uses exactly the same per-second displacements as row `X`; the only difference is that position is a scalar along a road polyline instead of an integrated heading. The gap between the two is therefore the heading component of the drift.


## Benchmark: under 100 m over a 1 km (800-1200 m), 60 s outage

| predictor | n | mean | median | p90 | % under 100 m |
|---|---|---|---|---|---|
| `viterbi_matched_graph_constant_velocity_dr` | 58 | 247.2 | 168.3 | 486.3 | 31% |
| `hmm_matched_graph_constant_velocity_dr` | 58 | 252.2 | 168.3 | 494.0 | 31% |
| `map_matched_graph_constant_velocity_dr` | 58 | 252.1 | 168.3 | 494.0 | 31% |
| `hmm_matched_graph_ins_dr` | 58 | 264.6 | 202.6 | 600.0 | 29% |
| `map_matched_graph_ins_dr` | 58 | 264.5 | 202.6 | 600.0 | 29% |
| `viterbi_matched_graph_ins_dr` | 58 | 273.3 | 208.1 | 605.2 | 29% |
| `ins_dr` | 60 | 380.4 | 292.1 | 790.1 | 8% |
| `constant_velocity_dr` | 60 | 387.3 | 316.0 | 868.3 | 12% |
| `viterbi_matched_graph_model` | 58 | 383.0 | 346.4 | 695.5 | 12% |
| `hmm_matched_graph_model` | 58 | 415.0 | 360.5 | 716.9 | 12% |
| `map_matched_graph_model` | 58 | 415.1 | 360.5 | 716.9 | 12% |
| `viterbi_matched_graph_model_speed_fused` | 58 | 510.4 | 516.3 | 857.3 | 9% |
| `hmm_matched_graph_model_speed_fused` | 58 | 501.6 | 516.3 | 867.2 | 10% |
| `map_matched_graph_model_speed_fused` | 58 | 501.6 | 516.3 | 867.2 | 10% |
| `model` | 60 | 707.0 | 589.2 | 1199.6 | 0% |
| `model_speed_fused` | 60 | 736.3 | 706.3 | 1203.5 | 0% |

## Mean drift (m) by duration

| predictor | 10 s | 30 s | 60 s |
|---|---|---|---|
| `constant_velocity_dr` | 18.1 | 98.8 | 285.4 |
| `hmm_matched_graph_constant_velocity_dr` | 12.3 | 47.7 | 125.8 |
| `hmm_matched_graph_ins_dr` | 25.4 | 85.2 | 192.8 |
| `hmm_matched_graph_model` | 60.1 | 169.0 | 327.9 |
| `hmm_matched_graph_model_speed_fused` | 63.7 | 188.8 | 376.6 |
| `ins_dr` | 28.9 | 120.6 | 310.5 |
| `map_matched_graph_constant_velocity_dr` | 12.1 | 46.4 | 119.6 |
| `map_matched_graph_ins_dr` | 25.2 | 84.0 | 187.1 |
| `map_matched_graph_model` | 60.0 | 167.8 | 322.7 |
| `map_matched_graph_model_speed_fused` | 63.6 | 187.6 | 370.8 |
| `model` | 66.9 | 245.2 | 606.4 |
| `model_speed_fused` | 71.3 | 266.3 | 642.4 |
| `viterbi_matched_graph_constant_velocity_dr` | 11.2 | 42.4 | 111.4 |
| `viterbi_matched_graph_ins_dr` | 24.5 | 80.4 | 181.9 |
| `viterbi_matched_graph_model` | 59.5 | 164.6 | 312.9 |
| `viterbi_matched_graph_model_speed_fused` | 63.1 | 187.1 | 362.9 |

## Median drift (m) by duration

| predictor | 10 s | 30 s | 60 s |
|---|---|---|---|
| `constant_velocity_dr` | 13.5 | 75.1 | 225.2 |
| `hmm_matched_graph_constant_velocity_dr` | 6.6 | 19.6 | 46.6 |
| `hmm_matched_graph_ins_dr` | 24.1 | 74.2 | 153.4 |
| `hmm_matched_graph_model` | 47.6 | 129.3 | 247.6 |
| `hmm_matched_graph_model_speed_fused` | 51.9 | 146.4 | 278.5 |
| `ins_dr` | 26.9 | 101.8 | 254.3 |
| `map_matched_graph_constant_velocity_dr` | 6.6 | 19.3 | 45.6 |
| `map_matched_graph_ins_dr` | 24.0 | 73.9 | 152.8 |
| `map_matched_graph_model` | 47.5 | 129.3 | 245.1 |
| `map_matched_graph_model_speed_fused` | 51.7 | 145.3 | 277.9 |
| `model` | 54.2 | 209.6 | 528.5 |
| `model_speed_fused` | 58.6 | 239.1 | 588.0 |
| `viterbi_matched_graph_constant_velocity_dr` | 6.5 | 19.0 | 45.0 |
| `viterbi_matched_graph_ins_dr` | 23.9 | 73.9 | 152.8 |
| `viterbi_matched_graph_model` | 47.0 | 129.3 | 243.3 |
| `viterbi_matched_graph_model_speed_fused` | 51.1 | 147.3 | 277.9 |

## Outages skipped

An outage is skipped when GPS accuracy at t0 exceeds 20 m, or when no road lies within the snap radius of the start fix. The harness skips slow starts (<5 m/s) for every predictor alike.

| predictor | scored | skipped | candidates |
|---|---|---|---|
| `constant_velocity_dr` | 3752 | 457 | 4209 |
| `hmm_matched_graph_constant_velocity_dr` | 3731 | 478 | 4209 |
| `hmm_matched_graph_ins_dr` | 3731 | 478 | 4209 |
| `hmm_matched_graph_model` | 3731 | 478 | 4209 |
| `hmm_matched_graph_model_speed_fused` | 3731 | 478 | 4209 |
| `ins_dr` | 3752 | 457 | 4209 |
| `map_matched_graph_constant_velocity_dr` | 3731 | 478 | 4209 |
| `map_matched_graph_ins_dr` | 3731 | 478 | 4209 |
| `map_matched_graph_model` | 3731 | 478 | 4209 |
| `map_matched_graph_model_speed_fused` | 3731 | 478 | 4209 |
| `model` | 3752 | 457 | 4209 |
| `model_speed_fused` | 3752 | 457 | 4209 |
| `viterbi_matched_graph_constant_velocity_dr` | 3731 | 478 | 4209 |
| `viterbi_matched_graph_ins_dr` | 3731 | 478 | 4209 |
| `viterbi_matched_graph_model` | 3731 | 478 | 4209 |
| `viterbi_matched_graph_model_speed_fused` | 3731 | 478 | 4209 |

## What the tracker did

Counts pooled over every outage and duration. `forced` junctions had only one branch to take, so the gyro was not consulted; `ambiguous` re-matches are the ones where more than one road was plausible and the estimate was therefore left unmodified.

| predictor | outages | junctions | forced | re-matches | applied | ambiguous | skipped (GPS) | skipped (no road) |
|---|---|---|---|---|---|---|---|---|
| `hmm_matched_graph_constant_velocity_dr` | 3731 | 3823 | 2339 | 7463 | 6433 | 1030 | 21 | 0 |
| `hmm_matched_graph_ins_dr` | 3731 | 3436 | 2134 | 7463 | 6429 | 1034 | 21 | 0 |
| `hmm_matched_graph_model` | 3731 | 4342 | 2653 | 7463 | 6427 | 1036 | 21 | 0 |
| `hmm_matched_graph_model_speed_fused` | 3731 | 4456 | 2672 | 7463 | 6428 | 1035 | 21 | 0 |
| `map_matched_graph_constant_velocity_dr` | 3731 | 3777 | 2335 | 7463 | 6438 | 1025 | 21 | 0 |
| `map_matched_graph_ins_dr` | 3731 | 3407 | 2133 | 7463 | 6430 | 1033 | 21 | 0 |
| `map_matched_graph_model` | 3731 | 4277 | 2643 | 7463 | 6431 | 1032 | 21 | 0 |
| `map_matched_graph_model_speed_fused` | 3731 | 4391 | 2666 | 7463 | 6430 | 1033 | 21 | 0 |
| `viterbi_matched_graph_constant_velocity_dr` | 3731 | 3664 | 2276 | 0 | 0 | 0 | 21 | 0 |
| `viterbi_matched_graph_ins_dr` | 3731 | 3339 | 2098 | 0 | 0 | 0 | 21 | 0 |
| `viterbi_matched_graph_model` | 3731 | 3950 | 2486 | 0 | 0 | 0 | 21 | 0 |
| `viterbi_matched_graph_model_speed_fused` | 3731 | 4013 | 2487 | 0 | 0 | 0 | 21 | 0 |

## Full statistics
```
                                 predictor  duration_s    n  mean_drift_m  median_drift_m  p90_drift_m  max_drift_m  mean_cae_m  mean_abs_cae_m
                      constant_velocity_dr        10.0 2499         18.11           13.50        38.11       128.25       -0.19           10.02
                      constant_velocity_dr        30.0  836         98.76           75.09       213.32       791.21        1.59           37.55
                      constant_velocity_dr        60.0  417        285.40          225.17       611.59      1070.75        7.35          100.85
    hmm_matched_graph_constant_velocity_dr        10.0 2486         12.34            6.61        27.58       315.12       -0.12            9.92
    hmm_matched_graph_constant_velocity_dr        30.0  831         47.70           19.63       106.63      1241.65        1.69           37.34
    hmm_matched_graph_constant_velocity_dr        60.0  414        125.83           46.59       327.03      2392.73        7.62          100.67
                  hmm_matched_graph_ins_dr        10.0 2486         25.36           24.07        39.33       307.41      -21.62           24.06
                  hmm_matched_graph_ins_dr        30.0  831         85.19           74.24       132.19      1174.22      -60.73           77.52
                  hmm_matched_graph_ins_dr        60.0  414        192.82          153.36       316.73      2231.77     -113.10          170.17
                   hmm_matched_graph_model        10.0 2486         60.08           47.55       134.52       315.75       17.90           59.02
                   hmm_matched_graph_model        30.0  831        168.96          129.32       396.50      1030.77       58.24          161.21
                   hmm_matched_graph_model        60.0  414        327.85          247.61       748.62      1867.99      119.81          301.68
       hmm_matched_graph_model_speed_fused        10.0 2486         63.72           51.88       135.53       325.10       17.88           62.59
       hmm_matched_graph_model_speed_fused        30.0  831        188.80          146.44       426.86      1267.58       60.75          182.33
       hmm_matched_graph_model_speed_fused        60.0  414        376.64          278.51       832.47      2393.57      126.15          353.43
                                    ins_dr        10.0 2499         28.87           26.86        46.67       136.64      -21.67           24.13
                                    ins_dr        30.0  836        120.61          101.77       219.54       793.77      -60.80           77.64
                                    ins_dr        60.0  417        310.53          254.28       610.64      1059.09     -113.22          170.34
    map_matched_graph_constant_velocity_dr        10.0 2486         12.15            6.57        27.41       315.12       -0.11            9.92
    map_matched_graph_constant_velocity_dr        30.0  831         46.41           19.29       104.71      1241.65        1.70           37.34
    map_matched_graph_constant_velocity_dr        60.0  414        119.64           45.57       318.67      2392.73        7.64          100.69
                  map_matched_graph_ins_dr        10.0 2486         25.19           24.03        39.22       307.41      -21.61           24.06
                  map_matched_graph_ins_dr        30.0  831         84.01           73.94       130.74      1174.22      -60.73           77.52
                  map_matched_graph_ins_dr        60.0  414        187.13          152.77       305.91      2231.77     -113.08          170.15
                   map_matched_graph_model        10.0 2486         59.96           47.50       134.48       315.75       17.90           59.02
                   map_matched_graph_model        30.0  831        167.79          129.32       391.64      1030.77       58.24          161.22
                   map_matched_graph_model        60.0  414        322.69          245.08       742.35      1852.56      119.83          301.71
       map_matched_graph_model_speed_fused        10.0 2486         63.58           51.72       135.19       325.10       17.89           62.59
       map_matched_graph_model_speed_fused        30.0  831        187.64          145.33       420.66      1267.58       60.76          182.34
       map_matched_graph_model_speed_fused        60.0  414        370.84          277.95       817.37      2393.57      126.17          353.45
                                     model        10.0 2499         66.92           54.20       138.75       315.85       17.76           59.01
                                     model        30.0  836        245.20          209.64       462.41       837.49       58.22          160.93
                                     model        60.0  417        606.36          528.52      1084.86      2086.74      120.02          301.53
                         model_speed_fused        10.0 2499         71.26           58.62       141.38       295.41       17.65           62.64
                         model_speed_fused        30.0  836        266.27          239.05       492.67       877.18       60.67          182.32
                         model_speed_fused        60.0  417        642.41          587.98      1125.63      1937.59      126.24          353.73
viterbi_matched_graph_constant_velocity_dr        10.0 2486         11.24            6.53        26.69       176.13       -0.10            9.91
viterbi_matched_graph_constant_velocity_dr        30.0  831         42.39           18.96       100.98       684.19        1.78           37.36
viterbi_matched_graph_constant_velocity_dr        60.0  414        111.35           45.04       310.35      1104.32        7.94          100.64
              viterbi_matched_graph_ins_dr        10.0 2486         24.45           23.93        38.70       153.29      -21.60           24.05
              viterbi_matched_graph_ins_dr        30.0  831         80.44           73.94       126.31       388.29      -60.67           77.49
              viterbi_matched_graph_ins_dr        60.0  414        181.95          152.77       316.73      1314.83     -112.91          170.03
               viterbi_matched_graph_model        10.0 2486         59.47           47.02       133.42       305.33       17.93           59.04
               viterbi_matched_graph_model        30.0  831        164.57          129.32       384.67       826.73       58.45          161.35
               viterbi_matched_graph_model        60.0  414        312.88          243.31       699.70      1352.65      120.78          302.43
   viterbi_matched_graph_model_speed_fused        10.0 2486         63.07           51.09       134.46       282.32       17.92           62.61
   viterbi_matched_graph_model_speed_fused        30.0  831        187.10          147.26       426.86       880.89       61.15          182.72
   viterbi_matched_graph_model_speed_fused        60.0  414        362.89          277.95       803.71      1546.18      127.00          354.20
```