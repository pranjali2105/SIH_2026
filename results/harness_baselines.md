# Deployable baselines through the outage harness

Drift = CRSE, the norm of the accumulated 2-D position error at the end of the outage.


`constant_velocity_dr` holds the speed and heading observed at t0 and extrapolates — it is windowed persistence fed through the harness, where it can only see its own previous output. It reads no IMU at all, so a model's margin over it is exactly the part that comes from the IMU rather than from knowing the speed regime.


## Mean drift (m)

| split | duration | INS DR | constant-velocity DR | n |
|---|---|---|---|---|
| validate | 10 s | 61.03 | 69.61 | 3601 |
| validate | 30 s | 175.39 | 173.43 | 1196 |
| validate | 60 s | 416.92 | 365.19 | 598 |
| test | 10 s | 28.87 | 18.11 | 2499 |
| test | 30 s | 120.61 | 98.76 | 836 |
| test | 60 s | 310.53 | 285.40 | 417 |

## Full statistics

```
    role            predictor  duration_s    n  mean_drift_m  median_drift_m  p90_drift_m  max_drift_m
    test constant_velocity_dr        10.0 2499         18.11           13.50        38.11       128.25
    test constant_velocity_dr        30.0  836         98.76           75.09       213.32       791.21
    test constant_velocity_dr        60.0  417        285.40          225.17       611.59      1070.75
    test               ins_dr        10.0 2499         28.87           26.86        46.67       136.64
    test               ins_dr        30.0  836        120.61          101.77       219.54       793.77
    test               ins_dr        60.0  417        310.53          254.28       610.64      1059.09
validate constant_velocity_dr        10.0 3601         69.61           34.62       225.10       618.10
validate constant_velocity_dr        30.0 1196        173.43          152.55       292.78       734.43
validate constant_velocity_dr        60.0  598        365.19          308.38       664.20      1710.21
validate               ins_dr        10.0 3601         61.03           16.95       240.85       653.22
validate               ins_dr        30.0 1196        175.39          138.48       353.95       819.98
validate               ins_dr        60.0  598        416.92          384.93       728.78      1934.92
```