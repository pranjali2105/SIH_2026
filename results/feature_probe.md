# Hand-crafted feature probes

The ablation that decides whether a 3.85 M-parameter CNN earns its complexity. 24 features = 4 statistics x 6 channels.


## Windowed MAE

| split | predictor | bucket-mean | pooled | 0-5 | 5-15 | 15-25 | >25 |
|---|---|---|---|---|---|---|---|
| validate | `mean` | **10.262** | 12.807 | 12.832 | 2.998 | 7.325 | 17.893 |
| validate | `ridge_probe` | **11.189** | 14.351 | 9.720 | 6.862 | 8.453 | 19.721 |
| validate | `xgb_probe` | **10.470** | 15.477 | 4.792 | 4.825 | 11.015 | 21.248 |
| test | `mean` | **9.050** | 9.200 | 11.961 | 3.756 | 7.771 | 12.713 |
| test | `ridge_probe` | **8.010** | 8.111 | 9.264 | 5.009 | 6.662 | 11.107 |
| test | `xgb_probe` | **5.944** | 5.577 | 6.345 | 6.280 | 4.096 | 7.055 |

## Top ridge weights (standardised features)

```
  absmean_acc_z    6.196
  absdiff_gyr_y    5.194
  absmean_gyr_z    4.602
  absmean_gyr_x    4.517
  absdiff_acc_x    4.334
  absdiff_acc_y    3.749
  absdiff_gyr_z    3.078
  std_gyr_y        2.283
```

## Mechanism check: 2 Hz sessions

At 2 Hz the Nyquist limit is 1 Hz, so the high-frequency vertical-acceleration texture the probe leans on at 10 Hz is largely gone. If the probe's advantage over predict-the-mean collapses, the vibration mechanism is demonstrated.

```
         regime  n_windows  probe_mae_m  mean_mae_m  improvement_pct
2Hz (own split)      13282        7.449       7.546            1.285
```