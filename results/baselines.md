# Untrained baselines

Targets the first trained model must beat. Neither predictor looks at the IMU.


- `mean`: predict the train mean, 13.846 m, always.
- `previous`: predict the previous second's displacement. Displacement is near-continuous, so this is a strong baseline; a model that cannot beat it has learned nothing the label's own autocorrelation does not already give.


**Bucket-mean MAE is the number of record.** Pooled MAE is shown alongside and is dominated by each split's regime mix.


## Results

| split | predictor | bucket-mean MAE | pooled MAE | MAE 0-5 | MAE 5-15 | MAE 15-25 | MAE >25 |
|---|---|---|---|---|---|---|---|
| validate | `mean` | **10.262** | 12.807 | 12.832 | 2.998 | 7.325 | 17.893 |
| validate | `previous` | **0.182** | 0.148 | 0.114 | 0.345 | 0.169 | 0.101 |
| test | `mean` | **9.050** | 9.200 | 11.961 | 3.756 | 7.771 | 12.713 |
| test | `previous` | **1.037** | 1.219 | 0.355 | 1.050 | 1.304 | 1.437 |

## Window counts per bucket

```
  validate mean      n=38117  dropped=0     0-5:2117  5-15:3895  15-25:11841  >25:20264
  validate previous  n=38109  dropped=8     0-5:2111  5-15:3895  15-25:11840  >25:20263
  test     mean      n=27937  dropped=0     0-5:2725  5-15:3962  15-25:12262  >25:8988
  test     previous  n=27933  dropped=4     0-5:2722  5-15:3961  15-25:12262  >25:8988
```

`dropped` counts windows with no in-session predecessor (first of a session, or following a gap); they are excluded rather than given a borrowed value.
