# Validation split check

Do the suspect validation sessions change what validation *says*? Three arbitrary, untrained predictors are ranked on both candidate splits; if the ranking holds, the suspect sessions are not distorting the signal.


Suspect cut: `label_stability_pct` > 5.21%. Sessions evaluated: 7 of 7; suspect: 0.


## Pooled MAE (metres), window-weighted

| predictor | all 7 | clean only (7) | delta |
|---|---|---|---|
| `mean` | 18.163 | 18.163 | +0.000 |
| `previous` | 0.148 | 0.148 | +0.000 |
| `accel_var` | 18.327 | 18.327 | +0.000 |

## Ranking (best first)

- all 7:      previous < mean < accel_var
- clean only: previous < mean < accel_var

**Ranking is STABLE.**


No suspect sessions remain among the 7: the two splits are identical, so the ranking is trivially stable. The check is retained as a regression guard — if a future change readmits a noisy session, this will catch it.


## Per session

```
session  label_stability_pct  suspect  n_windows  mae_mean  mae_previous  mae_accel_var
  S-T10             0.909542    False        657  6.712948      0.180436       6.751149
  S-T11             2.854148    False        538  4.293320      0.242416       4.195296
   S-T2             0.227074    False       7536 20.712414      0.121430      20.823561
   S-T3             0.405787    False       6450 16.494676      0.183966      16.639705
   S-T7             0.246991    False      10134 17.957238      0.120481      18.146599
   S-T8             0.340822    False       4822 18.982059      0.176700      19.183805
   S-T9             0.296115    False       7979 18.745971      0.152554      18.952325
```