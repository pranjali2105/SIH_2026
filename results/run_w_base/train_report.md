# Training run

- best single-epoch validation 60 s drift: **709.07 m** (epoch 29) -> `best.pt`
- best 3-epoch moving-average drift: **781.66 m** (epoch 31) -> `best_ma3.pt`
- drift standard error (last epoch, 66 outages): 53.79 m
- median epoch-to-epoch change in mean drift: 30.33 m
- **Noise dominates**: SE is comparable to or larger than the epoch-to-epoch movement, so best-single-epoch selection is arbitrary; the MA3 checkpoint is the one to use.

Log: `train_log.jsonl`
