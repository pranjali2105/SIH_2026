# Training run

- best single-epoch validation 60 s drift: **1069.96 m** (epoch 25) -> `best.pt`
- best 3-epoch moving-average drift: **1095.03 m** (epoch 38) -> `best_ma3.pt`
- drift standard error (last epoch, 66 outages): 74.97 m
- median epoch-to-epoch change in mean drift: 17.62 m
- **Noise dominates**: SE is comparable to or larger than the epoch-to-epoch movement, so best-single-epoch selection is arbitrary; the MA3 checkpoint is the one to use.

Log: `train_log.jsonl`
