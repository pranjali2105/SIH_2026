# Training run

- best single-epoch validation 60 s drift: **697.35 m** (epoch 26) -> `best.pt`
- best 3-epoch moving-average drift: **720.13 m** (epoch 41) -> `best_ma3.pt`
- drift standard error (last epoch, 66 outages): 49.29 m
- median epoch-to-epoch change in mean drift: 39.26 m
- **Noise dominates**: SE is comparable to or larger than the epoch-to-epoch movement, so best-single-epoch selection is arbitrary; the MA3 checkpoint is the one to use.

Log: `train_log.jsonl`
