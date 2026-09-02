# Training run

- best single-epoch validation 60 s drift: **721.21 m** (epoch 25) -> `best.pt`
- best 3-epoch moving-average drift: **786.33 m** (epoch 26) -> `best_ma3.pt`
- drift standard error (last epoch, 66 outages): 56.35 m
- median epoch-to-epoch change in mean drift: 31.23 m
- **Noise dominates**: SE is comparable to or larger than the epoch-to-epoch movement, so best-single-epoch selection is arbitrary; the MA3 checkpoint is the one to use.

Log: `train_log.jsonl`
