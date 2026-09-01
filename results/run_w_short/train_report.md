# Training run

- best single-epoch validation 60 s drift: **686.73 m** (epoch 41) -> `best.pt`
- best 3-epoch moving-average drift: **714.63 m** (epoch 43) -> `best_ma3.pt`
- drift standard error (last epoch, 66 outages): 50.34 m
- median epoch-to-epoch change in mean drift: 67.02 m
- **Signal exceeds noise**: single-epoch selection resolves real differences.

Log: `train_log.jsonl`
