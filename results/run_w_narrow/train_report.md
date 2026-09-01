# Training run

- best single-epoch validation 60 s drift: **782.70 m** (epoch 10) -> `best.pt`
- best 3-epoch moving-average drift: **821.48 m** (epoch 44) -> `best_ma3.pt`
- drift standard error (last epoch, 66 outages): 53.63 m
- median epoch-to-epoch change in mean drift: 60.22 m
- **Signal exceeds noise**: single-epoch selection resolves real differences.

Log: `train_log.jsonl`
