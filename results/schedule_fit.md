# Scheduled sampling for the HMM observation model

Inner predictor: `model`. Schedule: inverse_sigmoid, 6 epochs, eps 1 -> 0, 60 s fitting rollouts.


Leave-one-session-out: each session is scored with parameters fitted on the other three, so every outage below is out-of-sample.


## Mean drift (m) by duration

| predictor | 10 s | 30 s | 60 s |
|---|---|---|---|
| `viterbi_base_model` | 59.5 | 164.5 | 312.4 |
| `viterbi_fitted_model` | 59.5 | 164.5 | 312.4 |

## Fitted parameters per fold

| held out | `class_speed_penalty` | `min_sigma_m` | `sigma_scale` | `sigma_turn_rad` |
|---|---|---|---|---|
| S-A5 | 2.000 | 1.500 | 1.000 | 0.192 |
| S-A6 | 2.000 | 1.500 | 1.000 | 0.209 |
| S-A7 | 2.000 | 1.500 | 1.000 | 0.192 |
| S-A8 | 2.000 | 1.500 | 1.000 | 0.192 |

Hand-set defaults: `class_speed_penalty`=2.000, `min_sigma_m`=1.500, `sigma_scale`=1.000, `sigma_turn_rad`=0.209


## Curriculum

The objective at eps>0 is teacher-forced and is NOT a deployment number; only the eps=0 row is comparable to the drift table above.

| held out | e0 (eps 1.00) | e1 (eps 0.87) | e2 (eps 0.64) | e3 (eps 0.36) | e4 (eps 0.13) | e5 (eps 0.00) |
|---|---|---|---|---|---|---|
| S-A5 | 10.9 | 11.6 | 13.7 | 21.2 | 56.3 | 171.6 |
| S-A6 | 11.0 | 11.8 | 14.2 | 22.1 | 58.2 | 196.8 |
| S-A7 | 11.9 | 12.9 | 15.6 | 24.8 | 64.9 | 222.5 |
| S-A8 | 10.3 | 11.1 | 13.8 | 21.3 | 59.8 | 196.1 |
