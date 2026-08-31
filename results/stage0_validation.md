# Stage 0 known-answer test

Classical INS dead reckoning vs Onyekpe et al. 2021 (Appl. Sci. 11, 1270), 10-second outages.


## Results

| scenario | n (ours/theirs) | our max | our mean | our min | our std | their max | max ratio |
|---|---|---|---|---|---|---|---|
| motorway | 9 / 9 | 34.63 | 29.53 | 24.29 | 2.54 | 30.11 | 1.150 |
| roundabout | 12 / 11 | 201.93 | 52.42 | 6.52 | 59.39 | 171.92 | 1.175 |
| quick_acceleration_change | 15 / 13 | 54.42 | 20.98 | 4.17 | 11.69 | 79.05 | 0.688 |
| hard_brake | 15 / 17 | 50.03 | 18.26 | 2.38 | 11.39 | 133.12 | 0.376 |
| sharp_cornering | 35 / 40 | 84.45 | 37.87 | 3.74 | 22.75 | 92.06 | 0.917 |

## Sequence counts

- motorway: 9 vs 9
- roundabout: 12 vs 11
- quick_acceleration_change: 15 vs 13
- hard_brake: 15 vs 17
- sharp_cornering: 35 vs 40

## Difficulty ordering

- ours (by max CRSE):        motorway < hard_brake < quick_acceleration_change < sharp_cornering < roundabout
- theirs (by published max): motorway < quick_acceleration_change < sharp_cornering < hard_brake < roundabout
- ours (by mean CRSE):       hard_brake < quick_acceleration_change < motorway < sharp_cornering < roundabout  *(not the comparison of record — shown for completeness)*

motorway lowest: **True**; roundabout highest: **True**


## Verdict

- max CRSE within 30% of published: **3/5** scenarios (need >= 4)
- difficulty ordering preserved: **True**

### FAIL


## Diagnosis by ratio

| scenario | max ratio | suggests |
|---|---|---|
| motorway | 1.150 | — |
| roundabout | 1.175 | — |
| quick_acceleration_change | 0.688 | — |
| hard_brake | 0.376 | — |
| sharp_cornering | 0.917 | — |

Ratio spread across scenarios: **3.13x**. Ratios are scenario-dependent, which RULES OUT a unit bug: a factor error (g, km/h, deg/rad, a stray 2) scales every scenario together instead of reordering them. 
None of the diagnostic ratios (9.81, 3.6, 2.0, 57.3) appears.


Scenarios where our error is materially SMALLER than published: quick_acceleration_change, hard_brake. Being too good is not a unit bug — it is consistent with our uniformly-strided outage windows not landing on the specific manoeuvre the paper's maximum came from.


## Notes

- [roundabout] V-Vfb02d: 1 outage(s) skipped (first: t0=69743.4: start speed 0.26 m/s below 5.0 m/s — GPS heading is meaningless at rest)
- [sharp_cornering] V-Vw6: 1 outage(s) skipped (first: t0=56904.2: start speed 2.19 m/s below 5.0 m/s — GPS heading is meaningless at rest)
- [sharp_cornering] V-Vw7: 3 outage(s) skipped (first: t0=57516.8: start speed 0.67 m/s below 5.0 m/s — GPS heading is meaningless at rest)
- [sharp_cornering] V-Vw8: 4 outage(s) skipped (first: t0=57680.0: start speed 4.02 m/s below 5.0 m/s — GPS heading is meaningless at rest)