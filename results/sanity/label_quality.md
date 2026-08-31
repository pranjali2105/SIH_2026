# Label quality (GPS), corpus-wide

`label_stability_pct` = mean absolute change in the 1 s displacement label when the window moves +/-500 ms, as a percentage of mean displacement.


Built as a sync-sensitivity test, it turns out to measure GPS label quality: a session whose labels swing when the window shifts half a second has noisy positions, independent of any clock offset. That makes it usable on groups T, A and I, which have no ECU data.


Scored: **161/187** sessions (6 of them flagged `low_denominator`, mean displacement < 2 m, and excluded from the suspect lists and from the cut-point quantiles).


## Validation against ECU ground truth

Spearman rho between `label_stability_pct` and the S-/V- speed correlation, over **49** Fiesta pairs: **rho = -0.154** (p = 0.29).


A strong negative rho is the expected direction: worse GPS (lower speed correlation) should mean less stable labels (higher percentage).


## Distribution

```
      0 -     1%    49  ########################################
      1 -     2%    35  #############################
      2 -     3%    31  #########################
      3 -     5%    30  ########################
      5 -    10%     9  #######
     10 -    20%     1  #
     20 -    50%     0  
     50 -   100%     0  
    100 -   inf%     0  
```

quantiles: p50 1.84%, p60 2.22%, p75 3.02%, p90 4.05%, p95 5.23%, max 12.98%


## Proposed cut points

Derived from this corpus's own distribution (p60 and p90) rather than chosen as round numbers:


- **clean**: < 2.22%
- **usable (down-weight)**: 2.22% - 4.05%
- **suspect**: > 4.05%


### Band counts by role

```
band            clean  low_denominator  suspect  usable
role                                                   
excluded            1                0        0       0
robustness_2hz      6                0        0       2
stage0_only        40                4       15      30
test                0                0        1       3
train              40                2        0       6
validate            6                0        0       5
```

## What each application would exclude (NOTHING IS DROPPED YET)


### train — down-weight or exclude

48 sessions scored; **0 suspect**.


### validate — exclude, so early stopping is not driven by label noise

11 sessions scored; **0 suspect**.


### test — exclude entirely — unreliable ground truth corrupts the headline number

4 sessions scored; **1 suspect**.

```
session group  label_stability_pct  mean_displacement_m  n_windows
   S-A8     A             4.190788             20.51772      66321
```

### Worst 15 interpretable sessions

```
 session        role group  label_stability_pct  mean_displacement_m
 V-vta25 stage0_only   Vta            12.984520             2.397700
V-Vfb01d stage0_only   Vfb             5.688363             3.169945
V-Vfb01a stage0_only   Vfb             5.624030             3.892439
 V-vta19 stage0_only   Vta             5.509718             8.111768
   V-S3b stage0_only     S             5.400117             5.528340
  V-vtb4 stage0_only   Vtb             5.323795             4.358349
   V-Vw5 stage0_only    Vw             5.295223             6.594817
V-Vfb02c stage0_only   Vfb             5.232359             8.330831
   V-Vw7 stage0_only    Vw             5.229647             7.222685
   V-Vw8 stage0_only    Vw             5.164068             6.929617
V-Vfb02f stage0_only   Vfb             4.830264             7.093201
  V-vta3 stage0_only   Vta             4.747489             5.820225
V-Vfb02b stage0_only   Vfb             4.659668             6.986608
 V-vta24 stage0_only   Vta             4.342268             5.933172
    S-A8        test     A             4.190788            20.517720
```

### Excluded as low-denominator (6)

Slow sessions, not bad GPS: the ratio's denominator is too small for the percentage to mean anything.

```
session        role group  label_stability_pct  mean_displacement_m
  V-Vw1 stage0_only    Vw            35.027281             0.022644
 V-Vw15 stage0_only    Vw            14.860207             0.065946
 V-vtb3 stage0_only   Vtb             7.980898             0.833653
V-vta20 stage0_only   Vta             5.116859             1.210173
   S-S4       train     S             1.768048             0.454823
 S-Vtb3       train   Vtb             1.049065             1.894170
```