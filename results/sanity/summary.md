# Sanity check summary

Sessions processed: **187**. Report only — nothing was repaired and the split was not changed.


## Trajectory invariants

| Invariant | pass | fail | n/a |
|---|---|---|---|
| start/end inside country box | 187 | 0 | 0 |
| path length vs sweep (±5%) | 185 | 0 | 2 |
| no fix jump beyond reported speed | 153 | 33 | 1 |
| loop closes (where a loop) | 21 | 8 | 158 |

`n/a` means the invariant does not apply (e.g. a route that never returns near its origin) or the input was missing.


### Sessions containing an implausible fix jump (33)

```
 session           role group  n_jumps    max_jump_m
    S-A1 robustness_2hz     A        2   5743.831594
   S-A13 robustness_2hz     A        2 126266.235460
    S-A2 robustness_2hz     A        2   6192.409585
    S-A3 robustness_2hz     A        2   5909.887446
    S-A9 robustness_2hz     A        1     32.286304
     V-M    stage0_only     M        8     44.143684
    V-S1    stage0_only     S        3     10.001700
   V-S3c    stage0_only     S       19     92.074861
   V-St1    stage0_only    St        7     22.934579
   V-St7    stage0_only    St        4     65.908320
 V-Vfa02    stage0_only   Vfa       32    139.079233
V-Vfb01a    stage0_only   Vfb        3    175.054490
  V-vtb1    stage0_only   Vtb        3      5.156200
   V-Vw2    stage0_only    Vw        3      8.768896
    V-Y1    stage0_only     Y     1129   1839.091917
    V-Y2    stage0_only     Y        7     22.934579
    S-A6           test     A        7   1231.870627
    S-A7           test     A        2     58.970445
    S-A8           test     A        8    163.070230
     S-M          train     M        1    226.406102
 S-Vta13          train   Vta        1    242.000588
  S-Vta2          train   Vta        1     22.537649
 S-Vta21          train   Vta        2    467.135681
 S-Vta30          train   Vta        1    227.593449
  S-Vtb1          train   Vtb        1   3986.366110
 S-Vtb10          train   Vtb        1    124.709628
   S-Vw4          train    Vw        1    321.998049
    S-Y1          train     Y        1    189.800282
    S-T1       validate     T      180    675.577969
    S-T3       validate     T       10    336.954903
    S-T4       validate     T      209    920.927413
    S-T5       validate     T      183    699.040896
    S-T6       validate     T      527    585.672043
```

### Sessions failing loop closure (8)

```
session        role group  end_to_start_m
   V-S4 stage0_only     S      221.260124
V-vta19 stage0_only   Vta      231.456389
V-vtb10 stage0_only   Vtb      222.860123
 V-vtb4 stage0_only   Vtb      163.456515
S-Vta19       train   Vta      231.104780
 S-Vta9       train   Vta      160.647504
S-Vtb10       train   Vtb      210.391679
 S-Vtb4       train   Vtb      157.928291
```

## Time synchronisation

Lag estimated for **173/187** sessions. Offsets are stored per session in `time_offsets.csv`; there is no global constant.


### Lag distribution by group

| group | n | median lag (ms) | p05 | p95 | median peak r |
|---|---|---|---|---|---|
| A | 12 | -1000 | -4615 | 4670 | -0.02 |
| I | 1 | 2900 | 2900 | 2900 | -0.09 |
| M | 2 | 200 | -70 | 470 | 0.42 |
| S | 12 | -100 | -2525 | 3460 | 0.43 |
| St | 4 | -100 | -100 | -100 | 0.90 |
| T | 11 | 1600 | -3500 | 4450 | -0.11 |
| Vfa | 4 | 300 | -100 | 955 | 0.25 |
| Vfb | 11 | -100 | -100 | -100 | 0.96 |
| Vta | 56 | 0 | -550 | 4025 | 0.57 |
| Vtb | 23 | -100 | -2830 | 2940 | 0.46 |
| Vw | 34 | -100 | -620 | 3445 | 0.65 |
| Y | 3 | -100 | -3430 | -100 | 0.89 |

### Lag distribution by role

| role | n | median lag (ms) | median peak r |
|---|---|---|---|
| ood_spotcheck | 1 | 2900 | -0.09 |
| robustness_2hz | 8 | -800 | -0.02 |
| stage0_only | 83 | -100 | 0.93 |
| test | 4 | -1250 | 0.01 |
| train | 66 | 1250 | -0.16 |
| validate | 11 | 1600 | -0.11 |

## Suspect sessions (93)

Flagged when `|best_lag_ms| > 500` or `|peak_correlation| < 0.3`. A weak peak means the cross-correlation never locked onto real structure — that is a finding about the session, not a tuning problem.

```
session           role group  best_lag_ms  peak_correlation  multi_axis_R  n_samples_used                   channel
   S-A4       excluded     A          NaN               NaN           NaN               0            no time column
    S-I  ood_spotcheck     I       2900.0         -0.094264      0.119890            5799 levelled (yaw,lag) search
   S-A1 robustness_2hz     A       5000.0         -0.022126      0.044731           29707 levelled (yaw,lag) search
  S-A10 robustness_2hz     A       -800.0          0.274443      0.282414           44425 levelled (yaw,lag) search
  S-A11 robustness_2hz     A      -5000.0          0.089200      0.095424           25770 levelled (yaw,lag) search
  S-A12 robustness_2hz     A       4400.0         -0.063473      0.096050           23360 levelled (yaw,lag) search
  S-A13 robustness_2hz     A      -4300.0          0.056257      0.056268           45915 levelled (yaw,lag) search
   S-A2 robustness_2hz     A      -3000.0         -0.019541      0.029372           58145 levelled (yaw,lag) search
   S-A3 robustness_2hz     A       4000.0         -0.017689      0.023324           55661 levelled (yaw,lag) search
   S-A9 robustness_2hz     A       -800.0         -0.367678      0.398378           59805 levelled (yaw,lag) search
V-Vta18    stage0_only   Vta          NaN               NaN           NaN               0        too few timestamps
V-vta19    stage0_only   Vta          NaN               NaN           NaN               0        too few timestamps
 V-vta9    stage0_only   Vta          NaN               NaN           NaN               0        too few timestamps
V-vtb10    stage0_only   Vtb          NaN               NaN           NaN               0        too few timestamps
  V-Vw1    stage0_only    Vw          NaN               NaN           NaN               0               flat signal
 V-Vw13    stage0_only    Vw          NaN               NaN           NaN               0        too few timestamps
 V-Vw15    stage0_only    Vw          NaN               NaN           NaN               0               flat signal
   S-A5           test     A      -1400.0          0.288291      0.290570           85044 levelled (yaw,lag) search
   S-A6           test     A       -900.0          0.085784      0.092058           85375 levelled (yaw,lag) search
   S-A7           test     A      -1100.0         -0.075520      0.075627           43871 levelled (yaw,lag) search
   S-A8           test     A      -1500.0         -0.123104      0.128530           66616 levelled (yaw,lag) search
    S-M          train     M        500.0         -0.095691      0.097405           61760 levelled (yaw,lag) search
   S-S1          train     S        700.0         -0.349164      0.349688           51745 levelled (yaw,lag) search
   S-S2          train     S       1300.0         -0.239751      0.239846           92011 levelled (yaw,lag) search
  S-S3a          train     S       2200.0         -0.255415      0.258371           24620 levelled (yaw,lag) search
  S-S3b          train     S      -5000.0         -0.065350      0.113195           27076 levelled (yaw,lag) search
  S-S3c          train     S       5000.0         -0.044819      0.050139           37182 levelled (yaw,lag) search
   S-S4          train     S       -500.0         -0.055088      0.055208           55780 levelled (yaw,lag) search
S-Vfa01          train   Vfa       1000.0         -0.472800      0.477669           11485 levelled (yaw,lag) search
S-Vfa02          train   Vfa        700.0         -0.411251      0.415007           67523 levelled (yaw,lag) search
S-Vta10          train   Vta      -3500.0          0.270505      0.273269            1502 levelled (yaw,lag) search
S-Vta13          train   Vta       3300.0         -0.295978      0.510910             404 levelled (yaw,lag) search
S-Vta14          train   Vta       5000.0          0.257687      0.274836            2884 levelled (yaw,lag) search
S-Vta15          train   Vta       2100.0          0.285196      0.341580             833 levelled (yaw,lag) search
S-Vta16          train   Vta        700.0          0.210649      0.216067           11334 levelled (yaw,lag) search
S-Vta17          train   Vta        600.0         -0.386063      0.386536            4530 levelled (yaw,lag) search
S-Vta19          train   Vta          NaN               NaN           NaN               0        too few timestamps
S-Vta20          train   Vta      -4700.0         -0.161017      0.303772            3222 levelled (yaw,lag) search
S-Vta21          train   Vta       1600.0         -0.355152      0.392291            2078 levelled (yaw,lag) search
S-Vta22          train   Vta       4100.0          0.259537      0.267207            1564 levelled (yaw,lag) search
S-Vta23          train   Vta       2100.0          0.391902      0.392913            1109 levelled (yaw,lag) search
S-Vta24          train   Vta       4100.0          0.307920      0.480095            1171 levelled (yaw,lag) search
S-Vta26          train   Vta       2500.0          0.286577      0.343278            1934 levelled (yaw,lag) search
S-Vta27          train   Vta       2600.0          0.494353      0.494965            2540 levelled (yaw,lag) search
S-Vta28          train   Vta        700.0          0.281310      0.281649            4210 levelled (yaw,lag) search
S-Vta29          train   Vta        700.0         -0.173015      0.191127           23698 levelled (yaw,lag) search
 S-Vta3          train   Vta       3300.0         -0.398705      0.516942             644 levelled (yaw,lag) search
S-Vta30          train   Vta       1300.0          0.377241      0.377746           17135 levelled (yaw,lag) search
 S-Vta4          train   Vta       3400.0         -0.307609      0.323590            1788 levelled (yaw,lag) search
 S-Vta5          train   Vta       -700.0          0.378214      0.654726             307 levelled (yaw,lag) search
 S-Vta6          train   Vta       3800.0         -0.161564      0.239199            1375 levelled (yaw,lag) search
 S-Vta7          train   Vta       4000.0         -0.438603      0.455588             840 levelled (yaw,lag) search
 S-Vta8          train   Vta       4000.0          0.355948      0.370435            3675 levelled (yaw,lag) search
 S-Vta9          train   Vta          NaN               NaN           NaN               0        too few timestamps
 S-Vtb1          train   Vtb       1400.0         -0.326201      0.343001           32458 levelled (yaw,lag) search
S-Vtb10          train   Vtb          NaN               NaN           NaN               0        too few timestamps
S-Vtb11          train   Vtb       3000.0         -0.417942      0.507040             360 levelled (yaw,lag) search
S-Vtb12          train   Vtb       1000.0         -0.527238      0.583245             447 levelled (yaw,lag) search
 S-Vtb2          train   Vtb       2400.0         -0.385296      0.396316            5711 levelled (yaw,lag) search
 S-Vtb4          train   Vtb       4600.0         -0.311386      0.324617             555 levelled (yaw,lag) search
 S-Vtb6          train   Vtb      -4100.0          0.276479      0.370954             497 levelled (yaw,lag) search
 S-Vtb7          train   Vtb      -2200.0          0.346855      0.471638             461 levelled (yaw,lag) search
 S-Vtb9          train   Vtb      -2900.0          0.463978      0.508255             451 levelled (yaw,lag) search
  S-Vw1          train    Vw          NaN               NaN           NaN               0               flat signal
 S-Vw10          train    Vw        400.0         -0.284255      0.292042             652 levelled (yaw,lag) search
 S-Vw11          train    Vw       4000.0          0.463695      0.465863            4908 levelled (yaw,lag) search
 S-Vw12          train    Vw       2100.0          0.224346      0.236907             917 levelled (yaw,lag) search
 S-Vw13          train    Vw          NaN               NaN           NaN               0        too few timestamps
S-Vw14b          train    Vw       1700.0          0.368883      0.375850           19588 levelled (yaw,lag) search
S-Vw14c          train    Vw        300.0         -0.235662      0.241780           15825 levelled (yaw,lag) search
 S-Vw15          train    Vw          NaN               NaN           NaN               0               flat signal
S-Vw16a          train    Vw       1400.0         -0.589761      0.595909            5878 levelled (yaw,lag) search
S-Vw16b          train    Vw       3900.0         -0.500045      0.502496            1126 levelled (yaw,lag) search
 S-Vw17          train    Vw      -1400.0          0.337598      0.495584             329 levelled (yaw,lag) search
  S-Vw2          train    Vw       1500.0          0.408122      0.413326           52712 levelled (yaw,lag) search
  S-Vw3          train    Vw       3000.0         -0.206136      0.206803            3861 levelled (yaw,lag) search
  S-Vw4          train    Vw       1700.0         -0.104417      0.106221          126525 levelled (yaw,lag) search
  S-Vw5          train    Vw       2400.0          0.299977      0.325371            1012 levelled (yaw,lag) search
  S-Vw6          train    Vw       1200.0          0.103061      0.210119            1280 levelled (yaw,lag) search
  S-Vw7          train    Vw       2500.0          0.154870      0.165525            1601 levelled (yaw,lag) search
  S-Vw8          train    Vw       3200.0         -0.189084      0.197452            1528 levelled (yaw,lag) search
  S-Vw9          train    Vw      -2200.0          0.552186      0.610488             552 levelled (yaw,lag) search
   S-Y1          train     Y      -3800.0          0.056735      0.058905           66908 levelled (yaw,lag) search
   S-T1       validate     T       -100.0          0.196763      0.226812            7384 levelled (yaw,lag) search
  S-T10       validate     T       1900.0         -0.448897      0.449028            6698 levelled (yaw,lag) search
   S-T2       validate     T       1900.0          0.222966      0.226003           75550 levelled (yaw,lag) search
   S-T3       validate     T       1300.0         -0.184966      0.186021           64693 levelled (yaw,lag) search
   S-T4       validate     T      -2500.0         -0.113847      0.115048           35803 levelled (yaw,lag) search
   S-T5       validate     T       4100.0         -0.146543      0.156825           20370 levelled (yaw,lag) search
   S-T6       validate     T       1600.0          0.085674      0.090656           25109 levelled (yaw,lag) search
   S-T7       validate     T       3900.0          0.272498      0.274779          101701 levelled (yaw,lag) search
   S-T8       validate     T       4800.0          0.243244      0.243828           48339 levelled (yaw,lag) search
   S-T9       validate     T      -4500.0         -0.127452      0.148854           79908 levelled (yaw,lag) search
```

### Single-channel vs multi-axis correlation

`multi_axis_R` is a least-squares fit of ALL accelerometer axes to d(speed)/dt at the best lag — mounting-invariant, so a weak single-axis peak cannot simply be blamed on picking the wrong axis.


| family | n | median peak r (single) | median R (multi-axis) |
|---|---|---|---|
| S- | 90 | 0.29 | 0.32 |

## Eye check

`vw12_jpg_comparison.png` — V-Vw12 plotted trajectory beside its supplied route map.


JPGs exist only for train-role and `stage0_only` sessions — none for groups T, A or I — so this is the only session where the plotted route can be compared against a real map.
