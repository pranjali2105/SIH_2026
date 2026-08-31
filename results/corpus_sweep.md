```
# IO-VNBD corpus sweep (diagnostic)

Discovered 564 CSVs; 187 unique (family, session) pairs after dedup.
Read with encoding='iso-8859-1'.

Chosen copies by tree:
  Synchronised    / Categorised     144
  Unsynchronised  / Categorised      17
  Unsynchronised  / Uncategorised    26

==========================================================================================
1. RATE DISTRIBUTION BY GROUP  (files / hours at each effective_hz)
==========================================================================================

This is the table the split decision rests on.

  (4 files with bimodal timestamps are bucketed by hz_tick, not by median delta.)
  (6 duplicate-heavy files use their DATE-derived duration.)

-- S- files --
  group               10 Hz              2 Hz           unknown   total h
  -----------------------------------------------------------------------
  A               4f / 7.8h         8f / 9.5h         1f / 0.0h      17.3
  I               1f / 0.2h                 -                 -       0.2
  M               1f / 1.7h                 -                 -       1.7
  S               6f / 8.0h                 -                 -       8.0
  T             11f / 13.1h                 -                 -      13.1
  Vfa             2f / 2.2h                 -                 -       2.2
  Vta            30f / 3.6h                 -                 -       3.6
  Vtb            12f / 3.2h                 -                 -       3.2
  Vw             20f / 7.3h                 -                 -       7.3
  Y               1f / 1.9h                 -                 -       1.9
  ALL           88f / 48.9h         8f / 9.5h         1f / 0.0h      58.4

-- V- files --
  group               10 Hz   total h
  -----------------------------------
  M               1f / 2.9h       2.9
  S               6f / 8.6h       8.6
  St              4f / 4.6h       4.6
  Vfa             2f / 2.2h       2.2
  Vfb            11f / 3.2h       3.2
  Vta            31f / 3.6h       3.6
  Vtb            13f / 3.2h       3.2
  Vw             20f / 7.3h       7.3
  Y               2f / 3.7h       3.7
  ALL           90f / 39.4h      39.4

  !! groups spanning more than one rate: A

-- files re-timed from DATE (dup_timestamp_frac > 0.05) --
  file              dup_frac  hz_from_TIME  hz_from_DATE  rows_per_s   hz_tick  dur_DATE_s
  ----------------------------------------------------------------------------------------
  S-I.csv              0.484         10.00         10.00       10.00     10.00       579.9
  S-T1.csv             0.673       1000.00       1000.00       34.88     10.00       738.4
  S-T4.csv             0.490         10.10         10.10       20.52     10.00      3580.2
  S-T5.csv             0.532         23.81         23.81       22.77     10.00      2037.0
  S-T6.csv             0.687       1000.00       1000.00       34.43     10.00      2510.9
  S-Vtb1.csv           0.201         10.00         10.00       10.00     10.00      3245.8

  rows_per_s >> hz means several rows share each tick (one row per
  sensor event), not a faster sampling rate.

==========================================================================================
2. DISTINCT COLUMN SETS
==========================================================================================

-- S- files: 3 distinct column set(s) --

  hash 0fa5bd883792  (25 cols, 1 files)
    groups: A
    (largest S- schema — reference)

  hash 6ec6776e0230  (24 cols, 87 files)
    groups: A, I, M, S, T, Vfa, Vta, Vtb, Vw, Y
    missing (1): ['UNNAMED: 24']
    extra   (0): -

  hash 0de5325f3c02  (18 cols, 9 files)
    groups: T
    missing (8): ['GPS SATELLITES IN RANGE', 'MAGNETIC FIELD X (Î¼T)', 'MAGNETIC FIELD Y (Î¼T)', 'MAGNETIC FIELD Z (Î¼T)', 'ORIENTATION (PITCH) (Â°)', 'ORIENTATION (ROLL ) (Â°)', 'ORIENTATION (YAW) (Â°)', 'UNNAMED: 24']
    extra   (1): ['SATELLITES IN RANGE']

  columns common to ALL S- files: 17

-- V- files: 1 distinct column set(s) --

  hash f7a4d7742d3e  (29 cols, 90 files)
    groups: M, S, St, Vfa, Vfb, Vta, Vtb, Vw, Y
    (largest V- schema — reference)

==========================================================================================
3. UNIT VERDICTS BY GROUP
==========================================================================================

  family-level majority:
    S-: speed_ratio 1.003   accel_mag_median 10.040
    V-: speed_ratio 3.624   accel_mag_median 0.057

  group  fam     n  speed_ratio    unit  accel_med      unit   flag
  --------------------------------------------------------------------------------------
  A      S      13        1.001     m/s      9.787     m/s^2   
  I      S       1          n/a       ?      9.862     m/s^2   no speed cross-check
  M      S       1        1.007     m/s      9.939     m/s^2   
  M      V       1        3.626    km/h      0.054         g   
  S      S       6        1.016     m/s      9.910     m/s^2   
  S      V       6        3.626    km/h      0.061         g   
  St     V       4        3.613    km/h      0.026         g   
  T      S      11        1.001     m/s      9.845     m/s^2   
  Vfa    S       2        1.009     m/s     10.079     m/s^2   
  Vfa    V       2        3.620    km/h      0.049         g   
  Vfb    V      11        3.610    km/h      0.044         g   
  Vta    S      30        1.006     m/s     10.146     m/s^2   
  Vta    V      31        3.628    km/h      0.069         g   
  Vtb    S      12        1.000     m/s     10.071     m/s^2   
  Vtb    V      13        3.626    km/h      0.059         g   
  Vw     S      20        1.006     m/s     10.148     m/s^2   
  Vw     V      20        3.604    km/h      0.050         g   
  Y      S       1        1.014     m/s      9.891     m/s^2   
  Y      V       2        3.623    km/h      0.054         g   

```
