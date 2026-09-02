# Per-scenario MAE (NHC falsification test)

Scenarios defined from window statistics; the corpus has no manoeuvre labels.


- `cornering`: mean |gyro_z| >= 0.05
- `straight`: mean |gyro_z| <= 0.02, label < 25.0 m
- `motorway`: mean |gyro_z| <= 0.02, label >= 25.0 m


## MAE (m)

```
run                 tcn_nhc  tcn_physics_3s
role     scenario                          
test     cornering    5.267          11.939
         motorway     3.881           3.411
         straight     6.430           6.189
validate cornering    7.556          10.327
         motorway    14.790          15.486
         straight     5.986           6.397
```

## Signed bias (predicted - true, m)

```
run                 tcn_nhc  tcn_physics_3s
role     scenario                          
test     cornering   -2.005          11.657
         motorway    -2.440          -1.971
         straight     4.774           4.622
validate cornering   -5.031           5.389
         motorway   -14.741         -15.455
         straight    -3.481          -4.128
```

## Window counts

```
run                 tcn_nhc  tcn_physics_3s
role     scenario                          
test     cornering   1217.0          1217.0
         motorway    7560.0          7560.0
         straight   15393.0         15393.0
validate cornering   2711.0          2711.0
         motorway   18336.0         18336.0
         straight   10934.0         10934.0
```