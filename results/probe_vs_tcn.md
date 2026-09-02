# XGBoost probe vs TCN through the outage harness

Both use raw levelled gyro for heading, so this compares the displacement model only.


## Mean drift (CRSE, m)

| split | predictor | 10 s | 30 s | 60 s |
|---|---|---|---|---|
| test | `constant_velocity_dr` | 18.1 | 98.8 | 285.4 |
| test | `ridge_probe` | 92.4 | 308.1 | 676.5 |
| test | `tcn_nhc` | 51.4 | 210.2 | 545.8 |
| test | `tcn_nhc_oracle` | 41.8 | 112.4 | 214.4 |
| test | `xgb_probe` | 70.1 | 252.4 | 590.2 |
| test | `xgb_probe_oracle` | 62.3 | 180.9 | 351.3 |
| validate | `constant_velocity_dr` | 69.6 | 173.4 | 365.2 |
| validate | `ridge_probe` | 163.7 | 484.5 | 998.9 |
| validate | `tcn_nhc` | 133.3 | 387.0 | 826.7 |
| validate | `tcn_nhc_oracle` | 129.4 | 341.6 | 664.2 |
| validate | `xgb_probe` | 170.9 | 508.2 | 1044.5 |
| validate | `xgb_probe_oracle` | 166.7 | 471.5 | 929.2 |

## Error character at 60 s

| split | predictor | CRSE | mean CAE | \|CAE\|/CRSE | reading |
|---|---|---|---|---|---|
| test | `constant_velocity_dr` | 285.4 | +7.4 | 0.35 | mixed |
| test | `ridge_probe` | 676.5 | -422.2 | 0.74 | systematic |
| test | `tcn_nhc` | 545.8 | +34.3 | 0.40 | mixed |
| test | `tcn_nhc_oracle` | 214.4 | +34.3 | 1.01 | systematic |
| test | `xgb_probe` | 590.2 | -250.4 | 0.61 | mixed |
| test | `xgb_probe_oracle` | 351.3 | -250.4 | 1.02 | systematic |
| validate | `constant_velocity_dr` | 365.2 | +3.7 | 0.48 | mixed |
| validate | `ridge_probe` | 998.9 | -820.3 | 0.87 | systematic |
| validate | `tcn_nhc` | 826.7 | -648.5 | 0.81 | systematic |
| validate | `tcn_nhc_oracle` | 664.2 | -648.5 | 1.01 | systematic |
| validate | `xgb_probe` | 1044.5 | -916.8 | 0.90 | systematic |
| validate | `xgb_probe_oracle` | 929.2 | -916.8 | 1.01 | systematic |

## 60 s drift by speed bucket
```
bucket                           0-5  15-25   5-15     >25
role     predictor                                        
test     constant_velocity_dr  325.6  281.3  297.5   283.6
         ridge_probe           477.9  641.9  349.0   836.4
         tcn_nhc               712.1  516.1  620.2   543.3
         tcn_nhc_oracle        661.7  150.4  483.7   166.4
         xgb_probe             409.4  559.1  410.3   693.3
         xgb_probe_oracle      359.4  266.2  265.6   468.8
validate constant_velocity_dr  442.5  422.4  355.8   332.5
         ridge_probe           619.2  715.6  446.5  1255.1
         tcn_nhc               171.2  653.4  402.1  1004.5
         tcn_nhc_oracle        174.7  407.5  206.5   892.2
         xgb_probe             250.6  790.9  432.2  1300.5
         xgb_probe_oracle      249.2  612.2  218.5  1235.1
```

## 60 s signed CAE by bucket
```
bucket                           0-5  15-25   5-15     >25
role     predictor                                        
test     constant_velocity_dr  272.9  -21.1  121.5   -14.7
         ridge_probe           440.8 -404.3  178.9  -689.0
         tcn_nhc               674.2   40.3  486.6  -158.3
         tcn_nhc_oracle        674.2   40.3  486.6  -158.3
         xgb_probe             365.1 -235.5  273.3  -476.5
         xgb_probe_oracle      365.1 -235.5  273.3  -476.5
validate constant_velocity_dr  419.4    5.2  146.6   -27.0
         ridge_probe           697.2 -479.4  151.9 -1195.8
         tcn_nhc               207.1 -401.8  -59.2  -897.9
         tcn_nhc_oracle        207.1 -401.8  -59.2  -897.9
         xgb_probe             295.1 -619.6  -67.7 -1242.5
         xgb_probe_oracle      295.1 -619.6  -67.7 -1242.5
```

## fraction of outages with negative CAE
```
bucket                         0-5  15-25  5-15   >25
role     predictor                                   
test     constant_velocity_dr  0.0   0.53  0.33  0.58
         ridge_probe           0.0   0.95  0.26  1.00
         tcn_nhc               0.0   0.41  0.03  0.85
         tcn_nhc_oracle        0.0   0.41  0.03  0.85
         xgb_probe             0.0   0.87  0.10  1.00
         xgb_probe_oracle      0.0   0.87  0.10  1.00
validate constant_velocity_dr  0.0   0.43  0.30  0.64
         ridge_probe           0.0   0.99  0.19  1.00
         tcn_nhc               0.0   0.94  0.63  1.00
         tcn_nhc_oracle        0.0   0.94  0.63  1.00
         xgb_probe             0.0   0.99  0.63  1.00
         xgb_probe_oracle      0.0   0.99  0.63  1.00
```