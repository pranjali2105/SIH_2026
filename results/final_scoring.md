# Final scoring

CRSE is the norm of the accumulated 2-D position error; CAE is the signed sum of per-second displacement errors.


**|CAE| / CRSE near 1** means the per-second errors share a sign — systematic bias compounding linearly. **Near 0** means they scatter and the amplification comes from elsewhere, most likely heading rotating the track away from truth.


## Mean CRSE by duration

| split | predictor | 10 s | 30 s | 60 s |
|---|---|---|---|---|
| validate | `constant_velocity_dr` | 69.6 | 173.4 | 365.2 |
| validate | `ins_dr` | 61.0 | 175.4 | 416.9 |
| validate | `model` | 139.9 | 403.4 | 843.0 |
| validate | `model_final` | 141.1 | 414.6 | 875.1 |
| validate | `model_final_true_heading` | 135.5 | 361.4 | 700.3 |
| validate | `model_true_heading` | 135.5 | 361.4 | 700.1 |
| validate | `model_zupt` | 143.2 | 414.9 | 863.8 |
| validate | `model_zupt_cal` | 143.6 | 415.8 | 865.1 |
| validate | `model_zupt_cal_true_heading` | 139.3 | 375.2 | 728.2 |
| validate | `model_zupt_true_heading` | 138.9 | 374.0 | 725.5 |
| validate | `ridge_probe` | 159.4 | 454.8 | 907.5 |
| validate | `xgb_probe` | 170.5 | 495.6 | 987.9 |
| test | `constant_velocity_dr` | 18.1 | 98.8 | 285.4 |
| test | `ins_dr` | 28.9 | 120.6 | 310.5 |
| test | `model` | 70.9 | 311.3 | 940.2 |
| test | `model_final` | 69.1 | 262.5 | 648.6 |
| test | `model_final_true_heading` | 57.9 | 155.1 | 294.8 |
| test | `model_true_heading` | 57.9 | 155.1 | 294.9 |
| test | `model_zupt` | 70.8 | 310.7 | 938.8 |
| test | `model_zupt_cal` | 68.2 | 305.4 | 930.2 |
| test | `model_zupt_cal_true_heading` | 54.7 | 146.3 | 277.4 |
| test | `model_zupt_true_heading` | 57.8 | 154.6 | 293.5 |
| test | `ridge_probe` | 82.9 | 259.8 | 537.4 |
| test | `xgb_probe` | 57.5 | 188.8 | 414.2 |

## CAE vs CRSE at 60 s

| split | predictor | CRSE | mean CAE | mean \|CAE\| | \|CAE\|/CRSE | reading |
|---|---|---|---|---|---|---|
| test | `constant_velocity_dr` | 285.4 | +7.4 | 100.8 | 0.35 | mixed |
| test | `ins_dr` | 310.5 | -113.2 | 170.3 | 0.55 | mixed |
| test | `model` | 940.2 | +239.3 | 301.5 | 0.32 | mixed |
| test | `model_final` | 648.6 | +239.3 | 301.5 | 0.46 | mixed |
| test | `model_final_true_heading` | 294.8 | +239.3 | 301.5 | 1.02 | systematic bias |
| test | `model_true_heading` | 294.9 | +239.3 | 301.5 | 1.02 | systematic bias |
| test | `model_zupt` | 938.8 | +237.7 | 300.1 | 0.32 | mixed |
| test | `model_zupt_cal` | 930.2 | +215.2 | 283.7 | 0.31 | mixed |
| test | `model_zupt_cal_true_heading` | 277.4 | +215.2 | 283.7 | 1.02 | systematic bias |
| test | `model_zupt_true_heading` | 293.5 | +237.7 | 300.1 | 1.02 | systematic bias |
| test | `ridge_probe` | 537.4 | -365.1 | 463.4 | 0.86 | systematic bias |
| test | `xgb_probe` | 414.2 | -174.9 | 306.1 | 0.74 | systematic bias |
| validate | `constant_velocity_dr` | 365.2 | +3.7 | 173.8 | 0.48 | mixed |
| validate | `ins_dr` | 416.9 | -134.0 | 254.0 | 0.61 | mixed |
| validate | `model` | 843.0 | -653.2 | 702.2 | 0.83 | systematic bias |
| validate | `model_final` | 875.1 | -653.3 | 702.3 | 0.80 | systematic bias |
| validate | `model_final_true_heading` | 700.3 | -653.3 | 702.3 | 1.00 | systematic bias |
| validate | `model_true_heading` | 700.1 | -653.2 | 702.2 | 1.00 | systematic bias |
| validate | `model_zupt` | 863.8 | -681.2 | 727.6 | 0.84 | systematic bias |
| validate | `model_zupt_cal` | 865.1 | -684.5 | 730.2 | 0.84 | systematic bias |
| validate | `model_zupt_cal_true_heading` | 728.2 | -684.5 | 730.2 | 1.00 | systematic bias |
| validate | `model_zupt_true_heading` | 725.5 | -681.2 | 727.6 | 1.00 | systematic bias |
| validate | `ridge_probe` | 907.5 | -792.4 | 851.8 | 0.94 | systematic bias |
| validate | `xgb_probe` | 987.9 | -932.5 | 954.6 | 0.97 | systematic bias |

## 10 s drift (CRSE) by speed bucket

```
bucket                                  0-5  15-25   5-15    >25
role     predictor                                              
test     constant_velocity_dr          34.0   16.9   25.0   16.6
         ins_dr                        25.2   26.1   25.3   33.4
         model                        161.8   64.1  145.2   50.0
         model_final                  161.5   62.7  149.5   45.5
         model_final_true_heading     157.8   53.4  136.0   32.8
         model_true_heading           157.8   53.4  136.0   32.8
         model_zupt                   155.6   64.1  144.8   50.0
         model_zupt_cal               154.9   61.3  142.6   47.6
         model_zupt_cal_true_heading  150.8   50.3  133.4   29.5
         model_zupt_true_heading      151.6   53.4  135.6   32.8
         ridge_probe                   93.3   67.8   60.5  107.5
         xgb_probe                     99.0   42.8   70.5   68.3
validate constant_velocity_dr         232.8   31.5   35.2   83.2
         ins_dr                       212.1   21.9   32.3   75.3
         model                        143.5   74.9   68.8  191.4
         model_final                  142.4   77.8   78.3  190.3
         model_final_true_heading     138.7   71.2   59.7  187.6
         model_true_heading           138.8   71.1   59.7  187.6
         model_zupt                   133.9   78.6   71.0  196.0
         model_zupt_cal               134.9   79.0   69.2  196.7
         model_zupt_cal_true_heading  130.2   75.4   60.3  193.0
         model_zupt_true_heading      129.3   74.9   62.0  192.3
         ridge_probe                  126.6   80.2   75.6  225.4
         xgb_probe                     81.8  104.8   55.4  239.4
```

## 30 s drift (CRSE) by speed bucket

```
bucket                                  0-5  15-25   5-15    >25
role     predictor                                              
test     constant_velocity_dr         129.0   91.4  128.7   95.1
         ins_dr                       108.5  112.3  120.7  129.8
         model                        485.9  289.0  451.3  279.9
         model_final                  469.1  244.8  455.5  207.0
         model_final_true_heading     453.9  152.4  370.5   72.3
         model_true_heading           453.9  152.4  370.5   72.3
         model_zupt                   473.0  289.0  448.7  279.9
         model_zupt_cal               468.9  282.6  443.2  275.9
         model_zupt_cal_true_heading  437.6  143.1  362.2   64.3
         model_zupt_true_heading      442.1  152.5  367.9   72.3
         ridge_probe                  281.7  215.7  200.3  325.2
         xgb_probe                    264.6  148.5  227.5  214.7
validate constant_velocity_dr         188.2  164.7  176.9  177.3
         ins_dr                       189.6  152.1  168.9  189.0
         model                        281.1  247.6  216.6  524.7
         model_final                  251.7  280.8  252.4  520.1
         model_final_true_heading     233.5  199.1  137.0  493.2
         model_true_heading           233.8  199.1  136.9  493.1
         model_zupt                   261.8  258.2  220.1  538.7
         model_zupt_cal               267.1  258.6  216.9  540.4
         model_zupt_cal_true_heading  220.9  212.1  138.8  510.1
         model_zupt_true_heading      217.3  211.0  142.1  508.0
         ridge_probe                  371.1  262.8  218.6  604.0
         xgb_probe                    201.1  324.6  179.1  652.1
```

## 60 s drift (CRSE) by speed bucket

```
bucket                                  0-5  15-25   5-15     >25
role     predictor                                               
test     constant_velocity_dr         325.6  281.3  297.5   283.6
         ins_dr                       276.5  305.9  265.3   332.7
         model                        804.5  885.2  973.0   990.7
         model_final                  775.0  641.9  856.7   576.5
         model_final_true_heading     724.4  296.3  735.8   119.5
         model_true_heading           724.6  296.3  735.8   119.5
         model_zupt                   751.9  885.3  969.7   990.7
         model_zupt_cal               757.9  875.6  958.4   983.4
         model_zupt_cal_true_heading  677.0  277.6  718.9   104.3
         model_zupt_true_heading      671.1  296.4  732.4   119.5
         ridge_probe                  508.1  475.7  342.7   671.1
         xgb_probe                    396.7  365.0  411.0   466.7
validate constant_velocity_dr         442.5  422.4  355.8   332.5
         ins_dr                       328.6  431.0  377.2   416.7
         model                        462.4  552.5  432.3  1081.0
         model_final                  328.8  676.2  499.0  1058.1
         model_final_true_heading     332.4  397.8  268.0   948.3
         model_true_heading           334.5  397.7  267.7   948.2
         model_zupt                   423.5  567.8  440.8  1107.9
         model_zupt_cal               438.4  569.0  432.7  1110.6
         model_zupt_cal_true_heading  321.9  421.7  273.8   982.7
         model_zupt_true_heading      308.0  418.4  283.3   978.6
         ridge_probe                  837.6  554.7  428.0  1186.6
         xgb_probe                    334.5  679.4  344.3  1277.7
```

## 60 s CAE (SIGNED) by speed bucket

A consistent sign within a bucket is a calibration problem in that regime specifically — narrower and more fixable than a general failure. Scatter about zero is not.

```
bucket                                  0-5  15-25   5-15     >25
role     predictor                                               
test     constant_velocity_dr         272.9  -21.1  121.5   -14.7
         ins_dr                       226.4 -135.3   50.3  -162.3
         model                        753.1  231.3  768.7    38.9
         model_final                  752.9  231.3  768.7    38.9
         model_final_true_heading     752.9  231.3  768.7    38.9
         model_true_heading           753.1  231.3  768.7    38.9
         model_zupt                   698.9  231.1  764.6    38.9
         model_zupt_cal               704.8  209.0  750.9    11.6
         model_zupt_cal_true_heading  704.8  209.0  750.9    11.6
         model_zupt_true_heading      698.9  231.1  764.6    38.9
         ridge_probe                  457.2 -344.3  228.9  -630.7
         xgb_probe                    343.8 -161.2  314.2  -383.5
validate constant_velocity_dr         419.4    5.2  146.6   -27.0
         ins_dr                       286.9  -91.4  143.9  -209.6
         model                        438.7 -378.0  129.5  -953.6
         model_final                  436.6 -378.2  129.1  -953.7
         model_final_true_heading     436.6 -378.2  129.1  -953.7
         model_true_heading           438.7 -378.0  129.5  -953.6
         model_zupt                   407.1 -400.0   95.6  -984.1
         model_zupt_cal               422.2 -404.7  100.4  -988.3
         model_zupt_cal_true_heading  422.2 -404.7  100.4  -988.3
         model_zupt_true_heading      407.1 -400.0   95.6  -984.1
         ridge_probe                  815.5 -444.6  210.9 -1178.3
         xgb_probe                    311.7 -620.2  -76.4 -1268.5
```

fraction of outages with NEGATIVE CAE (under-prediction):

```
bucket                                 0-5  15-25  5-15   >25
role     predictor                                           
test     constant_velocity_dr         0.00   0.53  0.33  0.58
         ins_dr                       0.00   0.86  0.34  0.99
         model                        0.00   0.18  0.00  0.37
         model_final                  0.00   0.18  0.00  0.37
         model_final_true_heading     0.00   0.18  0.00  0.37
         model_true_heading           0.00   0.18  0.00  0.37
         model_zupt                   0.00   0.18  0.00  0.37
         model_zupt_cal               0.00   0.18  0.00  0.44
         model_zupt_cal_true_heading  0.00   0.18  0.00  0.44
         model_zupt_true_heading      0.00   0.18  0.00  0.37
         ridge_probe                  0.00   0.89  0.18  1.00
         xgb_probe                    0.00   0.73  0.08  1.00
validate constant_velocity_dr         0.00   0.43  0.30  0.64
         ins_dr                       0.17   0.62  0.35  0.78
         model                        0.00   0.93  0.28  1.00
         model_final                  0.00   0.93  0.28  1.00
         model_final_true_heading     0.00   0.93  0.28  1.00
         model_true_heading           0.00   0.93  0.28  1.00
         model_zupt                   0.00   0.93  0.30  1.00
         model_zupt_cal               0.00   0.94  0.30  1.00
         model_zupt_cal_true_heading  0.00   0.94  0.30  1.00
         model_zupt_true_heading      0.00   0.93  0.30  1.00
         ridge_probe                  0.00   0.98  0.17  1.00
         xgb_probe                    0.00   0.98  0.63  1.00
```