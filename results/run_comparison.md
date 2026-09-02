# Run comparison

Primary success criterion, stated in advance: validate's `>25` bucket CAE moving substantially off **-953.6** (fraction negative 1.00).


## Summary

```
              run  params  window  val_bucket_mae  test_bucket_mae  val_drift60  test_drift60  test_drift60_oracle  val_gt25_cae  val_gt25_frac_neg
resnet_unweighted 3848196     100           8.452            8.388      875.138       648.568              294.850      -953.655              1.000
    resnet_w_base 3848196     100           7.767            8.976      800.484       660.922              303.856      -809.459              0.997
  resnet_w_narrow  965636     100           8.725            7.272      882.673       612.461              249.458      -959.915              1.000
resnet_w_short_3s 3848196      30           8.489            8.460      823.406       613.908              249.650      -852.801              0.997
   tcn_physics_3s   78982      30           8.713            7.171      852.236       579.435              214.127      -915.142              1.000
      tcn_no_accz   78982      30          12.304           12.636     1144.023      1084.131             1041.015     -1340.349              1.000
      tcn_despike   78982      30           8.998            7.763      844.870       582.406              219.264      -896.867              1.000
          tcn_nhc   78982      30           7.668            7.094      826.668       545.830              214.368      -897.948              1.000
```

## resnet_unweighted

```
60 s CAE (signed) by bucket, model:
bucket      0-5  15-25   5-15    >25
role                                
test      752.9  231.3  768.7   38.9
validate  436.6 -378.2  129.1 -953.7

fraction negative:
bucket    0-5  15-25  5-15   >25
role                            
test      0.0   0.18  0.00  0.37
validate  0.0   0.93  0.28  1.00
```

## resnet_w_base

```
60 s CAE (signed) by bucket, model:
bucket      0-5  15-25   5-15    >25
role                                
test      849.2  268.8  846.4   48.6
validate  433.3 -270.0  186.0 -809.5

fraction negative:
bucket    0-5  15-25  5-15   >25
role                            
test      0.0   0.13  0.00  0.32
validate  0.0   0.85  0.26  1.00
```

## resnet_w_narrow

```
60 s CAE (signed) by bucket, model:
bucket      0-5  15-25   5-15    >25
role                                
test      594.2  202.2  718.1  -38.6
validate  491.4 -304.6  205.7 -959.9

fraction negative:
bucket    0-5  15-25  5-15  >25
role                           
test      0.0   0.15  0.00  0.6
validate  0.0   0.84  0.26  1.0
```

## resnet_w_short_3s

```
60 s CAE (signed) by bucket, model:
bucket      0-5  15-25   5-15    >25
role                                
test      723.3  202.8  683.3   25.5
validate  386.7 -338.4  129.5 -852.8

fraction negative:
bucket    0-5  15-25  5-15   >25
role                            
test      0.0   0.17  0.00  0.37
validate  0.0   0.90  0.28  1.00
```

## tcn_physics_3s

```
60 s CAE (signed) by bucket, model:
bucket      0-5  15-25   5-15    >25
role                                
test      649.3  142.0  619.9  -76.0
validate  439.8 -329.2  171.1 -915.1

fraction negative:
bucket    0-5  15-25  5-15  >25
role                           
test      0.0   0.21  0.00  0.8
validate  0.0   0.88  0.26  1.0
```

## tcn_no_accz

```
60 s CAE (signed) by bucket, model:
bucket      0-5   15-25   5-15     >25
role                                  
test      -97.4 -1038.0 -406.1 -1337.1
validate  383.3  -747.2 -119.3 -1340.3

fraction negative:
bucket     0-5  15-25  5-15  >25
role                            
test      0.86   1.00  0.97  1.0
validate  0.00   0.95  0.61  1.0
```

## tcn_despike

```
60 s CAE (signed) by bucket, model:
bucket      0-5  15-25   5-15    >25
role                                
test      659.6  166.8  632.6  -70.2
validate  444.3 -323.2  168.4 -896.9

fraction negative:
bucket    0-5  15-25  5-15   >25
role                            
test      0.0   0.18  0.00  0.71
validate  0.0   0.87  0.28  1.00
```

## tcn_nhc

```
60 s CAE (signed) by bucket, model:
bucket      0-5  15-25   5-15    >25
role                                
test      674.2   40.3  486.6 -158.3
validate  207.1 -401.8  -59.2 -897.9

fraction negative:
bucket    0-5  15-25  5-15   >25
role                            
test      0.0   0.41  0.03  0.85
validate  0.0   0.94  0.63  1.00
```