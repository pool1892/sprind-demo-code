# Measured forecasts and first-call opportunities

Generated from [metrics.json](metrics.json); the [selection receipt](selection.json) records fitting, calibration and provenance. NA means the denominator or both outcome classes were absent. Lead is seconds from the first possible supervisor call to the unassisted failure, among caught failures. It is not rescue time.

## development_held_out

420 episodes, 181 failures, 83528 active command rows; robots w42/r04, w42/r08, w42/r12.

| Horizon (s) | Method | Brier | AUROC | Caught / failures | False calls / episodes | False calls / successes | Median lead (s) |
|---|---|---|---|---|---|---|---|
| 1 | telemetry | 0.006 | 0.999 | 181/181 | 9/420 | 9/239 | 1.55 |
| 1 | time_only | 0.001 | 0.999 | 180/181 | 0/420 | 0/239 | 1.20 |
| 1 | timeout | 0.036 | 0.899 | 181/181 | 15/420 | 15/239 | 5.30 |
| 1 | stall | 0.042 | 0.500 | 27/181 | 18/420 | 18/239 | 5.75 |
| 2 | telemetry | 0.016 | 1.000 | 181/181 | 9/420 | 9/239 | 3.45 |
| 2 | time_only | 0.002 | 1.000 | 180/181 | 2/420 | 2/239 | 2.40 |
| 2 | timeout | 0.055 | 0.918 | 181/181 | 15/420 | 15/239 | 5.30 |
| 2 | stall | 0.082 | 0.501 | 27/181 | 18/420 | 18/239 | 5.75 |
| 5 | telemetry | 0.041 | 0.997 | 181/181 | 17/420 | 17/239 | 6.15 |
| 5 | time_only | 0.007 | 0.999 | 181/181 | 15/420 | 15/239 | 5.30 |
| 5 | timeout | 0.022 | 0.986 | 181/181 | 15/420 | 15/239 | 5.30 |
| 5 | stall | 0.183 | 0.503 | 27/181 | 18/420 | 18/239 | 5.75 |

### Five-second discrimination within elapsed-time strata

Aggregate AUROC above includes the predictable terminal timeout. These strata compare telemetry with elapsed time on the same command rows.

| Steps | Positive / rows | Telemetry AUROC | Time-only AUROC | Telemetry Brier | Time-only Brier |
|---|---|---|---|---|---|
| 0-40 | 0/16800 | NA | NA | 0.008 | 0.000 |
| 40-80 | 0/16800 | NA | NA | 0.009 | 0.000 |
| 80-120 | 0/15985 | NA | NA | 0.021 | 0.000 |
| 120-160 | 36/11620 | 0.659 | 0.596 | 0.084 | 0.003 |
| 160-200 | 3640/7761 | 0.937 | 0.975 | 0.171 | 0.058 |
| 200-240 | 7224/7355 | 0.755 | 0.674 | 0.045 | 0.018 |
| 240-280 | 7200/7207 | 0.870 | 0.912 | 0.019 | 0.001 |

### Telemetry reliability

Each cell is rows; mean predicted probability; observed failure frequency. The row populations differ by horizon.

| Probability bin | 1 s | 2 s | 5 s |
|---|---|---|---|
| 0.0-0.1 | 76721; 0.014; 0.000 | 57704; 0.044; 0.000 | 29586; 0.070; 0.000 |
| 0.1-0.2 | 1537; 0.134; 0.003 | 11643; 0.139; 0.001 | 22333; 0.139; 0.000 |
| 0.2-0.3 | 473; 0.245; 0.013 | 3569; 0.244; 0.006 | 5968; 0.243; 0.007 |
| 0.3-0.4 | 344; 0.349; 0.038 | 2095; 0.348; 0.003 | 2625; 0.347; 0.014 |
| 0.4-0.5 | 284; 0.450; 0.088 | 1070; 0.442; 0.050 | 2442; 0.448; 0.062 |
| 0.5-0.6 | 261; 0.549; 0.192 | 430; 0.542; 0.377 | 1926; 0.546; 0.265 |
| 0.6-0.7 | 288; 0.650; 0.278 | 403; 0.661; 0.935 | 2047; 0.652; 0.583 |
| 0.7-0.8 | 310; 0.751; 0.500 | 924; 0.757; 0.997 | 3165; 0.753; 0.891 |
| 0.8-0.9 | 427; 0.853; 0.911 | 2081; 0.852; 1.000 | 5588; 0.855; 0.985 |
| 0.9-1.0 | 2883; 0.974; 0.999 | 3609; 0.964; 1.000 | 7848; 0.951; 0.999 |

## miniature_transfer

416 episodes, 157 failures, 80677 active command rows; robots w10/r01, w10/r02, w10/r03, w10/r04, w10/r05, w10/r06, w10/r07, w10/r08.

| Horizon (s) | Method | Brier | AUROC | Caught / failures | False calls / episodes | False calls / successes | Median lead (s) |
|---|---|---|---|---|---|---|---|
| 1 | telemetry | 0.008 | 0.998 | 157/157 | 26/416 | 26/259 | 2.00 |
| 1 | time_only | 0.002 | 0.997 | 154/157 | 8/416 | 8/259 | 1.20 |
| 1 | timeout | 0.032 | 0.896 | 157/157 | 37/416 | 37/259 | 5.30 |
| 1 | stall | 0.038 | 0.503 | 9/157 | 21/416 | 21/259 | 7.75 |
| 2 | telemetry | 0.020 | 0.997 | 157/157 | 32/416 | 32/259 | 4.85 |
| 2 | time_only | 0.006 | 0.995 | 154/157 | 20/416 | 20/259 | 2.40 |
| 2 | timeout | 0.052 | 0.909 | 157/157 | 37/416 | 37/259 | 5.30 |
| 2 | stall | 0.073 | 0.501 | 9/157 | 21/416 | 21/259 | 7.75 |
| 5 | telemetry | 0.062 | 0.976 | 157/157 | 45/416 | 45/259 | 6.45 |
| 5 | time_only | 0.029 | 0.986 | 157/157 | 37/416 | 37/259 | 5.30 |
| 5 | timeout | 0.037 | 0.966 | 157/157 | 37/416 | 37/259 | 5.30 |
| 5 | stall | 0.166 | 0.500 | 9/157 | 21/416 | 21/259 | 7.75 |

### Five-second discrimination within elapsed-time strata

Aggregate AUROC above includes the predictable terminal timeout. These strata compare telemetry with elapsed time on the same command rows.

| Steps | Positive / rows | Telemetry AUROC | Time-only AUROC | Telemetry Brier | Time-only Brier |
|---|---|---|---|---|---|
| 0-40 | 0/16640 | NA | NA | 0.014 | 0.000 |
| 40-80 | 0/16640 | NA | NA | 0.012 | 0.000 |
| 80-120 | 100/15411 | 0.490 | 0.600 | 0.024 | 0.006 |
| 120-160 | 120/10627 | 0.329 | 0.537 | 0.106 | 0.011 |
| 160-200 | 3160/7740 | 0.860 | 0.920 | 0.234 | 0.104 |
| 200-240 | 6160/7136 | 0.249 | 0.527 | 0.134 | 0.135 |
| 240-280 | 6160/6483 | 0.579 | 0.769 | 0.049 | 0.050 |

### Telemetry reliability

Each cell is rows; mean predicted probability; observed failure frequency. The row populations differ by horizon.

| Probability bin | 1 s | 2 s | 5 s |
|---|---|---|---|
| 0.0-0.1 | 73925; 0.015; 0.001 | 52866; 0.046; 0.000 | 25073; 0.070; 0.002 |
| 0.1-0.2 | 1746; 0.134; 0.006 | 13385; 0.138; 0.004 | 23934; 0.141; 0.005 |
| 0.2-0.3 | 456; 0.247; 0.013 | 4281; 0.244; 0.011 | 6113; 0.239; 0.007 |
| 0.3-0.4 | 318; 0.349; 0.013 | 2359; 0.344; 0.004 | 2666; 0.353; 0.009 |
| 0.4-0.5 | 284; 0.448; 0.014 | 995; 0.444; 0.002 | 2562; 0.445; 0.010 |
| 0.5-0.6 | 259; 0.549; 0.089 | 300; 0.536; 0.023 | 1405; 0.546; 0.046 |
| 0.6-0.7 | 345; 0.655; 0.099 | 167; 0.658; 0.725 | 1388; 0.653; 0.210 |
| 0.7-0.8 | 329; 0.749; 0.286 | 434; 0.759; 0.806 | 2957; 0.758; 0.745 |
| 0.8-0.9 | 401; 0.854; 0.810 | 2578; 0.860; 0.925 | 6084; 0.855; 0.942 |
| 0.9-1.0 | 2614; 0.977; 0.995 | 3312; 0.966; 0.997 | 8495; 0.951; 0.842 |

## w42

420 episodes, 181 failures, 83528 active command rows; robots w42/r04, w42/r08, w42/r12.

| Horizon (s) | Method | Brier | AUROC | Caught / failures | False calls / episodes | False calls / successes | Median lead (s) |
|---|---|---|---|---|---|---|---|
| 1 | telemetry | 0.006 | 0.999 | 181/181 | 9/420 | 9/239 | 1.55 |
| 1 | time_only | 0.001 | 0.999 | 180/181 | 0/420 | 0/239 | 1.20 |
| 1 | timeout | 0.036 | 0.899 | 181/181 | 15/420 | 15/239 | 5.30 |
| 1 | stall | 0.042 | 0.500 | 27/181 | 18/420 | 18/239 | 5.75 |
| 2 | telemetry | 0.016 | 1.000 | 181/181 | 9/420 | 9/239 | 3.45 |
| 2 | time_only | 0.002 | 1.000 | 180/181 | 2/420 | 2/239 | 2.40 |
| 2 | timeout | 0.055 | 0.918 | 181/181 | 15/420 | 15/239 | 5.30 |
| 2 | stall | 0.082 | 0.501 | 27/181 | 18/420 | 18/239 | 5.75 |
| 5 | telemetry | 0.041 | 0.997 | 181/181 | 17/420 | 17/239 | 6.15 |
| 5 | time_only | 0.007 | 0.999 | 181/181 | 15/420 | 15/239 | 5.30 |
| 5 | timeout | 0.022 | 0.986 | 181/181 | 15/420 | 15/239 | 5.30 |
| 5 | stall | 0.183 | 0.503 | 27/181 | 18/420 | 18/239 | 5.75 |

### Five-second discrimination within elapsed-time strata

Aggregate AUROC above includes the predictable terminal timeout. These strata compare telemetry with elapsed time on the same command rows.

| Steps | Positive / rows | Telemetry AUROC | Time-only AUROC | Telemetry Brier | Time-only Brier |
|---|---|---|---|---|---|
| 0-40 | 0/16800 | NA | NA | 0.008 | 0.000 |
| 40-80 | 0/16800 | NA | NA | 0.009 | 0.000 |
| 80-120 | 0/15985 | NA | NA | 0.021 | 0.000 |
| 120-160 | 36/11620 | 0.659 | 0.596 | 0.084 | 0.003 |
| 160-200 | 3640/7761 | 0.937 | 0.975 | 0.171 | 0.058 |
| 200-240 | 7224/7355 | 0.755 | 0.674 | 0.045 | 0.018 |
| 240-280 | 7200/7207 | 0.870 | 0.912 | 0.019 | 0.001 |

### Telemetry reliability

Each cell is rows; mean predicted probability; observed failure frequency. The row populations differ by horizon.

| Probability bin | 1 s | 2 s | 5 s |
|---|---|---|---|
| 0.0-0.1 | 76721; 0.014; 0.000 | 57704; 0.044; 0.000 | 29586; 0.070; 0.000 |
| 0.1-0.2 | 1537; 0.134; 0.003 | 11643; 0.139; 0.001 | 22333; 0.139; 0.000 |
| 0.2-0.3 | 473; 0.245; 0.013 | 3569; 0.244; 0.006 | 5968; 0.243; 0.007 |
| 0.3-0.4 | 344; 0.349; 0.038 | 2095; 0.348; 0.003 | 2625; 0.347; 0.014 |
| 0.4-0.5 | 284; 0.450; 0.088 | 1070; 0.442; 0.050 | 2442; 0.448; 0.062 |
| 0.5-0.6 | 261; 0.549; 0.192 | 430; 0.542; 0.377 | 1926; 0.546; 0.265 |
| 0.6-0.7 | 288; 0.650; 0.278 | 403; 0.661; 0.935 | 2047; 0.652; 0.583 |
| 0.7-0.8 | 310; 0.751; 0.500 | 924; 0.757; 0.997 | 3165; 0.753; 0.891 |
| 0.8-0.9 | 427; 0.853; 0.911 | 2081; 0.852; 1.000 | 5588; 0.855; 0.985 |
| 0.9-1.0 | 2883; 0.974; 0.999 | 3609; 0.964; 1.000 | 7848; 0.951; 0.999 |

## Selected settings

Calibration robots only; at most 5% of successful episodes called. Within that constraint, maximize total failure lead divided by all failures. This selection does not price operator time or prove claim 4.

| Horizon (s) | Telemetry threshold | Time-only threshold | Timeout telemetry step | Stall consecutive steps |
|---|---|---|---|---|
| 1 | 0.14 | 0.01 | 173 | 10 |
| 2 | 0.30 | 0.01 | 173 | 10 |
| 5 | 0.47 | 0.14 | 173 | 10 |
