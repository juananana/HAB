# Pilot Experiment Results

Scope: PhysioNet EEGBCI pilot, subjects 1-5, leave-subject-5-out, left/right motor imagery runs 4/8/12.

## Main metrics
| method        |      acc |   macro_f1 |      kappa |      nll |       ece |    brier |   params_m |   latency_ms |
|:--------------|---------:|-----------:|-----------:|---------:|----------:|---------:|-----------:|-------------:|
| Bandpower-LDA | 0.533333 |   0.347826 |  0         | 7.43978  | 0.466667  | 0.933333 |   0        |     0        |
| Raw-LogReg    | 0.444444 |   0.443345 | -0.0997067 | 1.27049  | 0.367456  | 0.760814 |   0        |     0        |
| EEGNet-Lite   | 0.466667 |   0.318182 |  0         | 0.695652 | 0.048611  | 0.502504 |   0.002122 |     1.18268  |
| HAB-Prototype | 0.533333 |   0.347826 |  0         | 0.691422 | 0.0158428 | 0.498276 |   0.043139 |     0.768829 |

## Risk metrics
|   target_coverage |   actual_coverage |   selective_acc |   selective_risk |   expected_cost |   confirmation_rate |   threshold | method        |
|------------------:|------------------:|----------------:|-----------------:|----------------:|--------------------:|------------:|:--------------|
|               1   |          1        |        0.533333 |         0.466667 |         2.33333 |            0        |  inf        | HAB-Prototype |
|               0.9 |          0.888889 |        0.55     |         0.45     |         2.11111 |            0.111111 |    0.482528 | HAB-Prototype |
|               0.8 |          0.8      |        0.583333 |         0.416667 |         1.86667 |            0.2      |    0.482522 | HAB-Prototype |

## World model proxy
| method          |   representation_mse |   next_step_acc | note                              |
|:----------------|---------------------:|----------------:|:----------------------------------|
| Last-state Copy |         nan          |             nan | Not evaluated in pilot            |
| HAB-Prototype   |           0.00524401 |             nan | Cognitive latent next-state proxy |