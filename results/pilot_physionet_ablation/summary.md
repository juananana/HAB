# Pilot HAB Ablation Results

Scope: PhysioNet EEGBCI pilot, subjects 1-5, leave-subject-5-out, left/right motor imagery runs 4/8/12.
These numbers are preliminary and should be reported only as pilot evidence.

## Decoding and calibration
| method | acc | macro_f1 | kappa | nll | ece | brier | params_m | latency_ms | best_val_acc | epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full HAB | 0.533333 | 0.347826 | 0 | 0.691082 | 0.00872085 | 0.497936 | 0.043139 | 0 | 0.527778 | 11 |
| w/o World Model | 0.533333 | 0.347826 | 0 | 0.691083 | 0.00877175 | 0.497937 | 0.043139 | 0 | 0.527778 | 11 |
| w/o Small-World Loss | 0.533333 | 0.347826 | 0 | 0.691081 | 0.00872085 | 0.497935 | 0.043139 | 0 | 0.527778 | 11 |
| w/o Neural-Cognitive Align | 0.533333 | 0.347826 | 0 | 0.691081 | 0.00871703 | 0.497935 | 0.043139 | 0 | 0.527778 | 11 |
| Dense Neural Graph | 0.533333 | 0.347826 | 0 | 0.691074 | 0.00872037 | 0.497928 | 0.043139 | 0 | 0.527778 | 11 |

## Selective decision simulation
| target_coverage | actual_coverage | selective_acc | selective_risk | expected_cost | confirmation_rate | threshold | method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.533333 | 0.466667 | 2.33333 | 0 | inf | Full HAB |
| 0.9 | 0.888889 | 0.5 | 0.5 | 2.33333 | 0.111111 | 0.475425 | Full HAB |
| 0.8 | 0.8 | 0.527778 | 0.472222 | 2.08889 | 0.2 | 0.475413 | Full HAB |
| 1 | 1 | 0.533333 | 0.466667 | 2.33333 | 0 | inf | w/o World Model |
| 0.9 | 0.888889 | 0.5 | 0.5 | 2.33333 | 0.111111 | 0.475476 | w/o World Model |
| 0.8 | 0.8 | 0.527778 | 0.472222 | 2.08889 | 0.2 | 0.475464 | w/o World Model |
| 1 | 1 | 0.533333 | 0.466667 | 2.33333 | 0 | inf | w/o Small-World Loss |
| 0.9 | 0.888889 | 0.5 | 0.5 | 2.33333 | 0.111111 | 0.475425 | w/o Small-World Loss |
| 0.8 | 0.8 | 0.527778 | 0.472222 | 2.08889 | 0.2 | 0.475413 | w/o Small-World Loss |
| 1 | 1 | 0.533333 | 0.466667 | 2.33333 | 0 | inf | w/o Neural-Cognitive Align |
| 0.9 | 0.888889 | 0.5 | 0.5 | 2.33333 | 0.111111 | 0.475422 | w/o Neural-Cognitive Align |
| 0.8 | 0.8 | 0.527778 | 0.472222 | 2.08889 | 0.2 | 0.475409 | w/o Neural-Cognitive Align |
| 1 | 1 | 0.533333 | 0.466667 | 2.33333 | 0 | inf | Dense Neural Graph |
| 0.9 | 0.888889 | 0.525 | 0.475 | 2.22222 | 0.111111 | 0.475424 | Dense Neural Graph |
| 0.8 | 0.8 | 0.5 | 0.5 | 2.2 | 0.2 | 0.475414 | Dense Neural Graph |

## World-model proxy
| hab_next_state_mse | last_state_copy_mse | method |
| --- | --- | --- |
| 0.00332197 | 1.49588e-11 | Full HAB |
| 0.0115941 | 1.49297e-11 | w/o World Model |
| 0.00332197 | 1.5e-11 | w/o Small-World Loss |
| 0.00332176 | 1.49713e-11 | w/o Neural-Cognitive Align |
| 0.00332191 | 1.20818e-11 | Dense Neural Graph |

## Cognitive-graph proxy metrics
| clustering_proxy | adjacency_l1_mean | edge_entropy | mean_max_edge | method |
| --- | --- | --- | --- | --- |
| 0.0277778 | 0.166667 | 1.79176 | 0.166759 | Full HAB |
| 0.0277778 | 0.166667 | 1.79176 | 0.166758 | w/o World Model |
| 0.0277778 | 0.166667 | 1.79176 | 0.166759 | w/o Small-World Loss |
| 0.0277778 | 0.166667 | 1.79176 | 0.166759 | w/o Neural-Cognitive Align |
| 0.0277778 | 0.166667 | 1.79176 | 0.166759 | Dense Neural Graph |