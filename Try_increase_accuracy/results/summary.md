# Accuracy experiments — comparison

Pooled over all 5 subject-wise folds (1,333,415 test segments, every user tested once).
**Selection is by validation macro-F1**; test is reported but never used to choose.

| configuration | val macro-F1 | accuracy | macro-F1 | balanced acc | kappa | Δ macro-F1 |
|---|---|---|---|---|---|---|
| Baseline RF (as submitted) | 0.2737 | 0.4733 | 0.3337 | 0.3566 | 0.2096 | +0.0000 |
| Baseline RF (as submitted) + prior-corrected | 0.2962 | 0.5155 | 0.3042 | 0.2867 | 0.1901 | -0.0294 |
| Baseline RF (as submitted) + tuned thresholds | 0.3866 | 0.5088 | 0.3882 | 0.3454 | 0.2459 | +0.0545 |
| RF min_samples_leaf=20 | 0.2777 | 0.4659 | 0.3301 | 0.3712 | 0.2142 | -0.0036 |
| RF min_samples_leaf=50 | 0.2787 | 0.4658 | 0.3282 | 0.3759 | 0.2245 | -0.0055 |
| RF min_samples_leaf=100 | 0.2828 | 0.4603 | 0.3244 | 0.3764 | 0.2252 | -0.0093 |
| Gradient boosting (HGB) | 0.2788 | 0.4408 | 0.3279 | 0.3522 | 0.1925 | -0.0057 |
| Two-stage still/moving | 0.2744 | 0.4761 | 0.3321 | 0.3527 | 0.2079 | -0.0015 |
| Per-user normalised | 0.2215 | 0.4374 | 0.2621 | 0.3050 | 0.1109 | -0.0716 |
| Raw + per-user normalised | 0.2545 | 0.4608 | 0.3058 | 0.3295 | 0.1456 | -0.0279 |
| + time of day | 0.3604 | 0.6265 | 0.4032 | 0.4114 | 0.4381 | +0.0695 |
| + time of day + prior-corrected | 0.3738 | 0.6516 | 0.3682 | 0.3401 | 0.4270 | +0.0345 |
| + time of day + tuned thresholds **←selected** | 0.4313 | 0.6597 | 0.4382 | 0.3979 | 0.4744 | +0.1045 |
| + time of day, leaf=100 | 0.3487 | 0.6122 | 0.3882 | 0.4307 | 0.4368 | +0.0546 |
| + time of day, leaf=100 + prior-corrected | 0.3773 | 0.6775 | 0.3958 | 0.3628 | 0.4735 | +0.0621 |
| + time of day, leaf=100 + tuned thresholds | 0.4266 | 0.6666 | 0.4422 | 0.3999 | 0.4829 | +0.1085 |
