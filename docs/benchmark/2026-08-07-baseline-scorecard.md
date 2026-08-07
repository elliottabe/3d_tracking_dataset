# Benchmark scorecard

## free_running (n=5)

| metric | median |
|---|---|
| hard_iou_median | 0.02366 |
| jitter_median | 0.003058 |
| jl_violation_rate | 0.0002032 |
| loo_px_median | 3.813 |
| reproj_px_leg_median | 4.082 |
| reproj_px_median | 4.132 |
| reproj_px_trunk_median | 3.531 |
| reproj_px_wing_median | 4.254 |
| soft_iou_median | 0.1464 |

## courtship_male (n=8)

| metric | median |
|---|---|
| hard_iou_apart | 0.02306 |
| hard_iou_close | 0.0207 |
| hard_iou_median | 0.0228 |
| jitter_apart | 0.003812 |
| jitter_close | 0.004897 |
| jitter_median | 0.002958 |
| jl_violation_rate | 9.358e-05 |
| loo_px_median | 5.024 |
| reproj_px_apart | 6.537 |
| reproj_px_close | 8.035 |
| reproj_px_leg_apart | 5.895 |
| reproj_px_leg_close | 8.079 |
| reproj_px_leg_median | 6.676 |
| reproj_px_median | 7.204 |
| reproj_px_trunk_apart | 9.167 |
| reproj_px_trunk_close | 9.772 |
| reproj_px_trunk_median | 9.424 |
| reproj_px_wing_apart | 11.67 |
| reproj_px_wing_close | 8.378 |
| reproj_px_wing_median | 11.6 |
| soft_iou_apart | 0.1408 |
| soft_iou_close | 0.1272 |
| soft_iou_median | 0.1382 |

## courtship_female (n=8)

| metric | median |
|---|---|
| hard_iou_apart | 0.02232 |
| hard_iou_close | 0 |
| hard_iou_median | 0.01344 |
| jitter_apart | 0.007706 |
| jitter_close | 0.01213 |
| jitter_median | 0.006175 |
| jl_violation_rate | 0.0001413 |
| loo_px_median | 54.39 |
| reproj_px_apart | 27.1 |
| reproj_px_close | 86.9 |
| reproj_px_leg_apart | 30.65 |
| reproj_px_leg_close | 96.34 |
| reproj_px_leg_median | 44.07 |
| reproj_px_median | 42.3 |
| reproj_px_trunk_apart | 20.85 |
| reproj_px_trunk_close | 93.32 |
| reproj_px_trunk_median | 37.08 |
| reproj_px_wing_apart | 27.59 |
| reproj_px_wing_close | 95.47 |
| reproj_px_wing_median | 59.94 |
| soft_iou_apart | 0.1369 |
| soft_iou_close | 0 |
| soft_iou_median | 0.07174 |

## Gap ratios (courtship / free_running, 1.0 = parity)

| metric | courtship_male | courtship_female |
|---|---|---|
| hard_iou_median | 0.964 | 0.568 |
| jitter_median | 0.967 | 2.02 |
| jl_violation_rate | 0.461 | 0.695 |
| loo_px_median | 1.32 | 14.3 |
| reproj_px_leg_median | 1.64 | 10.8 |
| reproj_px_median | 1.74 | 10.2 |
| reproj_px_trunk_median | 2.67 | 10.5 |
| reproj_px_wing_median | 2.73 | 14.1 |
| soft_iou_median | 0.944 | 0.49 |
