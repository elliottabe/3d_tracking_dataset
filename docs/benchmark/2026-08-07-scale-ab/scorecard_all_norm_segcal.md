# Benchmark scorecard

## free_running (n=5)

| metric | median |
|---|---|
| hard_iou_median | 0.02389 |
| jitter_median | 0.003129 |
| jl_violation_rate | 7.701e-05 |
| loo_px_median | 3.813 |
| reproj_px_leg_median | 4.223 |
| reproj_px_median | 4.294 |
| reproj_px_trunk_median | 3.404 |
| reproj_px_wing_median | 4.686 |
| soft_iou_median | 0.1474 |

## courtship_male (n=8)

| metric | median |
|---|---|
| hard_iou_apart | 0.02347 |
| hard_iou_close | 0.02211 |
| hard_iou_median | 0.0238 |
| jitter_apart | 0.002852 |
| jitter_close | 0.003011 |
| jitter_median | 0.002569 |
| jl_violation_rate | 0.0001406 |
| loo_px_median | 5.027 |
| reproj_px_apart | 22.02 |
| reproj_px_close | 24.19 |
| reproj_px_leg_apart | 21.81 |
| reproj_px_leg_close | 24.06 |
| reproj_px_leg_median | 23.01 |
| reproj_px_median | 23.22 |
| reproj_px_trunk_apart | 21.53 |
| reproj_px_trunk_close | 21.83 |
| reproj_px_trunk_median | 21.72 |
| reproj_px_wing_apart | 30.29 |
| reproj_px_wing_close | 47.73 |
| reproj_px_wing_median | 29.61 |
| soft_iou_apart | 0.1389 |
| soft_iou_close | 0.1299 |
| soft_iou_median | 0.1367 |

## courtship_female (n=8)

| metric | median |
|---|---|
| hard_iou_apart | 0.02309 |
| hard_iou_close | 0 |
| hard_iou_median | 0.01505 |
| jitter_apart | 0.01373 |
| jitter_close | 0.0177 |
| jitter_median | 0.004932 |
| jl_violation_rate | 0.0001077 |
| loo_px_median | 54.39 |
| reproj_px_apart | 34.9 |
| reproj_px_close | 85.18 |
| reproj_px_leg_apart | 36.67 |
| reproj_px_leg_close | 91.89 |
| reproj_px_leg_median | 64.86 |
| reproj_px_median | 58.44 |
| reproj_px_trunk_apart | 23.91 |
| reproj_px_trunk_close | 83.66 |
| reproj_px_trunk_median | 55.82 |
| reproj_px_wing_apart | 32.74 |
| reproj_px_wing_close | 141.6 |
| reproj_px_wing_median | 76.56 |
| soft_iou_apart | 0.1413 |
| soft_iou_close | 0 |
| soft_iou_median | 0.06532 |

## Gap ratios (courtship / free_running, 1.0 = parity)

| metric | courtship_male | courtship_female |
|---|---|---|
| hard_iou_median | 0.996 | 0.63 |
| jitter_median | 0.821 | 1.58 |
| jl_violation_rate | 1.83 | 1.4 |
| loo_px_median | 1.32 | 14.3 |
| reproj_px_leg_median | 5.45 | 15.4 |
| reproj_px_median | 5.41 | 13.6 |
| reproj_px_trunk_median | 6.38 | 16.4 |
| reproj_px_wing_median | 6.32 | 16.3 |
| soft_iou_median | 0.927 | 0.443 |
