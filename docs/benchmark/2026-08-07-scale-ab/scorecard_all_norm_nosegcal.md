# Benchmark scorecard

## free_running (n=5)

| metric | median |
|---|---|
| hard_iou_median | 0.02405 |
| jitter_median | 0.003199 |
| jl_violation_rate | 0.000422 |
| loo_px_median | 3.813 |
| reproj_px_leg_median | 4.221 |
| reproj_px_median | 3.903 |
| reproj_px_trunk_median | 3.289 |
| reproj_px_wing_median | 4.554 |
| soft_iou_median | 0.1482 |

## courtship_male (n=8)

| metric | median |
|---|---|
| hard_iou_apart | 0.0238 |
| hard_iou_close | 0.02251 |
| hard_iou_median | 0.02426 |
| jitter_apart | 0.003021 |
| jitter_close | 0.003203 |
| jitter_median | 0.002327 |
| jl_violation_rate | 0.0002122 |
| loo_px_median | 5.027 |
| reproj_px_apart | 19.49 |
| reproj_px_close | 21.17 |
| reproj_px_leg_apart | 17.79 |
| reproj_px_leg_close | 19.54 |
| reproj_px_leg_median | 18.38 |
| reproj_px_median | 19.93 |
| reproj_px_trunk_apart | 21.39 |
| reproj_px_trunk_close | 20.43 |
| reproj_px_trunk_median | 20.31 |
| reproj_px_wing_apart | 31.64 |
| reproj_px_wing_close | 48.06 |
| reproj_px_wing_median | 30.5 |
| soft_iou_apart | 0.139 |
| soft_iou_close | 0.1312 |
| soft_iou_median | 0.1406 |

## courtship_female (n=8)

| metric | median |
|---|---|
| hard_iou_apart | 0.02313 |
| hard_iou_close | 0 |
| hard_iou_median | 0.0136 |
| jitter_apart | 0.01167 |
| jitter_close | 0.01754 |
| jitter_median | 0.003623 |
| jl_violation_rate | 0.000113 |
| loo_px_median | 54.39 |
| reproj_px_apart | 33 |
| reproj_px_close | 84.41 |
| reproj_px_leg_apart | 34.59 |
| reproj_px_leg_close | 92.53 |
| reproj_px_leg_median | 67.3 |
| reproj_px_median | 57.71 |
| reproj_px_trunk_apart | 21.52 |
| reproj_px_trunk_close | 84.89 |
| reproj_px_trunk_median | 53.02 |
| reproj_px_wing_apart | 40.47 |
| reproj_px_wing_close | 146.6 |
| reproj_px_wing_median | 79.72 |
| soft_iou_apart | 0.1413 |
| soft_iou_close | 0 |
| soft_iou_median | 0.06834 |

## Gap ratios (courtship / free_running, 1.0 = parity)

| metric | courtship_male | courtship_female |
|---|---|---|
| hard_iou_median | 1.01 | 0.566 |
| jitter_median | 0.727 | 1.13 |
| jl_violation_rate | 0.503 | 0.268 |
| loo_px_median | 1.32 | 14.3 |
| reproj_px_leg_median | 4.35 | 15.9 |
| reproj_px_median | 5.11 | 14.8 |
| reproj_px_trunk_median | 6.17 | 16.1 |
| reproj_px_wing_median | 6.7 | 17.5 |
| soft_iou_median | 0.948 | 0.461 |
