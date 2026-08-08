# Benchmark scorecard

## free_running (n=5)

| metric | median |
|---|---|
| hard_iou_median | 0.02347 |
| jitter_median | 0.002978 |
| jl_violation_rate | 0.0001806 |
| loo_px_median | 3.813 |
| reproj_px_leg_median | 4.022 |
| reproj_px_median | 4.115 |
| reproj_px_trunk_median | 3.666 |
| reproj_px_wing_median | 4.605 |
| soft_iou_median | 0.1454 |

## courtship_male (n=8)

| metric | median |
|---|---|
| hard_iou_apart | 0.02033 |
| hard_iou_close | 0.01626 |
| hard_iou_median | 0.02016 |
| jitter_apart | 0.002982 |
| jitter_close | 0.003446 |
| jitter_median | 0.002648 |
| jl_violation_rate | 0.0001693 |
| loo_px_median | 5.027 |
| reproj_px_apart | 21.24 |
| reproj_px_close | 21.83 |
| reproj_px_leg_apart | 20.73 |
| reproj_px_leg_close | 21.78 |
| reproj_px_leg_median | 21.14 |
| reproj_px_median | 21.13 |
| reproj_px_trunk_apart | 20.41 |
| reproj_px_trunk_close | 20.18 |
| reproj_px_trunk_median | 20.31 |
| reproj_px_wing_apart | 26.59 |
| reproj_px_wing_close | 41.63 |
| reproj_px_wing_median | 26.17 |
| soft_iou_apart | 0.1247 |
| soft_iou_close | 0.1011 |
| soft_iou_median | 0.123 |

## courtship_female (n=8)

| metric | median |
|---|---|
| hard_iou_apart | 0.02277 |
| hard_iou_close | 0 |
| hard_iou_median | 0.01599 |
| jitter_apart | 0.008483 |
| jitter_close | 0.01554 |
| jitter_median | 0.003348 |
| jl_violation_rate | 0.0001168 |
| loo_px_median | 54.39 |
| reproj_px_apart | 33.44 |
| reproj_px_close | 84.51 |
| reproj_px_leg_apart | 35.19 |
| reproj_px_leg_close | 90.58 |
| reproj_px_leg_median | 65.71 |
| reproj_px_median | 58.03 |
| reproj_px_trunk_apart | 24.41 |
| reproj_px_trunk_close | 84.28 |
| reproj_px_trunk_median | 55.75 |
| reproj_px_wing_apart | 35.41 |
| reproj_px_wing_close | 145 |
| reproj_px_wing_median | 77.91 |
| soft_iou_apart | 0.1392 |
| soft_iou_close | 0 |
| soft_iou_median | 0.07776 |

## Gap ratios (courtship / free_running, 1.0 = parity)

| metric | courtship_male | courtship_female |
|---|---|---|
| hard_iou_median | 0.859 | 0.681 |
| jitter_median | 0.889 | 1.12 |
| jl_violation_rate | 0.937 | 0.647 |
| loo_px_median | 1.32 | 14.3 |
| reproj_px_leg_median | 5.26 | 16.3 |
| reproj_px_median | 5.14 | 14.1 |
| reproj_px_trunk_median | 5.54 | 15.2 |
| reproj_px_wing_median | 5.68 | 16.9 |
| soft_iou_median | 0.846 | 0.535 |
