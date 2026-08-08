# Benchmark scorecard

## free_running (n=5)

| metric | median |
|---|---|
| hard_iou_median | 0.02358 |
| jitter_median | 0.003074 |
| jl_violation_rate | 0.0001155 |
| loo_px_median | 3.813 |
| reproj_px_leg_median | 3.892 |
| reproj_px_median | 3.778 |
| reproj_px_trunk_median | 3.329 |
| reproj_px_wing_median | 4.411 |
| soft_iou_median | 0.1466 |

## courtship_male (n=8)

| metric | median |
|---|---|
| hard_iou_apart | 0.01865 |
| hard_iou_close | 0.01388 |
| hard_iou_median | 0.01882 |
| jitter_apart | 0.002926 |
| jitter_close | 0.00332 |
| jitter_median | 0.002645 |
| jl_violation_rate | 0.0003213 |
| loo_px_median | 5.027 |
| reproj_px_apart | 19.94 |
| reproj_px_close | 20.79 |
| reproj_px_leg_apart | 18.08 |
| reproj_px_leg_close | 19.92 |
| reproj_px_leg_median | 19.06 |
| reproj_px_median | 20.15 |
| reproj_px_trunk_apart | 23.67 |
| reproj_px_trunk_close | 20.39 |
| reproj_px_trunk_median | 20.8 |
| reproj_px_wing_apart | 31.09 |
| reproj_px_wing_close | 37.86 |
| reproj_px_wing_median | 29.23 |
| soft_iou_apart | 0.1162 |
| soft_iou_close | 0.08753 |
| soft_iou_median | 0.1166 |

## courtship_female (n=8)

| metric | median |
|---|---|
| hard_iou_apart | 0.0228 |
| hard_iou_close | 0 |
| hard_iou_median | 0.01699 |
| jitter_apart | 0.007096 |
| jitter_close | 0.01501 |
| jitter_median | 0.003199 |
| jl_violation_rate | 0.0001432 |
| loo_px_median | 54.39 |
| reproj_px_apart | 33.29 |
| reproj_px_close | 84.96 |
| reproj_px_leg_apart | 33.92 |
| reproj_px_leg_close | 92.79 |
| reproj_px_leg_median | 65.57 |
| reproj_px_median | 57.86 |
| reproj_px_trunk_apart | 22.93 |
| reproj_px_trunk_close | 83.85 |
| reproj_px_trunk_median | 55.12 |
| reproj_px_wing_apart | 43.26 |
| reproj_px_wing_close | 146 |
| reproj_px_wing_median | 82.66 |
| soft_iou_apart | 0.1396 |
| soft_iou_close | 0 |
| soft_iou_median | 0.0764 |

## Gap ratios (courtship / free_running, 1.0 = parity)

| metric | courtship_male | courtship_female |
|---|---|---|
| hard_iou_median | 0.798 | 0.721 |
| jitter_median | 0.861 | 1.04 |
| jl_violation_rate | 2.78 | 1.24 |
| loo_px_median | 1.32 | 14.3 |
| reproj_px_leg_median | 4.9 | 16.8 |
| reproj_px_median | 5.34 | 15.3 |
| reproj_px_trunk_median | 6.25 | 16.6 |
| reproj_px_wing_median | 6.63 | 18.7 |
| soft_iou_median | 0.796 | 0.521 |
