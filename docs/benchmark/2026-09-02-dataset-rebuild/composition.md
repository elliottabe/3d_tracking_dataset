# red_data_3d_v8_gm_only -- composition

Source: `general_model` ONLY, 21 subsets, 19929 image refs -> **19334 unique contents** (content/md5 dedup).

Two-fly images after merging the per-fly pairs: **499**.

Shipped: train 17143 img / 17250 ann, val 1778 img / 2024 ann (**9.4%** of images, 10.5% of annotations).

## Per subset

| subset | sex | source | train img | train ann | val img | val ann |
|---|---|---|---:|---:|---:|---:|
| 20_04_female_climbing | female | dirname | 105 | 105 | 0 | 0 |
| S6male | male | dirname | 3335 | 3335 | 0 | 0 |
| S8_male_R_amp | male | dirname | 1330 | 1330 | 0 | 0 |
| S9_male_L_amp | male | dirname | 2085 | 2085 | 0 | 0 |
| courtship_11_50_female | female | dirname | 0 | 0 | 103 | 103 |
| courtship_11_50_male | male | dirname | 0 | 0 | 105 | 105 |
| courtship_25_51_female | female | dirname | 0 | 0 | 157 | 157 |
| courtship_25_51_male | male | dirname | 0 | 0 | 150 | 150 |
| courtship_28_34_female | female | dirname | 251 | 251 | 0 | 0 |
| courtship_28_34_male | male | dirname | 302 | 302 | 0 | 0 |
| courtship_V2 | male | user_statement_2026_09_02 | 3220 | 3220 | 0 | 0 |
| courtship_V3 | male | user_statement_2026_09_02 | 0 | 0 | 958 | 958 |
| courtship_V4 | male | user_statement_2026_09_02 | 539 | 539 | 0 | 0 |
| female | female | dirname | 616 | 616 | 70 | 70 |
| grooming | male | user_statement_2026_09_02 | 4592 | 4592 | 0 | 0 |
| headless_22_50_female | female | dirname | 0 | 0 | 140 | 140 |
| headless_24_04_1_male | male | dirname | 0 | 0 | 341 | 341 |
| headless_24_04_male | male | dirname | 462 | 462 | 0 | 0 |
| headless_56_42_1_female | female | dirname | 168 | 168 | 0 | 0 |
| headless_56_42_female | female | dirname | 217 | 217 | 0 | 0 |
| wall_frames | female | user_statement_2026_09_02 | 28 | 28 | 0 | 0 |

## Per sex (image label `mixed` = one female AND one male annotation)

| sex | train img | train ann | val img | val ann |
|---|---:|---:|---:|---:|
| female | 1139 | 1385 | 221 | 470 |
| male | 15619 | 15865 | 1305 | 1554 |
| mixed | 246 | 0 | 249 | 0 |

## Per calibration group

| group | train img | train ann | val img | val ann |
|---|---:|---:|---:|---:|
| A | 13140 | 13386 | 712 | 858 |
| B | 3864 | 3864 | 958 | 958 |
| C | 0 | 0 | 105 | 208 |

## Per sex provenance

| sex_source | train img | train ann | val img | val ann |
|---|---:|---:|---:|---:|
| dirname | 8625 | 8871 | 817 | 1066 |
| user_statement_2026_09_02 | 8379 | 8379 | 958 | 958 |

## Female budget

train 1385 female annotations, val 470 (25.3% of all female).
Female share of all shipped annotations: 9.62%.
