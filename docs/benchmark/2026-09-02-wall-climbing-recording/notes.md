# 2025_10_12_15_06_46 male climbing — the BLOCKED recording, unblocked

**Root:** `/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v10_wall0902`
**New subset:** `wall_frames_15_06_46_male` (staged at
`red_data/courtship_labels_2026_09_02/wall_frames_15_06_46_male`)
**Source root (`--gm`):** `red_data/general_model_2026_09_02_v10` (symlink farm;
`general_model` untouched)
**Figures:** `figures/2026-09-02-climbing-recording/`
(`wall_climbing_conversion_check.png`, `all_frames.png`, `same_capture_contact.png`)
Regenerate: `python scripts/viz/wall_climbing_label_check.py`

## What changed

Task 18 reported this recording BLOCKED as `2025_10_12_10_56_07`: no video, no
frames, labels disjoint from `wall_frames`. **The timestamp was simply wrong.**
The user corrected it to `2025_10_12_15_06_46`, renamed the export directory and
extracted the frames. The label data is unchanged — the same 31 frame numbers
(0, 4012, 4121, 6321, 6403, 487938 … 1447731) under a corrected name.

## The video is still not usable — the jpgs are

`Video_recordings/Clip/Session6/2025_10_12_15_06_46/videos/Cam*.mp4` exists but
is a **symlink to a 921-frame excerpt** (`Cam*_frames_1161383_1162303.mp4`,
1936×448, 800 fps). The labels run to index **1,447,731** and the excerpt
covers 1161383–1162303, which contains **none** of them. So nothing was
decoded: the 217 pre-extracted jpgs under
`2025_10_12_15_06_46_male_climbing/Cam*/` are the image source, symlinked
read-only. `red3d2jarvis.py` gained one generalisation for that — a
`--link-images-from` directory may now hold `Cam*/Frame_*.jpg` at its own top
level, not only under `train/`+`val/`.

Dimensions come from the calibration yaml (1936×448) **and every one of the 217
jpgs is checked against it** before conversion, because `image_height` is what
the vertical flip subtracts; a mismatch would put the keypoints off the animal,
so it raises rather than warns.

## Same capture as `wall_frames` — corroborated by CONTENT

Task 18 could only record the user's mapping. The extracted frames make a real
content test possible, and it passes. The naive version of the test **fails**,
which is worth recording: a median background over the *whole* new recording is
59–68 grey levels from `wall_frames`, worse than 15 of 20 other subsets. That is
a confound, not a finding — the recording spans frame 0 → 1,447,731 and the
scene dims noticeably in its later half, while `wall_frames` sits entirely at
4031–8943. Restricted to the temporally comparable frames:

| test | new recording | best OTHER subset | worst |
|---|---:|---:|---:|
| median-background MAE vs `wall_frames`, Cam2012630 | **3.09** grey | 9.23 | 83.17 |
| " Cam2012862 | **2.74** | 9.47 | 72.83 |
| " Cam2012855 | **2.58** | 5.43 | 58.47 |

and pairwise, `Frame_4012` (new) vs `Frame_4031` (`wall_frames`) — 19 frames,
24 ms apart at 800 fps — agree to **2.5–3.2 grey levels outside the fly**,
against **11.6–13.7** for the same frame compared with a different capture
(`courtship_V2`). Visually (`same_capture_contact.png`, row C of the acceptance
figure) the arena, wall, lighting and the fly's position are continuous.

This is corroboration, not byte-level proof: the two frame-number ranges never
intersect, so there are no identical pixels to compare, and the only surviving
video contains neither set. The **calibration match is NOT evidence** — all 7
cameras are byte-identical to `wall_frames`, but so are 14 of the 21
`general_model` subsets; it is a rig-calibration epoch, not a capture.

## Join, not replace

`wall_frames`' 11 framesets (4031–8943) and this recording's 31 (0–1,447,731)
**do not intersect**. Replacement would delete 28 annotations and buy nothing.
Both are kept, and the user's standing "keep the wall frames for now" is
untouched.

The gain is real: **28 complete 7-camera framesets / 196 annotated images**
against `wall_frames`' 7 viable framesets / 28 annotations — about a fourfold
increase in the corpus's only wall-adjacent supervision, the regime this
pipeline is documented to fail at. Every one of the 31 framesets was looked at
(`all_frames.png`): all show one male on the arena wall, so `BEHAVIOR` is
`wall`, the same as `wall_frames`.

Three framesets — **0, 6321, 1261176** — carry no labels on any camera (21 of
the 217 images). Frame 0 is a startup frame with a different exposure; 6321 and
1261176 do contain a fly, simply unlabelled. The remaining 28 are labelled on
**all 7** cameras, which is the property `wall_frames` lacks (33 annotations
over 77 images, no complete frameset).

## CAPTURE GROUPS — the leak class this creates, made declarative

`wall_frames` and `wall_frames_15_06_46_male` share **no bytes**, so the
content merge cannot see that they are one capture. That is exactly
`courtship_V2/V3/V4`, where three "recordings" were one capture and holding one
out put **958 of 1,778 v8 val images (54%) on both sides** with every
name-, frame-, fly- and content-level audit reading 0.

`build_generalmodel_split.py` now carries a `CAPTURE_GROUPS` declaration and
`check_capture_groups()` **asserts against the shipped split** that every member
landed on the same side — not against the policy that was intended, because two
entries sitting in `TRAIN_COMPONENTS_FORCE` is not evidence they ended up
together. A straddle is a `SystemExit`. Two groups are declared: the wall
capture, and the pre-existing `2025_10_20_13_20_04` + `2026_08_26_16_05_15`
pair. Guarded by `test_build_generalmodel_split.py` (5 tests, including the
straddle regression).

Build output:

```
capture group OK: 2025_10_12_15_06_46 male climbing (arena wall) -> ['train']
capture group OK: 2025_10_20_13_20_04 courtship pair              -> ['train']
```

## Keypoint order — proven for THIS recording, not inherited

Exact reproduction against `general_model` is **impossible here**: the frames
are disjoint from `wall_frames`, so there is nothing to reproduce. Two
independent checks stand in.

1. **Cross-modal.** Index *k* of `keypoints3d.csv` projects through this
   recording's own raw DLT onto index *k* of every camera's 2D CSV at a
   **median 0.00053 px over 8,662 point-observations**, all 7 cameras
   (per-camera medians 0.00050–0.00061, p99 0.0077). A random permutation of
   the 3D keypoint axis gives **85.5 px — 160,000× worse**. A permutation
   between the 3D and 2D files could not survive this.
   *(The raw DLT projects into TOP-origin pixels while raw CSV `v` is
   BOTTOM-origin; comparing against unflipped `v` gives a median of 161 px with
   a permutation control no worse — i.e. the flip is load-bearing in this check
   too, and getting it wrong makes the check blind rather than loud.)*
2. **Rigid invariants, names resolved BY NAME.** All 16 homologous left/right
   segment pairs agree to **≤ 9.6%** on median length (femur/tibia, the
   well-defined segments, 0.4–7%); wing veins near-constant (robust CV
   3.4–5.6%, the short V12→V13 pair 7–18% on ~0.09 mm); Antenna_Base→Abd_tip
   **2.42 mm**. Looser than the 0.03–2.0% of `2025_10_20_13_20_04_male` because
   this is 27 frames of a wall-climbing fly with occluded tarsi versus 677 —
   the loose pairs are all tarsal. A scrambled order breaks symmetry by tens to
   hundreds of percent.
   `T2R`'s distance-from-trochanter is non-monotonic (0.687 → 0.653 at the
   knee): the middle-right leg is **folded**, which is a posture, not an order
   defect — its segment lengths are symmetric with T2L to 0.7–5%.
3. **The figure**, read back below.

The converter also **asserts** the CSVs' declared skeleton basename is
`fly50.json` (here `/home/user/red_data/skeleton/fly50.json`), so a different
skeleton cannot be silently converted with fly50 names.

## `--scale_10x` — True, verified against the shipped calibration

All 7 emitted yamls are **byte-identical** to both the shipped
`.../2025_10_12_15_06_46_male_climbing/calibration/Cam*.yaml` and to
`general_model/wall_frames/calib_params/2026_07_30_13_28_99/Cam*.yaml`
(`scale: 10` included). Negative control: `--no-scale-10x` produces
`2f2f725c…` against the shipped `cfdfc9f2…` — it does **not** match. Not guessed.

## Figure — expectation first, then read back

Expectation: cyan keypoints must sit ON the fly; **without** the
`y = image_height − v` flip the skeleton must LEAVE the animal; and the
same-capture pair must look continuous.

Read back (`wall_climbing_conversion_check.png`):
- **A** cyan sits on the fly in all four panels — `Cam2012862/489102` (48/50),
  `Cam2012630/4121` (37/50), `Cam2012855/1447731` (29/50),
  `Cam2012857/950756` (46/50). The fly is against the wall at the frame edge in
  every one; that is the hard case, not a flattering one.
- **B** the no-flip control (red) is mirrored about the 448-px midline and off
  the animal in all four. `Cam2012855` is the "almost plausible" one — red is
  only ~100 px away, on the dark band rather than the fly — which is why the
  0.00053 px numeric is what actually settles it.
- **C** `Cam2012853` Frame_4012 (cyan) vs `wall_frames` Frame_4031 (white),
  and `Cam2012631` Frame_6403 vs Frame_6708: same arena, same wall, same
  lighting, fly in nearly the same place, each drawn with its own stored
  annotation.

## New split

20 subsets, 17 recordings. The pixel corpus grows by exactly the new images.

| | v9 | **v10** | Δ |
|---|---:|---:|---:|
| train images | 18,123 | **18,319** | +196 |
| train annotations | 18,230 | **18,426** | +196 |
| train framesets | 2,630 | **2,658** | +28 |
| val images | 819 | **819** | 0 |
| val annotations | 1,066 | **1,066** | 0 |
| female share of annotations | 9.47% | **9.37%** | −0.10 pp |

val = 4.280% of images / 5.470% of annotations. Female annotations are
**unchanged** (train 1,357 / val 470); the share ticks down only because 196
male annotations were added — as expected, this recording is male.

**Zero-overlap assertion**, re-hashed independently from the shipped jsons
(not from `content_md5` or the build report):

```
train: 18319 images, 18426 annotations
val:     819 images,  1066 annotations
hashing 19138 files ...
train contents 18319, val contents 819, INTERSECTION 0
ASSERTION: train content-md5 set  INTERSECT  val content-md5 set  ==  EMPTY.  PASS
framesets straddling train/val: 0  PASS
```

Build audit: `cross_recording_leaks 0`, `cross_fly_leaked_frames 0`,
`cross_capture_leaks 0`, `content_overlap 0`.

**Unchanged consequence:** this val set still **cannot measure wall-adjacent
performance** — all of it is in train, now more so. Judge that regime GT-free
(rendered overlays, the frozen 13-bout benchmark).

## Source trees verified unmodified

* `Video_recordings` — manifest (path + size + mtime, 3,160 entries)
  **byte-identical** before and after.
* `general_model` image CONTENT — all **19,334** md5s recorded by the v9 root
  (built 17:49, before this session) re-hashed and **unchanged**.
* `_climbonly_src`, `courtship_label_2026_09_02`, `general_model`,
  `red3d2jarvis.py` — **0 files** modified after 21:25 (the first write of this
  session was at 21:45). The newest mtime anywhere in those trees is 21:20,
  the user's own frame extraction.
* `red3d2jarvis.py` md5 `e1e90f502684658a8af9f0b2fdbd10b5`.
* Full-tree manifest (21,954 entries, md5+symlink targets) recorded at
  `/tmp/task19/manifest_labels_AFTER.txt`, digest
  `90335136abbdad454b52da0b2a12c3d2`, as the baseline for the next task.

## Concerns

1. **Same-capture is corroborated, not proved.** The evidence is strong
   (background agreement 4–5× better than any other capture, visual continuity)
   but it is not byte identity, which is unobtainable here. Both members are in
   train, so the cost of being wrong is a marginally conservative split; the
   cost of the opposite error is the 54%-val-leak class.
2. **v8/v9/v10 val numbers are comparable to each other but not to v8.** val is
   byte-for-byte the same set as v9, so v9↔v10 A/B is clean. v8's courtship-male
   val was 54% leakage and must not be compared with either.
3. **val is 4.28% of images** and still has **no wall coverage at all** — by
   choice, restated above.
4. **21 of the 217 new images are unlabelled** (framesets 0, 6321, 1261176).
   They ship as images with no annotation, exactly as the raw export has them.
5. The other 8 recordings in `courtship_label_2026_09_02` still have no
   extracted frames and are unchanged; they remain covered by the existing
   `general_model` images.
