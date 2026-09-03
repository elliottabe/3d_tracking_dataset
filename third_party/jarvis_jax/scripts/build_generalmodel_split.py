"""Build a content-keyed train/val root sourcing frames from general_model ONLY.

    python third_party/jarvis_jax/scripts/build_generalmodel_split.py \
        --gm  .../red_data/general_model \
        --out .../red_data/red_data_3d_v8_gm_only

READ-ONLY on `general_model` and on `_climbonly_src`. general_model is now the
ONLY surviving copy of this corpus -- `red_data_unified_V3`,
`red_data_3d_v5_valfix`, `red_data_3d_v6_contentsplit` and
`red_data_3d_v7_generalmodel` were all deleted on 2026-09-02, along with every
`manifest.json` under the data root. Nothing here may write inside `--gm`.

Emits a new tree whose `images/` and `masks/` are symlinks back at
general_model and whose `annotations/` carry a split that is disjoint BY BYTES.

WHY THIS EXISTS. `red_data_3d_v5*` / `v6_contentsplit` merged general_model AND
red_data_unified_V3, which duplicated most of the corpus: 14,182 of V3's 14,973
image contents were byte-identical to a general_model image, and 18,430 of its
20,751 annotations matched a general_model annotation to under 5 px. The user's
ruling is general_model only, and V3 no longer exists to reconsider.

WHAT CHANGED 2026-09-02 (this rebuild vs the deleted v7 root): sex is now known
for all 21 subsets and is read from `<subset>/sex.json` (see SEX below), so
nothing is "unknown" and there is no table in this file; the
--sex-fallback-manifest path is gone because no manifest survived; five
headless subsets were renamed and THREE of them are female, not male, as their
old names claimed; and the train/val policy was redesigned from scratch (see
VAL_RECORDINGS below) rather than inherited from build_content_split.py.

THE ONE THING general_model FILES DIFFERENTLY. A two-fly courtship capture is
filed as TWO subsets under TWO recording timestamps, one per animal:

    courtship_25_51_female/  ->  2026_06_19_11_09_36
    courtship_25_51_male/    ->  2026_06_18_19_23_03    (same footage, 1 s apart)

so a name-keyed merge sees two single-fly recordings where there is one two-fly
capture. This module merges by CONTENT: one image per md5, both animals'
annotations attached, both on the same side of the split. Measured over
general_model's 19,929 image references -> 19,334 unique contents, that
recovers ~499 genuinely two-fly images (bbox centroids 241-392 px apart --
checked, not assumed; two labels on one animal sit under 10 px).

The only OTHER cross-subset content sharing in the corpus is 7 contents shared
by headless_56_42 and headless_56_42_1 -- one frame of one capture filed twice
for the SAME animal, not two animals. R15 (below) catches it and records both
slots absent, and the two recordings are held together as one alias component.

THE V2/V3/V4 MERGE (2026-09-02), and the leak it retires.
`courtship_V2`, `courtship_V3` and `courtship_V4` were not three recordings.
They were three arbitrary, non-overlapping frame ranges of ONE capture,
`2025_10_20_13_20_04` (Session0), filed under three fabricated recording ids
(2026_03_09_14_39_40 / 2026_03_18_15_31_22 / 2026_04_01_16_23_08). Proof, not
inference: the 2026-09-02 raw label export ships that recording whole, and its
image set is EXACTLY the union of the three -- 4,739 references, zero on either
side of the difference -- while all three subsets carry byte-identical
calibration.

That means the previous split's largest val block was a leak. It held V3 out
"whole" as an unseen session while training on V2 and V4, which are the same
fly in the same session minutes away; 958 of 1,778 val images, 54% by count.
Any courtship-male val number from the v8 root is optimistic by an unknown
amount and must not be compared with a number from this root.

The three subsets are replaced by ONE, `courtship_20_04_male`, converted from
the raw export by `scripts/data_prep/red3d2jarvis.py` (which documents how the
keypoint order and the vertical flip were established). It reproduces all 4,721
previously-existing annotations EXACTLY -- keypoints, visibility and bbox -- and
adds the 18 images V2/V3/V4 left unannotated, which is the whole reason the user
asked for the re-export. It is forced to train, below.

THE WALL CAPTURE (2026-09-02, added after v9). `general_model/wall_frames` and
the new subset `wall_frames_15_06_46_male` are different frames of ONE capture,
`2025_10_12_15_06_46` (male climbing). The second was reported BLOCKED under a
wrong timestamp (`2025_10_12_10_56_07`, for which no video and no frames
exist); the user corrected the timestamp, renamed the export directory and
extracted the 217 jpgs, and it converts like any other recording -- except that
its only surviving video is a 921-frame excerpt (1161383-1162303) while its
labels run to index 1,447,731, so the shipped jpgs are the sole image source.

It JOINS wall_frames rather than replacing it: the two label different moments
(4031-8943 vs 0-1,447,731, no intersection), so a replacement would simply
delete 28 annotations. Together they are the corpus's only wall-adjacent
supervision, and this subset multiplies it about fourfold: 28 complete
7-camera framesets against wall_frames' 7 viable ones.

Their frame numbers never intersect, so the content merge cannot see that they
are one capture -- exactly the V2/V3/V4 situation. They are therefore declared
in CAPTURE_GROUPS below and the declaration is ENFORCED against the shipped
split.

WHAT THIS ROOT DOES NOT HAVE. V3 was the only source of the second-fly labels
on the courtship_V2 capture (V3 recordings 2026_04_07_11_33_33 and
2026_04_08_14_59_45): 2,321 annotations, 2,014 of them the second animal of a
two-fly frame, and 1,225 of them on 791 image contents general_model does not
hold at all. So this root carries ~499 two-fly images where the deleted
V3-based root carried 1,673. That is a real, measured cost of the sourcing
decision; it is recorded in build_report.json under
`two_fly_deficit_vs_v3_root`, and it is no longer reversible.

IDENTITY COMES FROM THE SUBSET, NOT FROM POSITION. `build_v5.merge_annotations`
assigns fly_id by position within a camera's annotation list, guarded by
CONTROLLER RULING R15 (a camera whose fly count disagrees with the frameset max
is recorded ABSENT rather than guessed). Here the subset directory names the
animal, so identity is read off the filing and is immune to the chimera failure
R15 exists to prevent. R15's spirit still applies in one place: if the two
subsets' annotations on one camera land on the SAME animal (median visible
keypoint displacement < 50 px -- 21 contents, a labelling defect), neither can
be attributed, so BOTH slots are recorded ABSENT for that camera.

SEX COMES FROM `<subset>/sex.json`, WHICH TRAVELS WITH THE DATA. No
general_model ANNOTATION record carries a sex field -- checked, all 19,382 --
so sex lives in a per-subset sidecar written 2026-09-02:

    {"subset": ..., "sex": "female"|"male",
     "sex_source": "dirname" | "user_statement_2026_09_02",
     "recordings": [...], "citation": ..., "note": ...}

There is deliberately NO sex table in this file. A hardcoded table in a script
is how these labels were lost the first time: the metadata gets separated from
the data, the data is re-derived or renamed, and the table silently describes
something else. `sex.json` travels with the subset through any derivation; a
table in a repo does not. A missing, unparseable, or self-inconsistent sex.json
is a HARD ERROR -- there is no fall back to name-parsing and no "unknown" tier,
because either fallback would quietly re-introduce the failure it replaces.
(Name-parsing is doubly unsafe here: "female" CONTAINS "male", so a substring
parser that tests male first labels every female subset male, and every
sex-conditioned metric in this repo then reads fine while measuring the wrong
animals. `headless_22_50_female` and the two `headless_56_42*_female` subsets
were male-named until 2026-09-02 and the name was simply wrong.)

Nothing is inferred from an image. Body size in particular is NOT usable: this
repo's one size heuristic is documented as counter-intuitive (the male has the
LARGER masked area, because of wing extension), so guessing from size would be
both a rule violation and backwards.

Sex is a property of the ANNOTATED animal, not of the frame. courtship_V2/V3/V4
are male-labelled captures that also contain an unlabelled female; a merged
courtship pair image carries one female AND one male annotation and is counted
as `mixed` at image level, `female`+`male` at annotation level.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from jarvis_jax.data.build_v5 import (                 # noqa: E402
    MIN_CAMS, _canonical_keypoint_names, _keypoint_remap, _pad_keypoints)
from jarvis_jax.data.calib_groups import CAM_GLOB, group_calibrations   # noqa: E402
from jarvis_jax.data.content_index import (            # noqa: E402
    alias_components, content_groups, hash_paths)
from jarvis_jax.data.split_v5 import (                 # noqa: E402
    audit_split, make_split, write_derived)

# ---------------------------------------------------------------------------
# THE SPLIT, chosen 2026-09-02 from scratch. general_model ships its own
# train/ and val/ directories; they are subset-INTERNAL frame-level splits
# (~50% of their val framesets have a train frameset within +/-3 frames) and
# they are ignored completely -- they are read only as a place to find images.
#
# The holdout unit is the ALIAS COMPONENT (recordings joined by byte identity),
# and within a component the atom is the whole capture (all 7 cameras of a
# frame, every fly annotated in it). A model that has seen six cameras of a
# frame has effectively seen the seventh, so nothing smaller is ever split.
#
# PRINCIPLE: hold a component out WHOLE wherever its regime survives in
# training through a sibling component -- that is the only way a val number
# means "generalizes to an unseen fly and session". Where a regime lives in
# exactly ONE component, holding it out whole would erase it from training, so
# either take a guarded temporal tail (only if the tail is a real sample) or
# keep it wholly in train and declare the regime unmeasurable here.
#
# Whole components to val. Each names a regime that survives in train:
VAL_RECORDINGS = [
    # courtship_V3 (2026_03_18_15_31_22) USED TO BE HELD OUT HERE. It is gone,
    # and holding it out was a LEAK -- see THE V2/V3/V4 MERGE at the top of this
    # file. V2, V3 and V4 were three arbitrary frame ranges of ONE recording
    # (2025_10_20_13_20_04), so "hold out V3, train on V2 and V4" was a
    # same-session, same-fly holdout wearing an unseen-session costume, and it
    # was 54% of the whole val set by image count. Courtship MALE val now comes
    # from the two forced two-fly components below, each of which is a genuinely
    # different session and a different animal.
    "2026_06_09_15_21_14",   # headless_24_04_1_male  MALE headless, 49 fs /
                             #   343 img. Male headless stays in train via
                             #   headless_24_04_male (66 fs). Paired with the
                             #   female headless holdout below on purpose: see
                             #   THE CONTROLLED SEX CONTRAST.
]
# Female-inclusive components forced WHOLE to val. Female preservation normally
# vetoes a whole holdout; it is spent here three times, deliberately, because
# these are the only components whose regime has a female SIBLING left in train
# -- the only way to get an unseen-session FEMALE number at all. Cost is stated
# in build_report.json["composition"]["female_budget"].
VAL_COMPONENTS_FORCE = [
    "2026_05_27_11_56_05",   # courtship_11_50 pair (female+male), 15 fs / 105 img
    "2026_06_18_19_23_03",   # courtship_25_51 pair (female+male), 23 fs / 161 img
                             #   The two SMALLEST two-fly captures. Two-fly and
                             #   female-in-courtship are the two documented
                             #   failure modes, so val must contain both; the
                             #   largest two-fly capture stays in train below.
    "2026_06_09_15_46_55",   # headless_22_50_female  FEMALE headless, 20 fs /
                             #   140 img. Female headless survives in train via
                             #   the headless_56_42 component (57 fs / 392 img),
                             #   so this holdout is affordable and it is the
                             #   cheapest unseen-session female number in the
                             #   corpus: 140 female annotations spent.
]
# THE CONTROLLED SEX CONTRAST. Every other sex comparison available here is
# confounded by regime -- female footage is climbing/wall/courtship, male
# footage is grooming/amputation/courtship -- so a female-vs-male MPJPE gap
# measured across the whole val set is partly a gap between different
# behaviours. Holding out headless_22_50_female AND headless_24_04_1_male gives
# ONE regime, headless, with an unseen session of each sex, where the sex
# contrast is not confounded by behaviour. Report that pair separately from the
# aggregate; it is the only clean sex comparison this split supports.
#
# Female-inclusive components kept WHOLLY in train, overriding the automatic
# 10%-tail rule. Each one is a female metric this split will NOT have, so each
# needs a reason:
TRAIN_COMPONENTS_FORCE = [
    "2026_06_15_12_12_33",   # courtship_28_34 pair -- the LARGEST two-fly
                             #   capture (46 fs, 322 img, 269 female anns).
                             #   Two-fly val is already covered by the two
                             #   components above, so this one is worth more as
                             #   training data than as a third val session.
    "2026_08_26_16_05_15",   # 20_04_female_climbing -- 15 fs, the ONLY female
                             #   climbing footage. A 10% tail is 2 framesets;
                             #   that measures nothing and costs the regime.
    "2026_06_09_15_38_35",   # headless_56_42_female + headless_56_42_1_female,
                             #   ONE component (they share 7 contents: one frame
                             #   of one capture filed twice). It is the sibling
                             #   that keeps female headless in training while
                             #   22_50 goes to val. Forced rather than left to
                             #   the tail rule so the female-headless val number
                             #   is purely unseen-session, not a blend of an
                             #   unseen session and a same-session tail.
    "2026_07_30_13_28_99",   # wall_frames -- 7 viable framesets (camera
                             #   coverage tops out at 6/7 and 4 of its 11 frames
                             #   fall below MIN_CAMS=3), ~28 annotations.
                             #   DECIDED, not defaulted -- and RE-DERIVED
                             #   2026-09-02 after the user corrected this
                             #   subset's sex to MALE (source recording
                             #   2025_10_12_10_56_07_male_climbing). The old
                             #   justification -- "100% of the corpus's
                             #   wall-adjacent FEMALE supervision" -- was simply
                             #   false and is void. It survives on its own
                             #   merits, which are unchanged by sex: as a metric
                             #   ~28 annotations from one fly in one session
                             #   cannot separate any realistic model change, and
                             #   because it is the ONLY wall footage a val-only
                             #   placement means the model never trains on
                             #   wall-adjacent poses, so the number would read
                             #   badly for reasons unrelated to the change under
                             #   test. As training data it is 100% of the
                             #   corpus's wall-adjacent supervision, a regime
                             #   this pipeline is documented to fail at.
                             #   CONSEQUENCE, unchanged: this val set CANNOT
                             #   measure wall-adjacent performance. Judge that
                             #   regime GT-free (rendered overlays, the frozen
                             #   13-bout benchmark), as this repo already does.
                             #   NOTE it no longer contributes to the female
                             #   budget at all -- it is male.
    "2025_10_12_15_06_46",   # wall_frames_15_06_46_male -- 28 labelled
                             #   framesets / 196 annotated images of a male on
                             #   the arena wall, converted 2026-09-02 after the
                             #   user corrected the timestamp (it was reported
                             #   BLOCKED as 2025_10_12_10_56_07, for which no
                             #   frames and no video existed).
                             #   TRAIN for the same reason wall_frames is, and
                             #   it is the SAME CAPTURE as wall_frames (see
                             #   CAPTURE_GROUPS), so the two could not go to
                             #   opposite sides in any case. Wall footage is
                             #   the ONE regime this corpus has almost none of
                             #   and the one this pipeline is documented to
                             #   fail at; putting the only 4x increase in that
                             #   supervision into val would leave the model
                             #   trained on essentially no wall-adjacent poses
                             #   in order to buy a val number of ~200 images
                             #   from one fly in one session. It also joins
                             #   rather than replaces wall_frames: the two
                             #   label DIFFERENT moments (4031-8943 vs
                             #   0-1,447,731, no intersection), so replacing
                             #   would simply delete 28 annotations.
                             #   CONSEQUENCE, unchanged from v9: this val set
                             #   still cannot measure wall-adjacent
                             #   performance; judge that regime GT-free.
    "2025_10_20_13_20_04",   # courtship_20_04_male -- the merged V2+V3+V4
                             #   recording, 677 fs / 4,739 img, the corpus's
                             #   largest courtship-male block. Forced to TRAIN
                             #   because it is ONE session of ONE animal: a
                             #   partial holdout would be the same leak that
                             #   the old V3 holdout was, and a whole holdout
                             #   would move 4,739 images (24% of the corpus)
                             #   out of training to buy a val number that the
                             #   11_50 and 25_51 male components already
                             #   provide from genuinely unseen sessions.
                             #   It is also the same CAPTURE as
                             #   20_04_female_climbing (the user's mapping;
                             #   that subset's internal recording id,
                             #   2026_08_26_16_05_15, does NOT carry the
                             #   capture timestamp so the builder cannot see
                             #   the relationship). Both are forced to train,
                             #   which is what keeps that hidden relationship
                             #   from becoming a cross-side leak.
]
# Left to the automatic guarded-tail rule: exactly ONE component, `female`
# (2026_01_29_14_09_33, 100 fs / 700 img), the only single-fly female general
# footage. Its val block is the last `female_val_frac` of its frames behind a
# `guard`-frame band -- SAME fly, SAME session, so it measures "new pose" and
# is OPTIMISTIC. It is reported separately from the unseen-session female
# numbers and must never be averaged into a headline female metric.

# Median visible-keypoint displacement below which two annotations are the SAME
# animal. Task 15 measured this distribution to be bimodal with an empty gap:
# 95% of best matches sit under 5 px, 0.08% land in 5-186 px, the other animal
# sits at >= 186 px. 50 px is inside the gap.
SAME_FLY_PX = 50.0

# Behaviour is not recorded on any general_model annotation (nor on any V3 one
# -- every root so far has `behavior: "unknown"` on every annotation). The
# subset directory is the only behaviour signal in the corpus, so it is written
# to the MANIFEST, which no loader reads for behaviour, and the annotation
# field is left "unknown" exactly as the v5/v6 roots have it -- changing it
# would silently re-weight V5Dataset2D.balanced_weights.
BEHAVIOR = {
    "20_04_female_climbing": "climbing", "S6male": "general",
    "S8_male_R_amp": "amputation", "S9_male_L_amp": "amputation",
    "courtship_11_50_female": "courtship", "courtship_11_50_male": "courtship",
    "courtship_25_51_female": "courtship", "courtship_25_51_male": "courtship",
    "courtship_28_34_female": "courtship", "courtship_28_34_male": "courtship",
    # Replaces courtship_V2 + courtship_V3 + courtship_V4, which were three
    # arbitrary slices of THIS ONE recording -- see THE V2/V3/V4 MERGE below.
    "courtship_20_04_male": "courtship",
    "female": "general", "grooming": "grooming",
    "headless_22_50_female": "headless", "headless_24_04_male": "headless",
    "headless_24_04_1_male": "headless", "headless_56_42_female": "headless",
    "headless_56_42_1_female": "headless", "wall_frames": "wall",
    # Different frames of the SAME capture as wall_frames -- see CAPTURE_GROUPS.
    # All 31 of its labelled framesets show one male on the arena wall, checked
    # by looking at every one of them, so the regime label is the same: wall.
    "wall_frames_15_06_46_male": "wall",
}


# ---------------------------------------------------------------------------
# CAPTURE GROUPS: recordings that are the SAME physical capture but share no
# bytes, so the content merge above cannot see the relationship.
#
# This is the courtship_V2/V3/V4 failure made declarative. Those were three
# arbitrary frame ranges of one capture under three fabricated recording ids;
# holding one out "as an unseen session" put 958 of 1,778 val images (54%) on
# both sides of the v8 split. Content hashing did not catch it and could not:
# the frames genuinely differ. Only a declaration can.
#
# Members must land on the SAME side of the split. That is not left to the
# accident of two independent entries in the force lists below -- it is
# asserted against the SHIPPED split in main(), so a future policy edit that
# separates two members fails the build instead of silently leaking.
CAPTURE_GROUPS = [
    {
        "name": "2025_10_12_15_06_46 male climbing (arena wall)",
        "recordings": ["2026_07_30_13_28_99",    # general_model/wall_frames
                       "2025_10_12_15_06_46"],   # wall_frames_15_06_46_male
        "evidence":
            "User mapping (2026-09-02): general_model/wall_frames is "
            "2025_10_12_..._male_climbing; only the timestamp was mistaken "
            "(10_56_07 -> 15_06_46). CORROBORATED BY CONTENT, not taken on "
            "trust: over the temporally comparable frames the median arena "
            "background of the two subsets agrees to 2.6-3.1 grey levels on "
            "every camera tested and is the CLOSEST of all 21 subsets -- the "
            "next nearest capture on the same rig is 5.4-9.5 and the worst is "
            "83. Pairwise, Frame_4012 (new) and Frame_4031 (wall_frames) are "
            "19 frames apart and agree to 2.5-3.2 grey levels outside the fly, "
            "against 11.6-13.7 for the same frame vs a different capture. Not "
            "byte-level proof -- their frame numbers never intersect "
            "(wall_frames 4031-8943; this export 0-1,447,731), so there are no "
            "identical pixels to compare, and the only surviving video is a "
            "921-frame excerpt that contains neither set.",
    },
    {
        "name": "2025_10_20_13_20_04 courtship pair",
        "recordings": ["2025_10_20_13_20_04",    # courtship_20_04_male
                       "2026_08_26_16_05_15"],   # 20_04_female_climbing
        "evidence":
            "User mapping. 20_04_female_climbing's internal recording id is "
            "fabricated and carries no capture timestamp, so the builder "
            "cannot derive the relationship from the data.",
    },
]


VALID_SEX = ("female", "male")

# Populated by load_sex(); deliberately EMPTY at import so that nothing in this
# module can read a sex that did not come off disk this run.
_SEX: dict[str, dict] = {}


def load_sex(gm_root: str, subsets: dict[str, str]) -> dict[str, dict]:
    """Read `<gm>/<subset>/sex.json` for every subset. Fails loudly.

    Every failure mode here is fatal on purpose. A subset with no sidecar, an
    unparseable one, a sex outside {female, male}, or a sidecar whose
    `recordings` disagree with the recording the subset's own calib_params
    names -- each means the metadata does not describe this data, and the
    correct response is to stop rather than to guess. The recordings
    cross-check is the one that catches a sidecar copied to the wrong subset,
    which a sex value alone never would."""
    out = {}
    for sub, rec in sorted(subsets.items()):
        path = os.path.join(gm_root, sub, "sex.json")
        if not os.path.exists(path):
            raise SystemExit(
                f"FATAL: {path} is missing. Sex must travel with the data; "
                f"this build will not name-parse and will not assume unknown.")
        try:
            blob = json.load(open(path))
        except Exception as e:
            raise SystemExit(f"FATAL: {path} is unparseable: {e!r}")
        for key in ("subset", "sex", "sex_source", "recordings"):
            if key not in blob:
                raise SystemExit(f"FATAL: {path} has no {key!r}")
        if blob["sex"] not in VALID_SEX:
            raise SystemExit(f"FATAL: {path} sex={blob['sex']!r}, "
                             f"expected one of {VALID_SEX}")
        if blob["subset"] != sub:
            raise SystemExit(f"FATAL: {path} names subset {blob['subset']!r} "
                             f"but sits in {sub!r} -- sidecar in the wrong place")
        if rec not in blob["recordings"]:
            raise SystemExit(
                f"FATAL: {path} lists recordings {blob['recordings']} but "
                f"{sub}/calib_params names {rec!r} -- this sidecar describes "
                f"different footage")
        out[sub] = blob
    _SEX.clear()
    _SEX.update(out)
    n_f = sum(1 for b in out.values() if b["sex"] == "female")
    print(f"sex.json: {len(out)} subsets, {n_f} female / {len(out) - n_f} male")
    return out


def subset_sex(subset: str) -> str:
    """Sex of the animal a subset labels, from its sex.json. Hard error if
    load_sex() has not read that subset -- never a silent "unknown", which
    would drop the subset out of every per-sex metric AND out of the
    female-preservation rule in make_split without anything looking wrong."""
    if subset not in _SEX:
        raise KeyError(f"{subset}: sex.json not loaded. Call load_sex() first; "
                       f"do NOT infer sex from the images or the name.")
    return _SEX[subset]["sex"]


def subset_sex_source(subset: str) -> str:
    """Provenance string straight out of that subset's sex.json."""
    if subset not in _SEX:
        raise KeyError(subset)
    return _SEX[subset]["sex_source"]


def _kp_xy(ann):
    k = ann["keypoints"]
    return [(k[3 * i], k[3 * i + 1], k[3 * i + 2]) for i in range(len(k) // 3)]


def same_fly(a, b) -> float:
    """Median displacement over keypoints visible in BOTH annotations."""
    ka, kb = _kp_xy(a), _kp_xy(b)
    d = [((ka[i][0] - kb[i][0]) ** 2 + (ka[i][1] - kb[i][1]) ** 2) ** 0.5
         for i in range(min(len(ka), len(kb)))
         if ka[i][2] > 0 and kb[i][2] > 0]
    return statistics.median(d) if d else float("inf")


class _DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def load_general_model(gm_root: str):
    """Every (subset, split) annotation file, flattened.

    Returns `records`: one entry per source image, and `subsets`: subset ->
    recording. Image paths are `<subset>/<split>/<file_name>` -- verified
    against the real tree (19,929/19,929 resolve)."""
    records, subsets, ann_paths = [], {}, []
    for sub in sorted(os.listdir(gm_root)):
        cp = os.path.join(gm_root, sub, "calib_params")
        if not os.path.isdir(cp):
            continue
        recs = sorted(os.listdir(cp))
        if len(recs) != 1:
            raise ValueError(f"{sub}: expected one recording in calib_params, got {recs}")
        subsets[sub] = recs[0]
        for split in ("train", "val"):
            p = os.path.join(gm_root, sub, "annotations", f"instances_{split}.json")
            if not os.path.exists(p):
                continue
            ann_paths.append(p)
            blob = json.load(open(p))
            by_img = collections.defaultdict(list)
            for a in blob["annotations"]:
                by_img[a["image_id"]].append(a)
            fs_of_img = {}
            for key, fsv in blob.get("framesets", {}).items():
                for iid in fsv["frames"]:
                    fs_of_img[iid] = (sub, split, key)
            for im in blob["images"]:
                rec, cam, fname = im["file_name"].split("/")
                records.append({
                    "subset": sub, "split": split, "recording": rec, "camera": cam,
                    "frame": int(fname.split("_")[1].split(".")[0]),
                    "path": os.path.join(gm_root, sub, split, im["file_name"]),
                    "width": im["width"], "height": im["height"],
                    "kp_names": blob.get("keypoint_names", []),
                    "skeleton": blob.get("skeleton", []),
                    "anns": sorted(by_img.get(im["id"], []), key=lambda a: a["id"]),
                    "frameset": fs_of_img.get(im["id"]),
                })
    return records, subsets, sorted(set(ann_paths))


def composition(out_root: str, out_images, calib_groups: dict) -> dict:
    """Per-subset / per-sex / per-calibration-group composition, derived from
    the instances_{train,val}.json actually on disk."""
    img_meta = {im["id"]: im for im in out_images}
    per_subset, per_sex, per_group, per_src = {}, {}, {}, {}
    totals = {}
    for side in ("train", "val"):
        blob = json.load(open(os.path.join(out_root, "annotations",
                                           f"instances_{side}.json")))
        ids = {im["id"] for im in blob["images"]}
        # image-level sex: the set of sexes annotated on that image.
        img_sex = collections.defaultdict(set)
        for a in blob["annotations"]:
            img_sex[a["image_id"]].add(a["sex"])
            for d, k in ((per_subset, a["subset"]), (per_sex, a["sex"]),
                         (per_src, a["sex_source"])):
                d.setdefault(k, {}).setdefault(side, {"images": 0, "anns": 0})
                d[k][side]["anns"] += 1
            g = calib_groups.get(img_meta[a["image_id"]]["recording"], "?")
            per_group.setdefault(g, {}).setdefault(side, {"images": 0, "anns": 0})
            per_group[g][side]["anns"] += 1
        # images are counted once per subset/group they carry an annotation
        # for; an image with two flies counts for BOTH subsets, which is why
        # per-subset images can sum above the unique-image total.
        seen = collections.defaultdict(set)
        for a in blob["annotations"]:
            seen[("subset", a["subset"])].add(a["image_id"])
            seen[("src", a["sex_source"])].add(a["image_id"])
            seen[("group", calib_groups.get(
                img_meta[a["image_id"]]["recording"], "?"))].add(a["image_id"])
        for (kind, k), v in seen.items():
            d = {"subset": per_subset, "src": per_src, "group": per_group}[kind]
            d[k][side]["images"] = len(v)
        for k, v in img_sex.items():
            lab = "mixed" if len(v) > 1 else next(iter(v))
            per_sex.setdefault(lab, {}).setdefault(side, {"images": 0, "anns": 0})
            per_sex[lab][side]["images"] += 1
        totals[side] = {
            "images": len(ids),
            "images_with_annotation": len(img_sex),
            "annotations": len(blob["annotations"]),
            "framesets": len(blob["framesets"]),
        }
        by_img_fly = collections.defaultdict(set)
        for a in blob["annotations"]:
            by_img_fly[a["image_id"]].add(a["fly_id"])
        totals[side]["two_fly_images"] = sum(1 for v in by_img_fly.values()
                                             if len(v) >= 2)
    n = totals["train"]["images"] + totals["val"]["images"]
    totals["val_image_frac"] = round(totals["val"]["images"] / max(n, 1), 4)
    na = totals["train"]["annotations"] + totals["val"]["annotations"]
    totals["val_annotation_frac"] = round(totals["val"]["annotations"] / max(na, 1), 4)
    fem = per_sex.get("female", {})
    fem_all = sum(fem.get(x, {}).get("anns", 0) for x in ("train", "val"))
    totals["female_annotation_share"] = round(fem_all / max(na, 1), 4)
    return {"totals": totals, "per_subset": per_subset, "per_sex": per_sex,
            "per_calib_group": per_group, "per_sex_source": per_src,
            "female_budget": {
                "female_anns_train": fem.get("train", {}).get("anns", 0),
                "female_anns_val": fem.get("val", {}).get("anns", 0),
                "female_val_share_of_all_female": round(
                    fem.get("val", {}).get("anns", 0) / max(fem_all, 1), 4)}}


def _cell(d, side):
    v = d.get(side, {})
    return f"{v.get('images', 0):6d}/{v.get('anns', 0):<6d}"


def print_composition(comp: dict) -> None:
    t = comp["totals"]
    print("\n==== COMPOSITION (images/annotations, from the shipped jsons) ====")
    for title, key in (("per subset", "per_subset"), ("per sex", "per_sex"),
                       ("per calib group", "per_calib_group"),
                       ("per sex_source", "per_sex_source")):
        print(f"\n-- {title} --")
        print(f"{'':28s} {'TRAIN img/ann':>14s} {'VAL img/ann':>14s}")
        for k in sorted(comp[key]):
            print(f"{k:28s} {_cell(comp[key][k], 'train'):>14s} "
                  f"{_cell(comp[key][k], 'val'):>14s}")
    print(f"\ntrain {t['train']['images']:6d} img  {t['train']['annotations']:6d} ann  "
          f"{t['train']['framesets']:5d} fs  {t['train']['two_fly_images']:5d} two-fly img")
    print(f"val   {t['val']['images']:6d} img  {t['val']['annotations']:6d} ann  "
          f"{t['val']['framesets']:5d} fs  {t['val']['two_fly_images']:5d} two-fly img")
    print(f"val fraction: {t['val_image_frac']:.3%} of images, "
          f"{t['val_annotation_frac']:.3%} of annotations")
    print(f"female share of all annotations: {t['female_annotation_share']:.3%}")
    fb = comp["female_budget"]
    print(f"female annotations: train {fb['female_anns_train']}, "
          f"val {fb['female_anns_val']} "
          f"({fb['female_val_share_of_all_female']:.1%} of all female)")


def check_capture_groups(split: dict, framesets: dict,
                         used_recs: list, groups: list | None = None) -> list:
    """Assert every CAPTURE_GROUPS member landed on the SAME side.

    Checked against the split that was actually produced, not against the
    policy that was intended: two recordings sitting in TRAIN_COMPONENTS_FORCE
    is not evidence that they ended up together, and the whole point of a
    capture group is that no content hash can catch the mistake if they did
    not. Raises rather than warns -- a leak of this class cost 54% of the v8
    val set and nothing in the numbers showed it."""
    if groups is None:
        groups = CAPTURE_GROUPS
    used = set(used_recs)
    report = []
    for grp in groups:
        if not (set(grp["recordings"]) & used):
            continue                      # group entirely absent: nothing to enforce
        missing = [r for r in grp["recordings"] if r not in used]
        if missing:
            raise SystemExit(
                f"FATAL: CAPTURE_GROUPS {grp['name']!r} names recording(s) "
                f"{missing}, which this build does not contain while it DOES "
                f"contain the rest of the group. A partially present capture "
                f"group cannot be enforced.")
        sides = collections.defaultdict(set)
        for k, v in split.items():
            rec = framesets[k]["recording"]
            if rec in grp["recordings"]:
                sides[rec].add(v)
        allsides = set().union(*sides.values()) if sides else set()
        if len(allsides) > 1:
            raise SystemExit(
                f"FATAL: capture group {grp['name']!r} is SPLIT across "
                f"{sorted(allsides)}: {dict(sides)}. These recordings are "
                f"different frames of ONE video; putting them on opposite "
                f"sides is the courtship_V2/V3/V4 leak, which content hashing "
                f"cannot see. Fix the policy, do not relax this check.")
        report.append({"name": grp["name"], "recordings": grp["recordings"],
                       "side": sorted(allsides), "evidence": grp["evidence"]})
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hash-cache", default=None)
    ap.add_argument("--workers", type=int, default=192)
    ap.add_argument("--female-val-frac", type=float, default=0.10)
    ap.add_argument("--guard", type=int, default=200)
    args = ap.parse_args()

    records, subsets, ann_paths = load_general_model(args.gm)
    print(f"{len(subsets)} subsets, {len(records)} source image references")
    # Read sex BEFORE anything else touches it, so a bad sidecar stops the
    # build instead of producing a root whose per-sex metrics are wrong.
    sex_meta = load_sex(args.gm, subsets)

    # ---- content identity -------------------------------------------------
    cache = {}
    if args.hash_cache and os.path.exists(args.hash_cache):
        cache = json.load(open(args.hash_cache))
    real = {r["path"]: os.path.realpath(r["path"]) for r in records}
    todo = sorted({p for p in real.values() if p not in cache})
    if todo:
        print(f"hashing {len(todo)} files ...", flush=True)
        cache.update(hash_paths(todo, workers=args.workers))
        if args.hash_cache:
            json.dump(cache, open(args.hash_cache, "w"))
    for r in records:
        r["md5"] = cache[real[r["path"]]]
    contents = collections.defaultdict(list)
    for r in records:
        contents[r["md5"]].append(r)
    print(f"{len(records)} image refs -> {len(contents)} unique contents")

    # A content must be one camera of one frame, or the capture atom below is
    # not well defined. Verified 0/0 on the 2026-09-02 tree; asserted so a
    # future ingest cannot break it silently.
    for h, lst in contents.items():
        if len({r["camera"] for r in lst}) > 1 or len({r["frame"] for r in lst}) > 1:
            raise ValueError(f"content {h[:8]} spans several cameras/frames: "
                             f"{[(r['recording'], r['camera'], r['frame']) for r in lst]}")

    # ---- captures: source framesets joined by shared content ---------------
    # NOT joined on frame number. Counters free-run per session, so two
    # recordings can collide on a frame number while sharing zero bytes
    # (measured: min |dframe| = 4 between unrelated recordings). Byte identity
    # is the only evidence used.
    dsu = _DSU()
    for h, lst in contents.items():
        fs = [r["frameset"] for r in lst if r["frameset"]]
        for f in fs:
            dsu.find(f)
        for f in fs[1:]:
            dsu.union(fs[0], f)
    cap_members = collections.defaultdict(set)
    for r in records:
        if r["frameset"]:
            cap_members[dsu.find(r["frameset"])].add(r["md5"])
    print(f"{len(cap_members)} captures")

    # ---- canonical image per content ---------------------------------------
    kp_names = _canonical_keypoint_names(ann_paths)
    skeleton = next((r["skeleton"] for r in records if r["skeleton"]), [])
    canon = {}
    for h, lst in contents.items():
        pick = min(lst, key=lambda r: (r["recording"], r["subset"], r["split"]))
        canon[h] = pick
    order = sorted(contents, key=lambda h: (canon[h]["recording"], canon[h]["camera"],
                                            canon[h]["frame"]))
    img_id = {h: i for i, h in enumerate(order)}
    out_images = [{
        "id": img_id[h], "width": canon[h]["width"], "height": canon[h]["height"],
        "recording": canon[h]["recording"],
        "file_name": f"{canon[h]['recording']}/{canon[h]['camera']}/Frame_{canon[h]['frame']}.jpg",
        "content_md5": h,
    } for h in order]

    # ---- annotations + framesets -------------------------------------------
    out_anns, framesets = [], {}
    next_ann = 0
    n_ambiguous = 0
    ambiguous_log = []
    rec_slots = collections.defaultdict(set)
    for cap in sorted(cap_members, key=lambda c: (canon[min(cap_members[c])]["recording"],
                                                  canon[min(cap_members[c])]["frame"])):
        hs = cap_members[cap]
        cams = sorted({canon[h]["camera"] for h in hs})
        by_cam = {canon[h]["camera"]: h for h in hs}
        cap_rec = min(canon[h]["recording"] for h in hs)
        cap_frame = canon[next(iter(hs))]["frame"]
        slots = sorted({r["subset"] for h in hs for r in contents[h] if r["anns"]})
        if not slots:
            continue
        rec_slots[cap_rec].update(slots)
        frames = [img_id[by_cam[c]] for c in cams]

        # per camera, per slot: the one annotation that slot contributes
        picked = {}
        for c in cams:
            h = by_cam[c]
            for r in contents[h]:
                for a in r["anns"]:
                    if (c, r["subset"]) in picked:
                        raise ValueError(f"{r['subset']} has >1 annotation on "
                                         f"{r['recording']}/{c}/Frame_{r['frame']}")
                    picked[(c, r["subset"])] = (r, a)

        # R15, detected properly: two slots on one camera that land on the SAME
        # animal cannot be attributed to either. Absent for both.
        drop = set()
        for c in cams:
            here = [(s, picked[(c, s)]) for s in slots if (c, s) in picked]
            for i in range(len(here)):
                for j in range(i + 1, len(here)):
                    d = same_fly(here[i][1][1], here[j][1][1])
                    if d < SAME_FLY_PX:
                        drop.add((c, here[i][0]))
                        drop.add((c, here[j][0]))
                        n_ambiguous += 1
                        ambiguous_log.append({
                            "content_md5": by_cam[c], "camera": c,
                            "recording": cap_rec, "frame": cap_frame,
                            "subsets": [here[i][0], here[j][0]],
                            "median_kp_px": round(d, 2)})

        for k, sub in enumerate(slots):
            ann_ids = []
            for c in cams:
                if (c, sub) not in picked or (c, sub) in drop:
                    ann_ids.append(None)
                    continue
                r, a = picked[(c, sub)]
                remap = (None if r["kp_names"] == kp_names
                         else _keypoint_remap(r["kp_names"], kp_names))
                out_anns.append({
                    "id": next_ann, "image_id": img_id[by_cam[c]],
                    "bbox": a["bbox"],
                    "keypoints": (a["keypoints"] if remap is None
                                  else _pad_keypoints(a["keypoints"], remap)),
                    "num_keypoints": a.get("num_keypoints", len(kp_names)),
                    "sex": subset_sex(sub),
                    "sex_source": subset_sex_source(sub),
                    "behavior": "unknown",     # see BEHAVIOR comment at module top
                    "fly_id": k, "subset": sub,
                    "src_ann_id": a["id"],     # general_model subset id space (masks)
                })
                ann_ids.append(next_ann)
                next_ann += 1
            if sum(a is not None for a in ann_ids) < MIN_CAMS:
                continue
            framesets[f"{cap_rec}/Frame_{cap_frame}/fly{k}"] = {
                "recording": cap_rec, "fly_id": k, "subset": sub,
                "frames": frames, "ann_ids": ann_ids}

    print(f"{len(out_anns)} annotations, {len(framesets)} fly-samples; "
          f"{n_ambiguous} camera/slot pairs recorded ABSENT because two subsets "
          f"labelled the same animal")

    merged = {"keypoint_names": kp_names, "skeleton": skeleton,
              "categories": [{"id": 1, "name": "fly", "num_keypoints": len(kp_names)}],
              "images": out_images, "annotations": out_anns, "framesets": framesets}

    # ---- manifest ----------------------------------------------------------
    rec_subsets = collections.defaultdict(set)
    for sub, rec in subsets.items():
        rec_subsets[rec].add(sub)
    calib_of = {rec: os.path.join(args.gm, sorted(rec_subsets[rec])[0],
                                  "calib_params", rec)
                for rec in rec_subsets}
    used_recs = sorted({im["recording"] for im in out_images})
    groups = group_calibrations({r: calib_of[r] for r in used_recs})
    calib_out = os.path.join(args.out, "calibrations")
    for rec, grp in groups.items():
        dst = os.path.join(calib_out, grp)
        if os.path.isdir(dst):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(calib_of[rec], CAM_GLOB))):
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))

    man = {"version": os.path.basename(args.out.rstrip("/")),
           "source_root": args.gm,
           "calib_groups": sorted(set(groups.values())), "recordings": {}}
    for rec in used_recs:
        slots = sorted(rec_slots[rec])
        sexes = {s: subset_sex(s) for s in slots}
        fly_sex = {f"fly{k}": sexes[s] for k, s in enumerate(slots)}
        srcs = sorted({subset_sex_source(s) for s in slots})
        # A recording-level `sex` is only meaningful when every fly in it
        # agrees; a merged courtship pair has one of each and is "mixed", with
        # the real per-animal answer in fly_sex (which _is_female_fly reads).
        uniq = set(sexes.values())
        sex = next(iter(uniq)) if len(uniq) == 1 else "mixed"
        src = "+".join(srcs)
        entry = {"subset": "+".join(slots), "calib_group": groups[rec],
                 "n_framesets": len({k for k in framesets if k.startswith(f"{rec}/")}),
                 "n_flies": len(slots),
                 "behavior": "+".join(sorted({BEHAVIOR.get(s, "unknown") for s in slots})),
                 "sex": sex, "sex_source": src, "has_masks": False, "split": None}
        # fly_sex is what split_v5._is_female_fly consults; write it ALWAYS,
        # not only when the slots disagree. Writing it only on disagreement
        # made the female rule depend on a recording-level fallback that is
        # "mixed" for exactly the two-fly captures that matter most.
        entry["fly_sex"] = fly_sex
        man["recordings"][rec] = entry

    # ---- media -------------------------------------------------------------
    # 19k symlinks on gpfs are metadata-latency bound, not CPU bound -- the
    # same reason hash_paths threads. Serial with a per-file makedirs+lexists
    # measured ~8 links/s (40 min for this tree); mkdir once per
    # (recording, camera) plus 64 threads brings it to seconds.
    # FileExistsError IS the idempotency guard, so there is no extra stat.
    def _link(job):
        src, dst = job
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass

    img_jobs, mask_jobs = [], []
    for h in order:
        r = canon[h]
        img_jobs.append((os.path.realpath(r["path"]),
                         os.path.join(args.out, "images", r["recording"],
                                      r["camera"], f"Frame_{r['frame']}.jpg")))
        src = os.path.join(args.gm, r["subset"], "sam3_masks", r["split"],
                           r["recording"], r["camera"], f"Frame_{r['frame']}.npz")
        if os.path.exists(src):
            mask_jobs.append((os.path.realpath(src),
                              os.path.join(args.out, "masks", r["recording"],
                                           r["camera"], f"Frame_{r['frame']}.npz")))
            man["recordings"][r["recording"]]["has_masks"] = True
    for jobs in (img_jobs, mask_jobs):
        for d in {os.path.dirname(dst) for _, dst in jobs}:
            os.makedirs(d, exist_ok=True)
        with ThreadPoolExecutor(max_workers=64) as ex:
            list(ex.map(_link, jobs))
    n_masks = len(mask_jobs)
    print(f"linked {len(img_jobs)} images, {n_masks} masks", flush=True)

    # ---- split -------------------------------------------------------------
    path_hash = {os.path.realpath(r["path"]): r["md5"] for r in records}
    path_rec = {os.path.realpath(r["path"]): r["recording"] for r in records}
    aliases = alias_components(path_hash, lambda p: path_rec[p])
    # after the content merge every image already carries its canonical
    # recording, so aliases collapse -- keep them anyway: holding a component
    # together is exactly what makes a partial holdout impossible.
    aliases = {canon_rec: aliases.get(canon_rec, canon_rec) for canon_rec in used_recs}
    image_hashes = {im["id"]: im["content_md5"] for im in out_images}

    # A policy name that does not resolve to a real component would silently
    # do NOTHING: make_split falls back to `aliases.get(r, r)`, which yields a
    # component id that matches no frameset, and the intended holdout just
    # never happens. Every prior defect in this dataset was a silent one, so
    # this is checked rather than trusted. The pairs merge under their
    # alphabetically-first recording, so a policy MUST name that one.
    for label, names in (("VAL_RECORDINGS", VAL_RECORDINGS),
                         ("VAL_COMPONENTS_FORCE", VAL_COMPONENTS_FORCE),
                         ("TRAIN_COMPONENTS_FORCE", TRAIN_COMPONENTS_FORCE)):
        for r in names:
            if r not in aliases:
                raise SystemExit(
                    f"FATAL: {label} names {r!r}, which is not a canonical "
                    f"recording of this build. After the content merge the "
                    f"surviving names are:\n  " + "\n  ".join(sorted(aliases)))
    comps = {r: aliases[r] for r in
             list(VAL_RECORDINGS) + list(VAL_COMPONENTS_FORCE)
             + list(TRAIN_COMPONENTS_FORCE)}
    print("policy -> component:", json.dumps(comps, indent=2))

    split = make_split(merged, man, val_recordings=VAL_RECORDINGS,
                       val_components_force=VAL_COMPONENTS_FORCE,
                       train_components_force=TRAIN_COMPONENTS_FORCE,
                       female_val_frac=args.female_val_frac, guard=args.guard,
                       aliases=aliases)
    cap_group_report = check_capture_groups(split, framesets, used_recs)
    for row in cap_group_report:
        print(f"capture group OK: {row['name']} -> {row['side']}")

    audit = audit_split(merged, split, guard=args.guard, aliases=aliases,
                        image_hashes=image_hashes)
    print("audit:", json.dumps({k: v for k, v in audit.items()
                                if not k.startswith("leaky")
                                and k != "min_guard_distance"}, indent=2))
    if (audit["cross_fly_leaked_frames"] or audit["cross_capture_leaks"]
            or audit["content_overlap"]):
        raise SystemExit(f"REFUSING to write a leaky split: {audit}")

    write_derived(merged, split, args.out, image_hashes=image_hashes)
    with open(os.path.join(args.out, "annotations", "instances.json"), "w") as f:
        json.dump(merged, f)

    for rec in man["recordings"]:
        man["recordings"][rec]["split"] = None
    for k, v in split.items():
        rec = framesets[k]["recording"]
        cur = man["recordings"][rec].get("split")
        man["recordings"][rec]["split"] = v if cur in (None, v) else "mixed"
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)

    groups_c = content_groups(path_hash)
    flies_on_image = collections.defaultdict(set)
    for a in out_anns:
        flies_on_image[a["image_id"]].add(a["fly_id"])
    n_two = sum(1 for v in flies_on_image.values() if len(v) >= 2)

    # ---- composition, computed from the SHIPPED jsons ----------------------
    # Read back what was written rather than what was intended: every prior
    # defect in this dataset survived because a build reported its plan.
    comp = composition(args.out, out_images, groups)
    print_composition(comp)
    with open(os.path.join(args.out, "content_index.json"), "w") as f:
        json.dump({"n_image_refs": len(records), "n_unique_content": len(contents),
                   "duplicate_groups": {h: sorted(v) for h, v in groups_c.items()
                                        if len(v) > 1}}, f, indent=1)
    with open(os.path.join(args.out, "build_report.json"), "w") as f:
        json.dump({
            "source": "general_model ONLY (red_data_unified_V3 deliberately excluded)",
            "n_subsets": len(subsets), "n_recordings": len(used_recs),
            "n_image_refs": len(records), "n_unique_content": len(contents),
            "n_captures": len(cap_members), "n_annotations": len(out_anns),
            "n_fly_samples": len(framesets),
            "two_fly_images": n_two,
            "two_fly_deficit_vs_v3_root": {
                "v3_based_root_two_fly_images": 1673,
                "this_root_two_fly_images": n_two,
                "lost_annotations": 2321,
                "lost_second_fly_annotations": 2014,
                "v3_only_contents": 791,
                "source_recordings": ["2026_04_07_11_33_33", "2026_04_08_14_59_45"],
                "note": "V3 is the only source of the second animal's labels on "
                        "the courtship_V2 capture. Dropping it costs those labels."},
            "sex": {"read_from": "<general_model>/<subset>/sex.json",
                    "sidecars": {k: {"sex": v["sex"],
                                     "sex_source": v["sex_source"],
                                     "citation": v.get("citation")}
                                 for k, v in sorted(sex_meta.items())},
                    "n_female_subsets": sum(1 for v in sex_meta.values()
                                            if v["sex"] == "female"),
                    "n_male_subsets": sum(1 for v in sex_meta.values()
                                          if v["sex"] == "male"),
                    "note": "No general_model annotation carries a sex field "
                            "(checked, all 19,382). Sex is read from the "
                            "per-subset sidecar that travels with the data; "
                            "there is no table in the build script and no "
                            "name-parsing fallback. Nothing is inferred from "
                            "an image; body size is explicitly NOT used."},
            "composition": comp,
            "split_policy": {
                "ignored_general_model_own_splits": True,
                "holdout_unit": "alias component (byte-identity), atom = whole "
                                "capture (all cameras of a frame, all flies)",
                "capture_groups": cap_group_report,
                "val_whole_components": VAL_RECORDINGS,
                "val_components_force_female_inclusive": VAL_COMPONENTS_FORCE,
                "train_components_force": TRAIN_COMPONENTS_FORCE,
                "female_tail_rule": "female-inclusive components not named "
                                    "above give their last `female_val_frac` "
                                    "of frames to val behind a `guard`-frame "
                                    "band; same-session, so OPTIMISTIC",
                "controlled_sex_contrast": {
                    "regime": "headless",
                    "female": "headless_22_50_female (val, unseen session)",
                    "male": "headless_24_04_1_male (val, unseen session)",
                    "why": "the only female-vs-male comparison in this split "
                           "that is not confounded by behaviour"},
                "female_val_is_two_tiered": {
                    "unseen_session": ["courtship_11_50_female",
                                       "courtship_25_51_female",
                                       "headless_22_50_female"],
                    "same_session_optimistic": ["female (guarded tail)"],
                    "why": "never average these into one headline number"},
                "regimes_with_no_val_coverage": [
                    "grooming (1 component, 4,592 img -- whole holdout would "
                    "erase 24% of the corpus's only grooming session)",
                    "general male S6male (1 component)",
                    "amputation S8/S9 (kept in train; not a pipeline eval target)",
                    "female climbing (1 component, 15 fs)",
                    "wall, MALE (one capture group: wall_frames + "
                    "wall_frames_15_06_46_male, ~7 + 28 viable framesets) -- "
                    "see TRAIN_COMPONENTS_FORCE and CAPTURE_GROUPS; judge "
                    "GT-free. It was described as FEMALE wall through v9; the "
                    "user corrected the sex on 2026-09-02 and the source "
                    "recording is male climbing."],
            },
            "ambiguous_same_animal_slots": n_ambiguous,
            "ambiguous_examples": ambiguous_log[:20],
            "split_audit": audit,
            "val_recordings": VAL_RECORDINGS,
            "val_components_force": VAL_COMPONENTS_FORCE,
            "guard": args.guard, "female_val_frac": args.female_val_frac,
        }, f, indent=2)
    print(f"two-fly images in this root: {n_two}  (V3-based root: 1673)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
