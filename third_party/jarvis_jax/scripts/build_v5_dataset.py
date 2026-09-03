"""Build red_data_3d_v5 end to end.

    python scripts/build_v5_dataset.py paths=hyak

Idempotent: re-running relinks nothing that already exists and rewrites the
annotation/split JSON from scratch.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import hydra

from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers
from jarvis_jax.data.build_v5 import (
    discover_sources, build_manifest, link_media, merge_annotations)
from jarvis_jax.data.content_index import alias_components, hash_paths
from jarvis_jax.data.split_v5 import make_split, audit_split, write_derived

register_resolvers()

# Held out WHOLE. Chosen to resemble bout 28: a Group-A courtship recording
# plus one from each other calibration group so per-group val is reportable.
#
# 2026_04_07_11_33_33 added 2026-08-31 to fix a coverage hole: the val split
# above had ZERO two-fly frames, so every detector val number (overall px,
# female px, the v4 gain, the mask-channel ablation delta) was silently a
# single-fly number even though this detector family's documented weakness
# is precisely overlap. All four two-fly recordings had landed in train
# because none of them were in VAL_RECORDINGS. Of the four,
# 2026_04_08_14_59_45 holds 82% of all two-fly data and must stay in train;
# the 2026_06_11_13_58_43/_45 pair are ~2s apart and would have to move
# together or not at all (a worse, correlated choice); 2026_04_07_11_33_33
# is standalone and is the largest two-fly sample left (181 two-fly / 217
# frames) -- and configs/detector_finetune.yaml already lists it as a val
# recording, so holding it out here has precedent. Note: the per-annotation
# `sex` field is "unknown" for all four two-fly recordings, but sex is still
# resolvable per fly via manifest.json's recordings[rec]["fly_sex"]["fly<k>"]
# (see jarvis_jax.data.v5_2d._resolve_sex) -- for this recording fly0=male,
# fly1=female, so the added val slice IS sexed. See split_v5.py's docstring
# for why female-inclusive recordings ignore this list entirely regardless.
#
# 2026-09-02: THREE OF THESE FOUR ARE ALIASES OF A TRAINING RECORDING and this
# list, as written, produces a split with 29.6% of val byte-identical to train
# (see jarvis_jax.data.content_index). 2026_06_15_12_12_33 is the same footage
# as 2026_06_15_12_12_34; 2026_05_27_11_57_05 as 2026_05_27_11_56_05;
# 2026_04_07_11_33_33 as part of 2026_03_09_14_39_40. Only
# 2026_03_18_15_31_22 is clean. The `aliases=` argument below expands each
# name to its whole alias component so the holdout is all-or-nothing, and the
# content assertion in write_derived refuses the build outright if any pixels
# still straddle the split. Do not re-add a recording here without checking
# `content_index.alias_components` for what it drags with it.
VAL_RECORDINGS = [
    "2026_03_18_15_31_22",   # group A, courtship  (137 framesets)
    "2026_06_15_12_12_33",   # group B, courtship male
    "2026_05_27_11_57_05",   # group C, courtship male
    "2026_04_07_11_33_33",   # group B, two-fly (181/217 frames two-fly)
]


# RETIRED 2026-09-02. This builder is superseded by
# scripts/build_generalmodel_split.py and cannot run any more: both source
# roots it reads (red_data_unified_V3) and the root it writes
# (red_data_3d_v5) were deleted, and `paths.v3_root`/`paths.v5_root` were
# removed from configs/paths/hyak.yaml with them.
#
# It is kept rather than deleted because it is the record of how v5 was
# built, and three of its decisions are the ones that were reversed:
#   * `discover_sources` prefers V3's annotations over general_model's for
#     the 12 overlapping recordings -- the sourcing the user retired.
#   * `VAL_RECORDINGS` above holds out 2026_04_07_11_33_33, later shown to
#     be a slice of the capture now filed as 2025_10_20_13_20_04; holding it
#     out put 54% of v8's val on both sides of the split.
#   * it splits by RECORDING NAME, which cannot see either of the above.
# Reviving it means fixing those three things, not restoring two path keys.
_RETIRED = (
    "scripts/build_v5_dataset.py is retired: red_data_unified_V3 and "
    "red_data_3d_v5 were deleted on 2026-09-02 and paths.v3_root/paths.v5_root "
    "were removed with them. Use scripts/build_generalmodel_split.py, which "
    "builds from general_model only and splits on content + capture group "
    "rather than on recording name. See this module's comment for the three "
    "decisions that were reversed.")


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg):
    if "v5_root" not in cfg.paths or "v3_root" not in cfg.paths:
        raise SystemExit(_RETIRED)
    out = cfg.paths.v5_root
    os.makedirs(out, exist_ok=True)

    srcs = discover_sources(cfg.paths.general_model, cfg.paths.v3_root)
    print(f"discovered {len(srcs)} recordings")

    merged = merge_annotations(srcs, out)
    print(f"merged {len(merged['framesets'])} fly-samples, "
          f"{len(merged['annotations'])} annotations")

    man = build_manifest(srcs, out)
    link_media(srcs, out)

    # Content identity, resolved through the symlinks into the two upstream
    # corpora. This is the only thing that can see two recording NAMES for one
    # capture, and it is cheap: ~490 files/s at 192 threads on gpfs, so the
    # whole tree is about a minute.
    real = {im["id"]: os.path.realpath(os.path.join(out, "images", im["file_name"]))
            for im in merged["images"]}
    digest = hash_paths(sorted(set(real.values())))
    image_hashes = {i: digest[p] for i, p in real.items()}
    rec_of = {real[im["id"]]: im["recording"] for im in merged["images"]}
    aliases = alias_components({p: digest[p] for p in real.values()},
                               lambda p: rec_of[p])
    aliased = sorted((rec, comp) for rec, comp in aliases.items() if rec != comp)
    if aliased:
        print("alias components (same footage under a second name):")
        for rec, comp in aliased:
            print(f"    {rec} -> component {comp}")

    split = make_split(merged, man, val_recordings=VAL_RECORDINGS, aliases=aliases)
    audit = audit_split(merged, split, aliases=aliases, image_hashes=image_hashes)
    print("split audit:", json.dumps(audit, indent=2)[:400])
    if (audit["cross_recording_leaks"] or audit["cross_fly_leaked_frames"]
            or audit["cross_capture_leaks"] or audit["content_overlap"]):
        raise SystemExit(f"REFUSING to write a leaky split: {audit}")
    write_derived(merged, split, out, image_hashes=image_hashes)

    for k, v in split.items():
        man["recordings"][merged["framesets"][k]["recording"]]["split"] = (
            "mixed" if man["recordings"][merged["framesets"][k]["recording"]]
            .get("split") not in (None, v) else v)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    with open(os.path.join(out, "build_report.json"), "w") as f:
        json.dump({"n_recordings": len(srcs),
                   "n_fly_samples": len(merged["framesets"]),
                   "n_annotations": len(merged["annotations"]),
                   "split_audit": audit,
                   "val_recordings": VAL_RECORDINGS}, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
