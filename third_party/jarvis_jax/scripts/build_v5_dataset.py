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
from jarvis_jax.data.split_v5 import make_split, audit_split, write_derived

register_resolvers()

# Held out WHOLE. Chosen to resemble bout 28: a Group-A courtship recording
# plus one from each other calibration group so per-group val is reportable.
VAL_RECORDINGS = [
    "2026_03_18_15_31_22",   # group A, courtship  (137 framesets)
    "2026_06_15_12_12_33",   # group B, courtship male
    "2026_05_27_11_57_05",   # group C, courtship male
]


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg):
    out = cfg.paths.v5_root
    os.makedirs(out, exist_ok=True)

    srcs = discover_sources(cfg.paths.general_model, cfg.paths.v3_root)
    print(f"discovered {len(srcs)} recordings")

    merged = merge_annotations(srcs, out)
    print(f"merged {len(merged['framesets'])} fly-samples, "
          f"{len(merged['annotations'])} annotations")

    man = build_manifest(srcs, out)
    link_media(srcs, out)

    split = make_split(merged, man, val_recordings=VAL_RECORDINGS)
    audit = audit_split(merged, split)
    print("split audit:", json.dumps(audit, indent=2)[:400])
    if audit["cross_recording_leaks"] or audit["cross_fly_leaked_frames"]:
        raise SystemExit(f"REFUSING to write a leaky split: {audit}")
    write_derived(merged, split, out)

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
