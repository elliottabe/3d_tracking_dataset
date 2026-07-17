"""One-off: canonicalize male -> fly1 across a whole processed session's bouts.

Runs on the login node (pure numpy; no GPU). Dry-run first to preview, then the
real pass performs the swaps and writes sex.json per bout.

  python scripts/canonicalize_session_sex.py recording=session0 ++dry_run=true
  python scripts/canonicalize_session_sex.py recording=session0
"""
import glob
import json
import os
from collections import Counter

import hydra
from omegaconf import DictConfig, OmegaConf

from jarvis_jax.tracking.sexing import canonicalize_bout, read_sex_meta, _swap_fly_dirs

# Register the `basename` OmegaConf resolver used by configs/outputs/default.yaml
# (out = .../${recording.name}/${basename:${recording.session_dir}}/pose). Same
# import-side-effect registration as scripts/run_bout.py / analyze_ik_error.py /
# slurm_bout_array.py; required for @hydra.main to compose configs/pipeline.yaml
# at all (cfg.outputs.out otherwise fails with UnsupportedInterpolationType).
OmegaConf.register_new_resolver(
    "basename", lambda p: os.path.basename(os.path.normpath(str(p))), replace=True)


def apply_manual_labels(run_root, labels, *, dry_run=False, male_slot=1):
    """Canonicalize male -> fly{male_slot} from explicit per-bout labels (current
    on-disk fly order). `labels` maps bout int -> male fly int. Idempotent: skips a
    bout whose sex.json is already method='manual'. Returns a list of result dicts."""
    out = []
    for bd in sorted(glob.glob(os.path.join(run_root, "bouts", "bout_*"))):
        bi = int(os.path.basename(bd).split("_")[1])
        if bi not in labels:
            continue
        sj = os.path.join(bd, "sex.json")
        if os.path.exists(sj):
            try:
                if json.load(open(sj)).get("method") == "manual":
                    print(f"bout {bi}: already manual, skip"); continue
            except (ValueError, OSError):
                pass
        male_cur = int(labels[bi])
        swap = male_cur != male_slot
        if swap and not dry_run:
            _swap_fly_dirs(bd)
        res = dict(male_fly=male_slot, original_male_fly=male_cur, applied_swap=swap,
                   confidence="user", method="manual", note="manual labels")
        if not dry_run:
            with open(sj, "w") as f:
                json.dump(res, f, indent=2)
        print(f"bout {bi}: male_current=fly{male_cur} -> swap={swap} -> male=fly{male_slot}")
        out.append(res)
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="pipeline")
def main(cfg: DictConfig):
    run_root = str(cfg.outputs.out)
    predictions_dir = str(cfg.recording.predictions_dir)
    dry = bool(cfg.get("dry_run", False))

    labels_path = cfg.get("labels", None)
    if labels_path:
        raw = json.load(open(str(labels_path)))
        labels = {int(k): int(v) for k, v in raw.items()}
        print(f"[canonicalize] manual labels from {labels_path} (dry_run={dry})")
        apply_manual_labels(run_root, labels, dry_run=dry)
        return

    sx = cfg.get("sexing", {}) or {}
    kp_names = list(cfg.model.KP_NAMES)

    bout_dirs = sorted(glob.glob(os.path.join(run_root, "bouts", "bout_*")))
    print(f"[canonicalize] {len(bout_dirs)} bouts under {run_root} (dry_run={dry})")
    print(f"{'bout':12s} {'male':5s} {'conf':8s} {'method':14s} {'swap':6s} "
          f"{'cv0':>7s} {'cv1':>7s} {'ratio':>6s}")

    conf_counts = Counter()
    unknown = []

    def fmt(x):
        return "    nan" if x is None else f"{x:7.3f}"

    for bd in bout_dirs:
        b = os.path.basename(bd)
        mask_npz = os.path.join(predictions_dir, b, "sam3_masks.npz")
        res = canonicalize_bout(
            bd, kp_names,
            mask_sex_meta=read_sex_meta(mask_npz),
            ratio_thr=float(sx.get("ratio_thr", 1.5)),
            high_ratio=float(sx.get("high_ratio", 2.5)),
            conf_min=float(sx.get("conf_min", 0.2)),
            min_frames=int(sx.get("min_frames", 20)),
            dry_run=dry)
        cv = res["wing_cv_original"]
        r = res["cv_ratio"]
        print(f"{b:12s} {str(res['male_fly']):5s} {res['confidence']:8s} "
              f"{res['method']:14s} {str(res['applied_swap']):6s} "
              f"{fmt(cv['fly0'])} {fmt(cv['fly1'])} "
              f"{('   nan' if r is None else f'{r:6.2f}')}")
        conf_counts[res["confidence"]] += 1
        if res["confidence"] == "unknown":
            unknown.append(b)

    print(f"\nconfidence counts: {dict(conf_counts)}")
    if unknown:
        print(f"UNKNOWN (manual review needed): {unknown}")


if __name__ == "__main__":
    main()
