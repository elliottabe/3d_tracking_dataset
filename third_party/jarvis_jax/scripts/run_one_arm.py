"""Run ONE Task-15 training arm against the heatmap cache
(jarvis_jax/train/train_3d_cached_hm.py). Invoked as a subprocess (one per
GPU) by scripts/train_arms.py -- kept a separate script (rather than a
function called in-process) so each arm gets its own fresh JAX/XLA process
and CUDA_VISIBLE_DEVICES restriction.

Usage:
    python scripts/run_one_arm.py --arm A4_c2f_aug --cache-dir $HMCACHE \
        --v5-root $V5 --runs-root $RUNS --steps 20000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from jarvis_jax.train.train_3d_cached_hm import HMCachedConfig, run_cached_hm_training

# Each entry is a dict of HMCachedConfig field overrides. See
# docs/benchmark/2026-08-29-c2f-3d/phase3-arms.md for what each arm tests.
ARMS = {
    "A1_base":              dict(rot_augment=False, refine_enabled=False),
    "A2_base_aug":          dict(rot_augment=True,  refine_enabled=False),
    "A3_c2f":               dict(rot_augment=False, refine_enabled=True),
    "A4_c2f_aug":           dict(rot_augment=True,  refine_enabled=True),
    "A5_hires":             dict(rot_augment=True,  refine_enabled=False,
                                 grid_size=96, grid_spacing=0.5, batch_size=4),
    "A6_c2f_aug_norecover": dict(rot_augment=True,  refine_enabled=True,
                                 exclude_second_fly=True),
    "A7_c2f_aug_femwt":     dict(rot_augment=True,  refine_enabled=True,
                                 female_weight=3.0),
    "A8_c2f_aug_seed2":     dict(rot_augment=True,  refine_enabled=True, seed=1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--v5-root", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    # Default raised from make_manager's own 3 -- that default pruned A7's
    # pre-divergence checkpoint before its NaN divergence could be
    # investigated (Task 15). Cheap in disk; buys forensics on any arm that
    # goes unstable later in a long resumed run.
    ap.add_argument("--max-ckpt-to-keep", type=int, default=10)
    args = ap.parse_args()

    overrides = dict(ARMS[args.arm])
    overrides.setdefault("batch_size", args.batch_size)
    cfg = HMCachedConfig(total_steps=args.steps, **overrides)

    run_dir = os.path.join(args.runs_root, f"v5_{args.arm}")
    out_dir = os.path.join(run_dir, "final")
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(run_dir, exist_ok=True)

    print(f"[run_one_arm] arm={args.arm} cfg={json.dumps(overrides)} "
          f"steps={args.steps} run_dir={run_dir}", flush=True)

    result = run_cached_hm_training(
        args.cache_dir, out_dir=out_dir, ckpt_dir=ckpt_dir, cfg=cfg,
        v5_root=args.v5_root, save_every=max(args.steps // 10, 1),
        log_every=50, eval_every=max(args.steps // 4, 1),
        max_ckpt_to_keep=args.max_ckpt_to_keep)

    print(f"[run_one_arm] {args.arm} DONE: {result}", flush=True)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
