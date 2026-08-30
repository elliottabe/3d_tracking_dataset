"""Launch the 8 Task-15 training arms, one per GPU, against the per-camera
heatmap cache (scripts/precompute_heatmap_cache.py +
jarvis_jax/train/train_3d_cached_hm.py -- see that module's docstring for
why this reads a HEATMAP cache rather than the pre-reprojected-VOLUME cache
scripts/precompute_repro_cache.py builds: rotation augmentation and
coarse-to-fine refinement are both properties of the reprojection step
itself, which a fixed-grid volume cache bakes away).

Each arm is a separate OS process (scripts/run_one_arm.py) restricted to one
GPU via CUDA_VISIBLE_DEVICES -- NOT a multi-device JAX mesh -- so this
launcher is a thin process pool, not a distributed-training driver.

RAMP 4 -> 8. A prior 8-way JAX run on a 128 GB cgroup died with
CUDA_ERROR_UNKNOWN; several agents hit RESOURCE_EXHAUSTED on this node
today. --max-parallel defaults to 4 and must be raised only after a 4-way
run is observed healthy (see docs/benchmark/2026-08-29-c2f-3d/phase3-arms.md
for what "healthy" was checked against on this run).
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

ARMS = ["A1_base", "A2_base_aug", "A3_c2f", "A4_c2f_aug", "A5_hires",
        "A6_c2f_aug_norecover", "A7_c2f_aug_femwt", "A8_c2f_aug_seed2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--v5-root", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--mem-fraction", type=float, default=0.9)
    # GPU index assigned to arms[i] is (gpu_offset + i) % 8. Needed because a
    # SECOND invocation of this launcher (the "ramp to 8" step, run after a
    # first batch is confirmed healthy) would otherwise enumerate its OWN
    # arm list from 0 and collide with GPUs the first invocation's arms are
    # still using -- e.g. --arms A1..A4 (offset 0 -> GPUs 0-3), then
    # --arms A5..A8 --gpu-offset 4 (-> GPUs 4-7).
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.runs_root, exist_ok=True)
    running = []
    for i, arm in enumerate(args.arms):
        gpu = (args.gpu_offset + i) % 8
        cmd = [sys.executable, "-u", os.path.join(_HERE, "run_one_arm.py"),
               "--arm", arm, "--cache-dir", args.cache_dir,
               "--v5-root", args.v5_root, "--runs-root", args.runs_root,
               "--steps", str(args.steps)]
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES=str(gpu),
                   XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.mem_fraction))
        print("LAUNCH", arm, "on GPU", gpu, " ".join(cmd))
        if args.dry_run:
            continue
        log_path = os.path.join(args.runs_root, f"v5_{arm}.log")
        log = open(log_path, "a")
        running.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=log))
        while len([p for p in running if p.poll() is None]) >= args.max_parallel:
            running[0].wait()
            running = [p for p in running if p.poll() is None]
    for p in running:
        p.wait()


if __name__ == "__main__":
    main()
