#!/usr/bin/env python3
"""
Submit `scripts/coarse_pass.py` as a SLURM array over its windows, instead of
one process sweeping them sequentially.

The windows are already independent by design (coarse_pass.py's own module
docstring: each window gets a FRESH SAM3 detection + `assign_identities` +
`canonicalize_male_fly` -- cross-window identity comes from the absolute
male=fly1 sexing signal, not boundary stitching), and `coarse_pass.py --mode
window --window-index N` already computes exactly one window and writes only
its `parts/window_NNNN.npz` (see that script's docstring). This submitter
does the scheduling, nothing else -- the SAM3/gate logic is unchanged.

    [array, 1 task/window]  --afterok-->  [collect, 1 task, CPU-only]
     coarse_pass.py --mode window          coarse_pass.py --mode collect
     (SAM3, GPU, queue)                     (numpy merge; refuses loudly if
                                              any window's part is missing)

Same total camera-frame cost as today's sequential sweep, but wall clock is
now roughly ONE window's SAM3 time instead of N windows' worth -- the same
win today's per-bout SAM3 array already gets over a sequential bout sweep
(see scripts/slurm_bout_array.py). Measured on real Session0 video: a
1000-coarse-frame, 7-camera window takes on the order of an hour end-to-end
(propagation ~4-4.5 it/s/camera plus a real per-camera extraction/session
overhead beyond the propagation loop itself -- see the coarse-pass-phase1.md
report for the measured breakdown), so `--time` defaults generously.

Companion to scripts/slurm_bout_array.py / scripts/slurm_dir_array.py (same
sbatch-script-builder + `sbatch --parsable` pattern), but plain argparse
rather than Hydra: coarse_pass.py itself takes plain CLI flags, no
recording/outputs config group to compose.

Usage:
    # See both scripts without submitting:
    python scripts/slurm_coarse_pass_array.py \\
        --session-dir /gscratch/.../Session0/2025_10_20_13_20_04 \\
        --out /gscratch/.../coarse_pass/coarse_tracks.npz --dry-run

    # Real submit:
    python scripts/slurm_coarse_pass_array.py \\
        --session-dir /gscratch/.../Session0/2025_10_20_13_20_04 \\
        --out /gscratch/.../coarse_pass/coarse_tracks.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"


# ---------------------------------------------------------------------------
# Window-count discovery (CPU-only, no SAM3/GPU -- see coarse_pass.py's
# module-level imports: sam3_driver's own top-level imports are torch-free,
# torch/SAM3 are lazily imported inside run()/build_tracker(), same
# precedent as scripts/slurm_bout_array.py's submit-time `ensure_sync_plan`
# import). Reuses coarse_pass.window_starts + video_frame_count_and_size
# directly rather than reimplementing the window grid math.
# ---------------------------------------------------------------------------

def n_windows_for(session_dir: str, *, stride: int, chunk_len: int, chunk_overlap: int,
                   start_frame: int, end_frame: int | None) -> int:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import coarse_pass as cp  # noqa: E402 -- see module docstring
    mp4s = sorted(glob.glob(os.path.join(session_dir, "*.mp4")))
    if not mp4s:
        raise FileNotFoundError(f"no *.mp4 found under {session_dir}")
    n_full, _, _ = cp.video_frame_count_and_size(mp4s[0])
    end = end_frame if end_frame is not None else n_full
    n_coarse = (end - start_frame) // stride
    return len(cp.window_starts(n_coarse, chunk_len, chunk_overlap))


# ---------------------------------------------------------------------------
# Pure sbatch-script builders (mirrors slurm_bout_array.py / slurm_dir_array.py)
# ---------------------------------------------------------------------------

def _array_spec(n_windows: int, max_concurrent: int | None = None) -> str:
    spec = f"0-{n_windows - 1}"
    if max_concurrent:
        spec += f"%{max_concurrent}"
    return spec


def _coarse_pass_cli(*, session_dir, out_path, stride, chunk_len, chunk_overlap,
                      num_animals, start_frame, end_frame, project) -> str:
    """The CLI args shared by every mode invocation of coarse_pass.py."""
    end_arg = f" --end-frame {end_frame}" if end_frame is not None else ""
    return (f"--session-dir {session_dir} --out {out_path} --stride {stride} "
            f"--chunk-len {chunk_len} --chunk-overlap {chunk_overlap} "
            f"--num-animals {num_animals} --start-frame {start_frame}{end_arg} "
            f"--project {project}")


def build_window_array_script(
    *, job_name: str, partition: str, account: str, cpus: int, mem: int, gpus: int,
    time_limit: str, requeue: bool, constraint: str, conda_env: str, n_windows: int,
    max_concurrent: int | None, out_dir: str, coarse_pass_cli: str,
) -> str:
    """One array task per window index [0, n_windows): `coarse_pass.py --mode
    window --window-index $SLURM_ARRAY_TASK_ID`. `resume` (coarse_pass.py's
    default) makes a requeued/re-run task a no-op if its part file already
    exists, so this is safe to resubmit."""
    requeue_line = "#SBATCH --requeue" if requeue else ""
    constraint_line = f"#SBATCH --constraint={constraint}" if constraint else ""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={gpus}
#SBATCH --mem={mem}G
#SBATCH --array={_array_spec(n_windows, max_concurrent)}
#SBATCH --open-mode=append
#SBATCH -o {out_dir}/slurm-coarsewin-%A_%a.out
{requeue_line}
{constraint_line}
set -x
source ~/.bashrc
micromamba activate {conda_env}
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Node: $SLURMD_NODENAME  job: $SLURM_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
cd {PROJECT_DIR}
python -u scripts/coarse_pass.py {coarse_pass_cli} --sam3-compile false \\
    --mode window --window-index ${{SLURM_ARRAY_TASK_ID}}
"""


def build_collect_script(
    *, job_name: str, partition: str, account: str, cpus: int, mem: int, time_limit: str,
    conda_env: str, out_dir: str, coarse_pass_cli: str, dependency: str,
) -> str:
    """Dependent (afterok on the whole array), CPU-only, no --gpus/--constraint
    (mirrors slurm_bout_array.py's build_aggregate_script / slurm_dir_array.py's
    build_finalize_script pattern for a cheap post-array reduction step).
    `coarse_pass.py --mode collect` itself refuses loudly (exits non-zero) if
    any window's part file is missing, so this job's own exit code is a
    correct completeness signal -- it does not need its own extra check."""
    dependency_line = f"#SBATCH --dependency={dependency}" if dependency else ""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --open-mode=append
#SBATCH -o {out_dir}/slurm-coarsecollect-%j.out
{dependency_line}
set -x
source ~/.bashrc
micromamba activate {conda_env}
cd {PROJECT_DIR}
python -u scripts/coarse_pass.py {coarse_pass_cli} --mode collect
"""


# ---------------------------------------------------------------------------
# Submission plumbing
# ---------------------------------------------------------------------------

def slurm_submit(script: str, *, dependency: str | None = None) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    try:
        out = subprocess.check_output(cmd, input=script, universal_newlines=True)
        return out.strip().split(";")[0].split()[-1]
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--out", required=True, help="output coarse_tracks.npz path")
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--chunk-len", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=120)
    p.add_argument("--num-animals", type=int, default=2)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None,
                   help="exclusive; default = whole recording")
    p.add_argument("--project", default="unified_V3_masked")
    p.add_argument("--partition", default="ckpt-all")
    p.add_argument("--account", default="portia")
    p.add_argument("--constraint", default="h200|a100|l40s|l40|a40",
                   help="GPU constraint for the window array (excludes GPUs too "
                        "small for SAM3 -- same set as configs/slurm/ckpt_all.yaml)")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", type=int, default=64, help="GB, window array tasks")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--time", default="03:00:00",
                   help="per-window-task time limit; measured real per-window "
                        "SAM3 time on Session0 is on the order of an hour "
                        "(7 cameras x ~10 min incl. extraction+propagation) "
                        "plus a one-time ~7-9 min model load -- see report")
    p.add_argument("--collect-time", default="00:20:00")
    p.add_argument("--collect-mem", type=int, default=16)
    p.add_argument("--conda-env", default="3d_tracking")
    p.add_argument("--max-concurrent", type=int, default=None,
                   help="cap on concurrently running array tasks (--array=...%%N)")
    p.add_argument("--requeue", dest="requeue", action="store_true", default=None)
    p.add_argument("--no-requeue", dest="requeue", action="store_false")
    p.add_argument("--dry-run", action="store_true",
                   help="print both scripts + dependency without submitting")
    args = p.parse_args()

    requeue = args.requeue
    if requeue is None:
        requeue = args.partition.startswith("ckpt")

    n_windows = n_windows_for(
        args.session_dir, stride=args.stride, chunk_len=args.chunk_len,
        chunk_overlap=args.chunk_overlap, start_frame=args.start_frame,
        end_frame=args.end_frame)
    if n_windows == 0:
        print("Error: 0 windows computed for this session/frame range", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.out) or "."
    tag = os.path.basename(os.path.normpath(args.session_dir))
    coarse_pass_cli = _coarse_pass_cli(
        session_dir=args.session_dir, out_path=args.out, stride=args.stride,
        chunk_len=args.chunk_len, chunk_overlap=args.chunk_overlap,
        num_animals=args.num_animals, start_frame=args.start_frame,
        end_frame=args.end_frame, project=args.project)

    print(f"Session : {tag}  ({args.session_dir})")
    print(f"Windows : {n_windows} (stride={args.stride}, chunk_len={args.chunk_len}, "
          f"chunk_overlap={args.chunk_overlap})")
    print(f"Out     : {args.out}")

    array_job_name = f"coarsewin_{tag}"[:60]
    array_script = build_window_array_script(
        job_name=array_job_name, partition=args.partition, account=args.account,
        cpus=args.cpus, mem=args.mem, gpus=args.gpus, time_limit=args.time,
        requeue=requeue, constraint=args.constraint, conda_env=args.conda_env,
        n_windows=n_windows, max_concurrent=args.max_concurrent, out_dir=out_dir,
        coarse_pass_cli=coarse_pass_cli)

    print(f"\n--- window array script{' (dry-run)' if args.dry_run else ''} ---")
    print(array_script)

    if args.dry_run:
        array_job_id = "<ARRAY_JOBID>"
    else:
        os.makedirs(out_dir, exist_ok=True)
        array_job_id = slurm_submit(array_script)
        print(f"Submitted window array: {array_job_id}  "
              f"(--array={_array_spec(n_windows, args.max_concurrent)})")

    dependency = f"afterok:{array_job_id}"
    collect_job_name = f"coarsecol_{tag}"[:60]
    collect_script = build_collect_script(
        job_name=collect_job_name, partition=args.partition, account=args.account,
        cpus=args.cpus, mem=args.collect_mem, time_limit=args.collect_time,
        conda_env=args.conda_env, out_dir=out_dir, coarse_pass_cli=coarse_pass_cli,
        dependency=dependency)

    print(f"\n--- collect script{' (dry-run)' if args.dry_run else ''} "
          f"(dependency={dependency}) ---")
    print(collect_script)

    if args.dry_run:
        print(f"\n[dry-run] would submit window array (--array="
              f"{_array_spec(n_windows, args.max_concurrent)}), then collect with "
              f"--dependency={dependency}. Submitting nothing.")
        return

    collect_job_id = slurm_submit(collect_script, dependency=dependency)
    print(f"Submitted collect: {collect_job_id}  (dependency={dependency})")
    print(f"\nDependency chain: window_array({array_job_id}) -> collect({collect_job_id})")
    print(f"Logs   : {out_dir}/slurm-coarsewin-{array_job_id}_*.out , "
          f"{out_dir}/slurm-coarsecollect-{collect_job_id}.out")
    print(f"Monitor: squeue -u $USER")


if __name__ == "__main__":
    main()
