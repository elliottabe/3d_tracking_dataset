#!/usr/bin/env python3
"""
Submit ONE predict job per recording under a SessionN/ directory.

Discovers the recording subdirs (those containing a calibration/ dir) and, for
each, submits the same 2-stage SAM3 -> JAX-predict job that
scripts/slurm_predict_session.py builds, writing into the processed tree
(processed/<dataset>/SessionN/<rec>/{sam3_masks,predictions}). Each job is
independently requeue-safe and reuse_masks-resumable, and uses the multi-GPU
SAM3 fan-out internally for its own bouts.

Usage:
    # Submit all recordings of Session1 with the run4 v2vNet:
    python scripts/slurm_predict_session_dir.py --run-name run4 \\
        --session-root /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1

    # A subset, and inspect without submitting:
    python scripts/slurm_predict_session_dir.py --run-name run4 \\
        --session-root .../Session1 --recordings 2026_04_02_11_52_43,2026_04_02_12_11_50 --dry-run
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
sys.path.insert(0, str(PROJECT_DIR / "scripts"))  # import the sibling launcher
import slurm_predict_session as sps


def find_recordings(session_root):
    """Immediate subdirs of session_root that contain a calibration/ dir."""
    out = []
    for name in sorted(os.listdir(session_root)):
        d = os.path.join(session_root, name)
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "calibration")):
            out.append(d)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-name', required=True, help='Trained v2vNet run_id (e.g. run4)')
    p.add_argument('--session-root', required=True,
                   help='A Video_recordings/<dataset>/SessionN directory')
    p.add_argument('--recordings', default=None,
                   help='Comma-separated recording names to include (default: all)')
    p.add_argument('--dataset', default=None,
                   help='Dataset name (default: derive from each recording path)')
    p.add_argument('--paths', default='hyak', help='Hydra paths config group (default: hyak)')
    p.add_argument('--slurm', default='ckpt_g2', help='Hydra slurm config group (default: ckpt_g2)')
    p.add_argument('--dry-run', action='store_true', help='Print each job script without submitting')
    args, passthrough = p.parse_known_args()

    if not PKG_DIR.is_dir():
        print(f"Error: jarvis_jax package not found at {PKG_DIR}", file=sys.stderr)
        sys.exit(1)

    cfg = sps.compose_cfg(args.paths, args.slurm, args.run_name, passthrough)
    from jarvis_jax.predict.paths_util import processed_dir_for, dataset_for
    sl = cfg.slurm

    recs = find_recordings(args.session_root)
    if args.recordings:
        keep = set(args.recordings.split(","))
        recs = [r for r in recs if os.path.basename(r) in keep]
    if not recs:
        print(f"No recordings (with calibration/) under {args.session_root}",
              file=sys.stderr)
        sys.exit(1)

    requeue_line = "#SBATCH --requeue" if sl.requeue else ""
    nodelist_line = (f"#SBATCH --nodelist={sl.nodelist}"
                     if getattr(sl, 'nodelist', None) else "")
    exclude_line = (f"#SBATCH --exclude={sl.exclude}"
                    if getattr(sl, 'exclude', None) else "")
    overrides_str = (" " + " ".join(passthrough)) if passthrough else ""

    print(f"Recordings: {len(recs)} under {args.session_root}")
    for session_dir in recs:
        dataset = args.dataset or dataset_for(session_dir)
        out_root = processed_dir_for(cfg.paths.processed_root, session_dir, dataset=dataset)
        masks_dir = f"{out_root}/sam3_masks"
        out = f"{out_root}/predictions"
        job_name = f"predsess_{os.path.basename(session_dir)}"[:60]
        script = sps.build_script(
            job_name=job_name, partition=sl.partition, account=sl.account,
            nodelist_line=nodelist_line, exclude_line=exclude_line,
            requeue_line=requeue_line, gpus=sl.gpus, cpus=sl.cpus, mem=sl.mem,
            time_limit=sl.time, conda_env=sl.conda_env, mail_user=sl.mail_user,
            pkg_dir=PKG_DIR, log_dir=out_root, run_name=args.run_name,
            paths=args.paths, session_dir=session_dir, masks_dir=masks_dir,
            out=out, overrides_str=overrides_str)
        print(f"\n=== {os.path.basename(session_dir)} -> {out_root} ===")
        if args.dry_run:
            print(script)
            continue
        Path(out_root).mkdir(parents=True, exist_ok=True)
        jid = sps.slurm_submit(script)
        print(f"Submitted {job_name}: {jid}  (log: {out_root}/slurm-{jid}.out)")


if __name__ == "__main__":
    main()
