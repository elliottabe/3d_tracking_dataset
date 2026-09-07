# mvq v2 training -- handoff for a fresh session on another cluster

Written 2026-09-06 for whoever (human or Claude) launches or resumes the mvq v2
training somewhere other than Hyak. Everything referenced is in this repo at
commit 97f249f or later on branch `elliottabe/paper_update_062026`.

## What is being trained

`mvq` is a per-view DINOv3 ViT-B/16 + affine ray-token + query-decoder lifter
that emits 3D keypoints for the two courtship flies from 7 synchronized views.
**v2** is the from-scratch, T=2 (two-frame, Delta in {1,4,16}) version trained
on the human labels plus three pseudo-label exports, unprompted (typed slots
female/male, no mask prompt).

- Spec (binding): `docs/specs/2026-09-05-mvq-v2-pseudolabel-t2-design.md`
- Plans: `docs/plans/2026-09-05-mvq-v2-plan-a-pseudolabels.md` (data) and
  `docs/plans/2026-09-05-mvq-v2-plan-b-t2-training.md` (training)
- Evidence and every decision: `docs/benchmark/2026-09-mvq/v2-notes.md`,
  `docs/benchmark/2026-09-mvq/v2-pseudolabel-notes.md`, `p3b-notes.md`
- Config: `third_party/jarvis_jax/configs/train/mvq_v2.yaml` (values are
  commented with the spec section they come from)
- Trainer: `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py`,
  entry `python -m jarvis_jax.scripts.train_mvq`
- Loader: `jarvis_jax/data/v12_windows.py` (+ `concat_windows.py`,
  `loader_workers.py`), losses `jarvis_jax/train/losses_mvq.py`

## Data to copy (about 65 GB, no video needed)

| what | Hyak path (under /gscratch/portia/eabe/data/Johnson_lab/) | size | config key |
|---|---|---|---|
| human v12 labels, train+val | `red_data/red_data_3d_v12_export0902` | 2.3 GB | `paths.data_root` |
| p3b pseudo-labels (human-reviewed) | `red_data_3d_v12_pseudo_p3b_20260905` | 42 GB | `train.pseudo_root` |
| single-fly pseudo-labels | `red_data_3d_v12_pseudo_singlefly_20260905` | 2.7 GB | `train.singlefly_root` |
| empty-window negatives | `red_data_3d_v12_pseudo_negatives_20260905` | 11 GB | `train.negatives_root` |
| HF cache: DINOv3 (gated), SAM3 weights, HF token | `sam3/` | 7.2 GB | `HF_HOME` |

For comparison / acceptance also copy `jax_mvq_runs/mvq_t1_b16_p3b_contact_20260905/final`
(the P3b baseline, 0.085 mm on the val split).

## Install

Follow "Install and run on a new cluster" in `docs/running_the_pipeline.md`
and the Hyak-specific inventory in `docs/portability-checklist.md`. The short
version: conda env from `environment-3d_tracking.yml` (jax 0.11 needs
flax >= 0.12.8), `pip install -e third_party/jarvis_jax`, submodules
(`git submodule update --init --recursive`), the sibling `fruitfly_body_models`
clone is NOT needed for training (only for IK). Create
`third_party/jarvis_jax/configs/paths/<cluster>.yaml` from `hyak.yaml` and
point `data_root`, `mvq_runs_root` at the copied roots; edit the three absolute
`*_root` lines in `mvq_v2.yaml`.

Runtime env every GPU job needs (adapt the CUDA module name):

```bash
module load cuda/12.9.1
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6   # conda libstdc++ ABI
unset LD_LIBRARY_PATH JAX_PLATFORMS                    # some shells pin CPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 TF_GPU_ALLOCATOR=cuda_malloc_async
export HF_HOME=<copied sam3 dir> HF_TOKEN=             # empty -> uses $HF_HOME/token
```

## Launch

```bash
cd third_party/jarvis_jax
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
setsid nohup python -u -m jarvis_jax.scripts.train_mvq model=mvq train=mvq_v2 paths=<cluster> \
  run_id=mvq_t2_v2_<date> "paths.runs_root=\${paths.mvq_runs_root}" > ../../slurm_logs/mvq_t2_v2_<date>.out 2>&1 &
```

Before the real launch, run the 200-step loss-share check (same command with
`+share_check_steps=200`): `other_fly_repulsion` must be 5-30 % of the total
(7.8 % on Hyak), negatives mass must print 0.0500, and the per-root
female-host lines must show the solved multipliers.

Resource facts from Hyak (8x L40S 46 GB, 32 cores): batch 32 = 4 per GPU uses
42 GB per GPU; 1.45 s/step with 24 loader processes; 40k steps ~16 h; eval
and checkpoint every 2000 steps; resumes from `ckpt/` with the same run_id.
Fewer GPUs: `train.batch_size` must be a multiple of the device count
(24 on 6 GPUs worked; a 7-device layout hung once, avoid it). Gradient
accumulation is NOT implemented, so a smaller batch is a smaller effective
batch. Host RAM: each loader process holds ~2.3 GB (24 workers = 55 GB).

## Known traps (all hit on Hyak; fixes are in the code)

1. **8-GPU hang with one device idle** was the prefetcher sharding batches
   device-to-device (fixed in `jarvis_jax/data/prefetch.py`, commit e8d8446).
   If it recurs: `pip install py-spy; py-spy dump --pid <trainer>`.
2. **Loader starvation** (GPU util cycling 100 % / 0 %): the thread loader is
   GIL-bound; `train.loader_workers: processes` (default in mvq_v2.yaml) fixes
   it. Watch for `[loader] pool ... did not finish` at run end (an
   unreproduced pool-close hang; shutdown is bounded).
3. **Calibration mismatch warnings** for recordings 2026_04_02_15_25_51 and
   17_28_34 are expected (`allow_calib_mismatch: true`): the rig moved that day
   and the human labels (group A) and the pseudo-labels (video-folder
   calibration C) use different calibrations; each root is self-consistent.
   Details and the bundle-adjustment verdict: v2-notes.md "Calibration".
4. `manifest_disagreements` and the `[v12_windows] ... annotation sex WINS`
   lines at start-up are informational.
5. DINOv3 is HF-gated: the token must be in `$HF_HOME/token`; an exported
   non-empty `HF_TOKEN` that is stale overrides it and fails silently-ish.

## What "done" looks like

- Training curve reference (Hyak run `mvq_t2_v2_20260906d`, unprompted val,
  oracle mm): 0.356 @2k, 0.154 @4k, 0.101 @6k. P3b baseline 0.085 mm.
- Early-stop rule: unprompted val flat for 3 consecutive evals after 20k ->
  stop, promote the best checkpoint.
- After training: fit calibration temperatures
  (`scripts/benchmark/mvq_calibrate.py`, writes `final/mvq_run.json["calibration"]`),
  run the acceptance suite (`scripts/benchmark/mvq_v2_acceptance.py --run <dir> --all --gpu`,
  thresholds = spec section 5; on Hyak needs the bout data under `processed/`),
  then promote via a `configs/mvq/v2.yaml` (copy `p3b.yaml`, change the
  checkpoint) and re-lift the campaign with it.

## Conventions that bite

Keypoints and cameras are indexed BY NAME only (see CLAUDE.md); units are
0.1 mm; cameras are affine; the trainer's val split is inside the human v12
root (`instances_val.json`); never write into a live run's `final/`.
