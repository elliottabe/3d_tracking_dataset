# Session-level driver for the mvq route (2026-09-05)

New: `scripts/slurm/mvq_session_pipeline.sh` (session driver),
`scripts/slurm/mvq_local_lift.sh` (worker loop, shared), and
`scripts/session_collect.py` (the `collect` stage). The driver **composes**
`scripts/slurm_bout_array.py` once per recording and adds one CPU-only
session job gated on every recording's aggregate:

```
lift[ts] -> precompute[ts] -> ik[ts] -> aggregate[ts]     (per recording)
                                           aggregate[*] -> collect
```

`precompute`'s `afterok` is on the WHOLE lift array (a single `--array=` job
id, so all tasks must succeed), confirmed in the dry run below:
`mvqlift_Session1 --array=1,…,11` then `ctprecomp_Session1 #SBATCH
--dependency=afterok:<mvq_lift_JOBID>`. That ordering is deliberate: the
per-fly body scale and per-fly marker offsets are pooled over the whole
recording, and seeding them from one bout measured 15.5 % low (the
scale-from-first-bout defect).

## Real dry run (nothing submitted)

```
bash scripts/slurm/mvq_session_pipeline.sh --dry-run \
  --session /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1 \
  --run-name pose_mvq_p3b_dry
```

exit 0; `grep -c '^Submitted' = 0`; no `pose_mvq_p3b_dry/` dir created; no
collect job in `squeue`. 4 of the 14 Session1 folders were skipped for having
no `sam3_masks.npz` (`11_52_43`, `13_03_01`, `15_12_14`, `17_13_56`); the other
10 (130 bouts) printed a full chain:

```
skip Session1/2026_04_02_11_52_43 (no bout_*/sam3_masks.npz under …/sam3_masks)
skip Session1/2026_04_02_13_03_01 …
skip Session1/2026_04_02_15_12_14 …
skip Session1/2026_04_02_17_13_56 …
submitted 10 recording chain(s) (Session1, pose_mvq_p3b_dry, mvq=p3b)

Dependency graph (dry-run, nothing submitted):
  lift[2026_04_02_12_11_50] -> precompute[…] -> ik[…] -> aggregate[…]   (11 bouts)
  lift[2026_04_02_14_54_28] -> precompute[…] -> ik[…] -> aggregate[…]   (19 bouts)
  lift[2026_04_02_15_25_51] -> precompute[…] -> ik[…] -> aggregate[…]   (30 bouts)
  lift[2026_04_02_15_44_42] -> precompute[…] -> ik[…] -> aggregate[…]   ( 8 bouts)
  lift[2026_04_02_16_03_48] -> precompute[…] -> ik[…] -> aggregate[…]   ( 7 bouts)
  lift[2026_04_02_16_21_32] -> precompute[…] -> ik[…] -> aggregate[…]   (11 bouts)
  lift[2026_04_02_16_39_56] -> precompute[…] -> ik[…] -> aggregate[…]   (18 bouts)
  lift[2026_04_02_16_56_37] -> precompute[…] -> ik[…] -> aggregate[…]   ( 8 bouts)
  lift[2026_04_02_17_28_34] -> precompute[…] -> ik[…] -> aggregate[…]   (15 bouts)
  lift[2026_04_02_17_52_50] -> precompute[…] -> ik[…] -> aggregate[…]   ( 3 bouts)
  aggregate[*] -> collect   (10 aggregate ids -> mvqcollect_Session1)
Monitor : squeue -u $USER
```

(each graph line carries the real job ids after the arrows; under `--dry-run`
they are `<stage_JOBID>` placeholders. Bout counts added here from the
`=== Session1 / <ts> : N bouts with masks ===` lines.)

The collect job is CPU-only on the same slurm config: `--gpus=0`, `--mem=16G`,
`--time=1:00:00`, and **no `--constraint`** — `ckpt_all`'s
`h200|a100|l40s|l40|a40` exists to keep the GPU stages off cards that are too
small, and carrying it would pin a pure-python JSON roll-up to a GPU node.

## `session_collect.py` on real runs

`Session0/pose_mvq_p3a_r2` (complete, 30 bouts / 60 bout-flies / 28 447 frames):
LOO median 0.77 px, p90 3.66 px, reproj median 2.15 px, IoU hard median 0.024,
**female missing 22.7 % vs male 0.7 %**, 0 bouts unsolvable, 0 bout-flies
without `stac_ik.h5`. The female/male asymmetry is the same one §1 of
`pipeline-audit-2026-09-05.md` measured (fly1 99.3 % of frames fully finite vs
fly0 77.3 %) — the collect table surfaces it per recording without re-deriving
anything.

Run against the *in-flight* `pose_mvq_p3b` root (lift done, no IK yet) it
reports 12 bouts, 0 bout-flies solved, 24 bout-flies missing `stac_ik.h5`,
containment-dropped 0.78 % — i.e. a partial run is a readable row, not a crash.
`p3a_r2` predates the containment gate and its cell is `-`, not `0.00`.

## Follow-up: the campaign wrapper is NOT yet a thin caller

`scripts/slurm/mvq_p3a_campaign.sh` was left byte-identical. A P3b campaign
(`--run-name pose_mvq_p3b --local-gpus 8`, pid 1896160) was running from this
checkout the whole time, and bash reads a script incrementally — editing the
file underneath a live `bash script.sh` corrupts its parse position. The
factored worker loop in `scripts/slurm/mvq_local_lift.sh` is a verbatim
extraction of the wrapper's (plus two glob fixes, below), so the switch is
mechanical once no `pgrep -f mvq_p3a_campaign.sh` matches:

- replace the wrapper's `RECORDINGS=(…)` loop body with a call to
  `scripts/slurm/mvq_session_pipeline.sh --session <VID>/Session{0,1}`, keeping
  `--only`, `--run-name`, `--local-gpus`, `--dry-run` pass-through;
- delete its `read_mvq_cfg`, `guard_pids` and inline worker loop in favour of
  `mvq_read_yaml`, `mvq_guard_pids`, `mvq_local_lift`.

Two defects found while extracting that loop, fixed in the shared helper only
(the wrapper still has them, and they are harmless there because it never sets
`nullglob`): `ls -d <glob>` with an unmatched glob under `nullglob` has **no
operand**, so it prints `.` — which the old `sed` pipeline turned into a phantom
bout, and the old `ls … | wc -l` kp3d count turned into a fake non-zero. Both
are now guarded shell loops. `tests/test_mvq_session_pipeline.py` covers the
first (a recording with a `bout_*` dir but no `sam3_masks.npz` must be skipped).
