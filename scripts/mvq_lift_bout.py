#!/usr/bin/env python
"""Lift SAM3-detected courtship bouts to pipeline keypoints with the mvq model.

The front half of the mvq pipeline route (spec
`docs/specs/2026-09-04-mvq-maskfree-frontend-design.md`, Task 7): for every
bout that already has SAM3 masks, place one 448-px multi-view window per fly
(merged into one when the two flies are within 3 mm), run the typed-slot mvq
lifter UNPROMPTED, and write the result in the pipeline's own bout-dir format:

    <out>/bouts/bout_<idx:05d>/fly0/{kp2d,kp3d}.npz      fly0 = FEMALE slot
    <out>/bouts/bout_<idx:05d>/fly1/{kp2d,kp3d}.npz      fly1 = MALE slot
    <out>/bouts/bout_<idx:05d>/{sex.json,mvq_meta.json}

`scripts/run_bout.py --config-name=pipeline pipeline.lifter=mvq mvq=p3a
outputs.out=<out>` then continues from those files: Stage A and Stage B are
skipped (their artifacts exist), the Stage-B gate signature is the mvq
checkpoint's rather than the DLT gates', and the keypoint filter -> body-scale
precompute -> STAC IK -> polish -> viz chain runs unchanged.

WHAT THE MASKS ARE USED FOR (`--identity`, default `mask`). The masks always
place the CROP (their per-camera centroids, triangulated). With
`--identity mask` they also ASSIGN IDENTITY: fly0 is mask fly 0 and fly1 is
mask fly 1 -- the human id review the canonicalized masks carry, which is the
top of this pipeline's identity precedence -- and per frame each mvq instance
is matched to whichever mask centre it is NEARER (by at least
`--mask-assign-margin-units`, and within `--mask-assign-max-units`), never to
an absolute radius: the female's SAM mask on 20_04 is poor enough that its
centre sits 13-35 units off her body while the other mask is ~50 away, so a
radius rejects a correct, unambiguous detection. `sex.json` then says
`method = "mask_human_id_review"`.

With `--identity sex` the masks do NOT assign identity: fly0 is the model's
FEMALE typed slot and fly1 its MALE one, per frame, from the network's own sex
head, and `sex.json` says `method = "mvq_sex_head"`. That was the original P3a
behaviour, and it is why the default changed: on 2025_10_20_13_20_04 the sex
head types the female as a male, which NaN'd fly0 on 40% of the recording and
let the male track jump onto her body on 2.5% of frames (see
`.superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b/female-miss-diagnosis.md`).
Either way `sexing.canonicalize_bout` treats the written sex.json as
authoritative instead of re-deciding the bout from a wing-song CV.

The masks do one more job under `--identity mask`: `--containment on` (the
default) runs the per-keypoint MASK-CONTAINMENT FILTER after the identity
assignment, which NaNs the individual keypoints of one fly that reproject
inside the OTHER fly's mask (and outside their own, dilated) in at least
`--containment-min-views` of the cameras where both masks are valid. Identity
is decided per FRAME, so it cannot see the residual failure it fixes: in
contact frames -- especially where the female's slot is empty -- the male
instance absorbs PART of her body (bout 1 frame 364: 17 male keypoints on the
female). The instance is mostly right, so the frame must not be dropped; only
those keypoints are. It is part of the gate string.

ORDER DISCIPLINE (CLAUDE.md). The written keypoint axis is
`cfg.model.KP_NAMES` (`--anatomy`), permuted from the model's own detector
order BY NAME, and asserted plus checked against the rigid EyeL-EyeR spacing.
The camera axis is the recording config's canonical order, which the mask npz
is permuted into by name and which `MVQRunner` asserts equals the calibration
glob order.

Usage:

    PYTHONPATH=third_party/jarvis_jax:. python scripts/mvq_lift_bout.py \\
        --session-dir  <video>/courtship/Session0/2025_10_20_13_20_04 \\
        --predictions-dir <processed>/.../sam3_masks \\
        --out          <processed>/.../pose_mvq_p3a \\
        --run          <jax_mvq_runs>/mvq_t1_b16_p3a_20260904/final \\
        --bout 28                                   # or --all

Idempotent: a bout whose kp3d.npz already carries this checkpoint's gates
string is skipped (`--force` re-runs it), so a preempted array task resumes
cleanly and a re-submitted array costs nothing.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax"))
sys.path.insert(0, ROOT)

from omegaconf import OmegaConf  # noqa: E402

from jarvis_jax.predict.synced_reader import load_plan, read_window  # noqa: E402
from jarvis_jax.tracking.lift_mvq import (BoutMaskStore, MVQRunner,  # noqa: E402
                                          bout_centres_3d, bout_lift_is_current,
                                          lift_masked_bout, mvq_gate_string,
                                          resolve_bout_frames, resolve_mask_identity)
from jarvis_jax.tracking.sexing import (load_review, read_sex_meta,  # noqa: E402
                                        review_key_for)

DEF_RECORDING_CFG = os.path.join(ROOT, "configs", "recording", "session0.yaml")
DEF_ANATOMY_CFG = os.path.join(ROOT, "configs", "anatomy", "v1.yaml")


def discover_bouts(predictions_dir):
    """Sorted bout indices with a sam3_masks.npz (mirrors
    `slurm_bout_array.bout_indices`, but a bout dir with no npz is not a bout
    this script can lift)."""
    idxs = []
    for d in sorted(glob.glob(os.path.join(predictions_dir, "bout_*"))):
        m = re.match(r"bout_(\d+)$", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "sam3_masks.npz")):
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-dir", required=True, help="recording dir (mp4s + calibration)")
    p.add_argument("--predictions-dir", default=None,
                   help="SAM3 mask tree (default <session-dir>/../sam3_masks is NOT "
                        "assumed -- pass the processed-tree path explicitly)")
    p.add_argument("--out", required=True,
                   help="pipeline run root; bouts land in <out>/bouts/bout_<idx:05d>")
    p.add_argument("--bout", type=int, action="append", default=[],
                   help="bout index (repeatable)")
    p.add_argument("--all", action="store_true",
                   help="every bout under --predictions-dir that has masks")
    p.add_argument("--run", required=True,
                   help="mvq RUN dir (with --step) or a final/ dir (without)")
    p.add_argument("--step", default=None,
                   help="load ckpt/<step> instead of final/. A CONCRETE int: 'latest' "
                        "names different weights on every save, so it cannot identify "
                        "a checkpoint in the Stage-B gate signature")
    p.add_argument("--exist-thresh", type=float, default=0.5,
                   help="typed-slot existence threshold; below it the fly's frame is "
                        "NaN (no fallback to an untyped slot). Part of the gate string")
    p.add_argument("--batch", type=int, default=8, help="windows per forward")
    p.add_argument("--merge-dist-units", type=float, default=30.0,
                   help="two flies closer than this (world units, 30 == 3 mm) share "
                        "one crop -- the model's own two-instance case")
    p.add_argument("--identity", choices=("mask", "sex"), default="mask",
                   help="who each written fly is. 'mask' (default): fly{f} IS SAM3 "
                        "mask fly f, per the HUMAN id review the masks carry -- the "
                        "model only says which instance sits on which mask. 'sex': "
                        "fly0/fly1 are the model's female/male typed slots. Part of "
                        "the gate string. A bout whose masks carry no human review "
                        "falls back to 'sex' with a warning")
    p.add_argument("--mask-assign-margin-units", dest="mask_assign_margin_units",
                   type=float, default=8.0,
                   help="--identity mask only: how much NEARER its own mask's "
                        "triangulated centre an instance must be than the other "
                        "mask's to be assigned by geometry (world units, 8 == 0.8 mm). "
                        "Below the margin the frame is ambiguous and geometry abstains "
                        "(the typed slot may then be used -- see assign_reason)")
    p.add_argument("--containment", choices=("on", "off"), default="on",
                   help="per-keypoint mask-containment filter (default on). NaNs a "
                        "keypoint that reprojects INSIDE the other fly's mask and "
                        "OUTSIDE its own (dilated) in --containment-min-views of the "
                        "cameras where both masks are valid, plus a conservative "
                        "temporal spike rule. Needs the masks' human id review, so it "
                        "is a no-op (and gated off) wherever --identity resolves to "
                        "'sex'. Part of the gate string")
    p.add_argument("--containment-min-views", dest="containment_min_views", type=int,
                   default=3,
                   help="cameras that must agree before a keypoint is called 'on the "
                        "other fly'. Recorded in mvq_meta.json, NOT in the gate string")
    p.add_argument("--containment-own-margin-px", dest="containment_own_margin_px",
                   type=float, default=6.0,
                   help="dilation of the OWN mask before the outside-own test, px. "
                        "Legs extend past a SAM mask, so this errs toward keeping")
    p.add_argument("--containment-max-step-units", dest="containment_max_step_units",
                   type=float, default=5.0,
                   help="per-keypoint frame-to-frame 3D step (world units; 5 == 0.5 mm "
                        "in 1/800 s) above which the keypoint is SUSPECT in both "
                        "frames -- dropped only if it also fails containment in "
                        "either, or the step is above twice this (a pure spike)")
    p.add_argument("--window-pref", dest="window_pref", choices=("own", "any"),
                   default="own",
                   help="--identity mask only: which window a fly is read from when "
                        "several hold it. 'own' (default) prefers the window centred "
                        "on that fly's OWN mask and falls back to the others only "
                        "when it has no candidate -- the partner's crop puts half the "
                        "fly at the edge and drives its conf3d to ~0.1, which gates "
                        "it out of the offsets sampler, the rigid repair and the "
                        "display. 'any' is the pre-2026-09-05 rule. In the gate "
                        "signature")
    p.add_argument("--mask-assign-max-units", dest="mask_assign_max_units",
                   type=float, default=60.0,
                   help="--identity mask only: garbage cap on that distance (world "
                        "units). The 448-px window's half-width is ~28 units, so 60 "
                        "excludes only instances that are in neither mask's crop")
    p.add_argument("--attn-impl", default=None,
                   help="override the run's attn_impl ('xla' to run cudnn weights on CPU)")
    p.add_argument("--cameras", default=None,
                   help="comma-separated CANONICAL camera order (default: the "
                        "recording config's)")
    p.add_argument("--recording-cfg", default=DEF_RECORDING_CFG,
                   help="hydra recording config that defines the canonical camera order")
    p.add_argument("--calib-dir", default=None, help="default <session-dir>/calibration")
    p.add_argument("--anatomy", default=DEF_ANATOMY_CFG,
                   help="anatomy config defining model.KP_NAMES -- the keypoint order "
                        "the written npz files speak")
    p.add_argument("--bouts-csv", default=None,
                   help="bout summary CSV; tried before the session's own (Session0's "
                        "unified CSV is a broken symlink, so its per-fly CSVs are used)")
    p.add_argument("--review", default=None,
                   help="id_review manifest, recorded in mvq_meta.json for comparison "
                        "only -- the typed slots decide identity here")
    p.add_argument("--force", action="store_true",
                   help="re-lift bouts whose kp3d.npz already carries these gates")
    p.add_argument("--n", type=int, default=0,
                   help="DEBUG: lift only the first N frames of each bout. The result "
                        "has T != the mask npz's T, so it is NOT ingestible by "
                        "run_bout.py -- use a scratch --out")
    p.add_argument("--progress-every", type=int, default=200,
                   help="frames between progress lines (0 = only the per-bout summary)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.bout and not args.all:
        raise SystemExit("give --bout <id> (repeatable) or --all")
    predictions_dir = args.predictions_dir
    if predictions_dir is None:
        raise SystemExit("--predictions-dir is required (the SAM3 mask tree)")

    cameras = ([c.strip() for c in args.cameras.split(",") if c.strip()]
               if args.cameras else
               [str(c) for c in OmegaConf.load(args.recording_cfg).cameras])
    model_names = [str(n) for n in OmegaConf.load(args.anatomy).model.KP_NAMES]
    calib_dir = args.calib_dir or os.path.join(args.session_dir, "calibration")

    bouts = discover_bouts(predictions_dir) if args.all else sorted(set(args.bout))
    if not bouts:
        raise SystemExit(f"no bouts with sam3_masks.npz under {predictions_dir}")

    # The runner owns the checkpoint, the calibration and the window geometry;
    # its constructor is where the canonical-vs-glob camera-order assertion
    # lives. Built ONCE for the whole invocation -- loading DINOv3-B per bout
    # would dominate the runtime of a --all pass.
    runner = MVQRunner(args.run, step=args.step, attn_impl=args.attn_impl,
                       calib_dir=calib_dir, cameras=cameras, batch=int(args.batch),
                       exist_thresh=float(args.exist_thresh), identity=args.identity,
                       containment=args.containment, window_pref=args.window_pref)
    print(f"[mvq-lift] checkpoint {runner.checkpoint} step {runner.step_label}; "
          f"K={runner.K} slots={runner.I} exist_thresh={runner.exist_thresh} "
          f"identity={runner.identity}; "
          f"unrestored={runner.meta.get('_unrestored_leaves', [])}", flush=True)
    print(f"[mvq-lift] gates (identity={runner.identity}, "
          f"containment={runner.containment}, window_pref={runner.window_pref}) "
          f"{runner.gates_string()}", flush=True)
    print(f"[mvq-lift] {len(bouts)} bout(s): {bouts}", flush=True)

    review = load_review(args.review) if args.review else {}
    plan = load_plan(args.session_dir)
    n_done = n_skipped = 0
    for bout in bouts:
        out_dir = os.path.join(args.out, "bouts", f"bout_{bout:05d}")
        mask_npz = os.path.join(predictions_dir, f"bout_{bout:05d}", "sam3_masks.npz")
        # The masks' sex_meta decides whether `--identity mask` can be honoured
        # for THIS bout, so it is read (cheap -- one small npz member) BEFORE
        # the skip check: gating a fallback bout as "mask" would let a re-lift
        # be skipped after a human review is canonicalized into its masks.
        mask_sex_meta = read_sex_meta(mask_npz)
        identity, fallback = resolve_mask_identity(args.identity, mask_sex_meta,
                                                   where=f"bout {bout}")
        if fallback:
            print(f"[mvq-lift] WARNING {fallback}", flush=True)
        gates_string = mvq_gate_string(runner.checkpoint, step=runner.step,
                                       exist_thresh=runner.exist_thresh,
                                       identity=identity,
                                       containment=args.containment,
                                       window_pref=args.window_pref)
        if not args.force and bout_lift_is_current(out_dir, gates_string):
            print(f"[mvq-lift] bout {bout}: skip (kp3d.npz already carries these gates, "
                  f"identity {identity})", flush=True)
            n_skipped += 1
            continue
        t0_wall = time.time()
        store = BoutMaskStore(mask_npz, cameras)
        abs_start, abs_end, n_csv = resolve_bout_frames(args.session_dir, bout,
                                                        bouts_csv=args.bouts_csv)
        if n_csv != store.T:
            raise RuntimeError(f"bout {bout}: the bouts CSV says {n_csv} frames "
                               f"({abs_start}..{abs_end}) but {mask_npz} has {store.T}")
        n = store.T if int(args.n) <= 0 else min(int(args.n), store.T)
        if n != store.T:
            print(f"[mvq-lift] WARNING bout {bout}: --n {args.n} truncates to {n} of "
                  f"{store.T} frames; the result is NOT ingestible by run_bout.py",
                  flush=True)
        centres, ok = bout_centres_3d(store, runner.cam_mats, n, 0)

        key = review_key_for(out_dir, pose_dir=os.path.basename(os.path.normpath(args.out)))
        entry = review.get(key) if key else None
        review_male = None if not entry else entry.get("reviewed_male_fly")

        res = lift_masked_bout(
            runner, read_window(args.session_dir, cameras, plan, abs_start, n),
            centres, ok, out_dir=out_dir, model_names=model_names,
            merge_dist_units=float(args.merge_dist_units),
            identity=args.identity, window_pref=args.window_pref,
            mask_assign_margin_units=float(args.mask_assign_margin_units),
            mask_assign_max_units=float(args.mask_assign_max_units),
            mask_store=store, containment=args.containment,
            containment_min_views=int(args.containment_min_views),
            containment_own_margin_px=float(args.containment_own_margin_px),
            containment_max_step_units=float(args.containment_max_step_units),
            force=args.force,
            progress_every=int(args.progress_every),
            mask_sex_meta=mask_sex_meta, review_male_fly=review_male,
            meta_extra={"bout": int(bout), "session_dir": str(args.session_dir),
                        "masks_npz": mask_npz, "bouts_csv": args.bouts_csv,
                        "frame_start": int(abs_start),
                        "bout_frame_range": [int(abs_start), int(abs_end)],
                        "review_key": key})
        el = time.time() - t0_wall
        print(f"[mvq-lift] bout {bout}: {n} frames in {el:.0f}s "
              f"({n / max(el, 1e-9):.1f} frames/s) -> {out_dir}", flush=True)
        n_done += 1
    print(f"[mvq-lift] done: {n_done} lifted, {n_skipped} already current", flush=True)


if __name__ == "__main__":
    main()
