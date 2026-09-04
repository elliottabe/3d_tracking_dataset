#!/usr/bin/env python
"""ViTPose+DLT baseline on the mvq v12 val split, for a like-for-like MPJPE
comparison against the mvq lifter (see docs/benchmark/2026-09-mvq/p2-t1-notes.md).

EXPECTATION (write before running): this is the "current pipeline" number mvq
has to beat. It runs the SAME v5vf_maskoff ViTPose checkpoint the free-running
pipeline uses (`configs/detector/vitpose_v3.yaml`, resolved with OmegaConf --
NOT hardcoded, so a config change here is a config change there too) on the
SAME v12 val framesets `V12WindowDataset` gives mvq, decodes 2D heatmap peaks,
and triangulates with the pipeline's own robust DLT gates (conf_thresh,
view_conf_thresh, reproj_resid_px). Ground truth is the dataset's own
DLT-of-human-2D-labels 3D (`kp3d_local`) -- 0 error by construction, so it is
not itself a baseline row; this script's number is what a competent
non-learned 3D lift gets INSTEAD of the human labels, from the SAME detector
2D. A `nan` cohort is a genuinely empty cohort (e.g. no group_C val
framesets), not a bug.

FAIRNESS (read before comparing numbers): the baseline's own MPJPE only
scores joints DLT could triangulate (>=2 confident views), silently
excluding the rest -- e.g. an all-occluded tarsus never enters its mean.
mvq's own reported val MPJPE (`final/mvq_run.json`) scores every
ground-truth joint, occluded or not. Comparing those two numbers directly
overstates the baseline: it is being graded on an easier, self-selected
subset. This script therefore ALSO restricts mvq's per-sample prediction to
exactly the same (sample, joint) pairs the baseline could triangulate, so
`mvq_same_joints_units` and `baseline_units` are a fair apples-to-apples
number; `mvq_all_joints_units` (copied straight from `final/mvq_run.json`,
no recomputation) is kept alongside it as the "official" mvq number, on the
harder full joint set.

No 4th (SAM-mask) channel is available at this granularity, so it is passed
as an all-zero channel -- EXACT for v5vf_maskoff, whose patch_embed kernel
weights for that channel are all identically 0.0 (verified in
tracking/predict_2d.py's `zero_mask_channel` docstring), so this is not an
approximation for this checkpoint.

Geometry: V12WindowDataset's M/t_local are CROP-LOCAL (uv_crop = M@X_local +
t_local, X_local = world - center3D), so both the ViTPose 2D and the
triangulated 3D stay in the same local frame as `kp3d_local` -- no round trip
through full-frame pixels is needed. `cam_mats_local[c]` is built to the exact
(4,3) = P.T convention `triangulate_dlt_batched` expects (row 2 = [0,0,0,1]):
cam_mats_local[:, :3, :2] = M.transpose(0, 2, 1); cam_mats_local[:, 3, :2] =
t_local; cam_mats_local[:, 3, 2] = 1.

TRAIN-SET-EXPOSURE CAVEAT (read before citing this baseline as "held-out"):
see `_train_exposure_note()` below and docs/benchmark/2026-09-mvq/p2-t1-notes.md
-- the detector was trained on a DIFFERENT root (red_data_3d_v5_valfix, since
deleted) than mvq's v12 val split, and this script cannot open that root to
check recording-level overlap directly. A known, well-documented failure mode
in this exact dataset lineage (docs/benchmark/2026-09-02-dataset-dedup/notes.md)
is byte-identical recordings filed under different session names landing on
opposite sides of a name-keyed split; this baseline's comparison is
PROVISIONAL until that is ruled out for v12's specific val recordings.
"""
import argparse, json, os, sys, time, warnings

import numpy as np
import jax, jax.numpy as jnp
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax")); sys.path.insert(0, ROOT)
from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.convert.build_checkpoint import load_vitpose
from jarvis_jax.data.device import normalize_image
from jarvis_jax.data.v12_windows import V12WindowDataset
from jarvis_jax.tracking.predict_2d import peaks_and_conf
from jarvis_jax.tracking.triangulate import triangulate_keypoints
from jarvis_jax.train.train_mvq import MM_PER_UNIT, normalize_crops

DEFAULT_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902"
# v5vf_maskoff's own training root -- deleted (2026-09-02 cleanup), kept here
# only so _train_exposure_note() can say plainly that it is gone.
DETECTOR_TRAIN_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5_valfix"
DETECTOR_RUN_DIR = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v5vf_maskoff"
VAL_RECORDINGS_CHECKED = ["2026_01_29_14_09_33", "2026_04_02_12_11_50", "2026_04_02_15_25_51",
                         "2026_06_09_15_21_14", "2026_06_09_15_46_55"]


def load_detector_cfg():
    """Resolve configs/detector/vitpose_v3.yaml against configs/paths/hyak.yaml
    with OmegaConf (NOT hardcoded), returning the plain dict of detector
    fields this script needs. Resolving only the `detector` subtree (not the
    whole merged config) sidesteps an unrelated interpolation
    (paths.data_dir -> ${dataset.name}) that has no value outside the full
    pipeline app and would otherwise fail OmegaConf.resolve()."""
    paths_cfg = OmegaConf.load(os.path.join(ROOT, "configs", "paths", "hyak.yaml"))
    det_cfg = OmegaConf.load(os.path.join(ROOT, "configs", "detector", "vitpose_v3.yaml"))
    cfg = OmegaConf.create({"paths": paths_cfg, "detector": det_cfg})
    return OmegaConf.to_container(cfg.detector, resolve=True)


def _train_exposure_note():
    """Best-effort check of whether v5vf_maskoff's OWN train/val split
    (which this baseline was never told about -- it just runs the trained
    checkpoint) could have included any of mvq's v12 val recordings. Returns
    a human-readable caveat string; never raises."""
    lines = []
    ov = os.path.join(DETECTOR_RUN_DIR, ".hydra", "overrides.yaml")
    if os.path.exists(ov):
        data_root = None
        for line in open(ov):
            line = line.strip().lstrip("- ").strip()
            if line.startswith("paths.data_root="):
                data_root = line.split("=", 1)[1].strip()
        lines.append(f"v5vf_maskoff trained on paths.data_root={data_root!r} "
                     f"(NOT v12's root); its own train/val split lived in "
                     f"{data_root}/annotations/*.json.")
    if not os.path.isdir(DETECTOR_TRAIN_ROOT):
        lines.append(f"{DETECTOR_TRAIN_ROOT} no longer exists on disk (deleted in the "
                     f"2026-09-02 dataset cleanup) -- cannot check recording-level overlap "
                     f"with v12's val recordings ({', '.join(VAL_RECORDINGS_CHECKED)}) directly.")
    lines.append(
        "KNOWN RISK (docs/benchmark/2026-09-02-dataset-dedup/notes.md): this exact dataset "
        "lineage's name-keyed splits have historically put byte-identical recordings (the "
        "same footage re-ingested under a different session name) on opposite sides of "
        "train/val -- e.g. a confirmed 'courtship_11_50_female'/'courtship_11_50_male' alias "
        "pair. v12's own val recording 2026_04_02_12_11_50 carries that same 'courtship_11_50' "
        "subset label; whether it is byte-identical to the aliased pair documented there "
        "(different calendar date, so NOT confirmed) could not be checked because the source "
        "root is deleted.")
    lines.append(
        "CAVEAT: this baseline's numbers below are therefore PROVISIONAL -- they may partly "
        "reflect the detector's own TRAIN-set performance rather than a genuinely held-out "
        "comparison. Treat the gap between mvq and this baseline as an upper bound on how "
        "much mvq currently trails, not a precise figure.")
    return " ".join(lines)


def cam_mats_local(M, t_local):
    """M (C,2,3), t_local (C,2) -> cam_mats_local (C,4,3), the (4,3)=P.T affine
    convention triangulate_dlt_batched expects (row 2 == [0,0,0,1])."""
    C = M.shape[0]
    cm = np.zeros((C, 4, 3), np.float64)
    cm[:, :3, :2] = np.transpose(M, (0, 2, 1))
    cm[:, 3, :2] = t_local
    cm[:, 3, 2] = 1.0
    return cm


def _load_mvq_model(final_dir):
    """Minimal mvq restore, mirroring scripts/viz/mvq_overlay.py::load_model
    (kept local/duplicated rather than imported from a sibling script, which
    Python does not treat as a normal importable module here)."""
    import orbax.checkpoint as ocp
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    meta = json.load(open(os.path.join(final_dir, "mvq_run.json")))
    cfg = MVQConfig(**meta["model"])
    model = nnx.eval_shape(lambda: MVQModel(cfg, rngs=nnx.Rngs(0)))
    gdef, state = nnx.split(model)
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    target = jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl), state)
    restored = ocp.StandardCheckpointer().restore(final_dir, target=target)
    model = nnx.merge(gdef, restored); model.eval()
    return model, meta


def _mvq_predict_xyz(model, s):
    """One val sample (from V12WindowDataset) -> (xyz_local (K,3), has3d (K,)
    bool) for the host-matched instance, unprompted. Mirrors
    scripts/viz/mvq_overlay.py's per-sample nearest-instance-to-GT match."""
    b = {k: jnp.asarray(v)[None] for k, v in s.items()}
    on = jnp.array([False])
    out = model(normalize_crops(b["crops"]), b["cam_valid"], b["M"], b["t_local"], b["prompt_mask"], prompt_on=on)
    xyz = np.asarray(out["xyz"][0]); gt = s["kp3d_local"][0, 0]; has = s["has3d"][0, 0]
    d = np.linalg.norm(xyz[:, 0] - gt[None], axis=-1)
    inst = int(np.argmin((d * has).sum(1)))
    return xyz[inst, 0], has


def run(args):
    det = load_detector_cfg()
    ds = V12WindowDataset(args.root, args.split, T=1, train=False)
    if det["kp_names"] != ds.keypoint_names:
        if len(det["kp_names"]) != len(ds.keypoint_names):
            raise ValueError(
                f"detector kp_names (configs/detector/vitpose_v3.yaml) has "
                f"{len(det['kp_names'])} entries but dataset keypoint_names "
                f"({args.root}/annotations/keypoint_names.json) has {len(ds.keypoint_names)} -- "
                f"cannot compare position-for-position. Fix the config or the dataset before "
                f"running this.")
        first = next(i for i, (a, b) in enumerate(zip(det["kp_names"], ds.keypoint_names)) if a != b)
        raise ValueError(
            f"detector kp_names (configs/detector/vitpose_v3.yaml) != dataset keypoint_names "
            f"({args.root}/annotations/keypoint_names.json) -- first mismatch at index {first}: "
            f"{det['kp_names'][first]!r} vs {ds.keypoint_names[first]!r}. Comparing them would "
            f"silently score the wrong anatomy against the wrong anatomy (see CLAUDE.md's "
            f"keypoint-order-bug history); fix the config or the dataset before running this.")
    n = len(ds) if args.n is None else min(args.n, len(ds))
    vit = load_vitpose(det["ckpt"], ViTPoseConfig(num_keypoints=det["num_keypoints"]))
    mvq_model = mvq_meta = None
    if args.mvq_run:
        mvq_model, mvq_meta = _load_mvq_model(args.mvq_run)

    n_empty_framesets = 0
    rows = []  # (mpjpe_units, n_valid, n_has, female, two_fly, group, mvq_same_units, mvq_n_same)
    t0 = time.time()
    for i in range(n):
        s = ds[i]
        crops = s["crops"][0]                                    # (C,448,448,3) u8
        cam_valid = s["cam_valid"][0]                            # (C,)
        # 4th (mask) channel: zeroed only when the config says the checkpoint was
        # trained with it zeroed (v5vf_maskoff's own train.mask_ablation=true --
        # EXACT for that checkpoint per the module docstring, since its channel-3
        # kernel weights are all identically 0.0); otherwise a normally-trained
        # checkpoint expects the REAL mask, so feed V12WindowDataset's own
        # prompt_mask (literal 0/1, matching normalize_image's un-scaled 4th
        # channel convention -- see data/device.py's docstring), not zeros.
        if det["zero_mask_channel"]:
            mask_ch = np.zeros(crops.shape[:-1] + (1,), np.uint8)
        else:
            mask_ch = s["prompt_mask"][0].astype(np.uint8)[..., None]           # (C,448,448,1)
        crops4 = np.concatenate([crops, mask_ch], axis=-1)
        hm = vit(normalize_image(jnp.asarray(crops4)), use_running_average=True)
        kp_crop, conf = peaks_and_conf(hm, decode_sharpen=det["decode_sharpen"])   # (C,K,2),(C,K)
        kp_crop = np.asarray(kp_crop, np.float32)
        conf = np.where(cam_valid[:, None], np.asarray(conf), 0.0).astype(np.float32)

        cm = cam_mats_local(s["M"], s["t_local"][0])
        kp3d_pred, _ = triangulate_keypoints(kp_crop[None], conf[None], cm,
                                             conf_thresh=det["conf_thresh"],
                                             view_conf_thresh=det["view_conf_thresh"],
                                             reproj_resid_px=det["reproj_resid_px"])
        gt = s["kp3d_local"][0, 0]; has = s["has3d"][0, 0]
        valid = has & np.isfinite(kp3d_pred[0]).all(-1)      # baseline's own finite-joint mask
        if has.any() and not valid.any():
            n_empty_framesets += 1
            warnings.warn(f"frameset {i} (ds index): DLT triangulated ZERO of "
                          f"{int(has.sum())} ground-truth joints", stacklevel=2)
        err = np.linalg.norm(np.where(valid[:, None], kp3d_pred[0], 0.0) - gt, axis=-1)
        mpjpe_i = float(err[valid].mean()) if valid.any() else float("nan")

        mvq_same_units, mvq_n_same = float("nan"), 0
        if mvq_model is not None:
            mvq_xyz, mvq_has = _mvq_predict_xyz(mvq_model, s)
            same = valid & mvq_has                                 # baseline-finite AND mvq has GT
            if same.any():
                mvq_same_units = float(np.linalg.norm(mvq_xyz[same] - gt[same], axis=-1).mean())
                mvq_n_same = int(same.sum())

        rows.append((mpjpe_i, int(valid.sum()), int(has.sum()), bool(ds.is_female(i)), ds.n_flies(i) > 1,
                    ds.calib_group(i), mvq_same_units, mvq_n_same))
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"[{i+1}/{n}] ({time.time()-t0:.0f}s)", flush=True)

    mp = np.array([r[0] for r in rows]); nv = np.array([r[1] for r in rows]); nh = np.array([r[2] for r in rows])
    ok = np.isfinite(mp) & (nv > 0)
    mvq_same = np.array([r[6] for r in rows]); mvq_n = np.array([r[7] for r in rows])
    mvq_ok = np.isfinite(mvq_same) & (mvq_n > 0)

    def cohort_mpjpe(mask):
        sel = mask & ok
        return float(np.average(mp[sel], weights=nv[sel])) if sel.any() else float("nan")

    def cohort_coverage(mask):
        return float(nv[mask].sum() / max(nh[mask].sum(), 1))

    def cohort_n(mask):
        return int(mask.sum()), int((mask & ok).sum())

    def cohort_mvq_same(mask):
        sel = mask & mvq_ok
        return float(np.average(mvq_same[sel], weights=mvq_n[sel])) if sel.any() else float("nan")

    female = np.array([r[3] for r in rows]); two_fly = np.array([r[4] for r in rows])
    groups = sorted({r[5] for r in rows})
    cohorts = {"overall": np.ones(len(rows), bool), "female": female, "two_fly": two_fly}
    for g in groups:
        cohorts[f"group_{g}"] = np.array([r[5] == g for r in rows])

    result = {
        "n_framesets": n, "n_empty_framesets": n_empty_framesets,
        "ckpt": det["ckpt"], "root": args.root, "split": args.split,
        "decode_sharpen": det["decode_sharpen"], "conf_thresh": det["conf_thresh"],
        "view_conf_thresh": det["view_conf_thresh"], "reproj_resid_px": det["reproj_resid_px"],
        "detector_zero_mask_channel": bool(det["zero_mask_channel"]),
        "train_exposure_caveat": _train_exposure_note(),
    }
    for name, mask in cohorts.items():
        n_tot, n_finite = cohort_n(mask)
        result[f"{name}_units"] = cohort_mpjpe(mask)
        result[f"{name}_coverage"] = cohort_coverage(mask)
        result[f"{name}_n_framesets"] = n_tot
        result[f"{name}_n_framesets_finite"] = n_finite
    if args.mvq_run:
        result["mvq_run"] = args.mvq_run
        result["mvq_meta_val"] = mvq_meta["val"]     # (c) mvq on ALL joints -- copied, not recomputed
        for name, mask in cohorts.items():
            result[f"mvq_same_joints_{name}_units"] = cohort_mvq_same(mask)   # (b)
    for k in list(result):
        if k.endswith("_units"):
            v = result[k]
            result[k.replace("_units", "_mm")] = v * MM_PER_UNIT if np.isfinite(v) else float("nan")

    print(json.dumps(result, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print("wrote", args.out)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=None, help="limit to first N val framesets")
    ap.add_argument("--mvq_run", default=None,
                    help="path to a gate-1-style final/ dir; when given, also reports "
                         "mvq's own MPJPE restricted to the SAME (sample,joint) pairs the "
                         "baseline could triangulate (items b/c) -- adds one CPU/GPU mvq "
                         "forward pass per frameset, so this is slow without a GPU.")
    ap.add_argument("--out", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
