#!/usr/bin/env python3
"""IK error-quantification driver (analysis Task 3).

For a configured (bout, fly), quantifies:

  Q1  local IK sensitivity: how much does the solved qpos move per unit of
      isotropic 3D marker noise (the marker Jacobian J = d(site_xpos)/dqpos at
      the solved pose, propagated through the per-keypoint noise model), plus
      a Monte-Carlo re-solve check (condition A / morph only) and the
      keypoint<->model fit residual.
  Q2  morph-vs-no-morph: how much does the per-segment (per-limb) SHAPE
      calibration (``cfg.model.SEGMENT_SCALES``, Task's "condition A") change
      the fit residual / solved qpos / implied bias relative to the unmorphed
      base model ("condition B"), for the SAME input keypoints.

The math core (marker_site_ids, marker_jacobian, qpos_sensitivity,
implied_bias, dof_units, the 3D-noise model, monte_carlo_qpos) lives in
``jarvis_jax.tracking.ik_error`` (tested separately). This script is
integration glue: it wires that math to a real bout's triangulated keypoints
and a real STAC solve (``jarvis_jax.tracking.stac``), mirroring the
segment-calibration + offsets + ik plumbing in ``scripts/run_bout.py``.

Usage:
    python scripts/analyze_ik_error.py recording=session0 ++bout_ids=1 \\
        ++ik_error.n_frames=80 ++ik_error.mc_n=20
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
import copy
import json

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mujoco

from jarvis_jax.tracking import ik_error as ike
from jarvis_jax.tracking.stac import fit_offsets_once, ik_only_bout
from jarvis_jax.tracking.resume import atomic_save_json, stage_done

# Register the `basename` OmegaConf resolver used by configs/outputs/default.yaml
# (mirrors scripts/run_bout.py; `replace=True` makes re-registration idempotent
# whether or not run_bout is also imported below).
OmegaConf.register_new_resolver(
    "basename", lambda p: os.path.basename(os.path.normpath(str(p))), replace=True)

# Reuse the courtship driver's plumbing (bout-frame lookup, high-confidence
# frame sampling, the per-segment shape-calibration M-step + cfg wiring)
# instead of duplicating it -- scripts/ is this file's own directory, so it is
# importable as a plain module both under `python scripts/analyze_ik_error.py`
# (script dir is sys.path[0]) and under pytest/hydra invocations from the repo
# root (added explicitly below just in case).
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import run_bout as rb  # noqa: E402  (after sys.path shim)


# ---------------------------------------------------------------------------
# kp3d / conf3d loading
# ---------------------------------------------------------------------------

def _load_or_triangulate_kp3d(cfg, bout_idx: int, fly: int):
    """Prefer the run's persisted ``kp3d.npz`` (run_bout.py Stage B); else
    triangulate it ourselves via ViTPose(masks) + DLT (mirrors run_bout.py
    Stage A+B exactly, minus the Stage B2 temporal-smoothing step -- kp3d.npz
    is itself the RAW/unfiltered triangulation, so this fallback matches that
    same semantics)."""
    run_root = str(cfg.outputs.out)
    bout_dir = os.path.join(run_root, "bouts", f"bout_{bout_idx:05d}", f"fly{fly}")
    kp3d_path = os.path.join(bout_dir, "kp3d.npz")
    if stage_done(kp3d_path):
        with np.load(kp3d_path) as z:
            return np.asarray(z["kp3d"]), np.asarray(z["conf3d"]), bout_dir

    print(f"[ik_error] {kp3d_path} not found -- triangulating via Stage A+B "
          f"(ViTPose + DLT) for bout {bout_idx} fly{fly}")
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.bout_masks import load_bout_masks, check_bout_camera_order
    from jarvis_jax.tracking.predict_2d import (
        load_detector, predict_bout_2d, reorder_detector_to_model)
    from jarvis_jax.tracking.triangulate import triangulate_keypoints
    from jarvis_jax.predict.synced_reader import load_plan, read_window

    predictions_dir = str(cfg.recording.predictions_dir)
    bout_npz = os.path.join(predictions_dir, f"bout_{bout_idx:05d}", "sam3_masks.npz")
    cameras = list(cfg.recording.cameras)

    rt = ReprojectionTool(cfg.recording.calib_dir)
    cam_mats = np.asarray(rt.camera_matrices, np.float32)
    sync_plan = load_plan(cfg.recording.session_dir)

    _cam = check_bout_camera_order(bout_npz, fly, cameras, rt)
    print(f"[ik_error][camera-order] bout {bout_idx} fly{fly}: {_cam['status']}"
          f" (identity={_cam.get('identity_resid')}, worst_cam={_cam.get('worst_cam')})")

    masks_dict = load_bout_masks(bout_npz, fly, expected_cameras=cameras)
    T = masks_dict["T"]
    start = rb.bout_start_frame(cfg, bout_idx)
    centroids = masks_dict["centroids"]

    def _frames_iter():
        for _frames, _present in read_window(
                cfg.recording.session_dir, cameras, sync_plan, start, T):
            yield _frames

    vit = load_detector(cfg.detector.ckpt, num_keypoints=int(cfg.detector.num_keypoints))
    kp2d, conf = predict_bout_2d(
        vit, _frames_iter(), masks_dict["masks"], centroids, masks_dict["valid"], cam_mats,
        crop=int(cfg.detector.crop), batch=int(cfg.detector.get("batch", 64)),
        decode_sharpen=float(cfg.detector.get("decode_sharpen", 1.0)))
    kp2d, conf = reorder_detector_to_model(
        kp2d, conf, list(cfg.detector.kp_names), list(cfg.model.KP_NAMES))

    _kf = cfg.detector.get("kp2d_filter", None)
    if _kf is not None and bool(_kf.get("enabled", False)):
        from jarvis_jax.tracking.kp2d_oneeuro import filter_kp2d_oneeuro
        kp2d = filter_kp2d_oneeuro(
            kp2d, conf, list(cfg.model.KP_NAMES),
            conf_thresh=float(cfg.detector.conf_thresh),
            min_cutoff=float(_kf.get("min_cutoff", 1.0)),
            beta=float(_kf.get("beta", 0.0)),
            d_cutoff=float(_kf.get("d_cutoff", 1.0)),
            preserve_raw_patterns=tuple(_kf.get("preserve_raw_patterns", ["Wing"])))

    kp3d, conf3d = triangulate_keypoints(
        kp2d, conf, cam_mats, conf_thresh=float(cfg.detector.conf_thresh))
    os.makedirs(bout_dir, exist_ok=True)
    from jarvis_jax.tracking.resume import atomic_save_npz
    atomic_save_npz(kp3d_path, kp3d=kp3d, conf3d=conf3d)
    return kp3d, conf3d, bout_dir


def _subsample_high_conf(kp3d, conf3d, n_frames: int):
    """Indices of the `n_frames` highest-mean-confidence frames (time order
    preserved), or all frames if n_frames <= 0 or >= T."""
    T = kp3d.shape[0]
    if n_frames <= 0 or n_frames >= T:
        return np.arange(T)
    with np.errstate(invalid="ignore"):
        mean_conf = np.nanmean(np.where(conf3d > 0, conf3d, np.nan), axis=1)
    mean_conf = np.nan_to_num(mean_conf, nan=-1.0)
    idx = np.argsort(-mean_conf)[:n_frames]
    return np.sort(idx)


# ---------------------------------------------------------------------------
# Per-fly-constant morph inputs (scale.json / segment_scales.json), shared
# with run_bout.py's on-disk cache under the SAME run_root.
# ---------------------------------------------------------------------------

def _load_or_compute_scale(cfg, kp3d, kp_names, run_root):
    scale_path = os.path.join(run_root, "scale.json")
    if stage_done(scale_path):
        with open(scale_path) as f:
            return float(json.load(f)["scale"])
    from jarvis_jax.tracking.scale import compute_trunk_scale
    _scale = compute_trunk_scale(
        kp3d, kp_names, cfg.silhouette.xml,
        trunk_names=list(cfg.scaling.trunk_keypoints),
        estimator=cfg.scaling.estimator,
        robust_stat=cfg.scaling.robust_stat,
        robust=cfg.scaling.robust)
    atomic_save_json(scale_path, {"scale": float(_scale),
                                  "trunk_keypoints": list(cfg.scaling.trunk_keypoints),
                                  "estimator": str(cfg.scaling.estimator)})
    return float(_scale)


def _load_or_compute_segment_scales(cfg, kp3d, kp_names, scale, run_root):
    seg_path = os.path.join(run_root, "segment_scales.json")
    if stage_done(seg_path):
        with open(seg_path) as f:
            return json.load(f)
    seg_entries = rb.compute_segment_scales(cfg, kp3d, kp_names, scale, run_root)
    atomic_save_json(seg_path, seg_entries)
    return seg_entries


# ---------------------------------------------------------------------------
# qpos DOF classification: authoritative (mj_model joint-type-based), NOT the
# name-substring heuristic alone.
# ---------------------------------------------------------------------------

def _dof_kinds_from_model(mj_model):
    """Per-qpos-index DOF kind, derived from the compiled model's joint
    TYPES/addresses (not names). stac-mjx's `_align_joint_dims` repeats the
    SAME joint name across every dim of a multi-dim joint (e.g. a free joint's
    x,y,z,qw,qx,qy,qz all share one name -- see stac_mjx/stac.py), so
    ``ike.dof_units``'s name-substring heuristic cannot split a free joint's 3
    translation DOFs from its 4 quaternion DOFs by name alone. This walks
    mj_model.jnt_type / jnt_qposadr directly instead, which IS authoritative."""
    nq = int(mj_model.nq)
    kind = [None] * nq
    for j in range(mj_model.njnt):
        jt = int(mj_model.jnt_type[j])
        adr = int(mj_model.jnt_qposadr[j])
        if jt == int(mujoco.mjtJoint.mjJNT_FREE):
            for k in range(3):
                kind[adr + k] = "root_trans"
            for k in range(3, 7):
                kind[adr + k] = "root_quat"
        elif jt == int(mujoco.mjtJoint.mjJNT_BALL):
            for k in range(4):
                kind[adr + k] = "root_quat"
        elif jt == int(mujoco.mjtJoint.mjJNT_SLIDE):
            kind[adr] = "root_trans"
        else:  # hinge
            kind[adr] = "hinge"
    assert all(k is not None for k in kind), f"unclassified qpos DOF(s) (nq={nq})"
    return kind


def _unit_factor(kind, scale, mocap_scale_factor):
    """Per-DOF natural-unit scalar multiplier (matches ike.dof_units's report
    convention): hinge rad->deg; root_trans model-units->mm (the inverse of
    the mm->model-units scaling stac.fit_offsets_once/ik_only_bout apply to
    kp3d, i.e. divide by scale*MOCAP_SCALE_FACTOR); root_quat small-angle
    proxy dtheta ~= 2*dq (near-identity quaternion) -> deg. All quantities
    here are PROXIES, same caveat as jarvis_jax.tracking.ik_error."""
    if kind == "hinge":
        return 180.0 / np.pi
    if kind == "root_trans":
        return 1.0 / (float(scale) * float(mocap_scale_factor))
    if kind == "root_quat":
        return 2.0 * 180.0 / np.pi
    return 1.0


def _mj_model_for_condition(cfg, morph: bool, seg_entries):
    """mj_model matching what stac_mjx.Stac actually solved on: unmorphed base
    model (condition B / no seg_entries), or the SAME rescale_per_segment
    morph Stac.__init__ applies when cfg.model.SEGMENT_SCALES is set
    (condition A) -- built independently here (not reused from Stac) because
    we need our own mjx model for the ike.marker_jacobian/FK verification."""
    from stac_mjx import rescale
    if not morph or not seg_entries:
        spec = mujoco.MjSpec.from_file(str(cfg.model.MJCF_PATH))
        rescale.dm_scale_spec(spec, float(cfg.model.SCALE_FACTOR))
        return spec.compile()
    spec = mujoco.MjSpec.from_file(str(cfg.model.MJCF_PATH))
    # Global scale FIRST, matching stac_mjx.Stac._create_body_sites (applied
    # unconditionally, before rescale_per_segment) -- see stac-mjx/stac_mjx/
    # stac.py:239. Currently a no-op in every anatomy config (SCALE_FACTOR=1),
    # but kept so this driver stays correct if SCALE_FACTOR is ever changed.
    rescale.dm_scale_spec(spec, float(cfg.model.SCALE_FACTOR))
    seg_list = [{"geom_body": e.get("geom_body", ""), "length_body": e.get("length_body", ""),
                "scale": float(e["scale"]),
                "scale_sites_on_body": e.get("scale_sites_on_body", "")}
                for e in seg_entries]
    rescale.rescale_per_segment(spec, seg_list)
    return spec.compile()


# ---------------------------------------------------------------------------
# Per-condition analysis (fit + per-frame Jacobian/sensitivity/residual/bias)
# ---------------------------------------------------------------------------

def _condition_analysis(cfg, kp3d_sub, conf3d_sub, kp_names, *, morph, out_dir, scale,
                        seg_entries, eps, mc_frames, mc_n, seed, run_mc):
    import jax.numpy as jnp
    import mujoco.mjx as mjx
    import stac_mjx.io_dict_to_hdf5 as ioh5

    cond_name = "morph" if morph else "nomorph"
    cond_dir = os.path.join(out_dir, cond_name)
    os.makedirs(cond_dir, exist_ok=True)

    # Independent cfg per condition: fit_offsets_once/ik_only_bout mutate
    # cfg.stac.* fields in place, and only condition A sets SEGMENT_SCALES.
    cfg_c = copy.deepcopy(cfg)
    if morph and seg_entries:
        rb.apply_segment_scales(cfg_c, seg_entries)

    sample_idx = rb.high_confidence_sample(kp3d_sub, max_frames=300)
    fit_offsets_once(cfg_c, kp3d_sub[sample_idx], kp_names,
                     offsets_path="offsets.h5", save_path=cond_dir, scale=scale)
    ik_only_bout(cfg_c, kp3d_sub, kp_names, offsets_path="offsets.h5",
                out_h5="stac_ik.h5", save_path=cond_dir, scale=scale)

    d = ioh5.load(os.path.join(cond_dir, "stac_ik.h5"))
    qpos = np.asarray(d["qpos"], np.float64)                       # (T,nq)
    names_qpos = [n.decode("utf-8") if isinstance(n, bytes) else str(n)
                 for n in d["names_qpos"]]
    T, nq = qpos.shape
    kp_data = np.asarray(d["kp_data"], np.float64).reshape(T, len(kp_names), 3)  # model units

    mj_model = _mj_model_for_condition(cfg_c, morph, seg_entries)
    site_ids = ike.marker_site_ids(mj_model, kp_names)
    mjx_model = mjx.put_model(mj_model)
    d0 = mjx.make_data(mjx_model)

    kw = dict(cfg.model.get("KEYPOINT_WEIGHTS") or {})
    w = np.array([float(kw.get(k, 1.0)) for k in kp_names], np.float64)
    W = np.diag(np.repeat(w, 3))

    mocap_scale = float(cfg.model.MOCAP_SCALE_FACTOR)
    sigma_kp_mm = ike.sigma_kp_from_conf(conf3d_sub, kp3d_sub)              # (K,) mm
    sigma_kp_model = sigma_kp_mm * float(scale) * mocap_scale                # model units
    Sigma2_model = ike.sigma2_diag(sigma_kp_model)                          # model-units^2

    transfers = np.zeros((T, nq))
    stds = np.zeros((T, nq))
    cond_numbers = np.zeros(T)
    residual_mm = np.zeros((T, len(kp_names)))
    implied_bias_mat = np.zeros((T, nq))
    for t in range(T):
        J = np.asarray(ike.marker_jacobian(mjx_model, d0, qpos[t], site_ids))   # (3K,nq)
        sens = ike.qpos_sensitivity(J, W, Sigma2_model, eps=eps)
        transfers[t] = np.asarray(sens["transfer"])
        stds[t] = np.asarray(sens["std"])
        H = J.T @ W @ J
        cond_numbers[t] = float(np.linalg.cond(np.asarray(H) + eps * np.eye(nq)))

        d_t = mjx.kinematics(mjx_model, d0.replace(qpos=jnp.asarray(qpos[t])))
        pred = np.asarray(d_t.site_xpos)[site_ids]                         # (K,3) model units
        diff = pred - kp_data[t]                                           # model units
        residual_mm[t] = np.linalg.norm(diff, axis=-1) / (float(scale) * mocap_scale)

        implied_bias_mat[t] = np.asarray(ike.implied_bias(J, W, diff.reshape(-1), eps=eps))

    kinds = _dof_kinds_from_model(mj_model)
    factor = np.array([_unit_factor(k, scale, mocap_scale) for k in kinds])
    # `transfer` is dqpos per one MODEL-LENGTH-UNIT of marker noise; std/bias
    # are already expressed in natural qpos units via `factor` alone (their
    # Sigma2_model input is built in mm, see sigma_kp_model above -- correct
    # as-is). To report transfer honestly as "per mm" it additionally needs
    # the input-side mm->model-units factor (scale * MOCAP_SCALE_FACTOR),
    # the SAME scale/mocap_scale used to build sigma_kp_model / residual_mm
    # above, so the whole thing reads as natural-qpos-unit per mm of 3D error.
    transfer_nat = transfers * factor[None, :] * (float(scale) * mocap_scale)
    std_nat = stds * factor[None, :]
    bias_nat = implied_bias_mat * factor[None, :]

    mc = None
    if run_mc and mc_frames > 0 and mc_n > 0:
        with np.errstate(invalid="ignore"):
            mean_conf = np.nanmean(np.where(conf3d_sub > 0, conf3d_sub, np.nan), axis=1)
        mean_conf = np.nan_to_num(mean_conf, nan=-1.0)
        mc_idx = np.sort(np.argsort(-mean_conf)[:min(int(mc_frames), T)])
        kp3d_mc = kp3d_sub[mc_idx]
        sigma_kp_mc_mm = ike.sigma_kp_from_conf(conf3d_sub[mc_idx], kp3d_mc)
        mc_qpos = ike.monte_carlo_qpos(
            cfg_c, kp3d_mc, kp_names, sigma_kp_mc_mm, offsets_path="offsets.h5",
            save_path=cond_dir, scale=scale, n=int(mc_n), seed=int(seed))     # (n,Tmc,nq)
        mc_std_nat = np.asarray(mc_qpos).std(axis=0) * factor[None, :]        # (Tmc,nq)
        jac_std_nat_mc = std_nat[mc_idx]                                     # (Tmc,nq)
        mc_std_med = np.median(mc_std_nat, axis=0)
        jac_std_med = np.median(jac_std_nat_mc, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(jac_std_med > 1e-9, mc_std_med / jac_std_med, np.nan)
        mc = dict(mc_idx=mc_idx, mc_std_median=mc_std_med,
                 jac_std_median=jac_std_med, agreement_ratio=ratio)

    return dict(cond=cond_name, cond_dir=cond_dir, qpos=qpos, names_qpos=names_qpos,
               kinds=kinds, factor=factor, transfer_nat=transfer_nat, std_nat=std_nat,
               bias_nat=bias_nat, residual_mm=residual_mm, cond_number=cond_numbers, mc=mc)


# ---------------------------------------------------------------------------
# Report (JSON) + plot
# ---------------------------------------------------------------------------

def _build_report(resA, resB, kp_names):
    def agg(res):
        return dict(
            transfer_median=np.median(res["transfer_nat"], axis=0),
            transfer_p95=np.percentile(res["transfer_nat"], 95, axis=0),
            std_median=np.median(res["std_nat"], axis=0),
            std_p95=np.percentile(res["std_nat"], 95, axis=0),
            bias_median=np.median(res["bias_nat"], axis=0),
            residual_mm_median_per_kp=np.median(res["residual_mm"], axis=0),
            residual_mm_p95_per_kp=np.percentile(res["residual_mm"], 95, axis=0),
            residual_mm_overall_median=float(np.median(res["residual_mm"])),
            cond_number_median=float(np.median(res["cond_number"])),
        )
    A, B = agg(resA), agg(resB)
    names_qpos = resA["names_qpos"]
    kinds = resA["kinds"]

    # Cross-check the Task-1/2 name-heuristic (ike.dof_units) against the
    # authoritative joint-structure classification; they are EXPECTED to
    # disagree on multi-dim joints (see _dof_kinds_from_model docstring) --
    # logged here so Task 4 can sanity-check (a free joint should yield 7
    # DOFs: 3 root_trans + 4 root_quat).
    heuristic = ike.dof_units(names_qpos)
    n_mismatch = sum(1 for h, k in zip(heuristic, kinds) if h["kind"] != k)
    dof_classification = dict(
        n_qpos=len(kinds),
        n_root_trans=kinds.count("root_trans"),
        n_root_quat=kinds.count("root_quat"),
        n_hinge=kinds.count("hinge"),
        n_mismatch_vs_name_heuristic=n_mismatch,
        note=("kind/unit here is the AUTHORITATIVE mj_model joint-type "
              "classification used for all unit conversions below; "
              "ike.dof_units's name-only heuristic (logged per-DOF as "
              "'heuristic_kind') is expected to mismatch on every dim of a "
              "multi-dim joint (free/ball) because stac-mjx's names_qpos "
              "repeats one joint name across all of that joint's dims."),
        per_dof=[dict(index=i, name=names_qpos[i], kind=kinds[i],
                     heuristic_kind=heuristic[i]["kind"])
                for i in range(len(kinds))],
    )

    q1 = dict(
        transfer_median=A["transfer_median"].tolist(),
        transfer_p95=A["transfer_p95"].tolist(),
        transfer_units="deg (hinge/root_quat) or mm (root_trans) of qpos change "
                       "PER MM of isotropic 3D marker noise",
        std_median=A["std_median"].tolist(),
        std_p95=A["std_p95"].tolist(),
        cond_number_median=A["cond_number_median"],
        per_keypoint_fit_residual_mm_median=dict(
            zip(kp_names, A["residual_mm_median_per_kp"].tolist())),
        per_keypoint_fit_residual_mm_p95=dict(
            zip(kp_names, A["residual_mm_p95_per_kp"].tolist())),
        overall_fit_residual_mm_median=A["residual_mm_overall_median"],
    )
    if resA["mc"] is not None:
        mc = resA["mc"]
        q1["monte_carlo"] = dict(
            mc_frame_idx=mc["mc_idx"].tolist(),
            mc_std_median=mc["mc_std_median"].tolist(),
            jacobian_std_median=mc["jac_std_median"].tolist(),
            agreement_ratio=mc["agreement_ratio"].tolist(),
            agreement_ratio_overall_median=float(np.nanmedian(mc["agreement_ratio"])),
        )

    qpos_delta_nat = (resA["qpos"] - resB["qpos"]) * resA["factor"][None, :]
    q2 = dict(
        residual_mm_median_morph=A["residual_mm_overall_median"],
        residual_mm_median_nomorph=B["residual_mm_overall_median"],
        residual_mm_delta=A["residual_mm_overall_median"] - B["residual_mm_overall_median"],
        per_keypoint_residual_mm_delta=dict(zip(
            kp_names,
            (A["residual_mm_median_per_kp"] - B["residual_mm_median_per_kp"]).tolist())),
        qpos_delta_median=np.median(qpos_delta_nat, axis=0).tolist(),
        qpos_delta_abs_p95=np.percentile(np.abs(qpos_delta_nat), 95, axis=0).tolist(),
        implied_bias_morph_median=A["bias_median"].tolist(),
        implied_bias_nomorph_median=B["bias_median"].tolist(),
    )

    return dict(dof_classification=dof_classification, names_qpos=names_qpos,
               kp_names=kp_names, Q1=q1, Q2=q2)


def _plot_report(resA, resB, png_path):
    names_qpos = resA["names_qpos"]
    transfer_med_A = np.median(resA["transfer_nat"], axis=0)
    qpos_delta_nat = (resA["qpos"] - resB["qpos"]) * resA["factor"][None, :]
    delta_med = np.median(qpos_delta_nat, axis=0)

    uniq = []
    for n in names_qpos:
        if n not in uniq:
            uniq.append(n)
    joint_transfer, joint_delta = [], []
    for n in uniq:
        m = [i for i, nm in enumerate(names_qpos) if nm == n]
        joint_transfer.append(float(np.max(transfer_med_A[m])))
        joint_delta.append(float(np.max(np.abs(delta_med[m]))))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.18 * len(uniq)), 8), sharex=True)
    x = np.arange(len(uniq))
    axes[0].bar(x, joint_transfer, color="#4c72b0")
    axes[0].set_ylabel("Q1: median qpos transfer\n(deg or mm / mm marker noise)")
    axes[0].set_title("IK error quantification: per-joint sensitivity + morph delta")
    axes[1].bar(x, joint_delta, color="#c44e52")
    axes[1].set_ylabel("Q2: |morph - nomorph|\nqpos delta (median, deg/mm)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(uniq, rotation=90, fontsize=6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def analyze_bout_fly(cfg, bout_idx: int, fly: int):
    run_root = str(cfg.outputs.out)
    ik_cfg = cfg.get("ik_error", {}) or {}
    n_frames = int(ik_cfg.get("n_frames", 80))
    mc_frames = int(ik_cfg.get("mc_frames", 10))
    mc_n = int(ik_cfg.get("mc_n", 20))
    eps = float(ik_cfg.get("eps", 1e-6))
    seed = int(ik_cfg.get("seed", cfg.get("seed", 0)))

    kp_names = list(cfg.model.KP_NAMES)
    kp3d, conf3d, _bout_dir = _load_or_triangulate_kp3d(cfg, bout_idx, fly)
    sub_idx = _subsample_high_conf(kp3d, conf3d, n_frames)
    kp3d_sub, conf3d_sub = kp3d[sub_idx], conf3d[sub_idx]
    print(f"[ik_error] bout {bout_idx} fly{fly}: {kp3d.shape[0]} frames -> "
          f"{kp3d_sub.shape[0]}-frame high-confidence subsample "
          f"(mc_frames={mc_frames}, mc_n={mc_n}, eps={eps})")

    scale = _load_or_compute_scale(cfg, kp3d, kp_names, run_root)
    seg_entries = None
    if bool(cfg.model.get("segment_calibration", True)):
        seg_entries = _load_or_compute_segment_scales(cfg, kp3d, kp_names, scale, run_root)
    else:
        print("[ik_error] cfg.model.segment_calibration is False -- "
              "condition A (morph) has no segment scales to apply and will "
              "match condition B (nomorph).")

    out_dir = os.path.join(run_root, "ik_error_analysis", f"bout_{bout_idx:05d}", f"fly{fly}")
    os.makedirs(out_dir, exist_ok=True)

    resA = _condition_analysis(
        cfg, kp3d_sub, conf3d_sub, kp_names, morph=True, out_dir=out_dir, scale=scale,
        seg_entries=seg_entries, eps=eps, mc_frames=mc_frames, mc_n=mc_n, seed=seed,
        run_mc=True)
    resB = _condition_analysis(
        cfg, kp3d_sub, conf3d_sub, kp_names, morph=False, out_dir=out_dir, scale=scale,
        seg_entries=None, eps=eps, mc_frames=mc_frames, mc_n=mc_n, seed=seed,
        run_mc=False)

    report = _build_report(resA, resB, kp_names)
    json_path = os.path.join(out_dir, "ik_error.json")
    atomic_save_json(json_path, report)
    png_path = os.path.join(out_dir, "ik_error.png")
    _plot_report(resA, resB, png_path)
    print(f"[ik_error] bout {bout_idx} fly{fly}: DOF classification nq="
          f"{report['dof_classification']['n_qpos']} "
          f"root_trans={report['dof_classification']['n_root_trans']} "
          f"root_quat={report['dof_classification']['n_root_quat']} "
          f"hinge={report['dof_classification']['n_hinge']} "
          f"(name-heuristic mismatches: "
          f"{report['dof_classification']['n_mismatch_vs_name_heuristic']})")
    print(f"[ik_error] bout {bout_idx} fly{fly}: wrote {json_path}, {png_path}")


def main_from_cfg(cfg: DictConfig):
    bout_ids = rb.resolve_bout_ids(cfg)
    fly_arg = cfg.get("fly", None)
    flies = [int(fly_arg)] if fly_arg is not None else list(range(int(cfg.recording.num_animals)))
    print(f"[ik_error] processing {len(bout_ids)} bout(s): {bout_ids}, flies={flies}")
    for bout_idx in bout_ids:
        for fly in flies:
            analyze_bout_fly(cfg, bout_idx, fly)


@hydra.main(version_base=None, config_path="../configs", config_name="pipeline")
def main(cfg: DictConfig):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
