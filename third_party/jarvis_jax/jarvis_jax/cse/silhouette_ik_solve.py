"""STAC solver-input assembly + integration probe for single-fly silhouette-landmark IK.

De-risks the ``stac_mjx`` batch jaxls IK API (``stac_mjx.stac_core_jaxls.JaxlsBatchSolver``)
for Phase 2 by reproducing, from a STAC ik-output h5 + the fly MJCF model, the exact input
assembly that ``stac_mjx.compute_stac`` / ``stac_mjx.stac.Stac`` use to call
``solve_trajectory``. Model-space only (no cameras) — cameras enter in a later task.

Site naming
-----------
STAC does not ship keypoint sites baked into the anatomy XML. ``stac_mjx.stac.Stac.
_create_body_sites`` adds one ``<site>`` per keypoint at compile time, parented to the
model body named by ``KEYPOINT_MODEL_PAIRS[kp_name]``, named ``kp_name`` itself, with an
initial local pos from ``KEYPOINT_INITIAL_OFFSETS[kp_name]``. The STAC ik h5 embeds the
run's resolved config (including both dicts) under the ``config`` dataset, so we rebuild
those sites the same way and then overwrite their local offsets with the ik h5's fitted
``offsets`` via ``stac_mjx.utils.set_site_pos`` (the per-recording calibrated marker
offsets, not the config's generic initial guess).
"""
from __future__ import annotations

import json
import os

import numpy as np
import mujoco
from mujoco import mjx
from omegaconf import OmegaConf

from stac_mjx import io as stac_io
from stac_mjx import utils as stac_utils
from stac_mjx.stac_core_jaxls import JaxlsBatchSolver

# MuJoCo joint-type -> qpos dims, and the "unconstrained" bound to use when a
# joint's jnt_range is the MuJoCo default [0, 0] (which means "no limit").
# Mirrors stac_mjx.stac._MUJOCO_JOINT_TYPE_DIMS / _MUJOCO_JOINT_TYPE_UNCONSTRAINED.
_JOINT_TYPE_DIMS = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}

_FREE_LB = np.concatenate([-np.inf * np.ones(3), -1.0 * np.ones(4)])
_FREE_UB = np.concatenate([np.inf * np.ones(3), 1.0 * np.ones(4)])
_HINGE_UNCONSTRAINED = (-2 * np.pi, 2 * np.pi)
_SLIDE_UNCONSTRAINED = (-np.inf, np.inf)
_BALL_UNCONSTRAINED = (-1.0, 1.0)


def _joint_bounds(mj_model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Per-qpos lower/upper bounds, matching stac_mjx.stac._align_joint_dims.

    Free joints are unbounded (translation +-inf, quat clipped to [-1, 1]).
    Hinge/slide/ball joints use ``jnt_range`` unless it is the MuJoCo default
    ``[0, 0]`` (meaning "unlimited"), in which case a generous unconstrained
    range is substituted.
    """
    lb_parts, ub_parts = [], []
    for j in range(mj_model.njnt):
        jtype = mj_model.jnt_type[j]
        dims = _JOINT_TYPE_DIMS[jtype]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            lb_parts.append(_FREE_LB)
            ub_parts.append(_FREE_UB)
            continue
        lo, hi = mj_model.jnt_range[j]
        if lo == 0.0 and hi == 0.0:
            if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                lo, hi = _HINGE_UNCONSTRAINED
            elif jtype == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = _SLIDE_UNCONSTRAINED
            else:
                lo, hi = _BALL_UNCONSTRAINED
        lb_parts.append(np.full(dims, lo))
        ub_parts.append(np.full(dims, hi))
    lb = np.concatenate(lb_parts)
    ub = np.concatenate(ub_parts)
    # Match stac_mjx.stac._align_joint_dims: lb is additionally clamped <= 0
    # (keeps qpos=0 always feasible, matters for the free-joint quat identity).
    lb = np.minimum(lb, 0.0)
    return lb, ub


def _build_keypoint_sites(
    xml_path: str,
    keypoint_model_pairs: dict,
    keypoint_initial_offsets: dict,
) -> mujoco.MjModel:
    """Compile ``xml_path`` with one <site> per keypoint added, STAC-style.

    Mirrors stac_mjx.stac.Stac._create_body_sites: for each keypoint name,
    add a site (named after the keypoint) to the model body it registers to,
    at the config's initial offset. The site's final offset is set later from
    the ik h5's fitted ``offsets`` via stac_mjx.utils.set_site_pos.
    """
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for kp_name, body_name in keypoint_model_pairs.items():
        parent = spec.body(body_name)
        pos = keypoint_initial_offsets[kp_name]
        if isinstance(pos, str):
            pos = [float(p) for p in pos.split(" ")]
        parent.add_site(
            name=kp_name,
            size=[0.005, 0.005, 0.005],
            rgba=(0, 0, 0, 0.8),
            pos=pos,
            group=3,
        )
    return spec.compile()


def build_solver_inputs(ik_h5: str, model_xml: str) -> dict:
    """Assemble stac_mjx JaxlsBatchSolver inputs from a STAC ik-output h5 + fly model.

    Args:
        ik_h5: Path to a STAC ik-only/fit-offsets h5 (as written by
            ``stac_mjx.io.save_data_to_h5``), containing ``qpos``, ``kp_data``,
            ``offsets``, ``kp_names``, ``names_qpos`` and the run's resolved
            ``config`` (used to rebuild the keypoint sites and joint names).
        model_xml: Path to the fly MJCF (e.g. fruitfly_v1_free.xml). Should
            match the ``MJCF_PATH`` the ik h5 was produced with.

    Returns:
        dict with keys: mjx_model, mjx_data, q_init (T,nq), kp_data (T,n_kp,3),
        kps_to_opt (n_kp*3,), qs_to_opt (nq,), lb (nq,), ub (nq,),
        site_idxs (n_kp,), q_reg_weights (nq,), kp_names (n_kp,).
    """
    config, stac_data = stac_io.load_stac_data(ik_h5)

    kp_names = list(stac_data.kp_names)
    n_kp = len(kp_names)

    keypoint_model_pairs = OmegaConf.to_container(
        config.model.KEYPOINT_MODEL_PAIRS, resolve=True
    )
    keypoint_initial_offsets = OmegaConf.to_container(
        config.model.KEYPOINT_INITIAL_OFFSETS, resolve=True
    )

    mj_model = _build_keypoint_sites(
        model_xml, keypoint_model_pairs, keypoint_initial_offsets
    )

    nq = int(mj_model.nq)
    assert nq == len(stac_data.names_qpos), (
        f"compiled model nq={nq} does not match ik h5 names_qpos len="
        f"{len(stac_data.names_qpos)}"
    )

    # site_idxs: model site id for each keypoint name, in kp_names order.
    site_idxs = np.array(
        [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in kp_names
        ],
        dtype=np.int32,
    )
    if np.any(site_idxs < 0):
        missing = [n for n, i in zip(kp_names, site_idxs) if i < 0]
        raise ValueError(f"Could not find sites for keypoints: {missing}")

    mjx_model, mjx_data = stac_utils.mjx_load(mj_model)

    # Set the site local offsets from the ik h5's fitted `offsets` (T-independent,
    # per-recording calibrated marker positions), not the config's generic guess.
    offsets = np.asarray(stac_data.offsets, dtype=np.float32).reshape(n_kp, 3)
    mjx_model = stac_utils.set_site_pos(mjx_model, offsets, site_idxs)

    q_init = np.asarray(stac_data.qpos, dtype=np.float32)  # (T, nq)
    T = q_init.shape[0]
    kp_data_flat = np.asarray(stac_data.kp_data, dtype=np.float32)  # (T, n_kp*3)
    kp_data = kp_data_flat.reshape(T, n_kp, 3)

    qs_to_opt = np.ones(nq, dtype=bool)
    kps_to_opt = np.ones(n_kp * 3, dtype=np.float32)
    lb, ub = _joint_bounds(mj_model)
    q_reg_weights = np.zeros(nq, dtype=np.float32)

    return dict(
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        q_init=q_init,
        kp_data=kp_data,
        kps_to_opt=kps_to_opt,
        qs_to_opt=qs_to_opt,
        lb=lb.astype(np.float32),
        ub=ub.astype(np.float32),
        site_idxs=site_idxs,
        q_reg_weights=q_reg_weights,
        kp_names=kp_names,
    )


def solve_ik(
    inputs: dict,
    *,
    smooth_weight: float = 0.1,
    n_iter: int = 50,
) -> np.ndarray:
    """Solve batch IK over a clip using stac_mjx's jaxls Levenberg-Marquardt solver.

    Thin wrapper around ``JaxlsBatchSolver(...).solve_trajectory(...)``. See
    ``build_solver_inputs`` for the expected ``inputs`` keys.

    Args:
        inputs: dict as returned by ``build_solver_inputs`` (optionally sliced
            along the time axis for ``q_init``/``kp_data``).
        smooth_weight: Temporal smoothness weight passed to JaxlsBatchSolver.
        n_iter: Max LM iterations.

    Returns:
        np.ndarray of shape (T, nq) — the solved qpos trajectory.
    """
    solver = JaxlsBatchSolver(n_iter=n_iter, smooth_weight=smooth_weight, use_se3_root=True)
    qposes = solver.solve_trajectory(
        q_init=inputs["q_init"],
        mjx_model=inputs["mjx_model"],
        mjx_data_template=inputs["mjx_data"],
        kp_data=inputs["kp_data"],
        qs_to_opt=inputs["qs_to_opt"],
        kps_to_opt=inputs["kps_to_opt"],
        lb=inputs["lb"],
        ub=inputs["ub"],
        site_idxs=inputs["site_idxs"],
        q_reg_weights=inputs["q_reg_weights"],
    )
    return np.asarray(qposes)


# ---------------------------------------------------------------------------
# Phase 2 / Task 4: single-fly driver wiring the silhouette wing-tip extractor
# (Task 1) + marker augmentation (Task 2) into the STAC solver (Task 3, above).
# ---------------------------------------------------------------------------

_DEFAULT_REFINED_CALIB_DIR = (
    "/gscratch/portia/eabe/data/Johnson_lab/cse_work/calib_refined/2026_03_18_15_31_22"
)
_DEFAULT_FACTORY_CALIB_DIR = (
    "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3/"
    "calib_params/2026_03_18_15_31_22"
)


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (s, R, t) minimizing ||s*R@src + t - dst||^2.

    Exactly mirrors the ``umeyama`` helper in ``silhouette_render_demo.py`` /
    ``silhouette_fit.py``: dst = s * (R @ src.T).T + t.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = Dc.T @ Sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((Sc ** 2).sum() / len(src))
    t = mu_d - s * R @ mu_s
    return s, R, t


def _model_to_mm(pts_model: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map (K,3) model-frame points -> mm frame under similarity (s, R, t)."""
    pts_model = np.asarray(pts_model, dtype=np.float64)
    return s * (R @ pts_model.T).T + t


def _mm_to_model(pts_mm: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse of ``_model_to_mm``: map (K,3) mm-frame points -> model frame."""
    pts_mm = np.asarray(pts_mm, dtype=np.float64)
    return ((pts_mm - t) @ R) / s


def _cam2img_for_frame(fs_imgids_row, id2file, cam_names) -> dict:
    """Map ReprojectionTool camera index -> coco image_id for one bout frame.

    Matches by camera name parsed from the image file path (mirrors
    ``silhouette_render_demo.py`` / ``demo_wingtip_triangulation.py``), rather
    than assuming positional alignment between ``fs_imgids`` columns and
    ``cam_names`` order (true for this dataset layout, but matching by name
    is the pattern used everywhere else in this module's reference files and
    is robust to either convention).
    """
    cam2img = {}
    for iid in fs_imgids_row:
        fn = id2file.get(int(iid), "")
        cam = fn.split("/")[1] if "/" in fn else ""
        if cam in cam_names:
            cam2img[cam_names.index(cam)] = int(iid)
    return cam2img


def _ann_for_image(id2ann_multi, image_id, ann_id_by_image=None):
    """Select one COCO annotation for an image.

    id2ann_multi maps image_id -> list of all anns for that image. If
    ann_id_by_image (per-fly image_id -> chosen ann id) is given and has this
    image, return the ann whose id matches; otherwise return the first ann
    (backward-compatible single-ann behavior). None if the image has no anns.
    """
    anns = id2ann_multi.get(int(image_id))
    if not anns:
        return None
    if ann_id_by_image is not None and int(image_id) in ann_id_by_image:
        want = int(ann_id_by_image[int(image_id)])
        for a in anns:
            if int(a["id"]) == want:
                return a
    return anns[0]


def _triangulate_kp_mm(rt, ik_kpnames, coco_kpnames, cam2img, id2ann_multi, ann_id_by_image=None):
    """Triangulate the STAC ik h5's named keypoints into the calib mm frame.

    For each ``ik_kpnames`` entry with a same-named coco keypoint, gathers
    the 2-D annotated (ground-truth) locations across cameras (>=2 visible)
    and DLT-triangulates via ``rt.reconstruct_point``. Returns (kp_mm (n,3),
    valid (n,) bool) — mirrors the ``kp_mm``/``kok`` construction in
    ``silhouette_render_demo.py``.

    ``id2ann_multi`` is image_id -> list[ann]; the ann used per camera is
    chosen by ``_ann_for_image(..., ann_id_by_image)`` (Phase 3 / Task 4:
    per-identity selection on multi-fly images). ``ann_id_by_image=None``
    (default) preserves the original first-ann-per-image behavior.
    """
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}
    n = len(ik_kpnames)
    kp_mm = np.zeros((n, 3))
    valid = np.zeros(n, dtype=bool)
    for j, nm in enumerate(ik_kpnames):
        ci = name2coco.get(nm)
        if ci is None:
            continue
        obs = np.zeros((rt.num_cameras, 2))
        cams = []
        for c, iid in cam2img.items():
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            kp = np.asarray(ann["keypoints"], dtype=float).reshape(-1, 3)
            if kp[ci, 2] > 0:
                obs[c] = kp[ci, :2]
                cams.append(c)
        if len(cams) >= 2:
            kp_mm[j] = rt.reconstruct_point(obs, cams_to_use=cams)
            valid[j] = True
    return kp_mm, valid


def _wing_fk_indices(mesh_npz: str) -> np.ndarray:
    """Full-vertex-array indices [L-prox, L-tip, R-prox, R-tip] for FK repose.

    ``silhouette_landmarks.wing_side_vertices`` returns indices into the
    ``fps_300``-subsampled point set (i.e. positions 0..299 within
    ``z["fps_300"]``), not raw indices into the canonical mesh's full
    ``vertices``/``vertices_local``/``vertex_geom`` arrays (length ~61666)
    that ``silhouette_ik.make_fk_repose``'s ``indices`` argument expects
    (mirrors ``vgeom_all[indices]`` in ``silhouette_ik.make_fk_repose``,
    where ``vgeom_all`` is the FULL per-vertex geom array — see its
    docstring/usage in ``silhouette_render_demo.py``, which always indexes
    the full array, never the fps subset). Passing the fps-relative indices
    straight into ``fk_repose`` silently selects the WRONG vertices (verified:
    for this mesh it selects two ``thorax_collision`` vertices for both wing
    sides, whose FK-reposed distance is ~constant regardless of wing pose,
    since thorax doesn't move when the wing joints rotate) instead of the
    intended wing-membrane vertices. This helper does the fps_300[idx]
    conversion needed to bridge ``wing_side_vertices``' fps-relative output
    into ``make_fk_repose``'s full-vertex-array index space.
    """
    from jarvis_jax.cse.silhouette_landmarks import wing_side_vertices

    z = np.load(mesh_npz, allow_pickle=True)
    fps = z["fps_300"] if "fps_300" in z.files else z[f"fps_{len(z['vertex_segment'])}"]
    sides = wing_side_vertices(mesh_npz)
    return np.array([
        fps[sides["left"]["prox"]], fps[sides["left"]["tip"]],
        fps[sides["right"]["prox"]], fps[sides["right"]["tip"]],
    ], dtype=np.int32)


def _wing_joint_qpos_indices(mj_model: mujoco.MjModel) -> dict:
    """qpos indices of the hinge joints on the ``wing_left``/``wing_right``
    bodies (yaw/roll/pitch), keyed by side.

    Used by ``run_single_fly``'s ``wing_angle_pred``/``wing_angle_stac``
    metrics: unlike the tip-prox FK distance (a geom-intrinsic constant,
    invariant to qpos -- removed, see module history), these joint angles
    directly reflect how the wing was actually posed by the solve.
    """
    out = {"left": [], "right": []}
    for j in range(mj_model.njnt):
        if mj_model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        body = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, mj_model.jnt_bodyid[j])
        if not body:
            continue
        bl = body.lower()
        if "wing" not in bl:
            continue
        if "left" in bl:
            out["left"].append(int(mj_model.jnt_qposadr[j]))
        elif "right" in bl:
            out["right"].append(int(mj_model.jnt_qposadr[j]))
    return out


def _load_sam_mask(root, split, file_name, ann_id):
    """Load the SAM mask matching ``ann_id`` for one camera image.

    Mirrors ``load_mask`` in ``demo_wingtip_triangulation.py`` /
    ``reproj_validate.py`` / ``silhouette_render_demo.py``: prefer a mask
    with ``ann_ids==ann_id & matched``, else fall back to any mask with
    ``ann_ids==ann_id``.
    """
    p = os.path.join(root, "sam3_masks", split, os.path.splitext(file_name)[0] + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    if z["masks"].shape[0] == 0:
        return None
    sel = np.where((z["ann_ids"] == ann_id) & z["matched"])[0]
    if not len(sel):
        sel = np.where(z["ann_ids"] == ann_id)[0]
    return z["masks"][sel[0]].astype(bool) if len(sel) else None


def _augment_wing_markers_stac_order(kp_data, kps_to_opt, tips_list, kp_names, *, wing_weight=0.5, min_cams=2, only_missing=True):
    """Apply ``marker_augment.augment_wing_markers`` to STAC-ordered kp arrays.

    ``augment_wing_markers.WING_MARKER_IDS`` hardcodes coco ``keypoint_names``
    ordering (WingL_V12=7, WingL_V13=8, WingR_V12=29, WingR_V13=30 — this is
    the coco order, e.g. ``instances_val.json["keypoint_names"]``). The STAC
    ik h5's ``kp_data``/``kp_names`` (from ``build_solver_inputs``) use a
    *different* permutation of the same 50 names (WingL_V12=6, WingL_V13=7,
    WingR_V12=8, WingR_V13=9 for this fly model) — calling
    ``augment_wing_markers`` directly on STAC-ordered arrays would silently
    overwrite the wrong keypoints (verified: index 7 in STAC order is
    WingL_V13, index 29 is a T2L leg marker, not a wing marker at all).

    Bridges this by permuting ``kp_data``/``kps_to_opt`` from STAC order into
    coco order (by keypoint name), calling ``augment_wing_markers`` unchanged,
    then permuting the result back to STAC order.
    """
    from jarvis_jax.cse.marker_augment import augment_wing_markers, WING_MARKER_IDS

    kp_names = list(kp_names)
    n_kp = len(kp_names)
    # coco order must be at least as long as needed to hold every referenced
    # index (WING_MARKER_IDS' max coco index) and be a valid permutation
    # target: build it as "STAC name order sorted into coco slot order" using
    # the same coco keypoint_names list the indices were defined against.
    coco_kpnames = _COCO_KEYPOINT_NAMES
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}
    missing = [n for n in kp_names if n not in name2coco]
    if missing:
        raise ValueError(f"kp_names not found in coco keypoint_names (needed for the "
                          f"WING_MARKER_IDS coco-order bridge): {missing}")
    # stac_idx_of_coco_slot[c] = the STAC-order index of the keypoint that
    # sits at coco slot c (only slots referenced by WING_MARKER_IDS matter).
    stac_idx_of_coco_slot = {name2coco[nm]: j for j, nm in enumerate(kp_names)}
    needed_slots = sorted({i for ids in WING_MARKER_IDS.values() for i in ids})
    for c in needed_slots:
        if c not in stac_idx_of_coco_slot:
            raise ValueError(f"coco wing-marker slot {c} ({coco_kpnames[c]}) has no "
                              f"matching STAC keypoint name")

    # Build a coco-ordered view: coco_kp[c] = kp_data[:, stac_idx_of_coco_slot[c]]
    # for the slots we care about; untouched slots are irrelevant (only wing
    # slots get written by augment_wing_markers) so we only need a big enough
    # coco-shaped array and can leave non-wing slots as zeros/placeholder.
    n_coco_slots = max(needed_slots) + 1
    T = kp_data.shape[0]
    coco_kp = np.zeros((T, n_coco_slots, 3), dtype=kp_data.dtype)
    coco_w = np.zeros(n_coco_slots * 3, dtype=kps_to_opt.dtype)
    for c in needed_slots:
        j = stac_idx_of_coco_slot[c]
        coco_kp[:, c, :] = kp_data[:, j, :]
        coco_w[c * 3:c * 3 + 3] = kps_to_opt[j * 3:j * 3 + 3]

    coco_kp2, coco_w2 = augment_wing_markers(
        coco_kp, coco_w, tips_list, wing_weight=wing_weight, min_cams=min_cams,
        only_missing=only_missing,
    )

    kp_data2 = np.array(kp_data, dtype=np.float64, copy=True)
    kps_to_opt2 = np.array(kps_to_opt, dtype=np.float64, copy=True)
    for c in needed_slots:
        j = stac_idx_of_coco_slot[c]
        kp_data2[:, j, :] = coco_kp2[:, c, :]
        kps_to_opt2[j * 3:j * 3 + 3] = coco_w2[c * 3:c * 3 + 3]
    return kp_data2, kps_to_opt2


# coco ``keypoint_names`` order that ``marker_augment.WING_MARKER_IDS`` is
# defined against (verified against
# red_data_unified_V3/annotations/instances_val.json). Kept as a fixed
# constant (rather than re-reading the coco json here) since
# WING_MARKER_IDS itself is a fixed constant in marker_augment.py -- both
# describe the same fixed coco keypoint schema, independent of any one
# recording's annotation file.
_COCO_KEYPOINT_NAMES = [
    "Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_A4", "Abd_tip",
    "WingL_base", "WingL_V12", "WingL_V13",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
    "WingR_base", "WingR_V12", "WingR_V13",
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
]


def extract_tips_for_frames(
    root: str,
    split: str,
    recording: str,
    fs_imgids: np.ndarray,
    calib_dir: str,
    model_xml: str,
    mesh_npz: str,
    q_init: np.ndarray,
    marker_sites: np.ndarray,
    kp_names,
    *,
    corridor: float = 12.0,
    ann_id_by_image: dict | None = None,
) -> list:
    """Per-frame SAM-silhouette wing-tip triangulation, returned in MODEL frame.

    Bridges the STAC MODEL frame (``q_init``/``marker_sites``/``kp_data``) and
    the calibrated mm/world frame (``ReprojectionTool``) exactly as
    ``silhouette_render_demo.py`` does:

      1. Repose the canonical wing tip/prox fps vertices under each frame's
         STAC qpos via ``silhouette_ik.make_fk_repose`` -> MODEL-frame prox/tip.
      2. Triangulate the recording's 50 coco keypoints for that frame into mm
         with the (possibly refined) calibration, then fit a similarity
         ``(s, R, t)`` from ``marker_sites[frame]`` -> ``kp_mm`` via
         ``_umeyama`` (matched by keypoint name, >=2-camera visible only).
         Map MODEL-frame prox/tip -> mm with this transform to get
         ``prox3d``/``predtip3d`` for ``triangulate_wing_tips``.
      3. ``triangulate_wing_tips`` returns the SAM wing tip in mm; map it back
         mm->MODEL with the inverse transform before appending to the
         returned ``tips_list``, so it lands in the same frame as ``kp_data``
         (consumed directly by ``marker_augment.augment_wing_markers``).

    Args:
        root: Dataset root (contains ``annotations/``, ``sam3_masks/``,
            ``calib_params/``).
        split: Dataset split (e.g. "val").
        recording: Recording name (e.g. "2026_03_18_15_31_22"); the coco-
            annotated fly on this recording (exactly one annotation per
            image here) is treated as the male.
        fs_imgids: (T, n_cam) bout-h5 array of coco image_ids per frame/camera.
        calib_dir: Calibration directory (refined or factory) for
            ``ReprojectionTool``.
        model_xml: Fly MJCF path, for the FK-repose anatomy.
        mesh_npz: Canonical mesh npz (wing tip/prox fps vertices + FK repose).
        q_init: (T, nq) STAC qpos trajectory (same T/order as ``fs_imgids``).
        marker_sites: (T, n_kp, 3) STAC-fitted MODEL-frame marker site
            positions (from the ik h5), used as the umeyama source per frame.
        kp_names: length-n_kp list of STAC keypoint names (same order as
            ``marker_sites``'s middle axis).
        corridor: Passed through to ``triangulate_wing_tips``.
        ann_id_by_image: Optional image_id -> chosen ann id (Phase 3 / Task 4
            per-identity selection), for solving a SPECIFIC fly on a
            multi-fly image. Default None selects the first ann per image
            (backward-compatible single-fly behavior).

    Returns:
        list of length T; each entry is a dict {"left": (X_model(3,), ncam)
        or None, "right": ...} in the MODEL frame, directly consumable by
        ``marker_augment.augment_wing_markers``.
    """
    import jax.numpy as jnp
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_landmarks import triangulate_wing_tips

    T = int(q_init.shape[0])
    assert fs_imgids.shape[0] == T, (
        f"fs_imgids has {fs_imgids.shape[0]} rows but q_init has {T} frames"
    )

    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    coco_kpnames = coco["keypoint_names"]

    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    cam_mats = [c.cameraMatrix for c in rt._camera_list]  # (3,4) per camera, DLT order

    anat = load_anatomy(model_xml, mesh_npz)
    fk = make_fk_repose(anat)

    fk_idx = _wing_fk_indices(mesh_npz)  # [L-prox, L-tip, R-prox, R-tip], full-vertex-array indices

    kp_names = list(kp_names)

    tips_list = []
    for t in range(T):
        q_t = np.asarray(q_init[t], dtype=np.float32)
        wing_verts = np.asarray(fk(jnp.asarray(q_t), indices=fk_idx))  # (4,3) MODEL frame
        prox_model = {"left": wing_verts[0], "right": wing_verts[2]}
        tip_model = {"left": wing_verts[1], "right": wing_verts[3]}

        cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
        kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann_multi, ann_id_by_image)

        if kok.sum() < 3:
            # Not enough triangulated markers this frame to fit a reliable
            # similarity transform -> skip silhouette extraction for it.
            tips_list.append({"left": None, "right": None})
            continue

        s, R, tr = _umeyama(marker_sites[t][kok], kp_mm[kok])

        prox3d_mm = {side: _model_to_mm(prox_model[side][None], s, R, tr)[0] for side in ("left", "right")}
        tip3d_mm = {side: _model_to_mm(tip_model[side][None], s, R, tr)[0] for side in ("left", "right")}

        # per-camera masks for the selected fly (this identity's ann on each
        # image), in rt camera order.
        masks = [None] * rt.num_cameras
        for c, iid in cam2img.items():
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            fn = id2file[iid]
            masks[c] = _load_sam_mask(root, split, fn, ann["id"])

        sam_tips_mm = triangulate_wing_tips(masks, cam_mats, prox3d_mm, tip3d_mm, corridor=corridor)

        frame_entry = {}
        for side in ("left", "right"):
            entry = sam_tips_mm.get(side)
            if entry is None:
                frame_entry[side] = None
                continue
            X_mm, ncam = entry
            X_model = _mm_to_model(X_mm[None], s, R, tr)[0]
            frame_entry[side] = (X_model, ncam)
        tips_list.append(frame_entry)

    return tips_list


def run_single_fly(
    recording: str,
    *,
    ik_h5: str,
    model_xml: str,
    mesh_npz: str,
    root: str,
    split: str = "val",
    calib_dir: str | None = None,
    use_silhouette: bool = True,
    wing_weight: float = 0.5,
    only_missing: bool = True,
    max_frames: int = 0,
    smooth_weight: float = 0.1,
    n_iter: int = 50,
    corridor: float = 12.0,
    out_dir: str,
    ann_id_by_image: dict | None = None,
) -> dict:
    """End-to-end single-fly silhouette-landmark IK driver (male, one recording).

    Pipeline: ``build_solver_inputs`` -> (optionally) extract SAM wing-tips
    per frame + ``augment_wing_markers`` to raise their weight in
    ``kps_to_opt`` -> ``solve_ik`` -> write qpos ``.npz`` + report dict.

    Args:
        recording: Recording name; used to find the bout h5
            (``{cse_work}/{recording}_bout.h5``, derived from ``ik_h5``'s
            directory) and the factory calib fallback.
        ik_h5: STAC ik-output h5 for this recording (see
            ``build_solver_inputs``).
        model_xml: Fly MJCF path.
        mesh_npz: Canonical mesh npz (FK repose + wing fps vertices).
        root: Dataset root (coco annotations + sam3_masks + calib_params).
        split: Dataset split the recording's frames were annotated under.
        calib_dir: Calibration directory. Defaults to the Phase-1 refined
            calibration dir if present, else the factory calibration under
            ``root/calib_params/{recording}``.
        use_silhouette: If True, augment wing markers from SAM silhouettes
            before solving (the Phase-2 pipeline); if False, solve on the
            raw STAC kp_data only (baseline, Task 5's ablation control).
        wing_weight: Passed to ``augment_wing_markers``. Default lowered to
            0.5 (from 2.0, Task 5 review) to avoid over-dragging the
            whole-body solve when a wing marker is filled.
        only_missing: Passed to ``augment_wing_markers`` (confidence gate).
            Default True: only fills wing markers that are missing/withheld
            (all-NaN) for that frame; a marker with real GT/tracked data is
            left untouched. This makes augmentation a no-op on full GT data
            (``use_silhouette=True`` must not worsen ``reproj_px`` relative
            to the ``use_silhouette=False`` baseline) and only actually fills
            markers in Task 5's withheld-keypoint ablation.
        max_frames: If >0, only process/solve the first ``max_frames`` frames
            (for fast smoke tests / dev iteration).
        smooth_weight: Passed to ``solve_ik``.
        n_iter: Passed to ``solve_ik``.
        corridor: Passed to ``extract_tips_for_frames``/``triangulate_wing_tips``.
        out_dir: Directory to write ``{recording}_qpos.npz`` into.
        ann_id_by_image: Optional image_id -> chosen ann id (Phase 3 / Task 4
            per-identity selection), forwarded to ``extract_tips_for_frames``
            and used for the reprojection-metric ann lookup, to solve a
            SPECIFIC fly on a multi-fly image. Default None preserves the
            single-fly (first-ann-per-image) behavior.

    Returns:
        dict with keys:
          qpos_shape: tuple, solved qpos.shape (T, nq).
          wing_tip_err_px / wing_tip_err_mm: per-side (left/right) mean
            distance, over frames with a valid SAM-triangulated tip, between
            the FK-reposed wing-tip vertex under the SOLVED qpos (mapped
            MODEL->mm via the per-frame umeyama bridge) and that frame's
            SAM-triangulated tip (also mm). ``_px`` additionally reprojects
            both into each visible camera and averages the pixel distance.
            NaN if ``n_frames_with_tips`` is 0 (e.g. ``use_silhouette=False``).
            Replaces the old ``wing_len_pred``/``wing_len_stac`` (tip-to-prox
            FK distance), which is a geom-intrinsic constant -- tip and prox
            share one rigid wing geom, so that distance does not move with
            qpos and was not a useful fit metric.
          wing_angle_pred / wing_angle_stac: dict {"left": [...], "right":
            [...]} of the wing hinge joint angle(s) (radians; yaw/roll/pitch,
            per ``_wing_joint_qpos_indices``), mean over frames, under the
            SOLVED qpos vs. the stored ik-h5 (STAC) q_init respectively --
            shows whether/how the wing pose actually changed.
          reproj_px: mean reprojection error (px) of the 50 kp sites (under
            solved qpos, mapped MODEL->mm via the per-frame umeyama bridge)
            against the refined-calib DLT triangulation of the annotated
            keypoints.
          n_frames_with_tips: number of frames where >=1 wing side had a
            valid (>=min_cams) SAM-triangulated tip used in the augmentation
            (0 if use_silhouette=False).
    """
    import jax.numpy as jnp
    import h5py
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    import stac_mjx.io_dict_to_hdf5 as ioh5

    if calib_dir is None:
        calib_dir = (
            _DEFAULT_REFINED_CALIB_DIR
            if os.path.exists(_DEFAULT_REFINED_CALIB_DIR)
            else _DEFAULT_FACTORY_CALIB_DIR
        )

    inputs = build_solver_inputs(ik_h5, model_xml)
    T_full = inputs["q_init"].shape[0]
    T = T_full if max_frames <= 0 else min(max_frames, T_full)

    q_init = inputs["q_init"][:T]
    kp_data = inputs["kp_data"][:T]
    kps_to_opt = inputs["kps_to_opt"]

    # marker_sites (STAC-fitted MODEL-frame per-keypoint marker positions),
    # needed as the umeyama bridge source; comes from the ik h5 directly
    # (not part of build_solver_inputs's returned dict).
    ik_raw = ioh5.load(ik_h5)
    marker_sites = np.asarray(ik_raw["marker_sites"])[:T]

    # The bout h5 lives one level up from the ik h5's directory, e.g.:
    #   cse_work/<recording>/Fruitfly_ik_v1_cse.h5
    #   cse_work/<recording>_bout.h5
    bout_h5 = os.path.join(os.path.dirname(os.path.dirname(ik_h5)), f"{recording}_bout.h5")
    with h5py.File(bout_h5, "r") as f:
        fs_imgids = f["fs_imgids"][()][:T]

    n_frames_with_tips = 0
    if use_silhouette:
        tips_list = extract_tips_for_frames(
            root, split, recording, fs_imgids, calib_dir, model_xml, mesh_npz,
            q_init, marker_sites, inputs["kp_names"], corridor=corridor,
            ann_id_by_image=ann_id_by_image,
        )
        n_frames_with_tips = sum(
            1 for f in tips_list if f.get("left") is not None or f.get("right") is not None
        )
        kp_data, kps_to_opt = _augment_wing_markers_stac_order(
            kp_data, kps_to_opt, tips_list, inputs["kp_names"],
            wing_weight=wing_weight, only_missing=only_missing,
        )
    else:
        tips_list = None

    small = dict(inputs)
    small["q_init"] = q_init
    small["kp_data"] = kp_data
    small["kps_to_opt"] = kps_to_opt

    qpos = solve_ik(small, smooth_weight=smooth_weight, n_iter=n_iter)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{recording}_qpos.npz")
    np.savez(out_path, qpos=qpos)

    # --- report metrics ---
    anat = load_anatomy(model_xml, mesh_npz)
    fk = make_fk_repose(anat)
    fk_idx = _wing_fk_indices(mesh_npz)  # [L-prox, L-tip, R-prox, R-tip], full-vertex-array indices

    # wing_angle_{pred,stac}: mean hinge joint angle(s) (yaw/roll/pitch, rad)
    # for each wing side, under the solved vs. stored qpos -- shows whether
    # the wing pose actually moved (replaces the old tip-prox FK distance,
    # which is invariant to qpos since tip+prox share one rigid wing geom).
    # Uses anat["m"] (raw mujoco.MjModel from model_xml, no mjx wrapping) --
    # mj_id2name/jnt_* introspection needs a real MjModel, not the mjx.Model
    # used for the solve itself; joint/qpos layout is identical either way
    # since the keypoint <site> elements build_solver_inputs adds don't
    # affect joints or qpos.
    wing_joint_idx = _wing_joint_qpos_indices(anat["m"])

    def _wing_angles(qtraj):
        qtraj = np.asarray(qtraj)
        return {
            side: [float(np.mean(qtraj[:, qi])) for qi in idxs]
            for side, idxs in wing_joint_idx.items()
        }

    wing_angle_pred = _wing_angles(qpos)
    wing_angle_stac = _wing_angles(np.asarray(inputs["q_init"][:T]))

    # reprojection error: solved qpos site positions (MODEL) -> mm (per-frame
    # umeyama bridge against refined-calib triangulated coco keypoints) ->
    # rt.reproject_point -> compare to the annotated 2-D keypoints (px).
    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    coco_kpnames = coco["keypoint_names"]
    kp_names = list(inputs["kp_names"])
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}

    reproj_errs = []
    # wing_tip_err_{mm,px}: FK-reposed wing-tip vertex under the SOLVED qpos
    # (mapped MODEL->mm via the same per-frame umeyama bridge) vs. that
    # frame's SAM-triangulated tip (also MODEL->mm), per side. Measures how
    # well the fitted wing actually reaches the silhouette-derived target.
    tip_err_mm = {"left": [], "right": []}
    tip_err_px = {"left": [], "right": []}
    site_idxs = inputs["site_idxs"]
    mjx_model, mjx_data = inputs["mjx_model"], inputs["mjx_data"]
    for t in range(T):
        data_t = mjx_data.replace(qpos=np.asarray(qpos[t]))
        data_t = stac_utils.kinematics(mjx_model, data_t)
        data_t = stac_utils.com_pos(mjx_model, data_t)
        sites_model = np.asarray(stac_utils.get_site_xpos(data_t, site_idxs))  # (n_kp,3)

        cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
        kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann_multi, ann_id_by_image)
        if kok.sum() < 3:
            continue
        s, R, tr = _umeyama(marker_sites[t][kok], kp_mm[kok])
        sites_mm = _model_to_mm(sites_model, s, R, tr)

        for j, nm in enumerate(kp_names):
            ci = name2coco.get(nm)
            if ci is None:
                continue
            uv_pred_all = rt.reproject_point(sites_mm[j])  # (n_cam, 2)
            for c, iid in cam2img.items():
                ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
                if ann is None:
                    continue
                kp2d = np.asarray(ann["keypoints"], dtype=float).reshape(-1, 3)
                if kp2d[ci, 2] > 0:
                    reproj_errs.append(float(np.linalg.norm(uv_pred_all[c] - kp2d[ci, :2])))

        if tips_list is not None:
            frame_tips = tips_list[t]
            wing_verts = np.asarray(fk(jnp.asarray(qpos[t].astype(np.float32)), indices=fk_idx))
            tip_model = {"left": wing_verts[1], "right": wing_verts[3]}
            for side in ("left", "right"):
                entry = frame_tips.get(side)
                if entry is None:
                    continue
                sam_tip_model, _ncam = entry
                pred_tip_mm = _model_to_mm(tip_model[side][None], s, R, tr)[0]
                sam_tip_mm = _model_to_mm(np.asarray(sam_tip_model)[None], s, R, tr)[0]
                tip_err_mm[side].append(float(np.linalg.norm(pred_tip_mm - sam_tip_mm)))

                uv_pred = rt.reproject_point(pred_tip_mm)  # (n_cam, 2)
                uv_sam = rt.reproject_point(sam_tip_mm)    # (n_cam, 2)
                for c in cam2img:
                    tip_err_px[side].append(float(np.linalg.norm(uv_pred[c] - uv_sam[c])))

    reproj_px = float(np.mean(reproj_errs)) if reproj_errs else float("nan")
    wing_tip_err_mm = {
        side: (float(np.mean(v)) if v else float("nan")) for side, v in tip_err_mm.items()
    }
    wing_tip_err_px = {
        side: (float(np.mean(v)) if v else float("nan")) for side, v in tip_err_px.items()
    }

    return dict(
        qpos_shape=tuple(qpos.shape),
        wing_tip_err_px=wing_tip_err_px,
        wing_tip_err_mm=wing_tip_err_mm,
        wing_angle_pred=wing_angle_pred,
        wing_angle_stac=wing_angle_stac,
        reproj_px=reproj_px,
        n_frames_with_tips=n_frames_with_tips,
    )


# ---------------------------------------------------------------------------
# Phase 2 / Task 5: keypoint-ablation validation (the silhouette-value gate).
# ---------------------------------------------------------------------------

# STAC-order kp_names indices of the distal wing markers (WingL_V12, WingL_V13,
# WingR_V12, WingR_V13) for this fly model -- verified in
# test_stac_order_bridge_writes_stac_indices_6_7_8_9_not_29 / the module
# docstring on _augment_wing_markers_stac_order. NOT the coco-order indices
# marker_augment.WING_MARKER_IDS uses (7, 8, 29, 30) -- kp_data here is in
# STAC order (build_solver_inputs's kp_names), so it must be withheld at
# these STAC indices for the STAC marker_cost finite-mask to actually drop
# them.
_STAC_WING_KP_IDX = {"left": (6, 7), "right": (8, 9)}


def _withhold_wing_kp(kp_data: np.ndarray) -> np.ndarray:
    """Return a copy of ``kp_data`` (T, n_kp, 3, STAC order) with the four
    distal wing marker rows (STAC idx 6,7,8,9) set to NaN for every frame,
    so the STAC marker_cost finite-mask drops them (Task 5 condition b/c).
    """
    kp2 = np.array(kp_data, dtype=np.float64, copy=True)
    wing_idx = [i for ids in _STAC_WING_KP_IDX.values() for i in ids]
    kp2[:, wing_idx, :] = np.nan
    return kp2


def _wing_tip_solved_mm(qpos, t, fk, fk_idx, marker_sites_t, kok_t, kp_mm_t):
    """FK-repose the wing-tip vertices under ``qpos[t]`` and map MODEL->mm.

    Shared helper for run_ablation's per-condition/per-frame tip evaluation:
    fits the same per-frame umeyama bridge (marker_sites -> triangulated
    kp_mm, both indexed by ``kok_t``) that ``run_single_fly``/
    ``extract_tips_for_frames`` use, then maps the FK-reposed left/right
    wing-tip vertices under the SOLVED qpos from MODEL into mm.

    Returns:
        dict {"left": (3,) mm or None, "right": ...}, or None if the frame's
        umeyama fit is not well-determined (``kok_t.sum() < 3``).
    """
    import jax.numpy as jnp

    if kok_t.sum() < 3:
        return None
    s, R, tr = _umeyama(marker_sites_t[kok_t], kp_mm_t[kok_t])
    wing_verts = np.asarray(fk(jnp.asarray(np.asarray(qpos[t]).astype(np.float32)), indices=fk_idx))
    tip_model = {"left": wing_verts[1], "right": wing_verts[3]}
    return {side: _model_to_mm(tip_model[side][None], s, R, tr)[0] for side in ("left", "right")}


def run_ablation(
    recording: str,
    *,
    ik_h5: str,
    model_xml: str,
    mesh_npz: str,
    root: str,
    split: str = "val",
    calib_dir: str | None = None,
    wing_weight: float = 0.5,
    max_frames: int = 0,
    smooth_weight: float = 0.1,
    n_iter: int = 50,
    corridor: float = 12.0,
    out_dir: str,
    ann_id_by_image: dict | None = None,
) -> dict:
    """Keypoint-ablation validation: does the silhouette recover a withheld wing?

    For the SAME frames, solves the IK three ways:

      (a) REFERENCE -- full GT keypoints, no silhouette. The "truth" the
          wing should match; also the source of the stored ik-h5 fit.
      (b) BASELINE -- wing keypoints WITHHELD (STAC kp_data idx 6,7,8,9 set
          to NaN for every frame, dropping them from marker_cost's finite
          mask), NO silhouette. The wing is unconstrained by any keypoint
          target and is expected to collapse/drift toward whatever the
          smoothness/regularization terms leave it at.
      (c) SILHOUETTE -- same wing keypoints withheld, but silhouette
          augmentation is ON: ``extract_tips_for_frames`` + confidence-gated
          ``augment_wing_markers(only_missing=True)`` fill those NaN wing
          rows with the SAM-triangulated tips before solving. Expected to
          recover the wing back toward (a).

    Args:
        recording, ik_h5, model_xml, mesh_npz, root, split, calib_dir,
            wing_weight, max_frames, smooth_weight, n_iter, corridor: see
            ``run_single_fly`` (identical semantics; this reuses the same
            input assembly / silhouette extraction / solve pieces).
        out_dir: directory to write each condition's ``{recording}_{cond}_
            qpos.npz`` into (cond in {"reference", "baseline", "silhouette"}).
        ann_id_by_image: Optional image_id -> chosen ann id (Phase 3 / Task 4
            per-identity selection), forwarded to ``extract_tips_for_frames``
            and used for the reprojection-metric ann lookup, to solve a
            SPECIFIC fly on a multi-fly image. Default None preserves the
            single-fly (first-ann-per-image) behavior.

    Returns:
        dict with keys:
          conditions: {"reference": {...}, "baseline": {...}, "silhouette":
            {...}} -- per-condition dict with keys ``wing_tip_err_mm``,
            ``wing_tip_err_px`` (dict {"left","right"}: mean distance
            between the FK-reposed SOLVED wing tip (under THIS condition's
            qpos) and that frame's SAM-triangulated tip, mm/px -- computed
            for all three conditions, regardless of whether that condition's
            solve actually used the SAM tip as a target, so "reference" and
            "baseline" show how far an unaugmented fit lands from the
            silhouette's wing extent; NaN where no valid tip/umeyama fit
            exists for that frame), ``dist_to_ref_mm``/``dist_to_ref_px``
            (dict {"left","right"}: mean 3-D/pixel distance between this
            condition's SOLVED wing tip and the REFERENCE (a) SOLVED wing
            tip for the same frame -- NaN for "reference" itself),
            ``reproj_px`` (mean reprojection error, px, of the NON-wing kp
            sites only -- sanity check that body/legs don't degrade), and
            ``qpos_shape``.
          recovery_ratio_mm / recovery_ratio_px: dict {"left","right"} =
            (dist_b_to_ref - dist_c_to_ref) / dist_b_to_ref, in mm and px
            respectively. >0 means the silhouette moved the withheld-wing
            fit back toward the REFERENCE (a) solve; ~1 means full recovery
            toward reference; <=0 means no/negative recovery toward it.
          recovery_ratio_to_sam_mm / recovery_ratio_to_sam_px: dict
            {"left","right"} = (wing_tip_err_b - wing_tip_err_c) /
            wing_tip_err_b -- the same recovery-ratio formula but measured
            against the SAM-triangulated tip directly (the silhouette's own
            ground truth for wing extent) rather than against the
            keypoint-only reference solve. NOTE: the reference (a) solve is
            itself keypoint-limited, not ground truth -- if the GT wing
            keypoints (V12/V13) are systematically offset from the true
            silhouette extent (e.g. annotation convention placing V13
            proximal to the visible membrane edge), *_to_ref and
            *_to_sam can disagree in sign: recovering the wing toward the
            silhouette's true extent (positive *_to_sam) can simultaneously
            move it away from the keypoint-only reference (negative
            *_to_ref). Both are reported so this can be diagnosed rather
            than hidden.
          n_frames_with_tips: number of frames with >=1 valid SAM-
            triangulated wing tip (shared across b/c -- extraction does not
            depend on which condition's kp_data is solved).
    """
    import jax.numpy as jnp
    import h5py
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    import stac_mjx.io_dict_to_hdf5 as ioh5

    if calib_dir is None:
        calib_dir = (
            _DEFAULT_REFINED_CALIB_DIR
            if os.path.exists(_DEFAULT_REFINED_CALIB_DIR)
            else _DEFAULT_FACTORY_CALIB_DIR
        )

    inputs = build_solver_inputs(ik_h5, model_xml)
    T_full = inputs["q_init"].shape[0]
    T = T_full if max_frames <= 0 else min(max_frames, T_full)

    q_init = inputs["q_init"][:T]
    kp_data_ref = inputs["kp_data"][:T]
    kps_to_opt_ref = inputs["kps_to_opt"]
    kp_names = list(inputs["kp_names"])

    ik_raw = ioh5.load(ik_h5)
    marker_sites = np.asarray(ik_raw["marker_sites"])[:T]

    bout_h5 = os.path.join(os.path.dirname(os.path.dirname(ik_h5)), f"{recording}_bout.h5")
    with h5py.File(bout_h5, "r") as f:
        fs_imgids = f["fs_imgids"][()][:T]

    # Condition (a): REFERENCE -- full GT keypoints, no silhouette.
    small_a = dict(inputs)
    small_a["q_init"] = q_init
    small_a["kp_data"] = kp_data_ref
    small_a["kps_to_opt"] = kps_to_opt_ref
    qpos_a = solve_ik(small_a, smooth_weight=smooth_weight, n_iter=n_iter)

    # Condition (b): BASELINE -- wing keypoints withheld, no silhouette.
    kp_data_withheld = _withhold_wing_kp(kp_data_ref)
    small_b = dict(inputs)
    small_b["q_init"] = q_init
    small_b["kp_data"] = kp_data_withheld
    small_b["kps_to_opt"] = kps_to_opt_ref
    qpos_b = solve_ik(small_b, smooth_weight=smooth_weight, n_iter=n_iter)

    # Silhouette extraction (shared by condition (c); independent of which
    # condition's kp_data is solved -- it repose-FKs the STORED q_init and
    # triangulates from the recording's coco annotations/SAM masks).
    tips_list = extract_tips_for_frames(
        root, split, recording, fs_imgids, calib_dir, model_xml, mesh_npz,
        q_init, marker_sites, kp_names, corridor=corridor,
        ann_id_by_image=ann_id_by_image,
    )
    n_frames_with_tips = sum(
        1 for f in tips_list if f.get("left") is not None or f.get("right") is not None
    )

    # Condition (c): SILHOUETTE -- wing keypoints withheld + augmentation ON.
    kp_data_c, kps_to_opt_c = _augment_wing_markers_stac_order(
        kp_data_withheld, kps_to_opt_ref, tips_list, kp_names,
        wing_weight=wing_weight, only_missing=True,
    )
    small_c = dict(inputs)
    small_c["q_init"] = q_init
    small_c["kp_data"] = kp_data_c
    small_c["kps_to_opt"] = kps_to_opt_c
    qpos_c = solve_ik(small_c, smooth_weight=smooth_weight, n_iter=n_iter)

    os.makedirs(out_dir, exist_ok=True)
    qpos_by_cond = {"reference": qpos_a, "baseline": qpos_b, "silhouette": qpos_c}
    for cond, qp in qpos_by_cond.items():
        np.savez(os.path.join(out_dir, f"{recording}_{cond}_qpos.npz"), qpos=qp)

    # --- shared per-frame machinery for tip/reproj metrics ---
    anat = load_anatomy(model_xml, mesh_npz)
    fk = make_fk_repose(anat)
    fk_idx = _wing_fk_indices(mesh_npz)

    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    coco_kpnames = coco["keypoint_names"]
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}
    wing_stac_idx = {_STAC_WING_KP_IDX["left"][0], _STAC_WING_KP_IDX["left"][1],
                     _STAC_WING_KP_IDX["right"][0], _STAC_WING_KP_IDX["right"][1]}

    # Precompute, per frame, the umeyama bridge inputs (kp_mm/kok) once --
    # identical across conditions (depends only on the recording's
    # annotations/calibration + the stored marker_sites, not on qpos).
    frame_bridge = []  # list of (cam2img, kp_mm, kok) or None if under-determined
    for t in range(T):
        cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
        kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann_multi, ann_id_by_image)
        frame_bridge.append((cam2img, kp_mm, kok) if kok.sum() >= 3 else None)

    def _solved_wing_tips_mm(qpos):
        """Per-frame {"left": mm(3,) or None, "right": ...} under this qpos."""
        out = []
        for t in range(T):
            fb = frame_bridge[t]
            if fb is None:
                out.append(None)
                continue
            _cam2img, kp_mm, kok = fb
            out.append(_wing_tip_solved_mm(qpos, t, fk, fk_idx, marker_sites[t], kok, kp_mm))
        return out

    def _non_wing_reproj_px(qpos):
        """Mean reprojection error (px) of the NON-wing kp sites only."""
        site_idxs = inputs["site_idxs"]
        mjx_model, mjx_data = inputs["mjx_model"], inputs["mjx_data"]
        errs = []
        for t in range(T):
            fb = frame_bridge[t]
            if fb is None:
                continue
            cam2img, kp_mm, kok = fb
            s, R, tr = _umeyama(marker_sites[t][kok], kp_mm[kok])
            data_t = mjx_data.replace(qpos=np.asarray(qpos[t]))
            data_t = stac_utils.kinematics(mjx_model, data_t)
            data_t = stac_utils.com_pos(mjx_model, data_t)
            sites_model = np.asarray(stac_utils.get_site_xpos(data_t, site_idxs))
            sites_mm = _model_to_mm(sites_model, s, R, tr)
            for j, nm in enumerate(kp_names):
                if j in wing_stac_idx:
                    continue
                ci = name2coco.get(nm)
                if ci is None:
                    continue
                uv_pred_all = rt.reproject_point(sites_mm[j])
                for c, iid in cam2img.items():
                    ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
                    if ann is None:
                        continue
                    kp2d = np.asarray(ann["keypoints"], dtype=float).reshape(-1, 3)
                    if kp2d[ci, 2] > 0:
                        errs.append(float(np.linalg.norm(uv_pred_all[c] - kp2d[ci, :2])))
        return float(np.mean(errs)) if errs else float("nan")

    tips_a = _solved_wing_tips_mm(qpos_a)
    tips_b = _solved_wing_tips_mm(qpos_b)
    tips_c = _solved_wing_tips_mm(qpos_c)

    def _tip_err_to_sam(tips_solved):
        """Mean mm/px distance between the solved wing tip and this frame's
        SAM-triangulated tip (tips_list), per side."""
        err_mm = {"left": [], "right": []}
        err_px = {"left": [], "right": []}
        for t in range(T):
            solved = tips_solved[t]
            if solved is None:
                continue
            frame_tips = tips_list[t]
            fb = frame_bridge[t]
            cam2img = fb[0] if fb is not None else {}
            for side in ("left", "right"):
                entry = frame_tips.get(side)
                if entry is None:
                    continue
                sam_tip_model, _ncam = entry
                _cam2img_fb, kp_mm, kok = fb
                s, R, tr = _umeyama(marker_sites[t][kok], kp_mm[kok])
                sam_tip_mm = _model_to_mm(np.asarray(sam_tip_model)[None], s, R, tr)[0]
                err_mm[side].append(float(np.linalg.norm(solved[side] - sam_tip_mm)))
                uv_pred = rt.reproject_point(solved[side])
                uv_sam = rt.reproject_point(sam_tip_mm)
                for c in cam2img:
                    err_px[side].append(float(np.linalg.norm(uv_pred[c] - uv_sam[c])))
        return (
            {s: (float(np.mean(v)) if v else float("nan")) for s, v in err_mm.items()},
            {s: (float(np.mean(v)) if v else float("nan")) for s, v in err_px.items()},
        )

    def _dist_to_ref(tips_solved):
        """Mean mm/px distance between the solved wing tip and the REFERENCE
        (a) solved wing tip for the same frame, per side."""
        dist_mm = {"left": [], "right": []}
        dist_px = {"left": [], "right": []}
        for t in range(T):
            solved = tips_solved[t]
            ref = tips_a[t]
            if solved is None or ref is None:
                continue
            fb = frame_bridge[t]
            cam2img = fb[0] if fb is not None else {}
            for side in ("left", "right"):
                dist_mm[side].append(float(np.linalg.norm(solved[side] - ref[side])))
                uv_pred = rt.reproject_point(solved[side])
                uv_ref = rt.reproject_point(ref[side])
                for c in cam2img:
                    dist_px[side].append(float(np.linalg.norm(uv_pred[c] - uv_ref[c])))
        return (
            {s: (float(np.mean(v)) if v else float("nan")) for s, v in dist_mm.items()},
            {s: (float(np.mean(v)) if v else float("nan")) for s, v in dist_px.items()},
        )

    wing_tip_err_mm_a, wing_tip_err_px_a = _tip_err_to_sam(tips_a)
    wing_tip_err_mm_b, wing_tip_err_px_b = _tip_err_to_sam(tips_b)
    wing_tip_err_mm_c, wing_tip_err_px_c = _tip_err_to_sam(tips_c)

    nan_sides = {"left": float("nan"), "right": float("nan")}
    dist_to_ref_mm_a, dist_to_ref_px_a = nan_sides, nan_sides
    dist_to_ref_mm_b, dist_to_ref_px_b = _dist_to_ref(tips_b)
    dist_to_ref_mm_c, dist_to_ref_px_c = _dist_to_ref(tips_c)

    conditions = {
        "reference": dict(
            qpos_shape=tuple(qpos_a.shape),
            wing_tip_err_mm=wing_tip_err_mm_a,
            wing_tip_err_px=wing_tip_err_px_a,
            dist_to_ref_mm=dist_to_ref_mm_a,
            dist_to_ref_px=dist_to_ref_px_a,
            reproj_px=_non_wing_reproj_px(qpos_a),
        ),
        "baseline": dict(
            qpos_shape=tuple(qpos_b.shape),
            wing_tip_err_mm=wing_tip_err_mm_b,
            wing_tip_err_px=wing_tip_err_px_b,
            dist_to_ref_mm=dist_to_ref_mm_b,
            dist_to_ref_px=dist_to_ref_px_b,
            reproj_px=_non_wing_reproj_px(qpos_b),
        ),
        "silhouette": dict(
            qpos_shape=tuple(qpos_c.shape),
            wing_tip_err_mm=wing_tip_err_mm_c,
            wing_tip_err_px=wing_tip_err_px_c,
            dist_to_ref_mm=dist_to_ref_mm_c,
            dist_to_ref_px=dist_to_ref_px_c,
            reproj_px=_non_wing_reproj_px(qpos_c),
        ),
    }

    def _recovery_ratio(err_b, err_c):
        out = {}
        for side in ("left", "right"):
            eb, ec = err_b[side], err_c[side]
            if not (np.isfinite(eb) and np.isfinite(ec)) or eb == 0:
                out[side] = float("nan")
            else:
                out[side] = float((eb - ec) / eb)
        return out

    # Two recovery-ratio views (both requested; they can disagree in sign --
    # see the "reference is keypoint-limited, not ground truth" note below):
    #   *_to_ref: (dist_b_to_ref - dist_c_to_ref) / dist_b_to_ref -- did the
    #     silhouette move the withheld-wing fit back toward the REFERENCE (a)
    #     solve's wing tip?
    #   *_to_sam: (wing_tip_err_b - wing_tip_err_c) / wing_tip_err_b -- did
    #     the silhouette move the withheld-wing fit closer to the actual
    #     SAM-triangulated tip (the silhouette's own ground truth for wing
    #     extent, independent of any keypoint-based reference)?
    recovery_ratio_mm = _recovery_ratio(dist_to_ref_mm_b, dist_to_ref_mm_c)
    recovery_ratio_px = _recovery_ratio(dist_to_ref_px_b, dist_to_ref_px_c)
    recovery_ratio_to_sam_mm = _recovery_ratio(wing_tip_err_mm_b, wing_tip_err_mm_c)
    recovery_ratio_to_sam_px = _recovery_ratio(wing_tip_err_px_b, wing_tip_err_px_c)

    return dict(
        conditions=conditions,
        recovery_ratio_mm=recovery_ratio_mm,
        recovery_ratio_px=recovery_ratio_px,
        recovery_ratio_to_sam_mm=recovery_ratio_to_sam_mm,
        recovery_ratio_to_sam_px=recovery_ratio_to_sam_px,
        n_frames_with_tips=n_frames_with_tips,
    )
