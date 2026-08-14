#!/usr/bin/env python3
"""Staged STAC IK: snapshot qpos after each solver stage.

``stac-mjx/stac_mjx/stac.py:296`` (``Stac.fit_offsets``) already runs
``root_optimization`` then ``pose_optimization`` (alternating with
``offset_optimization`` across ``N_ITERS`` iterations). The explainer needs
the INTERMEDIATE states between those calls, so this module mirrors
``fit_offsets``'s control flow up through the first ``root_optimization`` +
``pose_optimization`` pair and records ``mjx_data.qpos`` after each stage,
instead of only keeping the final result. Same solver calls, same argument
order -- the "misaligned" state Act 3 opens on is the optimiser's own frame
zero, not anything hand-drawn. ``offset_optimization`` (marker-offset
recalibration) is deliberately NOT run here: it isn't one of the four visual
beats the explainer needs, and skipping it keeps this driver a strict
subset of ``fit_offsets`` rather than a reimplementation.

Four stages, in order:
  1. ``default`` -- the generic body model at its MJCF rest pose, residual
     measured against the RAW (unscaled) triangulated keypoints.
  2. ``scaled``  -- SAME model, SAME rest qpos (the model geometry is never
     touched -- no mesh morphing). What changes is the keypoint target: a
     single global body-size scale factor is applied to the keypoints
     themselves, computed by ``compute_shared_scale`` -- the SAME Procrustes/
     Umeyama trunk-marker scale ``scripts/preprocess_keypoints_for_ik.py``
     computes during real preprocessing (``cfg.preprocessing.scaling``,
     reused by import here, not reimplemented). This is deliberately NOT
     ``stac_mjx.rescale.rescale_per_segment`` (per-segment mesh morphing) --
     scaling belongs to preprocessing, not to a mesh-morph inside the IK
     stage. The visual "scale" beat is therefore the keypoint cloud
     contracting/expanding to match a FIXED mesh, not the mesh changing size.
     Because the model doesn't change, ``qpos_scaled`` is bit-identical to
     ``qpos_default`` -- only the residual (measured against the rescaled
     keypoints) differs.
  3. ``root``    -- ``compute_stac.root_optimization`` on the SAME model,
     fitting the globally-rescaled keypoints: translates the free-joint
     root to the ``ROOT_OPTIMIZATION_KEYPOINT`` (Scutellum) at
     ``frame_for_stills``.
  4. ``pose``    -- ``compute_stac.pose_optimization`` (the jaxls trajectory
     LM solver, since ``anatomy=v1`` sets ``USE_JAXLS: True``) over the WHOLE
     clip, fitting the rescaled keypoints; this is where per-frame
     ORIENTATION actually gets fit (via ``JAXLS_ORIENTATION_KEYPOINTS``
     warm-start) and where all 58 joint angles get solved.

IMPORTANT, read from the solver rather than assumed: ``anatomy/v1.yaml`` sets
``TRUNK_OPTIMIZATION_KEYPOINTS: {}`` (empty). ``root_optimization``'s own
keypoint-fit loss is weighted by ``trunk_kps`` (see
``stac_core.q_loss``: ``residual = residual * kps_to_opt``), so with an empty
trunk set that loss is uniformly zero and the ROOT solve only performs the
manual translation already coded into ``root_optimization`` (q0[:3] set to
the root keypoint) -- it does NOT explicitly fit a rotation; the quaternion
carried into the ``root`` stage is whatever the model's rest orientation is.
The rotational alignment ("orients") the narrative expects is explicitly
fit only inside ``pose_optimization``'s jaxls warm-start
(``_estimate_orientation_from_keypoints`` over ``JAXLS_ORIENTATION_
KEYPOINTS: rear=Scutellum, left=WingL_base, right=WingR_base,
front=Antenna_Base``). This module does NOT paper over that -- it renders
the ``root`` stage exactly as the solver leaves it. EMPIRICALLY, at
``frame_for_stills=450``, the ``root`` panel still comes out anatomically
correct (head keypoints land on the mesh's head, not the abdomen) -- this
is because the model's built-in rest quaternion happens to be close to
this frame's real heading, not because ``root_optimization`` fit it. That
is a property of this frame, not a guarantee for every frame; a bout where
the fly's heading differs a lot from the model's rest orientation could
show a visibly wrong ``root`` panel even though the underlying solve is
identical. The QC step renders and reports plainly rather than assuming.

Units: mm throughout (kp3d, offsets, residuals). SCALE_FACTOR and
MOCAP_SCALE_FACTOR are both 1 in anatomy/v1.yaml, i.e. the model's own MJCF
units already correspond to mm (independently validated in Task 6: body
length 2.4 mm vs. an independent baseline's 2.395 mm) -- but that only
validates the DATA's units, not whether the model's own rest-pose size
matches this fly, which is exactly what the ``compute_shared_scale`` step
above checks and corrects.

Outputs (both under ``<clip>/ik_explainer/``):
  predictions/06_stages.npz -- qpos_default/scaled/root/pose (nq,),
    qpos_seq (N,nq), residual_mm (4,), stage_names (4,), kp_names (50,),
    shared_scale (scalar).
  predictions/05_stac_ik.h5 -- packaged via ``Stac._package_data`` +
    ``stac_mjx.io.save_data_to_h5`` (the SAME packaging/saving code the real
    pipeline uses), from this single root+pose pass. NOTE: since
    ``offset_optimization`` is skipped, this h5 is not the full production
    fit -- it exists to back the explainer, not to replace a real STAC run.
    ``qvel`` is written as zeros (not computed) for the same reason.
  qc/05_ik_stages.png -- two rows x four stage columns. Row 1 uses the
    model's ``hero`` camera (matches the explainer video's framing): grey
    mesh + cyan keypoint cloud + residual (mm) burned in. Row 2 is a wide
    diagnostic camera that always keeps the full keypoint cloud in frame
    (hero's fixed relative offset crops most of it when the model sits far
    from the keypoints, e.g. ``default``/``scaled``) with keypoints coloured
    by anatomical group (red=head, yellow=thorax, blue=abdomen,
    magenta=legs) so a head/abdomen flip would be directly visible instead
    of inferred from an undifferentiated cyan cloud.

EXPECTATION for the QC figure: residual decreases monotonically default ->
scaled -> root -> pose, and by the ``root`` panel the mesh is anatomically
oriented (head end at the head keypoints). FALSIFICATION: residual flat or
rising, or the mesh 180 deg off with its head sitting on the abdomen
keypoints -- a flipped fit can score similarly to a correct one on a
roughly symmetric marker set, so this is only catchable by eye (see the
docstring section above: this IS a real risk given TRUNK_OPTIMIZATION_
KEYPOINTS is empty in this config).
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
import time
from pathlib import Path

import hydra
import imageio.v2 as imageio
import mujoco
import numpy as np
from jax import numpy as jp

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))
sys.path.insert(0, str(_REPO / "stac-mjx"))

from scripts.viz.ik_explainer import clip_io, draw  # noqa: E402
from scripts.viz.ik_explainer.kp_colors import jarvis_kp_colors_rgb01  # noqa: E402
from scripts.preprocess_keypoints_for_ik import compute_shared_scale  # noqa: E402
from stac_mjx import compute_stac, io as stac_io  # noqa: E402
from stac_mjx import utils as stac_utils  # noqa: E402
from stac_mjx.stac import Stac  # noqa: E402
from utils.path_utils import register_custom_resolvers  # noqa: E402
from viz.core.colors import PALETTE  # noqa: E402

register_custom_resolvers()

STAGE_NAMES = ["default", "scaled", "root", "pose"]
CAMERA = "hero"
CYAN = np.array([0.0, 0.9, 1.0, 1.0], dtype=np.float32)


def marker_residual_mm(mjx_model, mjx_data, kp_frame, body_site_idxs):
    """Mean Euclidean distance (mm) between model marker sites and keypoints."""
    sites = np.asarray(mjx_data.site_xpos)[np.asarray(body_site_idxs)]
    tgt = np.asarray(kp_frame).reshape(-1, 3)
    return float(np.nanmean(np.linalg.norm(sites - tgt, axis=-1)))


def _build_cfg():
    """Compose the same unified config tree `scripts/run_stac.py` uses.

    `dataset=courtship` (this clip's session is a courtship recording) is
    selected so `cfg.preprocessing.scaling.*` (the shared-scale parameters,
    reused below) and `cfg.stac.*` interpolations resolve; we never touch
    `cfg.stac.data_path`/`save_path` since kp3d is loaded directly from
    `04_kp3d_filt.npz`, not the preprocessing h5 pipeline.
    `paths.body_model_dir` is overridden to this repo's `models/` (identical
    byte-for-byte to the external `fruitfly_body_models` copy anatomy/v1.yaml
    points at by default) so the model path matches the brief exactly.
    """
    with hydra.initialize_config_dir(config_dir=str(_REPO / "configs"), version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "anatomy=v1",
                "dataset=courtship",
                f"paths.body_model_dir={_REPO}/models",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ],
        )
    return cfg


def _tracking_site_map(mj_model, kp_names):
    """keypoint name -> `tracking[name]` site index in `mj_model` (present only)."""
    out = {}
    for name in kp_names:
        idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f"tracking[{name}]")
        if idx >= 0:
            out[name] = idx
    return out


def _rest_snapshot(stac_obj):
    """Mirror `fit_offsets`' first block (stac.py:305-312): load mjx, set the
    initial marker offsets, run forward kinematics. Returns (mjx_model, mjx_data)
    at the model's rest qpos."""
    mjx_model, mjx_data = stac_utils.mjx_load(stac_obj._mj_model)
    offsets = jp.copy(stac_utils.get_site_pos(mjx_model, stac_obj._body_site_idxs))
    stac_obj._offsets = offsets
    mjx_model = stac_utils.set_site_pos(mjx_model, offsets, stac_obj._body_site_idxs)
    mjx_data = stac_utils.kinematics(mjx_model, mjx_data)
    mjx_data = stac_utils.com_pos(mjx_model, mjx_data)
    return mjx_model, mjx_data


def run(clip: str = clip_io.CLIP_DEFAULT, frame_for_stills: int = 450) -> dict:
    dirs = clip_io.out_dirs(clip)
    npz_path = dirs["predictions"] / "04_kp3d_filt.npz"
    with np.load(npz_path, allow_pickle=True) as z:
        kp3d = np.asarray(z["kp3d"], dtype=np.float64)  # (T, 50, 3) mm, MODEL order
        kp_names = [str(n) for n in z["kp_names"]]

    T, K, _ = kp3d.shape
    print(f"[stage_ik] {npz_path}: kp3d {kp3d.shape}, frame_for_stills={frame_for_stills}")

    # Keypoint-order guard: the ordering bug CLAUDE.md records scrambled
    # anatomy while every numeric QC stayed green. `04_kp3d_filt.npz` claims
    # MODEL order; verify against the same source `Stac` reads keypoint
    # names from (configs/anatomy/v1.yaml:KP_NAMES via clip_io).
    expected_names = clip_io.model_kp_names()
    if kp_names != expected_names:
        raise ValueError(
            "kp_names from 04_kp3d_filt.npz do not match configs/anatomy/"
            "v1.yaml:KP_NAMES (MODEL order) -- refusing to run STAC on "
            "possibly-mis-ordered keypoints. First mismatch: "
            f"{next((a, b) for a, b in zip(kp_names, expected_names) if a != b)}"
        )

    # --- Units sanity check (CLAUDE.md: a 38x body-scale error was silently
    # absorbed by marker offsets and invisible to residual/NaN checks) ---
    ab_i = kp_names.index("Antenna_Base")
    tip_i = kp_names.index("Abd_tip")
    body_len_mm = np.linalg.norm(kp3d[:, ab_i] - kp3d[:, tip_i], axis=-1)
    median_len = float(np.nanmedian(body_len_mm))
    if not (1.5 <= median_len <= 4.0):
        raise ValueError(
            f"median Antenna_Base->Abd_tip distance = {median_len:.3f} mm, "
            "outside the expected [1.5, 4.0] mm fly body-length range -- "
            "this is exactly the 38x body-scale failure mode CLAUDE.md "
            "records (marker offsets silently absorbing a scale bug); "
            "refusing to run STAC."
        )
    print(f"[stage_ik] units check OK: median Antenna_Base-Abd_tip = {median_len:.3f} mm")

    cfg = _build_cfg()
    xml_path = str(cfg.model.MJCF_PATH)
    print(f"[stage_ik] xml: {xml_path}")
    assert list(cfg.model.KEYPOINT_MODEL_PAIRS.keys()) == kp_names, (
        "cfg.model.KEYPOINT_MODEL_PAIRS key order != kp_names order -- "
        "Stac's site index map would silently misalign to the wrong keypoint."
    )

    t0 = time.time()

    # ---------------- Single model: no mesh morphing, ever -----------------
    stac = Stac(xml_path, cfg, kp_names)
    mjx_model, mjx_data0 = _rest_snapshot(stac)

    kp_flat_raw_np = kp3d.reshape(T, -1)  # (T, 150): keypoint-major, xyz contiguous

    # ---------------- Stage 1: default (rest pose, RAW keypoints) ----------
    qpos_default = np.array(mjx_data0.qpos)
    residual_default = marker_residual_mm(
        mjx_model, mjx_data0, kp_flat_raw_np[frame_for_stills], stac._body_site_idxs
    )
    print(f"[stage_ik] default: nq={qpos_default.shape[0]} residual={residual_default:.3f} mm")

    # ---------------- Stage 2: scaled (SAME rest pose, RESCALED keypoints) -
    # Global body-size scale, computed exactly as real preprocessing does
    # (scripts/preprocess_keypoints_for_ik.py:compute_shared_scale, reused by
    # import) -- NOT stac_mjx.rescale.rescale_per_segment (no mesh morphing).
    scaling_cfg = cfg.preprocessing.scaling
    skeleton_to_mujoco = _tracking_site_map(stac._mj_model, kp_names)
    shared_scale = compute_shared_scale(
        kp3d, kp_names, stac._mj_model, skeleton_to_mujoco,
        scale_keypoints=list(scaling_cfg.get("trunk_keypoints",
                                              ["Scutellum", "WingL_base", "WingR_base", "Abd_A4", "Abd_tip"])),
        robust_stat=str(scaling_cfg.get("robust_stat", "median")),
        estimator=str(scaling_cfg.get("estimator", "umeyama")),
        robust=str(scaling_cfg.get("robust", "none")),
    )
    if shared_scale is None:
        raise RuntimeError(
            "compute_shared_scale returned None (too few valid trunk-marker "
            "frames) -- cannot proceed without a body-size scale."
        )
    # NOTE on magnitude: this is expected to be far from 1.0, and that is NOT
    # the CLAUDE.md 38x-scale failure mode. The model's `<mesh scale="0.1 0.1
    # 0.1"/>` default plus its body-graph `pos` values put its own rest-pose
    # Antenna_Base->Abd_tip distance at ~0.289 (model units) -- a deliberately
    # inflated, numerically-convenient body size (a common MuJoCo/STAC
    # convention for tiny-animal solver conditioning), not literal mm. Real
    # mm data (median 2.405 mm here) must be shrunk by roughly that same
    # ~0.289/2.405 = 0.120 to land in the model's own units, so a measured
    # `shared_scale` near 0.12 is CORROBORATING evidence the fit is set up
    # correctly, not a red flag. The units guard that matters (CLAUDE.md's
    # 38x scenario) is the earlier body-length assert on the raw mm data.
    print(f"[stage_ik] shared_scale (preprocessing-style, Umeyama trunk fit) = {shared_scale:.4f}")

    kp3d_scaled = kp3d * shared_scale
    kp_flat_np = kp3d_scaled.reshape(T, -1)
    kp_flat = jp.asarray(kp_flat_np)

    qpos_scaled = qpos_default.copy()  # model untouched -- only the kp target changed
    residual_scaled = marker_residual_mm(
        mjx_model, mjx_data0, kp_flat_np[frame_for_stills], stac._body_site_idxs
    )
    print(f"[stage_ik] scaled:  residual={residual_scaled:.3f} mm")

    # ---------------- Stage 3: root_optimization ---------------------------
    mjx_data_root = compute_stac.root_optimization(
        stac.stac_core_obj, mjx_model, mjx_data0, kp_flat,
        stac._root_kp_idx, stac._lb, stac._ub,
        stac._body_site_idxs, stac._trunk_kps, frame=frame_for_stills,
    )
    qpos_root = np.array(mjx_data_root.qpos)
    residual_root = marker_residual_mm(
        mjx_model, mjx_data_root, kp_flat_np[frame_for_stills], stac._body_site_idxs
    )
    print(f"[stage_ik] root:    residual={residual_root:.3f} mm")

    # ---------------- Stage 4: pose_optimization (full sequence) -----------
    (mjx_data_pose, qposes, xposes, xquats, marker_sites, _frame_time, _frame_error) = (
        compute_stac.pose_optimization(
            stac.stac_core_obj, mjx_model, mjx_data_root, kp_flat,
            stac._lb, stac._ub, stac._body_site_idxs,
            stac._indiv_parts, stac._kp_weights,
        )
    )
    qposes_np = np.array(qposes)
    qpos_pose = qposes_np[frame_for_stills]
    mjx_data_pose_frame = mjx_data_root.replace(qpos=jp.asarray(qpos_pose))
    mjx_data_pose_frame = stac_utils.kinematics(mjx_model, mjx_data_pose_frame)
    mjx_data_pose_frame = stac_utils.com_pos(mjx_model, mjx_data_pose_frame)
    residual_pose = marker_residual_mm(
        mjx_model, mjx_data_pose_frame, kp_flat_np[frame_for_stills], stac._body_site_idxs
    )
    print(f"[stage_ik] pose:    residual={residual_pose:.3f} mm")

    solve_time = time.time() - t0
    print(f"[stage_ik] solve wall time: {solve_time:.1f} s")

    residual_mm = np.array([residual_default, residual_scaled, residual_root, residual_pose])
    if not np.all(np.diff(residual_mm) <= 1e-9):
        raise RuntimeError(
            "residuals did not decrease monotonically across "
            f"default->scaled->root->pose: {residual_mm}. Monotonic decrease "
            "is the explainer video's core claim (each stage is a real "
            "improvement over the last) -- refusing to produce output that "
            "contradicts it rather than scrolling past a warning."
        )
    print(f"[stage_ik] residuals monotonically decreasing: {residual_mm}")

    # ---------------- Save 06_stages.npz -----------------------------------
    stages_path = dirs["predictions"] / "06_stages.npz"
    np.savez(
        stages_path,
        qpos_default=qpos_default, qpos_scaled=qpos_scaled,
        qpos_root=qpos_root, qpos_pose=qpos_pose, qpos_seq=qposes_np,
        residual_mm=residual_mm, stage_names=np.array(STAGE_NAMES),
        kp_names=np.array(kp_names), frame_for_stills=np.array(frame_for_stills),
        shared_scale=np.array(shared_scale),
    )
    print(f"[stage_ik] saved {stages_path}")

    # ---------------- Save 05_stac_ik.h5 (stac_mjx's own packaging/io) -----
    packaged = stac._package_data(
        mjx_model, qposes_np, np.array(xposes), np.array(xquats),
        np.array(marker_sites), kp_flat_np,
    )
    packaged.qvel = np.zeros_like(packaged.qpos)  # not computed; see module docstring
    h5_path = dirs["predictions"] / "05_stac_ik.h5"
    stac_io.save_data_to_h5(config=cfg, file_path=str(h5_path), **packaged.as_dict())
    print(f"[stage_ik] saved {h5_path}")

    return {
        "qpos_default": qpos_default, "qpos_scaled": qpos_scaled,
        "qpos_root": qpos_root, "qpos_pose": qpos_pose, "qpos_seq": qposes_np,
        "residual_mm": residual_mm, "stage_names": STAGE_NAMES, "kp_names": kp_names,
        "mj_model": stac._mj_model,
        "kp3d_raw": kp3d, "kp3d_scaled": kp3d_scaled,
        "frame_for_stills": frame_for_stills, "clip": clip,
        "solve_time_s": solve_time, "shared_scale": shared_scale,
    }


def _add_sphere(scene, pos, rgba, size):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
        np.asarray(pos, dtype=float), np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_bone(scene, p_from, p_to, rgba, radius):
    """Thin capsule connector between two 3D points -- a skeleton "bone" --
    via `mjv_connector` (task-14 round 3, Act 3's skeleton redesign).
    `mjv_initGeom` must run first to set colour/other geom properties;
    `mjv_connector` then overwrites (type, size, pos, mat) to place a
    CAPSULE-type connector spanning `p_from`->`p_to` with the given radius."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, float(radius),
        np.asarray(p_from, dtype=np.float64), np.asarray(p_to, dtype=np.float64),
    )
    scene.ngeom += 1


def _bgr255_to_rgb01(bgr):
    """viz/core/colors.py's PALETTE is BGR/0-255 (cv2 convention); mjv_initGeom
    wants RGB/0-1. Derive rather than restate so the two cannot diverge."""
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


# --- Wing-visibility guard (Change 1, task-14) ------------------------------
# fruitfly_v1_free.xml's `wing_left_inertial`/`wing_right_inertial` geoms are
# BOX (type 6), group 1, with `rgba="0 0 0 0"` baked in via the `wing-inertial`
# default class -- deliberately invisible placeholders for the wing's inertia,
# sitting on top of the real wing meshes (`wing_left_brown`/`_membrane` etc,
# type 7 MESH). Every render below tints `geom_rgba` across ALL geoms
# (`geom_rgba[:, :3] = grey`, `geom_rgba[:, 3] = alpha`) to colour/fade the
# mesh uniformly; a blanket alpha assignment overwrites that alpha=0 and turns
# the invisible inertial boxes into opaque white rectangles over/behind the
# wings -- exactly the bug this guard exists to prevent. Capture each geom's
# ORIGINAL alpha once, at load, then re-zero alpha on originally-invisible
# geoms after every `geom_rgba` mutation; also skip recolouring them (the
# rgb channels of an alpha=0 geom are never visible, but "only recolour geoms
# whose original alpha > 0" is the safest general form and costs nothing).
def capture_geom_alpha(mj_model) -> np.ndarray:
    """Call ONCE, right after `MjModel.from_xml_path`, before any geom_rgba
    mutation. Returns the model's as-shipped per-geom alpha (nan-safe copy)."""
    return np.array(mj_model.geom_rgba[:, 3], copy=True)


def set_mesh_rgba(mj_model, orig_alpha: np.ndarray, rgb=None, alpha=None) -> None:
    """Recolour/re-alpha only geoms that were ORIGINALLY visible
    (`orig_alpha > 0`); force every originally-invisible geom's alpha back to
    0 regardless (a no-op if nothing ever touched it, a fix if something did).
    `rgb`/`alpha` are each optional so callers can set just one channel."""
    visible = orig_alpha > 0.0
    if rgb is not None:
        mj_model.geom_rgba[visible, :3] = rgb
    if alpha is not None:
        mj_model.geom_rgba[visible, 3] = alpha
    mj_model.geom_rgba[~visible, 3] = 0.0


def qc_stages(result: dict, out_png=None, size=(480, 480)) -> Path:
    """Two MuJoCo panels per stage: grey mesh + keypoint cloud + residual (mm).

    Row 1 uses the model's `hero` camera (matches the explainer video's
    framing). Row 2 is a wide diagnostic camera that always keeps the WHOLE
    keypoint cloud in frame (hero's fixed relative offset can crop most of
    the cloud when the model sits far from it, e.g. `default`/`scaled`) and
    colours keypoints per the JARVIS per-limb-chain scheme (`kp_colors.py`,
    each leg its own colour) so a 180-degree flip -- head keypoints landing
    on the abdomen -- is directly visible rather than inferred from an
    undifferentiated cyan cloud.
    """
    clip = result["clip"]
    dirs = clip_io.out_dirs(clip)
    out_png = Path(out_png) if out_png is not None else dirs["qc"] / "05_ik_stages.png"

    frame = result["frame_for_stills"]
    kp_names = result["kp_names"]
    mj_model = result["mj_model"]
    orig_alpha = capture_geom_alpha(mj_model)   # BEFORE any geom_rgba mutation
    set_mesh_rgba(mj_model, orig_alpha, rgb=0.55, alpha=1.0)  # grey mesh,
    # every panel -- wings' originally-invisible inertial boxes stay alpha=0.
    # `tracking[name]` sites, not bare `name` -- reuse the same helper `run()`
    # uses for `compute_shared_scale` rather than re-deriving the lookup.
    # `mj_name2id` returns -1 (not an error) on a miss, so this is guarded
    # explicitly: a silent -1 would make every "missing" keypoint alias site
    # index -1 (the model's LAST site), corrupting `mesh_ctr` without any
    # visible error -- exactly the silent index-mismatch class CLAUDE.md warns
    # about.
    site_map = _tracking_site_map(mj_model, kp_names)
    missing = [n for n in kp_names if n not in site_map]
    if missing:
        raise ValueError(f"tracking[...] site missing for keypoints: {missing}")
    body_site_idxs = np.asarray([site_map[n] for n in kp_names])

    kp_rgb01 = jarvis_kp_colors_rgb01(kp_names)
    kp_rgb01_by_idx = [kp_rgb01[n] for n in kp_names]

    stage_qpos = {
        "default": result["qpos_default"], "scaled": result["qpos_scaled"],
        "root": result["qpos_root"], "pose": result["qpos_pose"],
    }
    stage_kp = {
        "default": result["kp3d_raw"][frame], "scaled": result["kp3d_scaled"][frame],
        "root": result["kp3d_scaled"][frame], "pose": result["kp3d_scaled"][frame],
    }

    W, H = size
    marker_r = mj_model.stat.extent * 0.018
    hero_row, wide_row = [], []
    # Persistent renderer, created ONCE and reused for every stage (matches
    # Act 4's fix -- repeated per-frame `with mujoco.Renderer(...)`
    # construction is the prime suspect for Act 4's mid-render EGL
    # resource-leak crash; only 4 stages here so this loop never hit that
    # failure, but it is touched by Change 1's wing-alpha guard and Change 2's
    # per-keypoint colours, so it is fixed too rather than left on the
    # known-bad pattern).
    with mujoco.Renderer(mj_model, height=H, width=W) as renderer:
        for stage in STAGE_NAMES:
            d = mujoco.MjData(mj_model)
            d.qpos[:] = stage_qpos[stage]
            mujoco.mj_forward(mj_model, d)
            kp_frame = stage_kp[stage]
            idx = STAGE_NAMES.index(stage)
            resid = result["residual_mm"][idx]

            # --- hero camera (video framing) ---
            renderer.update_scene(d, camera=CAMERA)
            scn = renderer.scene
            for p in kp_frame:
                if np.all(np.isfinite(p)):
                    _add_sphere(scn, p, CYAN, marker_r)
            hero_img = np.ascontiguousarray(renderer.render())

            # --- wide diagnostic camera: always frames mesh + full cloud ---
            mesh_ctr = d.site_xpos[body_site_idxs].mean(axis=0)
            kp_ctr = np.nanmean(kp_frame, axis=0)
            lookat = (mesh_ctr + kp_ctr) / 2.0
            spread = max(
                float(np.nanmax(np.linalg.norm(kp_frame - lookat, axis=-1))),
                float(np.linalg.norm(mesh_ctr - lookat)),
                mj_model.stat.extent * 0.5,
            )
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(mj_model, cam)
            cam.lookat[:] = lookat
            cam.distance = spread * 2.6
            cam.azimuth, cam.elevation = 120.0, -20.0
            renderer.update_scene(d, camera=cam)
            scn = renderer.scene
            for i, p in enumerate(kp_frame):
                if np.all(np.isfinite(p)):
                    rgba = (*kp_rgb01_by_idx[i], 1.0)
                    _add_sphere(scn, p, np.array(rgba, dtype=np.float32), marker_r * 1.3)
            wide_img = np.ascontiguousarray(renderer.render())

            for img, row in ((hero_img, hero_row), (wide_img, wide_row)):
                img = draw.stage_title(img, stage, f"residual {resid:.2f} mm")
                img = draw.label(img, f"frame {frame}", (16, H - 16), scale=0.55, color=(200, 200, 200))
                row.append(img)

    hero_strip = np.concatenate(hero_row, axis=1)
    wide_strip = np.concatenate(wide_row, axis=1)
    hero_strip = draw.label(hero_strip, "hero camera (video framing)",
                             (16, hero_strip.shape[0] - 44), scale=0.55, color=(150, 220, 150))
    wide_strip = draw.label(
        wide_strip, "wide diagnostic camera -- JARVIS per-limb-chain colours (kp_colors.py)",
        (16, wide_strip.shape[0] - 44), scale=0.55, color=(150, 220, 150),
    )
    grid = np.concatenate([hero_strip, wide_strip], axis=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_png, grid)
    print(f"[stage_ik] QC figure saved to {out_png}")
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    ap.add_argument("--frame", type=int, default=450, help="frame index used for the stage stills/residuals")
    args = ap.parse_args()

    result = run(args.clip, args.frame)
    qc_stages(result)


if __name__ == "__main__":
    main()
