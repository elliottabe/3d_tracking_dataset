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
