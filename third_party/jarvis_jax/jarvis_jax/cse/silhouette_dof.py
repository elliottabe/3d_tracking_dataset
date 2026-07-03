"""Appendage DOF + vertex selection for the DOF-restricted silhouette factor.

The silhouette factor should move only the appendage DOFs the keypoints track
poorly (wings, legs, abdomen), leaving the root pose to keypoints. These helpers
name-match the model's joints and the mesh's segments. NOTE: the abdomen pattern
is `abdomen` (NOT a bare `abd`) so it does not also catch the leg `coxa_abduct`
joints.
"""
from __future__ import annotations
import re
import numpy as np

from jarvis_jax.cse.silhouette_targets import silhouette_fk_indices

APPENDAGE_PATTERNS = {
    "wing": r"wing",
    "leg": r"coxa|femur|tibia|tarsus|claw|trochanter",
    "abdomen": r"abdomen",
}


def _pattern(include):
    if not include:
        return re.compile(r"(?!)")        # never matches -> empty selection
    return re.compile("|".join(APPENDAGE_PATTERNS[k] for k in include))


def build_appendage_dof_mask(model, include=("wing", "leg", "abdomen")):
    """(nq,) bool: True for appendage hinge qpos DOFs; free (root) joint False."""
    import mujoco
    rex = _pattern(include)
    mask = np.zeros(int(model.nq), bool)
    for j in range(int(model.njnt)):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if nm and rex.search(nm.lower()):
            mask[int(model.jnt_qposadr[j])] = True   # hinge/slide = 1 qpos each
    return mask


def appendage_vertex_indices(mesh_npz, subset="fps_300", include=("wing", "leg", "abdomen")):
    """Full-array vertex indices in `subset` that belong to appendage segments."""
    z = np.load(mesh_npz, allow_pickle=True)
    rex = _pattern(include)
    seg_ids = np.asarray(z["seg_ids"])
    seg_names = [str(s) for s in z["seg_names"]]
    exclude = [int(sid) for sid, nm in zip(seg_ids, seg_names) if not rex.search(nm.lower())]
    return silhouette_fk_indices(mesh_npz, subset=subset, exclude_seg_ids=exclude)
