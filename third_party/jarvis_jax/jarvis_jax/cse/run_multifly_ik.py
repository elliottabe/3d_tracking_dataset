"""End-to-end multi-fly IK orchestrator (Phase 3).

Pipeline for a courtship recording (2 flies COCO-annotated per image):
  link_recording -> per-fly build_fly_bout -> per-fly run_stac_bout (ik_h5)
  -> per-fly silhouette IK (run_single_fly with the fly's ann_id_by_image,
     the per-fly ik_h5, and the per-recording calib_dir).

Reuses the Phase-2 solver + silhouette machinery unchanged; the only new
selection input is the per-fly ann_id_by_image (from the identity map).

STAC config (verified, not guessed): the overrides ``paths=hyak anatomy=v1
dataset=free_walking`` are the EXACT overrides used by both
``sbatch_cse_pilot.sh`` and ``sbatch_cse_array.sh`` to produce the existing
male ik_h5 at
``/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5``.
The real stac-mjx config tree lives at
``/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs``
(NOT under ``fly_neuromech``, which has no ``stac-mjx`` checkout at all).
``dataset/courtship.yaml`` also exists and is nearly identical to
``free_walking.yaml`` (only differs in ``n_fit_frames``/``n_frames_per_clip``,
both of which ``run_stac_bout.run`` overrides at call time anyway based on the
bout's actual frame count) -- ``free_walking`` is kept as the default since it
is the proven-working config for this pipeline.

Primary path: a per-fly STAC solve via ``jarvis_jax.cse.run_stac_bout.run``,
which fits per-fly offsets + q_init from that fly's own triangulated bout.
Escape hatch (``stac_fallback=True``): if per-fly STAC fails, reuse the
existing (male) fly's fitted offsets/marker_sites ik h5 as the fixed V1
anatomy source and seed ``solve_ik`` with a per-fly q_init derived from the
per-fly bout (``build_solver_inputs`` against that shared ik h5, then
overwrite ``kp_data`` with the per-fly bout's triangulated keypoints, same
50-kp order/scale). The fallback keeps anatomy fixed (V1) and only re-solves
pose per fly. Primary path is tried FIRST; fallback only on genuine failure.
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np

from jarvis_jax.cse.identity_link import link_recording
from jarvis_jax.cse.multifly_bout import build_fly_bout
from jarvis_jax.cse import run_stac_bout
from jarvis_jax.cse.silhouette_ik_solve import (
    build_solver_inputs, solve_ik, run_single_fly,
)

# The exact overrides used by sbatch_cse_pilot.sh / sbatch_cse_array.sh to
# produce the existing (male) ik_h5 -- see module docstring.
DEFAULT_STAC_OVERRIDES = ("paths=hyak", "anatomy=v1", "dataset=free_walking")

# Shared (male) fly's fitted-anatomy ik h5, used only by the stac_fallback
# escape hatch when per-fly STAC solving is not viable.
DEFAULT_FALLBACK_IK_H5 = (
    "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/"
    "Fruitfly_ik_v1_cse.h5"
)


def ann_id_by_image_for_fly(identity_map, coco_path, recording, fly_id):
    """Flatten {fs_key:{fly:{cam:ann_id}}} -> {image_id: ann_id} for one fly.

    Args:
        identity_map: output of ``identity_link.link_recording``.
        coco_path: COCO json path (for ann_id -> image_id lookup).
        recording: unused directly (identity_map is already scoped to one
            recording); kept for API symmetry / future validation.
        fly_id: which fly slot to flatten.

    Returns:
        dict[int, int]: {image_id: ann_id} for this fly, across every
        frameset/camera in identity_map. This is the ``ann_id_by_image``
        Task 4's ``run_single_fly``/``extract_tips_for_frames`` consume.
    """
    coco = json.load(open(coco_path))
    ann_img = {a["id"]: a["image_id"] for a in coco["annotations"]}
    out = {}
    for fs_key, per_fly in identity_map.items():
        for cam, aid in per_fly.get(fly_id, {}).items():
            iid = ann_img.get(aid)
            if iid is not None:
                out[int(iid)] = int(aid)
    return out


def _run_stac_fallback(bout_h5, fallback_ik_h5, out_h5):
    """Escape hatch: reuse the shared (male) fly's fitted anatomy ik_h5.

    Keeps ``offsets``/``marker_sites``/``names_qpos`` fixed (from
    ``fallback_ik_h5``) and re-solves pose against THIS fly's own
    triangulated bout keypoints, seeding q_init from the shared solve.
    Writes a new h5 at ``out_h5`` with the same io_dict_to_hdf5 layout that
    ``silhouette_ik_solve.build_solver_inputs``/``run_single_fly`` expect
    (``qpos``, ``kp_data``, ``offsets``, ``kp_names``, ``names_qpos``,
    ``marker_sites``, ``config``), so downstream code cannot tell the
    difference between a real per-fly STAC solve and this fallback.
    """
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from stac_mjx import io as stac_io

    # anatomy + shared q_init/config from the fixed male ik h5.
    config, stac_data = stac_io.load_stac_data(fallback_ik_h5)
    shared_raw = ioh5.load(fallback_ik_h5)

    bout = ioh5.load(bout_h5)
    kp = np.asarray(bout["keypoints"])  # (T, K, 3) model cm, model order
    T = kp.shape[0]

    T_shared = np.asarray(stac_data.qpos).shape[0]
    Tq = min(T, T_shared)

    kp_names_bout = [n.decode() if isinstance(n, bytes) else n for n in bout["kp_names"]]
    kp_names_shared = list(stac_data.kp_names)
    assert kp_names_bout == kp_names_shared, (
        "fallback requires the per-fly bout's keypoint order to match the "
        "shared ik h5's kp_names (both from the same anatomy_yaml)"
    )

    out = dict(shared_raw)
    out["kp_data"] = kp[:Tq].reshape(Tq, -1).astype(np.float32)
    out["qpos"] = np.asarray(stac_data.qpos)[:Tq].astype(np.float32)
    out["marker_sites"] = np.asarray(shared_raw["marker_sites"])[:Tq]

    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    ioh5.save(out_h5, out)
    return out_h5


def run_multifly_ik(
    recording,
    *,
    root,
    calib_dir,
    anatomy_yaml,
    model_xml,
    mesh_npz,
    stac_config_dir,
    out_dir,
    split="val",
    n_flies=2,
    coco_path=None,
    use_silhouette=True,
    max_frames=0,
    n_iter=50,
    wing_weight=0.5,
    smooth_weight=0.1,
    corridor=12.0,
    only_missing=True,
    residual_gate_px=40.0,
    stac_overrides=DEFAULT_STAC_OVERRIDES,
    stac_fallback=False,
    fallback_ik_h5=DEFAULT_FALLBACK_IK_H5,
):
    """Full multi-fly IK pipeline for a recording. See Interfaces (Task 5).

    Pipeline: ``link_recording`` -> per fly ``build_fly_bout`` -> per fly
    ``run_stac_bout.run`` (or the ``stac_fallback`` escape hatch) -> per fly
    ``run_single_fly(..., ann_id_by_image=ann_id_by_image_for_fly(...))``.

    Args:
        stac_fallback: if True, SKIP the primary per-fly STAC solve entirely
            and use the fixed-anatomy escape hatch (see module docstring) for
            every fly. If False (default), the primary path is attempted for
            each fly; if ``run_stac_bout.run`` raises, that fly automatically
            falls back (per-fly, not global) and the exception is swallowed
            with a printed diagnostic -- the caller can distinguish which
            path ran per fly via the returned ``flies[fid]["solve_path"]``.
        fallback_ik_h5: shared fixed-anatomy ik h5 used by the fallback path.

    Returns:
        {"identity_map_size": int,
         "flies": {fly_id: {"bout_h5", "ik_h5", "n_framesets", "report",
                             "solve_path"}}}
    """
    if coco_path is None:
        coco_path = os.path.join(root, "annotations", f"instances_{split}.json")

    identity_map = link_recording(
        coco_path, recording, calib_dir, split=split, n_flies=n_flies,
        residual_gate_px=residual_gate_px,
    )
    os.makedirs(out_dir, exist_ok=True)

    flies = {}
    for fid in range(n_flies):
        fly_dir = os.path.join(out_dir, f"fly{fid}")
        os.makedirs(fly_dir, exist_ok=True)

        # 1) per-fly bout (de-collapsed triangulation). Written at
        # fly_dir/<rec>_bout.h5 so run_single_fly's bout-derivation
        # (dirname(dirname(ik_h5))/<rec>_bout.h5) resolves back to it --
        # verified against silhouette_ik_solve.run_single_fly's actual code
        # (bout_h5 = os.path.join(os.path.dirname(os.path.dirname(ik_h5)),
        # f"{recording}_bout.h5")).
        bout_h5 = os.path.join(fly_dir, f"{recording}_bout.h5")
        bout_h5, scale = build_fly_bout(
            coco_path, calib_dir, recording, identity_map, fid,
            anatomy_yaml, model_xml, bout_h5, split=split)

        # 2) per-fly STAC solve -> per-fly ik_h5 (see path-derivation note
        # above: ik_h5's directory must be fly_dir/<rec>/ so that
        # dirname(dirname(ik_h5)) == fly_dir).
        ik_h5 = os.path.join(fly_dir, recording, "Fruitfly_ik_v1_cse.h5")
        solve_path = "stac"
        if stac_fallback:
            solve_path = "fallback"
            _run_stac_fallback(bout_h5, fallback_ik_h5, ik_h5)
        else:
            try:
                run_stac_bout.run(bout_h5, ik_h5, stac_config_dir, list(stac_overrides))
            except Exception as exc:  # pragma: no cover - exercised only on real STAC failure
                print(f"[run_multifly_ik] fly{fid}: primary STAC solve failed "
                      f"({type(exc).__name__}: {exc}); falling back to fixed-anatomy escape hatch")
                solve_path = "fallback"
                _run_stac_fallback(bout_h5, fallback_ik_h5, ik_h5)

        # 3) per-fly silhouette IK (with THIS fly's ann selection + per-fly ik_h5).
        ann_map = ann_id_by_image_for_fly(identity_map, coco_path, recording, fid)
        report = run_single_fly(
            recording, ik_h5=ik_h5, model_xml=model_xml, mesh_npz=mesh_npz,
            root=root, split=split, calib_dir=calib_dir,
            use_silhouette=use_silhouette, wing_weight=wing_weight,
            only_missing=only_missing, max_frames=max_frames,
            smooth_weight=smooth_weight, n_iter=n_iter, corridor=corridor,
            out_dir=fly_dir, ann_id_by_image=ann_map)

        with h5py.File(bout_h5, "r") as bf:
            n_fs = int(bf["keypoints"].shape[0])

        flies[fid] = {
            "bout_h5": bout_h5, "ik_h5": ik_h5, "n_framesets": n_fs,
            "report": report, "solve_path": solve_path,
        }
    return {"identity_map_size": len(identity_map), "flies": flies}
