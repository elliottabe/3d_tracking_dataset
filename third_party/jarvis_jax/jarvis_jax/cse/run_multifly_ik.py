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
existing (male) fly's fitted anatomy/qpos/marker_sites ik h5 verbatim as a
seed, replacing only ``kp_data`` with the per-fly bout's triangulated
keypoints (same 50-kp order/scale). This does NOT itself re-solve pose --
it writes an ik h5 whose ``qpos``/``marker_sites`` are still the shared
(male) fly's solved values, paired with THIS fly's own keypoints. The
actual per-fly pose re-solve happens downstream, inside ``run_single_fly``
(called by ``run_multifly_ik``/``run_multifly_ablation`` after
``_prepare_fly_ik`` returns), via its own ``build_solver_inputs``/
``solve_ik`` call on this ik h5. Primary path is tried FIRST; fallback only
on genuine failure.
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
    build_solver_inputs, solve_ik, run_single_fly, run_ablation,
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

    Coverage invariant (relied upon downstream): this map covers a SUPERSET of
    the images that end up in this fly's bout ``fs_imgids`` — ``build_fly_bout``
    keeps only framesets where the fly has >=2 assigned cameras, and every kept
    (camera) image_id is present here by construction. ``_ann_for_image`` falls
    back to the FIRST annotation when an image is absent from this map, so a
    caller that fed ``run_single_fly`` a ``fs_imgids`` referencing an image NOT
    in this fly's map would silently select the other fly's annotation there.
    Keep the bout and this map built from the SAME ``identity_map``/``fly_id``.
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
    """Escape hatch: reuse the shared (male) fly's fitted anatomy ik_h5 as a seed.

    Copies ``offsets``/``qpos``/``marker_sites``/``names_qpos`` VERBATIM
    from ``fallback_ik_h5`` (the shared/male solve) and replaces only
    ``kp_data`` with THIS fly's own triangulated bout keypoints. This
    function does NOT re-solve pose itself -- it does not call
    ``build_solver_inputs`` or ``solve_ik``; ``qpos``/``marker_sites`` here
    are still the shared fly's solved values, now paired with a different
    fly's keypoints. Writes a new h5 at ``out_h5`` with the same
    io_dict_to_hdf5 layout that ``silhouette_ik_solve.build_solver_inputs``/
    ``run_single_fly`` expect (``qpos``, ``kp_data``, ``offsets``,
    ``kp_names``, ``names_qpos``, ``marker_sites``, ``config``). The actual
    per-fly pose re-solve happens downstream, inside ``run_single_fly``,
    which loads this h5 via ``build_solver_inputs`` and calls ``solve_ik``
    to fit qpos to the (now per-fly) ``kp_data``.
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


def _prepare_fly_ik(
    recording, fly_dir, coco_path, calib_dir, identity_map, fly_id,
    anatomy_yaml, model_xml, stac_config_dir, split, stac_overrides,
    *, stac_fallback=False, fallback_ik_h5=DEFAULT_FALLBACK_IK_H5,
):
    """Build one fly's de-collapsed bout + per-fly STAC ik_h5.

    Shared by ``run_multifly_ik`` and ``run_multifly_ablation`` so the
    bout-build + STAC-invocation block (with its fallback escape hatch) is
    not duplicated. Primary path: real per-fly STAC solve via
    ``run_stac_bout.run``. If ``stac_fallback=True`` OR the primary solve
    raises, falls back to the fixed-anatomy escape hatch (see module
    docstring / ``_run_stac_fallback``), which seeds the returned ik_h5 with
    the shared (male) fly's anatomy/qpos/marker_sites and this fly's own
    ``kp_data`` -- it does NOT itself re-solve pose. Either way, the
    per-fly pose re-solve happens downstream when the caller feeds the
    returned ``ik_h5`` into ``run_single_fly``.

    Returns:
        (bout_h5: str, ik_h5: str, solve_path: "stac" | "fallback").
    """
    os.makedirs(fly_dir, exist_ok=True)

    # 1) per-fly bout (de-collapsed triangulation). Written at
    # fly_dir/<rec>_bout.h5 so run_single_fly's/run_ablation's bout-derivation
    # (dirname(dirname(ik_h5))/<rec>_bout.h5) resolves back to it -- verified
    # against silhouette_ik_solve's actual code (bout_h5 =
    # os.path.join(os.path.dirname(os.path.dirname(ik_h5)),
    # f"{recording}_bout.h5")).
    bout_h5 = os.path.join(fly_dir, f"{recording}_bout.h5")
    bout_h5, _ = build_fly_bout(
        coco_path, calib_dir, recording, identity_map, fly_id,
        anatomy_yaml, model_xml, bout_h5, split=split)

    # 2) per-fly STAC solve -> per-fly ik_h5 (see path-derivation note above:
    # ik_h5's directory must be fly_dir/<rec>/ so that
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
            print(f"[run_multifly_ik] fly{fly_id}: primary STAC solve failed "
                  f"({type(exc).__name__}: {exc}); falling back to fixed-anatomy escape hatch")
            solve_path = "fallback"
            _run_stac_fallback(bout_h5, fallback_ik_h5, ik_h5)

    return bout_h5, ik_h5, solve_path


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

        # 1)+2) per-fly bout (de-collapsed triangulation) + per-fly STAC
        # solve -> per-fly ik_h5 (or the fixed-anatomy fallback escape hatch).
        bout_h5, ik_h5, solve_path = _prepare_fly_ik(
            recording, fly_dir, coco_path, calib_dir, identity_map, fid,
            anatomy_yaml, model_xml, stac_config_dir, split, stac_overrides,
            stac_fallback=stac_fallback, fallback_ik_h5=fallback_ik_h5)

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


def run_multifly_ablation(
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
    ablate_fly_id=1,
    coco_path=None,
    max_frames=0,
    n_iter=50,
    wing_weight=0.5,
    smooth_weight=0.1,
    corridor=12.0,
    residual_gate_px=40.0,
    stac_overrides=DEFAULT_STAC_OVERRIDES,
    stac_fallback=False,
    fallback_ik_h5=DEFAULT_FALLBACK_IK_H5,
):
    """Courtship validation: 2nd-fly keypoint-withhold ablation (Task 6).

    Demonstrates the spec's "female rides on silhouette": withholds
    ``ablate_fly_id``'s wing keypoints and confirms the silhouette condition
    recovers wing extent toward the SAM-triangulated tip, while BOTH flies
    still reproject cleanly.

    Pipeline: ``link_recording`` -> per-fly ``_prepare_fly_ik`` (bout + STAC)
    for BOTH flies -> ``run_ablation`` on ``ablate_fly_id`` with its
    ``ann_id_by_image`` -> ``run_single_fly`` on the other fly to confirm
    clean reprojection.

    Args:
        ablate_fly_id: which fly slot (0 or 1) has its wing keypoints
            withheld for the ablation. Default 1 (the "female" slot per
            user decision 2), but the identity map's fly0/fly1 slots are
            otherwise arbitrary (sex disambiguation is out of scope).

    Returns:
        {"ablation": <run_ablation dict for ablate_fly_id>,
         "other_fly": {"fly_id": int, "reproj_px": float},
         "ablate_fly_id": int}
    """
    if coco_path is None:
        coco_path = os.path.join(root, "annotations", f"instances_{split}.json")
    identity_map = link_recording(
        coco_path, recording, calib_dir, split=split, n_flies=2,
        residual_gate_px=residual_gate_px,
    )
    os.makedirs(out_dir, exist_ok=True)

    other_fly_id = 1 - ablate_fly_id

    # ablated fly: build bout + STAC, then run_ablation with its ann selection.
    ab_dir = os.path.join(out_dir, f"fly{ablate_fly_id}")
    _, ab_ik, _ = _prepare_fly_ik(
        recording, ab_dir, coco_path, calib_dir, identity_map, ablate_fly_id,
        anatomy_yaml, model_xml, stac_config_dir, split, stac_overrides,
        stac_fallback=stac_fallback, fallback_ik_h5=fallback_ik_h5)
    ab_ann = ann_id_by_image_for_fly(identity_map, coco_path, recording, ablate_fly_id)
    ablation = run_ablation(
        recording, ik_h5=ab_ik, model_xml=model_xml, mesh_npz=mesh_npz, root=root,
        split=split, calib_dir=calib_dir, wing_weight=wing_weight,
        max_frames=max_frames, smooth_weight=smooth_weight, n_iter=n_iter,
        corridor=corridor, out_dir=ab_dir, ann_id_by_image=ab_ann)

    # other fly: build bout + STAC, run_single_fly to confirm clean reprojection.
    ot_dir = os.path.join(out_dir, f"fly{other_fly_id}")
    _, ot_ik, _ = _prepare_fly_ik(
        recording, ot_dir, coco_path, calib_dir, identity_map, other_fly_id,
        anatomy_yaml, model_xml, stac_config_dir, split, stac_overrides,
        stac_fallback=stac_fallback, fallback_ik_h5=fallback_ik_h5)
    ot_ann = ann_id_by_image_for_fly(identity_map, coco_path, recording, other_fly_id)
    ot_report = run_single_fly(
        recording, ik_h5=ot_ik, model_xml=model_xml, mesh_npz=mesh_npz, root=root,
        split=split, calib_dir=calib_dir, use_silhouette=True, wing_weight=wing_weight,
        max_frames=max_frames, smooth_weight=smooth_weight, n_iter=n_iter,
        corridor=corridor, out_dir=ot_dir, ann_id_by_image=ot_ann)

    return {
        "ablation": ablation,
        "other_fly": {"fly_id": other_fly_id, "reproj_px": ot_report["reproj_px"]},
        "ablate_fly_id": ablate_fly_id,
    }
