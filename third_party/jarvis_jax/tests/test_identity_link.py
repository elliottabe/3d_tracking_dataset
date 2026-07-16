import os

import numpy as np
import pytest
from jarvis_jax.tracking.affine_camera import factor_affine, reconstruct_affine, project_affine
from jarvis_jax.tracking.identity_link import score_assignment, _dlt_affine, link_frameset
from jarvis_jax.tracking.identity_link import link_recording

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
REC = "2026_04_07_11_33_33"
CALIB = f"{ROOT}/calib_params/{REC}"
COCO = f"{ROOT}/annotations/instances_val.json"

# A real telecentric DLT (from the rig) + 6 rotated copies -> 7 affine cameras
# with angular diversity, so affine triangulation is well-conditioned.
P_REAL = np.array([[8.1001, 0.0074869, -0.031773, -2.828],
                   [0.0093308, -8.0788, -0.17912, 462.78],
                   [0, 0, 0, 1.0]])


def _rig(n_cam=7):
    K2, R, t = factor_affine(P_REAL)
    cams = [P_REAL]
    for k in range(1, n_cam):
        th = np.deg2rad(15.0 * k)
        Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        cams.append(reconstruct_affine(K2, Ry @ R, t))
    return np.stack(cams, 0)  # (n_cam, 3, 4)


def _project_fly(cam_mats, kp3d):
    """kp3d (K,3) -> per-camera (n_cam, K, 3) [u, v, v=2] (all visible)."""
    n_cam, K = cam_mats.shape[0], kp3d.shape[0]
    out = np.zeros((n_cam, K, 3))
    for c in range(n_cam):
        for j in range(K):
            out[c, j, :2] = project_affine(cam_mats[c], kp3d[j])
            out[c, j, 2] = 2.0
    return out


def _make_two_flies(rng, cam_mats, K=8, sep=6.0):
    base = np.array([120.0, 30.0, 11.0])  # inside P_REAL's field of view
    fly0 = base + rng.normal(scale=1.5, size=(K, 3))
    fly1 = base + np.array([sep, 0.0, 0.0]) + rng.normal(scale=1.5, size=(K, 3))
    return fly0, fly1


def test_dlt_affine_recovers_known_point():
    cam_mats = _rig(7)
    X = np.array([118.0, 33.0, 12.5])
    pts = np.zeros((7, 2))
    for c in range(7):
        pts[c] = project_affine(cam_mats[c], X)
    Xhat = _dlt_affine(cam_mats, list(range(7)), pts)
    assert np.linalg.norm(Xhat - X) < 1e-4


def test_score_assignment_low_for_true_and_high_for_swapped():
    rng = np.random.default_rng(0)
    cam_mats = _rig(7)
    fly0, fly1 = _make_two_flies(rng, cam_mats)
    obs0 = _project_fly(cam_mats, fly0)
    obs1 = _project_fly(cam_mats, fly1)
    cams = list(range(7))
    # consistent single-fly observations -> ~0 residual (subpixel)
    good = score_assignment(obs0, cams, cam_mats)
    assert good < 1.0
    # mix cam 3 with the OTHER fly's detection -> a large residual
    bad = obs0.copy()
    bad[3] = obs1[3]
    assert score_assignment(bad, cams, cam_mats) > 5.0 * max(good, 1e-3)


def _anns_from_obs(obs0, obs1, order_per_cam):
    """Build anns_by_cam with per-camera ann ordering controlled by order_per_cam[c]
    (0 -> [fly0, fly1], 1 -> [fly1, fly0]). ann_id encodes (cam*10 + slot)."""
    anns_by_cam = {}
    for c in range(obs0.shape[0]):
        f0 = {"id": c * 10 + 0, "keypoints": obs0[c].reshape(-1).tolist()}
        f1 = {"id": c * 10 + 1, "keypoints": obs1[c].reshape(-1).tolist()}
        anns_by_cam[c] = [f0, f1] if order_per_cam[c] == 0 else [f1, f0]
    return anns_by_cam


def test_link_frameset_recovers_grouping_under_shuffle():
    rng = np.random.default_rng(1)
    cam_mats = _rig(7)
    fly0, fly1 = _make_two_flies(rng, cam_mats)
    obs0, obs1 = _project_fly(cam_mats, fly0), _project_fly(cam_mats, fly1)
    # shuffle each camera's ann order arbitrarily
    order = [0, 1, 1, 0, 1, 0, 1]
    anns = _anns_from_obs(obs0, obs1, order)
    linked = link_frameset(anns, cam_mats, n_flies=2)
    # each fly's chosen ann per cam must all carry the SAME true-fly slot
    def true_slot(ann_id):
        return ann_id % 10  # 0 -> obs0(fly0), 1 -> obs1(fly1)
    slots_fly = {fid: {true_slot(a) for a in linked[fid].values()} for fid in linked}
    assert all(len(s) == 1 for s in slots_fly.values())          # internally consistent
    assert slots_fly[0] != slots_fly[1]                          # the two flies differ
    assert set(linked[0]) == set(range(7)) and set(linked[1]) == set(range(7))


def test_link_frameset_robust_to_order_swap_and_missing_cam():
    rng = np.random.default_rng(2)
    cam_mats = _rig(7)
    fly0, fly1 = _make_two_flies(rng, cam_mats)
    obs0, obs1 = _project_fly(cam_mats, fly0), _project_fly(cam_mats, fly1)
    order = [0, 1, 0, 1, 0, 1, 0]
    anns = _anns_from_obs(obs0, obs1, order)
    # camera 5 sees only ONE fly (drop its second ann)
    anns[5] = [anns[5][0]]
    linked = link_frameset(anns, cam_mats, n_flies=2)

    def true_slot(ann_id):
        return ann_id % 10
    slots_fly = {fid: {true_slot(a) for a in linked[fid].values()} for fid in linked}
    assert all(len(s) == 1 for s in slots_fly.values())
    assert slots_fly[0] != slots_fly[1]
    # camera 5's single ann is assigned to exactly one fly (the nearest-reproj one)
    assigned_cam5 = [fid for fid in linked if 5 in linked[fid]]
    assert len(assigned_cam5) == 1


def test_link_frameset_returns_empty_when_residual_exceeds_gate():
    rng = np.random.default_rng(3)
    cam_mats = _rig(7)
    K = 8
    # Pure noise: each camera's two anns are independent random pixel coords,
    # so there is no true shared 3-D point and reprojection residual is huge.
    anns_by_cam = {}
    for c in range(7):
        f0 = {"id": c * 10 + 0,
              "keypoints": np.concatenate(
                  [rng.uniform(-500, 500, size=(K, 2)), np.full((K, 1), 2.0)], axis=1
              ).reshape(-1).tolist()}
        f1 = {"id": c * 10 + 1,
              "keypoints": np.concatenate(
                  [rng.uniform(-500, 500, size=(K, 2)), np.full((K, 1), 2.0)], axis=1
              ).reshape(-1).tolist()}
        anns_by_cam[c] = [f0, f1]
    linked = link_frameset(anns_by_cam, cam_mats, n_flies=2)
    assert linked == {0: {}, 1: {}}


def test_link_frameset_degenerate_full_camera_does_not_crash():
    rng = np.random.default_rng(4)
    cam_mats = _rig(7)
    fly0, fly1 = _make_two_flies(rng, cam_mats)
    obs0, obs1 = _project_fly(cam_mats, fly0), _project_fly(cam_mats, fly1)
    order = [0, 1, 0, 1, 0, 1, 0]
    anns = _anns_from_obs(obs0, obs1, order)
    # Camera 3 is "full" (has n_flies anns) but every keypoint is invisible
    # (v=0), so no permutation at that camera triangulates -> best_perm stays
    # None. Must not raise, and the camera must simply be dropped.
    for ann in anns[3]:
        kp = np.asarray(ann["keypoints"], float).reshape(-1, 3)
        kp[:, 2] = 0.0
        ann["keypoints"] = kp.reshape(-1).tolist()

    linked = link_frameset(anns, cam_mats, n_flies=2)

    def true_slot(ann_id):
        return ann_id % 10
    slots_fly = {fid: {true_slot(a) for cam, a in linked[fid].items() if cam != 3}
                 for fid in linked}
    assert all(len(s) == 1 for s in slots_fly.values())
    assert slots_fly[0] != slots_fly[1]
    # the degenerate camera is absent from both flies' maps
    assert 3 not in linked[0] and 3 not in linked[1]
    # the remaining good cameras are still resolved
    assert set(linked[0]) == set(range(7)) - {3}
    assert set(linked[1]) == set(range(7)) - {3}


def test_link_frameset_returns_empty_when_no_full_camera():
    rng = np.random.default_rng(5)
    cam_mats = _rig(7)
    fly0, fly1 = _make_two_flies(rng, cam_mats)
    obs0, obs1 = _project_fly(cam_mats, fly0), _project_fly(cam_mats, fly1)
    anns = _anns_from_obs(obs0, obs1, [0] * 7)
    # every camera has only 1 ann (< n_flies=2) -> no full camera at all
    for c in anns:
        anns[c] = [anns[c][0]]
    linked = link_frameset(anns, cam_mats, n_flies=2)
    assert linked == {0: {}, 1: {}}


@pytest.mark.skipif(not os.path.exists(COCO), reason="courtship coco not present")
def test_link_recording_two_fly_framesets_reproject_cleanly():
    m = link_recording(COCO, REC, CALIB, split="val", n_flies=2)
    assert len(m) > 0, "no framesets found for the recording"

    # Most framesets should resolve BOTH flies. NOTE on threshold: the brief's
    # ">= 50" assumed 181 "two-ann frames" (per-camera image count) meant 181
    # two-fly FRAMESETS, but this recording's instances_val.json only contains
    # 31 framesets total for REC (the 181 figure is per-camera frames across
    # those framesets, ~7 cams x ~26 framesets). Measured on real data: 30/31
    # framesets (97%) resolve both flies fully in all 7 cameras; the single
    # excluded frameset genuinely has only one fly annotated in every camera
    # (1-fly frameset, handled by the graceful-degrade branch). Since 50 two-fly
    # framesets is mathematically impossible out of 31 total, the count gate is
    # lowered to the actual achievable ceiling with margin.
    two_fly = [k for k, v in m.items() if len(v.get(0, {})) >= 2 and len(v.get(1, {})) >= 2]
    assert len(two_fly) >= 28, f"only {len(two_fly)} two-fly framesets linked"

    # NON-VACUOUS: each linked fly's chosen anns must triangulate + reproject
    # to a small residual (the linker's own objective) and the two flies must
    # pick DISJOINT anns per camera (no identity collapse).
    import json
    import numpy as np
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.identity_link import score_assignment, _ann_kp
    coco = json.load(open(COCO))
    id2ann = {}
    for a in coco["annotations"]:
        id2ann.setdefault(a["image_id"], []).append(a)
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    rt = ReprojectionTool(CALIB)
    cam_names = list(rt.cameras.keys())
    cam_mats = np.stack([c.cameraMatrix for c in rt._camera_list], 0)  # (n_cam,3,4)

    resid_all, disjoint_ok = [], 0
    for fs_key in two_fly[:30]:
        v = m[fs_key]
        # disjoint per-camera anns between fly0 and fly1
        shared = set(v[0].items()) & set(v[1].items())
        if not shared:
            disjoint_ok += 1
        for fly in (0, 1):
            n_cam = len(cam_names)
            K = _ann_kp(next(iter(ann_by_id.values()))).shape[0]
            obs = np.zeros((n_cam, K, 3))
            for cam, aid in v[fly].items():
                obs[cam] = _ann_kp(ann_by_id[aid])
            r = score_assignment(obs, list(v[fly]), cam_mats)
            if np.isfinite(r):
                resid_all.append(r)
    assert disjoint_ok >= 25, "flies share anns in too many framesets (identity collapse)"
    assert np.mean(resid_all) < 15.0, (
        f"post-link mean reprojection residual {np.mean(resid_all):.2f}px too high"
    )
