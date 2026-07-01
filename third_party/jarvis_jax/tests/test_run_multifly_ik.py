import os
import numpy as np
import pytest

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
REC = "2026_04_07_11_33_33"
CALIB = f"{ROOT}/calib_params/{REC}"
COCO = f"{ROOT}/annotations/instances_val.json"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
# Real stac-mjx config tree lives under 3d_tracking_dataset (NOT fly_neuromech) --
# verified against the sbatch scripts that produced the existing male ik_h5
# (sbatch_cse_pilot.sh / sbatch_cse_array.sh) and test_multifly_bout.py's own
# ANATOMY default.
ANATOMY = os.environ.get(
    "STAC_ANATOMY_V1",
    "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs/anatomy/v1.yaml",
)
STAC_CFG = os.environ.get(
    "STAC_CONFIG_DIR",
    "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs",
)


def test_ann_id_by_image_for_fly_flattens_map():
    from jarvis_jax.cse.run_multifly_ik import ann_id_by_image_for_fly
    # minimal synthetic identity map + a matching coco stub
    import json, tempfile
    coco = {
        "images": [{"id": 10, "file_name": "val/CamA/Frame_0.jpg"},
                   {"id": 11, "file_name": "val/CamB/Frame_0.jpg"}],
        "annotations": [
            {"id": 100, "image_id": 10}, {"id": 101, "image_id": 10},
            {"id": 110, "image_id": 11}, {"id": 111, "image_id": 11},
        ],
        "framesets": {"fs0": {"datasetName": REC, "frames": [10, 11]}},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(coco, f); path = f.name
    identity_map = {"fs0": {0: {0: 100, 1: 110}, 1: {0: 101, 1: 111}}}
    m0 = ann_id_by_image_for_fly(identity_map, path, REC, 0)
    m1 = ann_id_by_image_for_fly(identity_map, path, REC, 1)
    assert m0 == {10: 100, 11: 110}
    assert m1 == {10: 101, 11: 111}
    os.remove(path)


_HAVE = all(os.path.exists(p) for p in (COCO, XML, MESH, ANATOMY, STAC_CFG))


@pytest.mark.skipif(not _HAVE, reason="courtship coco / model / mesh / stac config not present")
def test_run_multifly_ik_both_flies_reproject_cleanly(tmp_path):
    """GPU real run on a small frame slice: BOTH flies produce an ik_h5 and a
    per-fly silhouette-IK report with reproj_px below an explicit threshold, and
    the two flies' solved keypoints are demonstrably different (de-collapsed)."""
    from jarvis_jax.cse.run_multifly_ik import run_multifly_ik
    out = run_multifly_ik(
        REC, root=ROOT, calib_dir=CALIB, anatomy_yaml=ANATOMY, model_xml=XML,
        mesh_npz=MESH, stac_config_dir=STAC_CFG, out_dir=str(tmp_path),
        split="val", n_flies=2, coco_path=COCO,
        use_silhouette=True, max_frames=16, n_iter=30,
    )
    assert out["identity_map_size"] > 0
    assert set(out["flies"]) == {0, 1}
    reproj = {}
    for fid in (0, 1):
        fly = out["flies"][fid]
        assert os.path.exists(fly["ik_h5"]), f"fly{fid} ik_h5 missing"
        assert os.path.exists(fly["bout_h5"])
        rep = fly["report"]
        assert rep["qpos_shape"][1] == 93
        assert np.isfinite(rep["reproj_px"])
        reproj[fid] = rep["reproj_px"]
    # BOTH flies must reproject cleanly (explicit px threshold vs annotated kps).
    assert reproj[0] < 12.0 and reproj[1] < 12.0, f"reproj_px too high: {reproj}"

    # de-collapse sanity: the two flies' solved qpos differ.
    q0 = np.load(os.path.join(str(tmp_path), "fly0", f"{REC}_qpos.npz"))["qpos"]
    q1 = np.load(os.path.join(str(tmp_path), "fly1", f"{REC}_qpos.npz"))["qpos"]
    n = min(len(q0), len(q1))
    assert np.linalg.norm(q0[:n] - q1[:n]) > 1e-3, "fly0/fly1 qpos identical (collapse)"


@pytest.mark.skipif(not _HAVE, reason="courtship coco / model / mesh / stac config not present")
def test_run_multifly_ablation_recovers_second_fly_wing(tmp_path):
    """GPU real run: withhold fly1's wing keypoints; the silhouette condition (c)
    must recover wing extent toward the SAM tip (recovery_to_sam > 0) while the
    OTHER fly still reprojects cleanly."""
    from jarvis_jax.cse.run_multifly_ik import run_multifly_ablation
    out = run_multifly_ablation(
        REC, root=ROOT, calib_dir=CALIB, anatomy_yaml=ANATOMY, model_xml=XML,
        mesh_npz=MESH, stac_config_dir=STAC_CFG, out_dir=str(tmp_path),
        split="val", ablate_fly_id=1, coco_path=COCO, max_frames=16, n_iter=30,
    )
    ab = out["ablation"]
    assert set(ab["conditions"]) == {"reference", "baseline", "silhouette"}
    assert out["ablate_fly_id"] == 1
    # both flies reproject cleanly: the non-ablated fly's reproj_px is bounded,
    # and the ablated fly's non-wing reproj_px (silhouette condition) is bounded.
    assert np.isfinite(out["other_fly"]["reproj_px"]) and out["other_fly"]["reproj_px"] < 12.0
    assert np.isfinite(ab["conditions"]["silhouette"]["reproj_px"])
    assert ab["conditions"]["silhouette"]["reproj_px"] < 15.0
    # the silhouette must recover the withheld wing toward the SAM extent on at
    # least one side (positive recovery vs the SAM-triangulated tip).
    assert ab["n_frames_with_tips"] > 0, "no SAM wing tips extracted for fly1"
    rec = ab["recovery_ratio_to_sam_mm"]
    assert any(np.isfinite(rec[s]) and rec[s] > 0.0 for s in ("left", "right")), (
        f"silhouette did not recover fly1's withheld wing toward SAM: {rec}"
    )
