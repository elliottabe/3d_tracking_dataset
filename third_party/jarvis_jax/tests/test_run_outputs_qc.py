import os
import numpy as np
import pytest


def test_module_importable_and_has_cli():
    import jarvis_jax.cse.run_outputs_qc as m
    assert hasattr(m, "run_outputs_qc") and callable(m.run_outputs_qc)
    assert hasattr(m, "main") and callable(m.main)
    assert hasattr(m, "resolve_calib_dir") and callable(m.resolve_calib_dir)


def test_resolve_calib_dir_prefers_refined(tmp_path):
    from jarvis_jax.cse.run_outputs_qc import resolve_calib_dir
    rec = "2026_03_18_15_31_22"
    root = tmp_path / "root"
    cse = tmp_path / "cse_work"
    refined = cse / "calib_refined" / rec
    factory = root / "calib_params" / rec
    factory.mkdir(parents=True)
    (factory / "Cam0.yaml").write_text("x")
    # no refined yet -> factory
    assert resolve_calib_dir(rec, root=str(root), cse_work_dir=str(cse)) == str(factory)
    # refined present -> refined wins
    refined.mkdir(parents=True)
    (refined / "Cam0.yaml").write_text("x")
    assert resolve_calib_dir(rec, root=str(root), cse_work_dir=str(cse)) == str(refined)


def test_resolve_calib_dir_raises_when_missing(tmp_path):
    from jarvis_jax.cse.run_outputs_qc import resolve_calib_dir
    with pytest.raises(FileNotFoundError):
        resolve_calib_dir("nope", root=str(tmp_path / "r"), cse_work_dir=str(tmp_path / "c"))


IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
CSE = "/gscratch/portia/eabe/data/Johnson_lab/cse_work"


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present (GPU-only heavy run)")
def test_run_outputs_qc_smoke_keys(tmp_path):
    """Tiny 2-frame real-data smoke (coordinator GPU): asserts the output h5
    keys + QC json keys exist. NOT a scientific magnitude check."""
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.cse.run_outputs_qc import run_outputs_qc
    rep = run_outputs_qc(
        "2026_03_18_15_31_22", ik_h5=IK, model_xml=XML, mesh_npz=MESH, root=ROOT,
        split="val", cse_work_dir=CSE, out_dir=str(tmp_path), mesh_subset="fps_500",
        max_frames=2, make_video=False)
    assert os.path.exists(rep["out_h5"]) and os.path.exists(rep["qc_json"])
    d = ioh5.load(rep["out_h5"])
    for k in ("qpos", "root_se3", "scale", "mesh_mm", "kp3d_mm",
              "mesh_vert_idx", "mesh_subset", "kp_names"):
        assert k in d
    assert np.asarray(d["mesh_mm"]).shape[0] == 2
    for k in ("per_camera_reproj_px", "loo_reproj_px", "silhouette_iou", "n_frames"):
        assert k in rep["qc"]
    assert rep["qc"]["n_frames"] == 2
