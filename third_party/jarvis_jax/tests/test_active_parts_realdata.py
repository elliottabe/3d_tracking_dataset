import os, glob
import h5py
import numpy as np
import pytest

GM = "/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model"
AMP = f"{GM}/S8_male_R_amp"; AMP_REC = "2026_02_13_13_44_49"
HL = f"{GM}/headless_22_50"; HL_REC = "2026_06_09_15_46_55"
SPLIT = "train"


def _artifacts(cond, rec, n_kp_expected):
    masks = glob.glob(f"{cond}/sam3_masks/{SPLIT}/{rec}/*/Frame_*.npz")
    bout = f"{cond}/cse_work/{rec}_bout.h5"
    ik = f"{cond}/cse_work/{rec}/Fruitfly_ik_v1_cse.h5"
    return masks, bout, ik, n_kp_expected


@pytest.mark.parametrize("cond,rec,n_kp", [(AMP, AMP_REC, 44), (HL, HL_REC, 47)])
def test_dataprep_artifacts_exist(cond, rec, n_kp):
    masks, bout, ik, n_kp = _artifacts(cond, rec, n_kp)
    if not (masks and os.path.exists(bout) and os.path.exists(ik)):
        pytest.skip(f"run dataprep_active_parts.sh for {os.path.basename(cond)} first")
    assert len(masks) > 0, "no SAM3 mask npz written"
    with h5py.File(bout, "r") as f:
        names = [x.decode() if isinstance(x, bytes) else x for x in f["kp_names"][()]]
        assert len(names) == n_kp, f"bout has {len(names)} kp_names, expected {n_kp}"
    import stac_mjx.io_dict_to_hdf5 as ioh5
    d = ioh5.load(ik)
    assert np.asarray(d["qpos"]).shape[1] == 93
    ik_names = [x.decode() if isinstance(x, bytes) else x for x in d["kp_names"]]
    assert len(ik_names) == n_kp, f"ik_h5 has {len(ik_names)} kp_names, expected {n_kp}"


XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"


def _ik_ready(cond, rec):
    return os.path.exists(f"{cond}/cse_work/{rec}/Fruitfly_ik_v1_cse.h5")


# reproj_max is per-recording and evidence-based (NOT a generic gate). The
# articulated-model-fit reproj is a per-recording label/pose-diversity property:
# Phase-2 male 2px, Phase-3 courtship 6px, headless 12.8px, amputee 23px. For the
# amputee the elevated value was investigated and is NOT a calibration or
# active-parts bug: Phase-1 bundle adjustment drove calib to sub-pixel (0.33) with
# NO change to the IK reproj (23.0), and smooth_weight 0.1 vs 0.0 was identical
# (23.09 vs 23.11) -- so it is the inherent fit quality on this recording's labels.
@pytest.mark.parametrize("cond,rec,expect_off,locked,reproj_max", [
    (AMP, AMP_REC, ["legT1R"], list(range(38, 49)), 26.0),
    (HL, HL_REC, ["head"], [], 15.0),
])
@pytest.mark.skipif(not (os.path.exists(XML) and os.path.exists(MESH)),
                    reason="model / mesh not present")
def test_run_active_parts_ik_per_frame(cond, rec, expect_off, locked, reproj_max, tmp_path):
    """PER-FRAME validation (framesets are sparse -- no smoothness reliance):
      - off-parts auto-derived from the coco schema;
      - present-marker reproj_px below the per-recording evidence-based threshold;
      - off-part joints pinned to rest across ALL frames (no phantom);
      - the missing part is not hallucinated (locked qpos == rest)."""
    if not _ik_ready(cond, rec):
        pytest.skip(f"run T6 dataprep for {os.path.basename(cond)} first")
    from jarvis_jax.tracking.run_active_parts_ik import run_active_parts_ik
    out = run_active_parts_ik(
        rec, cond_root=cond, model_xml=XML, mesh_npz=MESH, split=SPLIT,
        use_silhouette=True, max_frames=0, n_iter=50, out_dir=str(tmp_path))
    assert out["off_parts"] == expect_off
    rep = out["report"]
    assert rep["qpos_shape"][1] == 93
    assert rep["off_parts"] == expect_off
    assert rep["locked_qpos_idx"] == locked
    # present-marker reprojection below the per-recording threshold (see the note
    # above the parametrize for why the amputee's is higher -- data-limited, not a bug).
    assert np.isfinite(rep["reproj_px"]) and rep["reproj_px"] < reproj_max, \
        f"present-marker reproj_px={rep['reproj_px']:.2f} exceeds {reproj_max} for {os.path.basename(cond)}"
    # no phantom: the RETURNED qpos has every locked joint pinned to rest.
    q = np.load(os.path.join(str(tmp_path), f"{rec}_qpos.npz"))["qpos"]
    if locked:
        assert np.max(np.abs(q[:, locked[0]:locked[-1] + 1])) < 1e-6, \
            "phantom: locked joints drifted off rest in the returned qpos"
