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
