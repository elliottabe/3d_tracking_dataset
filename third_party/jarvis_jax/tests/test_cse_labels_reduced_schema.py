import os
import h5py
import numpy as np
import pytest

GM = "/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
ANATOMY = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs/anatomy/v1.yaml"
AMP = f"{GM}/S8_male_R_amp"
AMP_REC = "2026_02_13_13_44_49"
AMP_COCO = f"{AMP}/annotations/instances_train.json"       # train=174 framesets
AMP_CALIB = f"{AMP}/calib_params"

_HAVE = all(os.path.exists(p) for p in (AMP_COCO, XML, ANATOMY))


def test_reorder_index_intersection_no_keyerror():
    """The core blocker: _reorder_index must be fed a dst intersected with src,
    so an absent canonical name never triggers pos[n] KeyError."""
    from jarvis_jax.cse.cse_labels import _reorder_index
    src = ["A", "B", "D"]                 # recording (missing C)
    dst_full = ["A", "B", "C", "D"]       # model order
    # intersecting first is the required pattern:
    dst = [n for n in dst_full if n in set(src)]
    idx = _reorder_index(src, dst)
    assert [src[i] for i in idx] == dst   # src[idx] == dst
    # and the OLD unguarded call would KeyError -- assert we no longer do that path
    with pytest.raises(KeyError):
        _reorder_index(src, dst_full)     # documents why the intersection is needed


@pytest.mark.skipif(not _HAVE, reason="amputee coco / model / anatomy not present")
def test_build_bout_amputee_writes_44_kp_names(tmp_path):
    from jarvis_jax.cse.cse_labels import build_bout
    out = str(tmp_path / f"{AMP_REC}_bout.h5")
    p, s = build_bout(AMP_COCO, AMP_CALIB, AMP_REC, ANATOMY, XML, out)
    assert os.path.exists(p)
    with h5py.File(p, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f["kp_names"][()]]
        assert len(names) == 44, f"expected 44 kp_names, got {len(names)}"
        # the 6 T1R distal markers are absent; T1R_ThxCx kept.
        for miss in ("T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip"):
            assert miss not in names
        assert "T1R_ThxCx" in names
        assert f["keypoints"].shape[1] == 44
        assert f["vis"].shape[1] == 44
        assert f["keypoints"].shape[0] == f["fs_imgids"].shape[0] > 0
    assert np.isfinite(s) and s > 0
