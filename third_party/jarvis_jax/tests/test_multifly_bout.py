import os

import h5py
import numpy as np
import pytest

from jarvis_jax.tracking.multifly_bout import build_fly_bout
from jarvis_jax.tracking.identity_link import link_recording

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
REC = "2026_04_07_11_33_33"
CALIB = f"{ROOT}/calib_params/{REC}"
COCO = f"{ROOT}/annotations/instances_val.json"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
# anatomy yaml giving the model KEYPOINT_MODEL_PAIRS order (same one build_bout uses):
ANATOMY = os.environ.get(
    "STAC_ANATOMY_V1",
    "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs/anatomy/v1.yaml",
)
# a known-good single-fly bout h5 (cse_labels.build_bout output) used only to
# pin down the keypoint magnitude CONVENTION (scaled model-cm, not raw mm).
REFERENCE_BOUT = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_04_01_16_23_08_bout.h5"

_HAVE = os.path.exists(COCO) and os.path.exists(XML) and os.path.exists(ANATOMY)


@pytest.mark.skipif(not _HAVE, reason="courtship coco / model / anatomy not present")
def test_build_fly_bout_two_flies_differ_and_schema_matches(tmp_path):
    m = link_recording(COCO, REC, CALIB, split="val", n_flies=2)
    out0 = str(tmp_path / f"{REC}_fly0_bout.h5")
    out1 = str(tmp_path / f"{REC}_fly1_bout.h5")
    p0, s0 = build_fly_bout(COCO, CALIB, REC, m, 0, ANATOMY, XML, out0, split="val")
    p1, s1 = build_fly_bout(COCO, CALIB, REC, m, 1, ANATOMY, XML, out1, split="val")

    with h5py.File(p0, "r") as f0, h5py.File(p1, "r") as f1:
        # EXACT schema (matches cse_labels.build_bout / run_stac_bout expectations).
        for f in (f0, f1):
            assert set(["keypoints", "kp_names", "vis", "fs_keys", "fs_imgids"]) <= set(f.keys())
            assert f["keypoints"].ndim == 3 and f["keypoints"].shape[1] == 50
            assert f["keypoints"].shape[2] == 3
            assert f["keypoints"].dtype == np.float32
            assert f["kp_names"].shape == (50,)
            assert f["vis"].shape == f["keypoints"].shape[:2]
            assert f["vis"].dtype == bool
            assert f["fs_imgids"].shape[0] == f["keypoints"].shape[0]
            assert "scale" in f.attrs and "recording" in f.attrs

        k0 = f0["keypoints"][()]; k1 = f1["keypoints"][()]
        v0 = f0["vis"][()]; v1 = f1["vis"][()]
        keys0 = set(f0["fs_keys"][()].astype(str)); keys1 = set(f1["fs_keys"][()].astype(str))
        img0 = f0["fs_imgids"][()]; img1 = f1["fs_imgids"][()]

    # Per-fly keypoints must be DEMONSTRABLY different on shared framesets
    # (identity de-collapse: fly0 != fly1).
    assert len(k0) > 10 and len(k1) > 10
    common = sorted(keys0 & keys1)
    assert len(common) > 10, "flies share too few framesets"
    with h5py.File(p0, "r") as f0:
        ks0 = list(f0["fs_keys"][()].astype(str))
    with h5py.File(p1, "r") as f1:
        ks1 = list(f1["fs_keys"][()].astype(str))
    i0 = {k: i for i, k in enumerate(ks0)}
    i1 = {k: i for i, k in enumerate(ks1)}
    diffs = []
    for k in common[:20]:
        a = k0[i0[k]]; b = k1[i1[k]]
        both = v0[i0[k]] & v1[i1[k]]
        if both.sum() >= 5:
            diffs.append(np.linalg.norm(a[both] - b[both], axis=1).mean())
    assert np.mean(diffs) > 0.05, (
        f"fly0 and fly1 keypoints nearly identical (mean {np.mean(diffs):.4f} model-cm)"
        " -- identity de-collapse failed"
    )

    # fs_imgids row must point at coco image ids that belong to THIS recording.
    import json
    coco = json.load(open(COCO))
    rec_img_ids = {im["id"] for im in coco["images"]
                   if im["file_name"].startswith(f"val/") and REC in im["file_name"]}
    assert set(int(x) for x in img0.reshape(-1) if x >= 0) <= rec_img_ids or len(rec_img_ids) == 0


@pytest.mark.skipif(not _HAVE, reason="courtship coco / model / anatomy not present")
@pytest.mark.skipif(not os.path.exists(REFERENCE_BOUT), reason="reference single-fly bout h5 not present")
def test_build_fly_bout_keypoint_convention_matches_existing_bout(tmp_path):
    """cse_labels.build_bout stores keypoints SCALED to model-cm (kp_mm * s),
    not raw triangulated mm -- see build_bout's docstring ``kp_stac = kp_mm *
    s`` and its variable name ``kp_scaled``. This per-fly builder must match
    that convention exactly (same order of magnitude), or the Task-5 STAC
    solve silently gets double-scaled/garbage input.
    """
    m = link_recording(COCO, REC, CALIB, split="val", n_flies=2)
    out0 = str(tmp_path / f"{REC}_fly0_bout.h5")
    p0, s0 = build_fly_bout(COCO, CALIB, REC, m, 0, ANATOMY, XML, out0, split="val")

    with h5py.File(REFERENCE_BOUT, "r") as fref:
        kref = fref["keypoints"][()]
        vref = fref["vis"][()]
        scale_ref = float(fref.attrs["scale"])
    with h5py.File(p0, "r") as f0:
        k0 = f0["keypoints"][()]
        v0 = f0["vis"][()]
        scale0 = float(f0.attrs["scale"])

    ref_med = float(np.nanmedian(np.abs(kref[vref])))
    new_med = float(np.nanmedian(np.abs(k0[v0])))
    ref_max = float(np.nanmax(kref[vref]))
    new_max = float(np.nanmax(k0[v0]))

    # Same regime: both are "model-cm" scale (order ~0.01-10), NOT raw mm
    # (order ~10-500) and not some other arbitrary unit. Guard against a
    # 10x+ mismatch in either direction, which is what double-scaling or
    # missing-scaling would produce.
    assert 0.1 < new_med / ref_med < 10.0, (
        f"keypoint magnitude regime mismatch: reference median|kp|={ref_med:.4f}, "
        f"new median|kp|={new_med:.4f} -- convention (scaled vs raw mm) likely wrong"
    )
    assert new_max < 50.0, (
        f"new bout keypoint max {new_max:.2f} looks like raw mm, not scaled model-cm "
        f"(reference max {ref_max:.2f}, reference scale attr {scale_ref:.5f})"
    )
    # scale attrs should also be the same order of magnitude (both are
    # mm->model-cm Umeyama scales for the same rig/model).
    assert 0.1 < scale0 / scale_ref < 10.0, (
        f"scale attr regime mismatch: reference scale={scale_ref:.5f}, new scale={scale0:.5f}"
    )
