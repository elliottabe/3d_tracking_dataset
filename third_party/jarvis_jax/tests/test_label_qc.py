"""Label QC: completeness and 2D-vs-3D reprojection on a raw red export.

Two defects this catches that nothing else in the repo did (both shipped in
the 2026-09-02 export and were found by hand on 2026-09-03):

  1. A recording whose `calibration/` is not the one its labels were
     triangulated with. Six export dirs carried one byte-identical
     calibration; for two of them the exported 3D reprojects 4-15 px off the
     exported 2D. Every existing guard (keypoint order, split leaks, sex) was
     blind to it, and the 3D trainer triangulates with the shipped group.
  2. Frames whose exported 3D is exactly 10x the value re-triangulated from
     their own 2D (five frames across three recordings).

Synthetic throughout: an affine camera pair with known projection matrices.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from jarvis_jax.data import label_qc

H, W = 448, 1936
K = 50
SKEL = "/home/user/red_data/skeleton/fly50.json"
CAMS = ["CamA", "CamB", "CamC"]


def _P(cam: str, wrong: bool = False) -> np.ndarray:
    """Affine DLT per camera (row 2 = [0,0,0,1]), three DIFFERENT viewing
    directions so triangulation is well posed; `wrong` shifts the image."""
    P = {"CamA": [[80.0, 0.0, 3.0, 500.0], [0.0, -80.0, 5.0, 400.0]],      # top
         "CamB": [[80.0, 2.0, 0.0, 800.0], [0.0, 4.0, -80.0, 300.0]],      # side
         "CamC": [[3.0, 80.0, 0.0, 1100.0], [5.0, 0.0, -80.0, 300.0]]}[cam]  # end
    P = np.array(P + [[0.0, 0.0, 0.0, 1.0]])
    if wrong:
        P[0, 3] += 12.0            # a 12 px horizontal shift
    return P


def _write_raw_csv(path, ndim, rows):
    with open(path, "w") as f:
        f.write(SKEL + "\n")
        for frame, arr in rows.items():
            parts = [str(frame)]
            for k, v in enumerate(arr):
                parts.append(str(k))
                parts += [repr(float(x)) for x in v]
            f.write(",".join(parts) + "\n")


def _write_yaml(path, P, scale=1):
    vals = ", ".join(repr(float(v)) for v in P.ravel())
    with open(path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write(f"image_width: {W}\nimage_height: {H}\n")
        f.write("projectionMatrix: !!opencv-matrix\n   rows: 3\n   cols: 4\n"
                f"   dt: d\n   data: [ {vals} ]\nscale: {scale}\n")


def _project(P, X):
    p = P @ np.append(X, 1.0)
    return p[:2] / p[2]


@pytest.fixture
def raw_export(tmp_path):
    """A raw recording dir whose 2D IS the projection of its 3D through the
    shipped DLT (v written bottom-origin, as red does), with:
      - kp 3 unlabelled everywhere (1e7 sentinel in 2D and 3D),
      - CamC missing every keypoint on frame 20,
      - frame 30's 3D written 10x too large (the real export's defect)."""
    rec = tmp_path / "2099_01_01_00_00_00_male"
    (rec / "calibration").mkdir(parents=True)
    rng = np.random.default_rng(1)
    frames = [10, 20, 30]
    X = {fr: np.c_[rng.uniform(6, 11, K), rng.uniform(0.5, 4.0, K),
                   rng.uniform(0.5, 2.5, K)] for fr in frames}
    kp2d = {c: {} for c in CAMS}
    for fr in frames:
        for c in CAMS:
            uv = np.array([_project(_P(c), X[fr][k]) for k in range(K)])
            uv[:, 1] = H - uv[:, 1]                     # bottom-origin v
            uv[3] = [1e7, 1e7]
            if c == "CamC" and fr == 20:
                uv[:] = 1e7
            kp2d[c][fr] = uv
    X3 = {fr: X[fr].copy() for fr in frames}
    for fr in frames:
        X3[fr][3] = 1e7
    X3[30] = np.where(X3[30] >= 1e6, 1e7, X3[30] * 10.0)
    for c in CAMS:
        _write_raw_csv(rec / f"{c}.csv", 2, kp2d[c])
        coefs = _P(c).ravel()[:11]
        (rec / "calibration" / f"{c}_dlt.csv").write_text(
            "\n".join(repr(float(v)) for v in coefs) + "\n")
        _write_yaml(rec / "calibration" / f"{c}.yaml", _P(c))
    _write_raw_csv(rec / "keypoints3d.csv", 3, X3)
    return rec


def test_reader_maps_sentinel_to_nan_and_keeps_frames(raw_export):
    raw = label_qc.read_raw_labels(raw_export)
    assert raw.cams == CAMS and raw.frames == [10, 20, 30]
    assert raw.image_hw["CamA"] == (H, W)
    assert np.isnan(raw.kp2d["CamA"][10][3]).all()
    assert np.isnan(raw.kp2d["CamC"][20]).all()
    assert np.isnan(raw.kp3d[10][3]).all()
    assert np.isfinite(raw.kp2d["CamA"][10][0]).all()


def test_shipped_dlt_reproduces_labels_and_flags_the_10x_frame(raw_export):
    raw = label_qc.read_raw_labels(raw_export)
    proj = label_qc.dlt_projection(raw_export / "calibration", raw.cams)
    rep = label_qc.reprojection_report(raw, proj)
    assert rep["median_px"] < 1e-6
    assert rep["n_frames"] == 3
    assert rep["per_camera_median_px"]["CamB"] < 1e-6
    bad = rep["inconsistent_frames"]
    assert [b["frame"] for b in bad] == [30]
    assert bad[0]["median_px"] > 100
    assert abs(bad[0]["scale_ratio"] - 10.0) < 0.05
    # kp 3 (unlabelled) and CamC/frame 20 (all missing) contribute nothing
    assert rep["n_points"] == (3 * 3 - 1) * (K - 1)


def test_yaml_projection_with_scale_matches_dlt(raw_export):
    raw = label_qc.read_raw_labels(raw_export)
    ydir = raw_export.parent / "yaml_scaled"
    ydir.mkdir()
    for c in CAMS:
        P = _P(c).copy()
        P[0:2, 0:3] *= 0.1            # the JARVIS scale-10 convention
        _write_yaml(ydir / f"{c}.yaml", P, scale=10)
    proj = label_qc.yaml_projection(ydir, raw.cams)
    assert proj["CamA"].scale == 10.0
    rep = label_qc.reprojection_report(raw, proj)
    assert rep["median_px"] < 1e-6


def test_wrong_calibration_is_measured_and_best_group_is_named(raw_export):
    raw = label_qc.read_raw_labels(raw_export)
    good = raw_export.parent / "good"
    bad = raw_export.parent / "bad"
    good.mkdir(); bad.mkdir()
    for c in CAMS:
        _write_yaml(good / f"{c}.yaml", _P(c))
        _write_yaml(bad / f"{c}.yaml", _P(c, wrong=True))
    best, per = label_qc.best_calibration_group(
        raw, {"A": str(bad), "B": str(good)})
    assert best == "B"
    assert per["B"]["median_px"] < 1e-6
    assert 11.0 < per["A"]["median_px"] < 13.0
    assert label_qc.calibration_consistent(per["B"])
    assert not label_qc.calibration_consistent(per["A"])


def test_recording_id_from_export_dir_name():
    f = label_qc.recording_id_of
    assert f("2026_04_02_15_25_51_female") == "2026_04_02_15_25_51"
    assert f("2025_10_20_13_20_04_female_climbing") == "2025_10_20_13_20_04"
    assert f("not_a_recording") is None


def _coco(recs):
    """Tiny merged COCO: recs = [(recording, camera, frame, [vis-mask,...])]."""
    images, anns = [], []
    for i, (rec, cam, fr, masks) in enumerate(recs):
        images.append({"id": i, "recording": rec, "width": W, "height": H,
                       "file_name": f"{rec}/{cam}/Frame_{fr}.jpg"})
        for j, m in enumerate(masks):
            kp = []
            for k in range(K):
                kp += ([10 + k, 20 + k, 1] if m[k] else [0, 0, 0])
            anns.append({"id": len(anns), "image_id": i, "keypoints": kp,
                         "num_keypoints": K, "subset": f"sub{j}", "fly_id": j})
    return images, anns


def test_completeness_counts_missing_by_recording_camera_and_keypoint():
    full = [True] * K
    partial = [True] * K
    partial[3] = partial[7] = False
    images, anns = _coco([
        ("R1", "CamA", 1, [full, partial]),     # two flies, one partial
        ("R1", "CamB", 1, [full]),
        ("R1", "CamC", 1, []),                  # image with no annotation
        ("R2", "CamA", 5, [partial]),
    ])
    names = [f"kp{i}" for i in range(K)]
    c = label_qc.completeness(images, anns, names)
    r1 = c["per_recording"]["R1"]
    assert r1["images"] == 3 and r1["images_without_annotation"] == 1
    assert r1["annotations"] == 3
    assert r1["missing_keypoints"] == 2
    assert r1["fully_labelled_annotations"] == 2
    assert r1["num_keypoints_field_mismatches"] == 1
    assert c["per_recording_camera"]["R1"]["CamA"]["missing_keypoints"] == 2
    assert c["per_recording_camera"]["R1"]["CamB"]["missing_keypoints"] == 0
    assert c["per_keypoint_missing"]["kp3"] == 2
    assert c["per_keypoint_missing"]["kp7"] == 2
    assert "kp0" not in c["per_keypoint_missing"]
    assert c["totals"]["missing_keypoints"] == 4
    assert c["totals"]["keypoint_slots"] == 4 * K
    assert c["per_subset"]["sub1"]["missing_keypoints"] == 2
