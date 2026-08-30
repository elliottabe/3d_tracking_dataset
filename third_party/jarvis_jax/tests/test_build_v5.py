import json
import os
from jarvis_jax.data.build_v5 import SourceRec, build_manifest, link_media

CAMS = ["Cam2012630", "Cam2012631"]

def _fake_source(tmp_path, rec, n_frames=3, first=8.1001, masks=True):
    img_root = tmp_path / "src" / rec / "images"
    for c in CAMS:
        (img_root / c).mkdir(parents=True)
        for i in range(n_frames):
            (img_root / c / f"Frame_{i:06d}.jpg").write_bytes(b"\xff\xd8fake")
    calib = tmp_path / "src" / rec / "calib"
    calib.mkdir(parents=True)
    for c in ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
              "Cam2012857", "Cam2012861", "Cam2012862"]:
        data = ", ".join(str(v) for v in [first, 0.0074869, -0.031773, -2.828,
                                          0.0093308, -8.0788, -0.17912, 462.78,
                                          0.0, 0.0, 0.0, 1.0])
        (calib / f"{c}.yaml").write_text(
            f"%YAML:1.0\n---\nprojectionMatrix: !!opencv-matrix\n   data: [ {data} ]\n")
    mask_root = None
    if masks:
        mask_root = tmp_path / "src" / rec / "masks"
        for c in CAMS:
            (mask_root / c).mkdir(parents=True)
            for i in range(n_frames):
                (mask_root / c / f"Frame_{i:06d}.npz").write_bytes(b"npz")
    return SourceRec(recording=rec, subset=f"sub_{rec}", ann_paths=[],
                     calib_dir=str(calib), image_root=str(img_root),
                     mask_root=str(mask_root) if mask_root else None)

def test_manifest_records_calib_group_and_mask_availability(tmp_path):
    srcs = {"rec_a": _fake_source(tmp_path, "rec_a", first=8.1001, masks=True),
            "rec_b": _fake_source(tmp_path, "rec_b", first=8.1333, masks=False)}
    out = tmp_path / "v5"
    man = build_manifest(srcs, str(out))
    assert man["recordings"]["rec_a"]["has_masks"] is True
    assert man["recordings"]["rec_b"]["has_masks"] is False
    assert man["recordings"]["rec_a"]["calib_group"] != man["recordings"]["rec_b"]["calib_group"]
    assert man["recordings"]["rec_a"]["sex"] == "unknown"
    assert json.load(open(out / "manifest.json")) == man

def test_calibrations_are_deduplicated_not_copied_per_recording(tmp_path):
    srcs = {f"rec_{i}": _fake_source(tmp_path, f"rec_{i}", first=8.1001)
            for i in range(3)}
    out = tmp_path / "v5"
    build_manifest(srcs, str(out))
    groups = sorted(os.listdir(out / "calibrations"))
    assert groups == ["A"], f"3 identical calibrations must dedup to 1 dir, got {groups}"
    assert len(os.listdir(out / "calibrations" / "A")) == 7

def test_link_media_creates_symlinks_with_no_split_dirs(tmp_path):
    srcs = {"rec_a": _fake_source(tmp_path, "rec_a")}
    out = tmp_path / "v5"
    build_manifest(srcs, str(out))
    link_media(srcs, str(out))
    p = out / "images" / "rec_a" / "Cam2012630" / "Frame_000000.jpg"
    assert p.is_symlink(), "images must be symlinked, not copied"
    assert (out / "masks" / "rec_a" / "Cam2012630" / "Frame_000000.npz").exists()
    # The whole point: no train/ or val/ directory anywhere in the media tree.
    for root, dirs, _ in os.walk(out / "images"):
        assert "train" not in dirs and "val" not in dirs
