import numpy as np, os
import cv2
from viz.core import io
import stac_mjx.io_dict_to_hdf5 as ioh5

def test_fly_dir_format():
    assert io.fly_dir("/run", 3, 1).endswith("/bouts/bout_00003/fly1")

def test_load_outputs_and_kp2d(tmp_path):
    d = tmp_path/"bouts"/"bout_00001"/"fly0"; d.mkdir(parents=True)
    ioh5.save(str(d/"outputs.h5"), {"kp3d_mm": np.zeros((2,50,3),np.float32),
                                    "mesh_mm": np.zeros((2,300,3),np.float32),
                                    "kp_names": np.array(["a","b"], dtype='S')})
    np.savez(d/"kp2d.npz", kp2d=np.zeros((2,7,50,2),np.float32), conf=np.ones((2,7,50),np.float32))
    o = io.load_outputs(str(tmp_path), 1, 0)
    assert o["kp3d_mm"].shape == (2,50,3) and o["mesh_mm"].shape == (2,300,3)
    kp2d, conf = io.load_kp2d(str(tmp_path), 1, 0)
    assert kp2d.shape == (2,7,50,2) and conf.shape == (2,7,50)


def test_load_data3d_csv(tmp_path):
    # Synthetic per-bout dense CSV: 2 header rows (row1 = base kp name
    # repeated x4, row2 = x,y,z,conf labels) + col0 = absolute frame index.
    p = tmp_path / "fly0.csv"
    header1 = "frame,kpA,kpA,kpA,kpA,kpB,kpB,kpB,kpB,kpC,kpC,kpC,kpC\n"
    header2 = "frame,x,y,z,conf,x,y,z,conf,x,y,z,conf\n"
    row1 = "100,1,2,3,0.9,4,5,6,0.8,7,8,9,0.7\n"
    row2 = "101,1.1,2.1,3.1,0.91,4.1,5.1,6.1,0.81,7.1,8.1,9.1,0.71\n"
    p.write_text(header1 + header2 + row1 + row2)

    kp3d, conf, names, frames = io.load_data3d_csv(str(p))
    assert kp3d.shape == (2, 3, 3)
    assert conf.shape == (2, 3)
    assert names == ["kpA", "kpB", "kpC"]
    assert list(frames) == [100, 101]
    np.testing.assert_allclose(kp3d[0, 0], [1, 2, 3])
    assert conf[0, 0] == 0.9
    np.testing.assert_allclose(kp3d[1, 2], [7.1, 8.1, 9.1])
    assert conf[1, 2] == 0.71


def test_write_video_writes_nonzero_mp4(tmp_path):
    out = tmp_path / "vids" / "out.mp4"
    frames = []
    for i in range(3):
        f = np.zeros((16, 16, 3), dtype=np.uint8)
        f[:] = i * 30
        frames.append(f)

    result = io.write_video(str(out), frames, fps=10)

    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None


def test_write_video_avc1_writes_nonzero_reopenable_mp4(tmp_path):
    # avc1 (H.264) is available in this cv2 build (4.13.0); even if it weren't,
    # write_video falls back to mp4v internally, so this should always produce
    # a nonzero, reopenable mp4 either way.
    out = tmp_path / "vids" / "out_avc1.mp4"
    frames = []
    for i in range(3):
        f = np.zeros((16, 16, 3), dtype=np.uint8)
        f[:] = i * 30
        frames.append(f)

    result = io.write_video(str(out), frames, fps=10, fourcc="avc1")

    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None


def _fly(tmp_path, refined=True, wingfit=None, outputs="SIG"):
    d = tmp_path / "bouts" / "bout_00001" / "fly0"
    d.mkdir(parents=True, exist_ok=True)
    if refined:
        np.savez(d / "qpos_refined.npz", qpos=np.zeros((4, 93), np.float32),
                 bridge_ok=np.ones(4, bool))
    if wingfit is not None:
        kw = {"qpos": np.ones((4, 93), np.float32)}
        if wingfit:                       # wingfit is the signature, or "" for none
            kw["wing_mask_fit_sig"] = wingfit
        np.savez(d / "qpos_wingfit.npz", **kw)
    (d / "outputs.h5").unlink(missing_ok=True)
    if outputs is not None:
        ioh5.save(str(d / "outputs.h5"),
                  {"kp3d_mm": np.zeros((4, 5, 3), np.float32),
                   "pose_source": np.asarray(str(outputs)).astype("S")})
    return d


def test_load_qpos_prefers_the_wing_fit_when_it_is_present_and_stamped(tmp_path):
    """The renderers drew qpos_refined.npz unconditionally, so `python -m viz
    sidebyside` -- the pipeline's DEFAULT visual QC artifact -- showed the
    PRE-FIT pose even on a clean run with the wing-mask fit enabled. A weight
    sweep judged by eye off that render would be a confident, wrong comparison.
    """
    _fly(tmp_path, wingfit=None)
    qpos, src = io.load_qpos(str(tmp_path), 1, 0)
    assert src == "qpos_refined.npz" and qpos.shape == (4, 93)
    assert np.allclose(qpos, 0.0)

    _fly(tmp_path, wingfit="SIG")
    qpos, src = io.load_qpos(str(tmp_path), 1, 0)
    assert src == "qpos_wingfit.npz", "a stamped wing fit must win under 'auto'"
    assert np.allclose(qpos, 1.0)


def test_load_qpos_ignores_an_unstamped_wing_fit(tmp_path):
    """No readable signature means no provenance -- the file could be from any
    config, so 'auto' must not silently prefer it."""
    _fly(tmp_path, wingfit="")            # written, but carries no signature
    qpos, src = io.load_qpos(str(tmp_path), 1, 0)
    assert src == "qpos_refined.npz" and np.allclose(qpos, 0.0)


def test_load_qpos_source_override(tmp_path):
    """Task 7 must be able to render BOTH arms deliberately, not hope 'auto'
    picks the one it meant."""
    import pytest
    _fly(tmp_path, wingfit="SIG")
    assert io.load_qpos(str(tmp_path), 1, 0, source="refined")[1] == "qpos_refined.npz"
    assert io.load_qpos(str(tmp_path), 1, 0, source="wingfit")[1] == "qpos_wingfit.npz"

    _fly(tmp_path, wingfit=None)
    (tmp_path / "bouts" / "bout_00001" / "fly0" / "qpos_wingfit.npz").unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError, match="qpos_wingfit.npz"):
        io.load_qpos(str(tmp_path), 1, 0, source="wingfit")
    with pytest.raises(ValueError, match="source"):
        io.load_qpos(str(tmp_path), 1, 0, source="nonsense")


def test_load_qpos_auto_defers_to_the_pose_outputs_h5_committed_to(tmp_path):
    """`auto` must agree with outputs.h5, not merely with what is on disk.

    The rigcam panel draws TWO things from TWO sources: `rig_world` from
    outputs.h5's kp3d_mm (the fitted sites) and `rig_qpos` from load_qpos (which
    drives the MuJoCo mesh). If those disagree the drawn markers sit off the
    rendered wings -- visually indistinguishable from an IK failure. So
    preferring a stamped qpos_wingfit.npz merely because it EXISTS is wrong:
    after a disable-after-enable, outputs.h5 has been rebuilt from the STAC pose
    and stamped "none" while the wing fit is still on disk.
    """
    _fly(tmp_path, wingfit="SIG", outputs="SIG")
    assert io.load_qpos(str(tmp_path), 1, 0)[1] == "qpos_wingfit.npz"

    # disable-after-enable: outputs.h5 rebuilt from the STAC pose
    _fly(tmp_path, wingfit="SIG", outputs="none")
    assert io.load_qpos(str(tmp_path), 1, 0)[1] == "qpos_refined.npz", \
        "auto must not draw a pose outputs.h5 does not hold"

    # a fit from a DIFFERENT config than the one outputs.h5 was built from
    _fly(tmp_path, wingfit="OTHER", outputs="SIG")
    assert io.load_qpos(str(tmp_path), 1, 0)[1] == "qpos_refined.npz"

    # no outputs.h5 at all -> nothing authoritative to agree with
    _fly(tmp_path, wingfit="SIG", outputs=None)
    assert io.load_qpos(str(tmp_path), 1, 0)[1] == "qpos_refined.npz"

    # an explicit override still wins, so an A/B can force either arm
    _fly(tmp_path, wingfit="SIG", outputs="none")
    assert io.load_qpos(str(tmp_path), 1, 0, source="wingfit")[1] == "qpos_wingfit.npz"


def test_outputs_pose_source_reads_the_stamp_or_none(tmp_path):
    d = _fly(tmp_path, outputs="SIG")
    assert io.outputs_pose_source(str(tmp_path), 1, 0) == "SIG"
    ioh5.save(str(d / "outputs.h5"), {"kp3d_mm": np.zeros((4, 5, 3), np.float32)})
    assert io.outputs_pose_source(str(tmp_path), 1, 0) is None, \
        "an outputs.h5 predating the stamp has no provenance"
    (d / "outputs.h5").unlink()
    assert io.outputs_pose_source(str(tmp_path), 1, 0) is None
