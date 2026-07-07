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
