import os, glob
import numpy as np
import pytest

# Pre-load huggingface_hub.file_download before jarvis.prediction.sam3_video_tracker
# pulls in sam3.model_builder: SAM3's tqdm patching otherwise corrupts tqdm's
# `set_lock` before huggingface_hub's lazy file_download import runs, breaking
# `from huggingface_hub import hf_hub_download` (see
# third_party/jarvis_jax/scripts/sam3_masks.py for the same workaround).
import huggingface_hub.file_download  # noqa: E402,F401

cv2 = pytest.importorskip("cv2")


def _make_video(path, n, h=16, w=16):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), i % 256, np.uint8)   # frame i is a solid gray = i
        vw.write(frame)
    vw.release()


def test_extract_with_positions_places_black_at_gaps(tmp_path):
    from jarvis.prediction.sam3_video_tracker import SAM3VideoTracker
    vid = str(tmp_path / "Cam.mp4"); _make_video(vid, 50)
    out = str(tmp_path / "frames"); os.makedirs(out, exist_ok=True)
    # 5 output slots: read positions 10,11, GAP, 12,13
    positions = [10, 11, None, 12, 13]
    # call the unbound static-ish extractor without constructing the heavy tracker:
    SAM3VideoTracker._extract_bout_frames(SAM3VideoTracker.__new__(SAM3VideoTracker),
                                          vid, 10, 5, out, positions=positions)
    jpgs = sorted(glob.glob(os.path.join(out, "*.jpg")))
    assert len(jpgs) == 5                       # contiguous 000000..000004, gap padded
    # gap frame (index 2) is all-black; present frames are non-black
    g = cv2.imread(jpgs[2]); assert int(g.max()) == 0
    f0 = cv2.imread(jpgs[0]); assert int(f0.max()) > 0


def test_extract_positions_none_is_positional(tmp_path):
    from jarvis.prediction.sam3_video_tracker import SAM3VideoTracker
    vid = str(tmp_path / "Cam.mp4"); _make_video(vid, 50)
    out = str(tmp_path / "frames"); os.makedirs(out, exist_ok=True)
    SAM3VideoTracker._extract_bout_frames(SAM3VideoTracker.__new__(SAM3VideoTracker),
                                          vid, 5, 4, out, positions=None)
    jpgs = sorted(glob.glob(os.path.join(out, "*.jpg")))
    assert len(jpgs) == 4                       # positional read of 4 frames from 5
