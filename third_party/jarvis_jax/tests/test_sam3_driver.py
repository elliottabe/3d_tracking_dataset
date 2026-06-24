import os, json
import numpy as np
import pytest
from jarvis_jax.predict.sam3_driver import (
    session_tag_for, parse_bouts, video_paths_for, bout_stats, build_manifest,
)


def test_session_tag_for():
    d = "/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04"
    assert session_tag_for(d) == "Session0/2025_10_20_13_20_04"
    assert session_tag_for(d + "/") == "Session0/2025_10_20_13_20_04"


def test_parse_bouts_filters_and_limits(tmp_path):
    csv = tmp_path / "b.csv"
    csv.write_text(
        "fly_id,bout_idx,start_frame,end_frame\n"
        "Session0/rec,1,100,200\n"
        "Session0/rec,2,300,350\n"
        "OtherSession/rec,3,0,10\n")
    rows = parse_bouts(str(csv), "Session0/rec")
    assert [r["bout_idx"] for r in rows] == [1, 2]
    assert rows[0] == {"bout_idx": 1, "start": 100, "end": 200, "n": 101}
    # limit
    assert [r["bout_idx"] for r in parse_bouts(str(csv), "Session0/rec", limit=1)] == [1]
    # explicit ids
    assert [r["bout_idx"] for r in parse_bouts(str(csv), "Session0/rec", bout_ids=[2])] == [2]


def test_video_paths_for_orders_by_camera(tmp_path):
    for cam in ("CamA", "CamB"):
        (tmp_path / f"{cam}.mp4").write_bytes(b"x")
    paths = video_paths_for(str(tmp_path), ["CamB", "CamA"])
    assert [os.path.basename(p) for p in paths] == ["CamB.mp4", "CamA.mp4"]
    with pytest.raises(FileNotFoundError):
        video_paths_for(str(tmp_path), ["CamMissing"])


class _FakeLoaded:
    # mimics LoadedBoutMasks surface used by bout_stats
    def __init__(self, valid, centroids):
        self.valid = valid                      # (A,C,F) bool
        self.centroids = centroids              # (A,C,F,2)
        self.num_animals_saved = valid.shape[0]
        self.num_cameras = valid.shape[1]
        self.num_frames = valid.shape[2]


def test_bout_stats():
    A, C, F = 2, 7, 10
    valid = np.zeros((A, C, F), bool); valid[:, :4, :] = True   # 4/7 cams valid
    cent = np.zeros((A, C, F, 2), np.float32)
    st = bout_stats(_FakeLoaded(valid, cent), num_animals=2)
    assert st["num_animals_saved"] == 2 and st["num_frames"] == 10
    assert abs(st["per_fly_valid_frac"][0] - 4 / 7) < 1e-6
    assert st["mean_cams_valid_per_frame"] == 4.0


def test_build_manifest_roundtrips(tmp_path):
    m = build_manifest("/s/dir", "S/rec", {"sam3_version": "sam3.1"},
                       [{"bout_idx": 1, "num_frames": 10}])
    assert m["session_tag"] == "S/rec" and m["n_bouts"] == 1
    assert m["sam3_settings"]["sam3_version"] == "sam3.1"
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    assert json.loads((tmp_path / "manifest.json").read_text())["n_bouts"] == 1


# ---------------------------------------------------------------------------
# GPU integration test (requires CUDA + SAM3 environment)
# ---------------------------------------------------------------------------

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


needs_sam3 = pytest.mark.skipif(not _has_cuda(), reason="needs CUDA+SAM3")

SESSION = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04"
# Bout 4 is the shortest bout (394 frames) — used to avoid OOM on a long first bout.
SHORT_BOUT_IDX = 4
# The JARVIS checkout that holds the projects/ directory (red_data_unified).
# third_party/JARVIS-HybridNet only contains models/code, not project configs.
JARVIS_ROOT = "/mmfs1/gscratch/portia/eabe/Research/Github/JARVIS-HybridNet"


@needs_sam3
def test_run_sam3_masks_one_bout(tmp_path):
    from jarvis_jax.predict.sam3_driver import run_sam3_masks, parse_bouts
    csv_path = os.path.join(SESSION, "courtship_bouts_unified_summary.csv")

    # Use the shortest bout (bout_idx=4, 394 frames) to reduce memory pressure.
    target = parse_bouts(csv_path, session_tag_for(SESSION),
                         bout_ids=[SHORT_BOUT_IDX])
    assert target, f"bout {SHORT_BOUT_IDX} not found in CSV for session tag"

    man = run_sam3_masks(
        project="red_data_unified",
        session_dir=SESSION,
        bouts_csv=csv_path,
        out=str(tmp_path),
        num_animals=2,
        bout_ids=[SHORT_BOUT_IDX],
        reuse_masks=False,
        jarvis_root=JARVIS_ROOT,
        sam3={"sam3_version": "sam3.1", "gpu_id": 0,
              "compile": False, "text_prompt": "insect"},
    )

    # --- manifest shape ---
    assert man["n_bouts"] == 1

    # --- npz exists ---
    npz = tmp_path / f"bout_{SHORT_BOUT_IDX:05d}" / "sam3_masks.npz"
    assert npz.is_file(), f"sam3_masks.npz not written: {npz}"

    # --- reload + identity check ---
    import importlib.util
    pmod_path = os.path.join(
        SESSION,
        "../../../../../../../../..",   # relative escape not reliable — use abs path
    )
    pmod_abs = "/mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet/tools/predict3D_multianimal.py"
    spec = importlib.util.spec_from_file_location("predict3D_multianimal", pmod_abs)
    pmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pmod)

    lm = pmod.LoadedBoutMasks(str(npz))

    # 2 fly identities saved
    assert lm.num_animals_saved == 2, (
        f"expected 2 animals, got {lm.num_animals_saved}")

    # masks valid in >=2 cameras for at least one frame
    assert lm.valid.any(axis=0).sum(axis=0).max() >= 2, (
        "never had >=2 cameras valid in any frame")

    # manifest bout stats
    st = man["bouts"][0]
    assert st["mean_cams_valid_per_frame"] > 0, (
        "mean_cams_valid_per_frame is 0 — all masks missing?")

    # --- temporal identity consistency check ---
    # Sample up to 20 consecutive valid frames and verify each fly's centroid
    # does NOT jump discontinuously (swap) from frame to frame.  We compare
    # the centroid displacement of the same fly across consecutive frames to
    # the cross-fly distance; a jump factor >0.9 in the majority of steps
    # would indicate identity swaps.
    valid = lm.valid        # (A, C, F)
    centroids = lm.centroids  # (A, C, F, 2)
    A, C, F = valid.shape
    # Find a camera with many valid frames for both flies
    cam_both = -1
    for c in range(C):
        if valid[0, c].sum() > 20 and valid[1, c].sum() > 20:
            cam_both = c
            break
    if cam_both >= 0:
        # frames where both flies valid in this camera
        both_valid = valid[0, cam_both] & valid[1, cam_both]
        frames_idx = np.where(both_valid)[0]
        # take up to 20 consecutive valid frames
        sampled = frames_idx[:20]
        if len(sampled) >= 3:
            c0 = centroids[0, cam_both, sampled]   # (N, 2) fly 0
            c1 = centroids[1, cam_both, sampled]   # (N, 2) fly 1
            # same-fly displacement between consecutive frames
            d0 = np.linalg.norm(np.diff(c0, axis=0), axis=1)  # (N-1,)
            d1 = np.linalg.norm(np.diff(c1, axis=0), axis=1)
            # cross-fly distance at each frame (not diff — actual separation)
            cross = np.linalg.norm(c0[:-1] - c1[:-1], axis=1)
            # A swap would appear as a same-fly displacement ~= cross distance.
            # Flag steps where displacement >= 0.8 * cross distance.
            swap_ratio_0 = (d0 >= 0.8 * cross).mean()
            swap_ratio_1 = (d1 >= 0.8 * cross).mean()
            assert swap_ratio_0 < 0.5, (
                f"fly 0 centroid jumps suspiciously often (swap_ratio={swap_ratio_0:.2f})")
            assert swap_ratio_1 < 0.5, (
                f"fly 1 centroid jumps suspiciously often (swap_ratio={swap_ratio_1:.2f})")
