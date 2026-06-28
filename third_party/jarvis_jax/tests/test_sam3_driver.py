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


def test_parse_bouts_no_fly_id_column_keeps_all(tmp_path):
    # Session1 good_bouts schema: no fly_id column -> every row is this recording's
    csv = tmp_path / "good.csv"
    csv.write_text(
        "bout_idx,start_frame,end_frame,n_frames,mean_score\n"
        "0,100,200,101,0.5\n"
        "1,300,350,51,0.4\n")
    rows = parse_bouts(str(csv), "Session1/whatever")
    assert [r["bout_idx"] for r in rows] == [0, 1]
    assert rows[0] == {"bout_idx": 0, "start": 100, "end": 200, "n": 101}


def test_parse_bouts_with_fly_id_still_filters(tmp_path):
    csv = tmp_path / "u.csv"
    csv.write_text(
        "fly_id,bout_idx,start_frame,end_frame\n"
        "S/rec,1,10,20\n"
        "OTHER/rec,2,30,40\n")
    assert [r["bout_idx"] for r in parse_bouts(str(csv), "S/rec")] == [1]


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


# ---------------------------------------------------------------------------
# Pure helpers for multi-GPU bout fan-out (Task 1)
# ---------------------------------------------------------------------------

from jarvis_jax.predict.sam3_driver import (
    resolve_gpus, split_bouts_contiguous, build_worker_cmd, merge_manifests,
)


def test_resolve_gpus_explicit_list():
    assert resolve_gpus([0, 1, 3], env={}, device_count=8) == [0, 1, 3]
    # OmegaConf-style list is just an iterable of ints
    assert resolve_gpus((2, 5), env={}, device_count=8) == [2, 5]


def test_resolve_gpus_from_cuda_visible_devices():
    assert resolve_gpus(None, env={"CUDA_VISIBLE_DEVICES": "2,3,5"},
                        device_count=8) == [2, 3, 5]
    # empty/whitespace entries ignored
    assert resolve_gpus(None, env={"CUDA_VISIBLE_DEVICES": "1, ,4"},
                        device_count=8) == [1, 4]


def test_resolve_gpus_auto_range():
    assert resolve_gpus(None, env={}, device_count=4) == [0, 1, 2, 3]
    # empty explicit list falls through to auto
    assert resolve_gpus([], env={}, device_count=2) == [0, 1]
    # CUDA_VISIBLE_DEVICES empty string -> auto
    assert resolve_gpus(None, env={"CUDA_VISIBLE_DEVICES": ""},
                        device_count=3) == [0, 1, 2]


def test_split_bouts_contiguous():
    assert split_bouts_contiguous([1, 2, 3, 4, 5], 2) == [[1, 2, 3], [4, 5]]
    assert split_bouts_contiguous([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
    # fewer bouts than GPUs -> trailing empty blocks, length == n_gpus
    assert split_bouts_contiguous([1, 2], 3) == [[1], [2], []]
    assert split_bouts_contiguous([], 2) == [[], []]


def test_build_worker_cmd():
    env, argv = build_worker_cmd(
        python="/py", script="/s/sam3_masks.py", gpu=3, bout_ids=[4, 5, 6],
        session_dir="/data/sess", out="/o", project="red_data_unified",
        bouts_csv="/data/b.csv", num_animals=2, reuse_masks=True,
        jarvis_root="/jr", sam3_version="sam3.1", sam3_compile=False,
        sam3_text="insect", sam3_checkpoint=None,
        manifest_name="manifest.gpu3.json", inductor_cache_dir="/tmp/ti_gpu3",
        base_env={"PATH": "/usr/bin"},
    )
    assert argv[0] == "/py" and argv[1] == "/s/sam3_masks.py"
    assert "sam3.gpus=[0]" in argv          # recursion guard
    assert "sam3.sam3_gpu=0" in argv
    assert "sam3.bout_ids='4,5,6'" in argv  # QUOTED so Hydra parses it as a string
    assert "sam3.limit=0" in argv
    assert "sam3.out=/o" in argv            # shared out dir
    assert "sam3.session_dir=/data/sess" in argv
    assert "sam3.bouts_csv=/data/b.csv" in argv
    assert "sam3.reuse_masks=true" in argv
    assert "sam3.jarvis_root=/jr" in argv
    assert "sam3.sam3_compile=false" in argv
    assert "sam3.sam3_checkpoint=null" in argv   # None -> Hydra null
    assert "sam3.manifest_name=manifest.gpu3.json" in argv
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/ti_gpu3"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["PATH"] == "/usr/bin"         # base_env preserved


def test_build_worker_cmd_jarvis_root_none():
    env, argv = build_worker_cmd(
        python="p", script="s", gpu=0, bout_ids=[1], session_dir="d", out="o",
        project="proj", bouts_csv="c", num_animals=2, reuse_masks=False,
        jarvis_root=None, sam3_version="sam3.1", sam3_compile=True,
        sam3_text="insect", sam3_checkpoint="/ckpt", manifest_name="m.json",
        inductor_cache_dir="/t", base_env={},
    )
    assert "sam3.jarvis_root=null" in argv
    assert "sam3.reuse_masks=false" in argv
    assert "sam3.sam3_compile=true" in argv
    assert "sam3.sam3_checkpoint=/ckpt" in argv


def test_build_worker_cmd_overrides_parse_through_hydra():
    """Regression: the worker overrides must actually compose under Hydra.
    A bare comma value (sam3.bout_ids=1,2,3) raises 'Ambiguous value'; this
    confirms the quoted form parses to the expected string. (The dispatcher's
    mocked Popen never exercised real Hydra, so this gap shipped a broken run.)"""
    from hydra import initialize_config_dir, compose
    from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers
    register_resolvers()
    _, argv = build_worker_cmd(
        python="/py", script="/s/sam3_masks.py", gpu=2, bout_ids=[17, 18, 19, 20],
        session_dir="/d/courtship/Session1/rec", out="/o", project="red_data_unified",
        bouts_csv="/d/b.csv", num_animals=2, reuse_masks=True, jarvis_root="/jr",
        sam3_version="sam3.1", sam3_compile=False, sam3_text="insect",
        sam3_checkpoint=None, manifest_name="manifest.gpu2.json",
        inductor_cache_dir="/tmp/ti2", base_env={})
    overrides = argv[2:]   # everything after [python, script]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=overrides)  # must NOT raise
    assert str(cfg.sam3.bout_ids) == "17,18,19,20"
    # the entrypoint parses it the same way it parses a user-supplied value
    assert [int(x) for x in str(cfg.sam3.bout_ids).split(",") if str(x).strip()] == [17, 18, 19, 20]


def test_merge_manifests(tmp_path):
    import json as _json
    p0 = tmp_path / "manifest.gpu0.json"
    p1 = tmp_path / "manifest.gpu1.json"
    p0.write_text(_json.dumps({"bouts": [{"bout_idx": 3, "num_frames": 5},
                                         {"bout_idx": 1, "num_frames": 7}]}))
    p1.write_text(_json.dumps({"bouts": [{"bout_idx": 2, "num_frames": 9}]}))
    base = {"session_dir": "/s", "session_tag": "S/rec",
            "sam3_settings": {"sam3_version": "sam3.1"}}
    m = merge_manifests([str(p0), str(p1)], base=base)
    assert [b["bout_idx"] for b in m["bouts"]] == [1, 2, 3]   # sorted
    assert m["n_bouts"] == 3
    assert m["session_tag"] == "S/rec"                        # base preserved
    assert m["sam3_settings"]["sam3_version"] == "sam3.1"


def test_merge_manifests_dedupes(tmp_path):
    import json as _json
    p0 = tmp_path / "a.json"; p1 = tmp_path / "b.json"
    p0.write_text(_json.dumps({"bouts": [{"bout_idx": 1}]}))
    p1.write_text(_json.dumps({"bouts": [{"bout_idx": 1}, {"bout_idx": 2}]}))
    m = merge_manifests([str(p0), str(p1)], base={})
    assert [b["bout_idx"] for b in m["bouts"]] == [1, 2]
    assert m["n_bouts"] == 2


def test_run_sam3_masks_multi_dispatch(tmp_path, monkeypatch):
    """Dispatcher splits bouts across GPUs, launches one worker per non-empty
    block, and merges per-worker partials into manifest.json — no real GPU."""
    import json as _json
    import subprocess
    from jarvis_jax.predict import sam3_driver

    # A bouts CSV with 4 bouts for our session tag.
    session = "/data/SessionT/rec"            # tag -> "SessionT/rec"
    csv = tmp_path / "b.csv"
    csv.write_text(
        "fly_id,bout_idx,start_frame,end_frame\n"
        "SessionT/rec,0,0,10\n"
        "SessionT/rec,1,20,30\n"
        "SessionT/rec,2,40,50\n"
        "SessionT/rec,3,60,70\n")
    out = tmp_path / "out"

    launched = []

    class _FakePopen:
        def __init__(self, argv, env=None):
            self.argv = argv
            # parse the overrides this worker received
            ov = {a.split("=", 1)[0]: a.split("=", 1)[1]
                  for a in argv if "=" in a and a.startswith("sam3.")}
            launched.append((env["CUDA_VISIBLE_DEVICES"], ov["sam3.bout_ids"].strip("'")))
            # emulate the worker: write a partial manifest for its bout subset
            ids = [int(x) for x in ov["sam3.bout_ids"].strip("'").split(",")]
            os.makedirs(ov["sam3.out"], exist_ok=True)
            partial = os.path.join(ov["sam3.out"], ov["sam3.manifest_name"])
            with open(partial, "w") as f:
                _json.dump({"bouts": [{"bout_idx": i, "num_frames": 11}
                                      for i in ids]}, f)

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    man = sam3_driver.run_sam3_masks_multi(
        gpus=[0, 1], project="red_data_unified", session_dir=session,
        bouts_csv=str(csv), out=str(out), num_animals=2,
        jarvis_root="/jr", sam3={"sam3_version": "sam3.1", "gpu_id": 0,
                                 "compile": False, "text_prompt": "insect",
                                 "checkpoint_path": None},
        python="/py", script="/s/sam3_masks.py")

    # two workers launched, contiguous split [0,1] / [2,3]
    assert sorted(launched) == [("0", "0,1"), ("1", "2,3")]
    # merged manifest written once, with all 4 bouts sorted
    merged = _json.loads((out / "manifest.json").read_text())
    assert [b["bout_idx"] for b in merged["bouts"]] == [0, 1, 2, 3]
    assert merged["n_bouts"] == 4
    assert merged["session_tag"] == "SessionT/rec"
    assert man["n_bouts"] == 4 and man.get("failures") == []


def test_run_sam3_masks_multi_raises_on_worker_failure(tmp_path, monkeypatch):
    """A failing worker (non-zero exit) makes the dispatcher raise AFTER writing
    the partial manifest, so a sequential pipeline does not proceed on partial
    masks and a requeue (reuse_masks) can finish the rest."""
    import json as _json
    import subprocess
    import pytest as _pytest
    from jarvis_jax.predict import sam3_driver

    session = "/data/SessionT/rec"
    csv = tmp_path / "b.csv"
    csv.write_text(
        "fly_id,bout_idx,start_frame,end_frame\n"
        "SessionT/rec,0,0,10\n"
        "SessionT/rec,1,20,30\n"
        "SessionT/rec,2,40,50\n"
        "SessionT/rec,3,60,70\n")
    out = tmp_path / "out"

    class _FakePopen:
        def __init__(self, argv, env=None):
            ov = {a.split("=", 1)[0]: a.split("=", 1)[1]
                  for a in argv if "=" in a and a.startswith("sam3.")}
            self._gpu = env["CUDA_VISIBLE_DEVICES"]
            ids = [int(x) for x in ov["sam3.bout_ids"].strip("'").split(",")]
            os.makedirs(ov["sam3.out"], exist_ok=True)
            # GPU 0 succeeds (writes its partial); GPU 1 "fails" (no partial).
            if self._gpu == "0":
                with open(os.path.join(ov["sam3.out"], ov["sam3.manifest_name"]),
                          "w") as f:
                    _json.dump({"bouts": [{"bout_idx": i} for i in ids]}, f)

        def wait(self):
            return 0 if self._gpu == "0" else 1

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    with _pytest.raises(RuntimeError, match="worker"):
        sam3_driver.run_sam3_masks_multi(
            gpus=[0, 1], project="red_data_unified", session_dir=session,
            bouts_csv=str(csv), out=str(out), num_animals=2, jarvis_root="/jr",
            sam3={"sam3_version": "sam3.1", "gpu_id": 0, "compile": False,
                  "text_prompt": "insect", "checkpoint_path": None},
            python="/py", script="/s/sam3_masks.py")

    # Partial manifest IS written before the raise (resume-safe): the succeeded
    # GPU-0 bouts [0,1] are present.
    merged = _json.loads((out / "manifest.json").read_text())
    assert [b["bout_idx"] for b in merged["bouts"]] == [0, 1]


def test_run_sam3_masks_multi_raises_on_clean_exit_no_manifest(tmp_path, monkeypatch):
    """A worker that exits 0 but writes no manifest is treated as a failure (its
    bouts would otherwise vanish silently from the merge)."""
    import json as _json
    import subprocess
    import pytest as _pytest
    from jarvis_jax.predict import sam3_driver

    session = "/data/SessionT/rec"
    csv = tmp_path / "b.csv"
    csv.write_text(
        "fly_id,bout_idx,start_frame,end_frame\n"
        "SessionT/rec,0,0,10\n"
        "SessionT/rec,1,20,30\n"
        "SessionT/rec,2,40,50\n"
        "SessionT/rec,3,60,70\n")
    out = tmp_path / "out"

    class _FakePopen:
        def __init__(self, argv, env=None):
            ov = {a.split("=", 1)[0]: a.split("=", 1)[1]
                  for a in argv if "=" in a and a.startswith("sam3.")}
            self._gpu = env["CUDA_VISIBLE_DEVICES"]
            os.makedirs(ov["sam3.out"], exist_ok=True)
            # GPU 0 writes its partial; GPU 1 exits 0 but writes nothing.
            if self._gpu == "0":
                ids = [int(x) for x in ov["sam3.bout_ids"].strip("'").split(",")]
                with open(os.path.join(ov["sam3.out"], ov["sam3.manifest_name"]),
                          "w") as f:
                    _json.dump({"bouts": [{"bout_idx": i} for i in ids]}, f)

        def wait(self):
            return 0   # both exit cleanly

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    with _pytest.raises(RuntimeError, match="worker"):
        sam3_driver.run_sam3_masks_multi(
            gpus=[0, 1], project="red_data_unified", session_dir=session,
            bouts_csv=str(csv), out=str(out), num_animals=2, jarvis_root="/jr",
            sam3={"sam3_version": "sam3.1", "gpu_id": 0, "compile": False,
                  "text_prompt": "insect", "checkpoint_path": None},
            python="/py", script="/s/sam3_masks.py")

    # Only GPU-0's bouts merged; GPU-1 (no manifest) was treated as failed.
    merged = _json.loads((out / "manifest.json").read_text())
    assert [b["bout_idx"] for b in merged["bouts"]] == [0, 1]


def test_enable_sam3_lowmem_on_module():
    import torch.nn as nn
    from jarvis_jax.predict.sam3_driver import _enable_sam3_lowmem

    class Leaf(nn.Module):
        def __init__(self):
            super().__init__()
            self.offload_output_to_cpu_for_eval = False
            self.trim_past_non_cond_mem_for_eval = False

    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Leaf()

    p = Parent()
    n = _enable_sam3_lowmem(p)
    assert n == 1
    assert p.child.offload_output_to_cpu_for_eval is True
    # trim is deliberately NOT enabled (KeyError in the text-prompt multiplex path)
    assert p.child.trim_past_non_cond_mem_for_eval is False


def test_enable_sam3_lowmem_on_wrapper_object():
    import torch.nn as nn
    from jarvis_jax.predict.sam3_driver import _enable_sam3_lowmem

    class Leaf(nn.Module):
        def __init__(self):
            super().__init__()
            self.offload_output_to_cpu_for_eval = False
            self.trim_past_non_cond_mem_for_eval = False

    class Wrapper:                    # not an nn.Module (request-handler style)
        def __init__(self):
            self.model = Leaf()

    w = Wrapper()
    n = _enable_sam3_lowmem(w)
    assert n == 1
    assert w.model.offload_output_to_cpu_for_eval is True
    assert w.model.trim_past_non_cond_mem_for_eval is False  # deliberately not set


def test_enable_sam3_lowmem_none_found_returns_zero():
    import torch.nn as nn
    from jarvis_jax.predict.sam3_driver import _enable_sam3_lowmem

    class Empty(nn.Module):
        pass

    assert _enable_sam3_lowmem(Empty()) == 0
