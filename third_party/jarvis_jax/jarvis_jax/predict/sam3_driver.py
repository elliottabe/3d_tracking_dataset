"""D2: SAM3 mask+identity front-end driver (PyTorch SAM3 reused; no JAX).

Pure helpers (torch-free) + run_sam3_masks (lazily imports the heavy JARVIS
PyTorch bits). Produces per-bout sam3_masks.npz + a session manifest for the
JAX D3 pipeline to consume.
"""
import csv
import os


def session_tag_for(session_dir: str) -> str:
    """'.../Session0/2025_10_20_13_20_04' -> 'Session0/2025_10_20_13_20_04'."""
    return "/".join(session_dir.rstrip("/").split("/")[-2:])


def parse_bouts(csv_path, session_tag, *, limit: int = 0, bout_ids=None):
    """Rows of the bouts CSV whose fly_id == session_tag, as
    {bout_idx, start, end, n}. limit>0 keeps the first N; bout_ids (iterable)
    keeps only those bout_idx."""
    ids = set(int(b) for b in bout_ids) if bout_ids else None
    out = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if session_tag and r.get("fly_id") != session_tag:
                continue
            bi = int(r["bout_idx"])
            if ids is not None and bi not in ids:
                continue
            s, e = int(r["start_frame"]), int(r["end_frame"])
            out.append({"bout_idx": bi, "start": s, "end": e, "n": e - s + 1})
    out.sort(key=lambda b: b["bout_idx"])
    if limit and limit > 0:
        out = out[:limit]
    return out


def video_paths_for(session_dir, camera_names):
    """Ordered <session_dir>/<cam>.mp4 for each camera name (raises if missing)."""
    paths = []
    for cam in camera_names:
        p = os.path.join(session_dir, f"{cam}.mp4")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing video for camera {cam}: {p}")
        paths.append(p)
    return paths


def bout_stats(loaded_masks, num_animals: int) -> dict:
    """Summary stats from a LoadedBoutMasks-like object (.valid (A,C,F))."""
    import numpy as np
    valid = np.asarray(loaded_masks.valid)            # (A,C,F)
    A, C, F = valid.shape
    per_fly = [float(valid[a].mean()) for a in range(A)]
    cams_valid_per_frame = valid.any(axis=0).sum(axis=0) if A else np.zeros(F)
    # mean over frames of (#cams with >=1 valid fly)
    mean_cams = float(valid.any(axis=0).sum(axis=0).mean()) if F else 0.0
    return {
        "num_animals_saved": int(loaded_masks.num_animals_saved),
        "num_cameras": int(C),
        "num_frames": int(F),
        "per_fly_valid_frac": per_fly,
        "mean_cams_valid_per_frame": mean_cams,
    }


def build_manifest(session_dir, session_tag, sam3_settings, per_bout) -> dict:
    return {
        "session_dir": session_dir,
        "session_tag": session_tag,
        "sam3_settings": dict(sam3_settings),
        "n_bouts": len(per_bout),
        "bouts": list(per_bout),
    }


def run_sam3_masks(*, project, session_dir, bouts_csv, out, num_animals=2,
                   limit=0, bout_ids=None, reuse_masks=True, sam3=None,
                   jarvis_root=None):
    """Run SAM3 video tracking + identity over a session's bouts, writing a
    per-bout sam3_masks.npz + a session manifest.

    Reuses the JARVIS PyTorch SAM3 stack (SAM3VideoTracker, save_bout_masks,
    LoadedBoutMasks from JARVIS-HybridNet). All heavy imports are lazy so the
    module stays torch-free at import time.

    Args:
        project: JARVIS project name (passed to ProjectManager.load).
        session_dir: Absolute path to the recording session directory.
        bouts_csv: Path (absolute or relative to session_dir) of the bouts CSV.
        out: Output root directory; per-bout subdirs are created here.
        num_animals: Number of animals to track (default 2).
        limit: If >0, process only the first N bouts (by bout_idx order).
        bout_ids: If given, only process bouts with these bout_idx values.
        reuse_masks: If True and the npz already exists, skip re-running SAM3.
        sam3: Dict of SAM3VideoTracker kwargs (sam3_version, gpu_id, compile,
              text_prompt, checkpoint_path). Defaults filled in if missing.
        jarvis_root: Absolute path to the JARVIS-HybridNet root that contains
              the `projects/` directory. If None, defaults to the
              JARVIS_ROOT env var, then falls back to the third_party copy
              (which has no projects — pass an explicit path or set the env
              var when the third_party copy is used).

    Returns:
        The manifest dict (also written to <out>/manifest.json).
    """
    import json
    import sys
    import time

    sam3 = dict(sam3 or {})

    # --- Lazy heavy imports: PyTorch / JARVIS ---
    # The code root (third_party/JARVIS-HybridNet) provides importable JARVIS
    # modules.  The *projects* root may differ — some installs keep projects in
    # a separate checkout.  We split these two concerns:
    #
    # code_root: used only for sys.path so that `import jarvis.*` resolves.
    #   Derived from this file's location (4 dirs up = third_party/,
    #   then sibling JARVIS-HybridNet/).
    # projects_root: the JARVIS-HybridNet directory that owns `projects/`.
    #   Resolved from the `jarvis_root` kwarg, then the JARVIS_ROOT env var,
    #   then falls back to code_root (works when both coincide).
    code_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "JARVIS-HybridNet")
    print(f"[sam3] code_root (imports): {code_root}")
    assert os.path.isdir(code_root), f"JARVIS-HybridNet not found at {code_root}"

    if jarvis_root is None:
        jarvis_root = os.environ.get("JARVIS_ROOT", code_root)
    print(f"[sam3] jarvis_root (projects): {jarvis_root}")
    assert os.path.isdir(jarvis_root), (
        f"jarvis_root not found: {jarvis_root} — set JARVIS_ROOT or pass jarvis_root=")

    for p in (code_root, os.path.join(code_root, "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from jarvis.config.project_manager import ProjectManager
    from jarvis.utils.reprojection import get_repro_tool
    from jarvis.prediction.sam3_video_tracker import SAM3VideoTracker
    import predict3D_multianimal as pmod  # save_bout_masks, LoadedBoutMasks, MASKS_FILENAME

    pm = ProjectManager()
    # Override parent_dir so ProjectManager looks for projects in jarvis_root.
    if os.path.realpath(pm.parent_dir) != os.path.realpath(jarvis_root):
        pm.parent_dir = jarvis_root
        print(f"[sam3] ProjectManager.parent_dir overridden -> {jarvis_root}")
    assert pm.load(project), f"Could not load JARVIS project '{project}'"
    cfg = pm.get_cfg()

    session_calib = os.path.join(session_dir, "calibration")
    repro_tool = get_repro_tool(
        cfg, session_calib if os.path.isdir(session_calib) else None)
    assert repro_tool is not None, "ReprojectionTool not available"

    tag = session_tag_for(session_dir)
    csv_path = (bouts_csv if os.path.isabs(bouts_csv)
                else os.path.join(session_dir, bouts_csv))
    bouts = parse_bouts(csv_path, tag, limit=limit, bout_ids=bout_ids)
    video_paths = video_paths_for(session_dir, list(repro_tool.cameras))
    os.makedirs(out, exist_ok=True)

    tracker = None
    tracker_load_time = None
    per_bout = []

    for b in bouts:
        bout_out = os.path.join(out, f"bout_{b['bout_idx']:05d}")
        os.makedirs(bout_out, exist_ok=True)
        npz_path = os.path.join(bout_out, pmod.MASKS_FILENAME)
        t0 = time.time()

        if reuse_masks and os.path.isfile(npz_path):
            lm = pmod.LoadedBoutMasks(npz_path)
            print(f"[sam3] bout {b['bout_idx']}: reusing existing {npz_path}")
        else:
            if tracker is None:
                t_load = time.time()
                tracker = SAM3VideoTracker(
                    gpu_id=sam3.get("gpu_id", 0),
                    text_prompt=sam3.get("text_prompt", "insect"),
                    sam3_version=sam3.get("sam3_version", "sam3.1"),
                    compile=sam3.get("compile", False),
                    checkpoint_path=sam3.get("checkpoint_path", None))
                tracker_load_time = round(time.time() - t_load, 1)
                print(f"[sam3] SAM3VideoTracker loaded in {tracker_load_time}s")

            bm = tracker.process_bout(
                video_paths, b["start"], b["n"], num_animals=num_animals)
            bm.assign_identities(repro_tool, num_animals=num_animals)
            pmod.save_bout_masks(bm, bout_out, num_animals)
            lm = pmod.LoadedBoutMasks(npz_path)

        st = bout_stats(lm, num_animals)
        st.update(
            bout_idx=b["bout_idx"],
            start=b["start"],
            end=b["end"],
            npz=npz_path,
            seconds=round(time.time() - t0, 1),
        )
        if tracker_load_time is not None and len(per_bout) == 0:
            st["sam3_load_seconds"] = tracker_load_time
        per_bout.append(st)
        print(f"[sam3] bout {b['bout_idx']}: {st['num_frames']} frames, "
              f"mean cams valid {st['mean_cams_valid_per_frame']:.2f}, "
              f"{st['seconds']}s -> {npz_path}")

    manifest = build_manifest(session_dir, tag, sam3, per_bout)
    manifest_path = os.path.join(out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[sam3] wrote {len(per_bout)} bouts + manifest.json to {out}")
    return manifest
