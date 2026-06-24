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
