"""D2: SAM3 mask+identity front-end driver (PyTorch SAM3 reused; no JAX).

Pure helpers (torch-free) + run_sam3_masks (lazily imports the heavy JARVIS
PyTorch bits). Produces per-bout sam3_masks.npz + a session manifest for the
JAX D3 pipeline to consume.
"""
import csv
import math
import os
import sys

# Pre-load huggingface_hub.file_download before the lazy torch/sam3 imports in
# run_sam3_masks corrupt `tqdm` (they leave it without `set_lock`, which breaks
# huggingface_hub's lazy file_download import and hence the hf_hub_download that
# sam3.model_builder needs). Loading it at driver-import time (tqdm still intact)
# caches it. See scripts/sam3_masks.py for the full rationale. (Verified fix.)
import huggingface_hub.file_download  # noqa: F401


def session_tag_for(session_dir: str) -> str:
    """'.../Session0/2025_10_20_13_20_04' -> 'Session0/2025_10_20_13_20_04'."""
    return "/".join(session_dir.rstrip("/").split("/")[-2:])


def parse_bouts(csv_path, session_tag, *, limit: int = 0, bout_ids=None):
    """Rows of the bouts CSV as {bout_idx, start, end, n}. Filtered to
    fly_id == session_tag ONLY when the CSV has a fly_id column; a fly_id-less
    CSV (e.g. Session1 good_bouts.csv) is already per-recording so all rows are
    kept. limit>0 keeps the first N; bout_ids (iterable) keeps only those bout_idx."""
    ids = set(int(b) for b in bout_ids) if bout_ids else None
    out = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        has_fly_id = reader.fieldnames is not None and "fly_id" in reader.fieldnames
        for r in reader:
            # Only filter by fly_id when the column exists; a fly_id-less CSV
            # (e.g. Session1 good_bouts.csv) is already per-recording -> keep all.
            if has_fly_id and session_tag and r.get("fly_id") != session_tag:
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


def append_cameras_to_npz(npz_path, cameras):
    """Append a `cameras` (C,) name-string array to an already-written
    sam3_masks.npz, WITHOUT reopening/recompressing the (often 50-500 MB)
    packed mask arrays -- this is what makes every future mask file
    self-identifying (see jarvis_jax.tracking.bout_masks.load_bout_masks
    / detect_camera_order, written to guard against the camera-axis-scramble
    bug: the packed masks' C axis previously carried NO identifying metadata).

    `np.savez`/`np.savez_compressed` just write a ZIP archive with one member
    per array (`<name>.npy`), so a new small member can be appended directly
    via `zipfile` in append mode -- far cheaper than reloading + re-saving
    the whole npz just to attach camera identity.

    `cameras` must be in the SAME order as the npz's C axis (i.e. the same
    ordered camera-name list already used to read the bout's videos, e.g.
    `list(repro_tool.cameras)`) -- this call does not reorder anything, it
    only records the order that's already there. A no-op if the npz already
    has a `cameras` member (e.g. re-invoked on an already-patched file).
    """
    import io
    import zipfile

    import numpy as np

    with zipfile.ZipFile(npz_path, mode="r") as zf:
        if "cameras.npy" in zf.namelist():
            return

    buf = io.BytesIO()
    np.save(buf, np.asarray(list(cameras)))
    with zipfile.ZipFile(npz_path, mode="a", allowZip64=True,
                         compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("cameras.npy", buf.getvalue())


def append_sync_stamp_to_npz(npz_path, plan):
    """Append a `sync` member recording which sync plan produced these masks, so a later
    run can detect masks made on a desynced (or no-plan) timeline. Zip-append (cheap),
    mirroring append_cameras_to_npz. No-op when plan is None."""
    if plan is None:
        return
    import io
    import zipfile

    import numpy as np

    stamp = np.array([str(plan.status), str(int(plan.delta_ns)), "1"])  # status, delta_ns, applied
    buf = io.BytesIO()
    np.save(buf, stamp, allow_pickle=False)
    with zipfile.ZipFile(npz_path, "a", zipfile.ZIP_STORED) as z:
        if "sync.npy" not in z.namelist():
            z.writestr("sync.npy", buf.getvalue())


def _read_sync_stamp(npz_path):
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as z:
        if "sync" not in z.files:
            return None
        s = z["sync"]
        return dict(plan_status=str(s[0]), delta_ns=int(s[1]))


def masks_are_stale(npz_path, plan):
    """True when the recording needs realignment (plan status trim/reindex) but the npz
    lacks a matching sync stamp. Clean/None plans are never stale (byte-identical path)."""
    if plan is None or plan.status == "clean":
        return False
    st = _read_sync_stamp(npz_path)
    if st is None:
        return True
    return not (st["plan_status"] == plan.status and st["delta_ns"] == int(plan.delta_ns))


def ensure_sync_plan(session_dir):
    """Generate <session_dir>/sync_plan.json from Cam*_meta.csv if absent; return the
    loaded SyncPlan (or None when there is no meta.csv). Logs status. When no meta.csv,
    runs a warn-only cross-camera decoded-frame-count check (needs cv2)."""
    import glob
    import json as _json

    from jarvis_jax.predict.frame_sync import analyze_recording
    from jarvis_jax.predict.synced_reader import load_plan

    sp = os.path.join(str(session_dir), "sync_plan.json")
    metas = glob.glob(os.path.join(str(session_dir), "Cam*_meta.csv"))
    if not metas:
        print(f"[sync] {session_dir}: no Cam*_meta.csv -> positional fallback")
        _warn_decoded_count_mismatch(session_dir)
        return None
    if not os.path.exists(sp):
        import tempfile

        plan = analyze_recording(str(session_dir))
        # Atomic write: the SAM3 stage runs as a parallel SLURM array (one task per
        # bout) and run_sam3_masks_multi fans out one worker per GPU, so several
        # processes may hit this concurrently. A unique temp (mkstemp -> distinct per
        # thread AND process) written then os.replace'd into place makes the visible
        # sync_plan.json always complete -- analyze_recording is deterministic, so
        # racing writers produce identical content and os.replace is an atomic
        # last-writer-wins, never a truncated/partial file.
        fd, tmp = tempfile.mkstemp(dir=str(session_dir), prefix=".sync_plan.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                _json.dump({k: v for k, v in plan.items() if not k.startswith("_")}, f,
                           separators=(",", ":"))
            os.replace(tmp, sp)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        print(f"[sync] {session_dir}: wrote plan status={plan['status']} "
              f"predict_len={plan['predict_len']} first_drop_slot={plan['first_drop_slot']}")
    return load_plan(session_dir)


def _warn_decoded_count_mismatch(session_dir):
    """Best-effort: warn if mp4 decoded frame counts differ across cameras (a drop we
    cannot correct without meta.csv). Needs cv2; silently skips if unavailable."""
    try:
        import glob

        import cv2

        counts = {}
        for mp4 in sorted(glob.glob(os.path.join(str(session_dir), "Cam*.mp4"))):
            cap = cv2.VideoCapture(mp4)
            counts[os.path.basename(mp4)] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        if counts and len(set(counts.values())) > 1:
            print(f"[sync] WARNING {session_dir}: mp4 frame counts differ across cameras "
                  f"{counts} and there is no meta.csv to realign -- 3D may be desynced.")
    except Exception:
        pass


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


def resolve_gpus(spec, *, env=None, device_count=None):
    """Resolve the sam3.gpus config value to a list of GPU ids.

    Order: explicit non-empty list `spec` -> CUDA_VISIBLE_DEVICES in `env`
    -> range(device_count). `env`/`device_count` are injected for testability
    (production passes os.environ and torch.cuda.device_count()).
    """
    if spec is not None:
        try:
            ids = [int(x) for x in list(spec)]
        except TypeError:
            ids = []
        if ids:
            return ids
    if env is None:
        env = os.environ
    cvd = env.get("CUDA_VISIBLE_DEVICES", "") or ""
    if cvd.strip():
        return [int(x) for x in cvd.split(",") if x.strip()]
    if device_count is None:
        import torch
        device_count = torch.cuda.device_count()
    return list(range(int(device_count)))


def split_bouts_contiguous(bout_ids, n_gpus):
    """Split bout_ids into n_gpus contiguous blocks (ceil per block). The result
    always has exactly n_gpus entries; trailing blocks are [] when there are
    fewer bouts than GPUs."""
    n = len(bout_ids)
    per = math.ceil(n / n_gpus) if n_gpus > 0 else n
    groups = []
    for gi in range(n_gpus):
        s = gi * per
        e = min(s + per, n)
        groups.append(list(bout_ids[s:e]) if s < n else [])
    return groups


def _ov(v):
    """Format a value as a Hydra override RHS (None -> 'null')."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def build_worker_cmd(*, python, script, gpu, bout_ids, session_dir, out, project,
                     bouts_csv, num_animals, reuse_masks, jarvis_root,
                     sam3_version, sam3_compile, sam3_text, sam3_checkpoint,
                     manifest_name, inductor_cache_dir, base_env=None):
    """Build the (env, argv) for one single-GPU SAM3 worker subprocess.

    Overrides every sam3.* field explicitly so the worker does not depend on the
    parent's Hydra groups. `sam3.gpus=[0]` is the recursion guard: with
    CUDA_VISIBLE_DEVICES=<gpu> the worker sees one device as cuda:0 and runs the
    single-GPU path on sam3.sam3_gpu=0.
    """
    bout_csv = ",".join(str(b) for b in bout_ids)
    argv = [
        python, script,
        "sam3=default",
        "sam3.gpus=[0]",
        "sam3.sam3_gpu=0",
        # Quote the comma value so Hydra parses it as a STRING (the entrypoint
        # does str(bout_ids).split(",")). Unquoted "1,2,3" -> Hydra "Ambiguous
        # value"; bracketed [1,2,3] -> a list that str().split(",") mangles.
        f"sam3.bout_ids='{bout_csv}'",
        "sam3.limit=0",
        f"sam3.session_dir={session_dir}",
        f"sam3.out={out}",
        f"sam3.bouts_csv={bouts_csv}",
        f"sam3.project={project}",
        f"sam3.num_animals={num_animals}",
        f"sam3.reuse_masks={_ov(bool(reuse_masks))}",
        f"sam3.jarvis_root={_ov(jarvis_root)}",
        f"sam3.sam3_version={sam3_version}",
        f"sam3.sam3_compile={_ov(bool(sam3_compile))}",
        f"sam3.sam3_text={sam3_text}",
        f"sam3.sam3_checkpoint={_ov(sam3_checkpoint)}",
        f"sam3.manifest_name={manifest_name}",
    ]
    env = dict(base_env if base_env is not None else os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env, argv


def merge_manifests(partial_paths, *, base):
    """Merge per-worker partial manifests into one. Concatenates each partial's
    `bouts`, dedupes by bout_idx, sorts by bout_idx; copies `base` for the
    session-level fields and sets n_bouts/bouts."""
    import json
    bouts = []
    seen = set()
    for p in partial_paths:
        with open(p) as f:
            m = json.load(f)
        for b in m.get("bouts", []):
            bi = b.get("bout_idx")
            if bi in seen:
                continue
            seen.add(bi)
            bouts.append(b)
    bouts.sort(key=lambda b: (b.get("bout_idx") is None, b.get("bout_idx")))
    merged = dict(base)
    merged["n_bouts"] = len(bouts)
    merged["bouts"] = bouts
    return merged


def _enable_sam3_lowmem(predictor):
    """Enable SAM3's long-video memory bounding on the multiplex tracker.

    Sets offload_output_to_cpu_for_eval=True (move per-frame outputs off-GPU) on
    every module that exposes it. This is the flag that bounds GPU memory on long
    bouts; it's read at propagation time, so setting it on the constructed tracker
    takes effect. Returns the number of modules updated (0 => warn upstream).

    NOTE: we deliberately do NOT set trim_past_non_cond_mem_for_eval — its trim
    path (_trim_past_out) assumes point-prompt inputs and raises
    KeyError('multistep_point_inputs') under our text-prompted multiplex tracking.
    offload_output_to_cpu_for_eval alone bounds the memory we need."""
    import torch.nn as nn
    n = 0
    seen = set()

    def _apply(obj):
        nonlocal n
        if hasattr(obj, "offload_output_to_cpu_for_eval"):
            obj.offload_output_to_cpu_for_eval = True
            n += 1

    def _walk(obj, depth=0):
        if id(obj) in seen or depth > 4:
            return
        seen.add(id(obj))
        if isinstance(obj, nn.Module):
            for m in obj.modules():
                _apply(m)
            return
        _apply(obj)
        d = getattr(obj, "__dict__", None)
        if d:
            for v in list(d.values()):
                if isinstance(v, nn.Module) or hasattr(v, "__dict__"):
                    _walk(v, depth + 1)

    _walk(predictor)
    return n


def _write_mask_overlay(out, bout_idx, *, session_dir, start_frame,
                        n_cams=3, n_frames=300):
    """Standard SAM-mask QC: stacked per-camera overlay video (fly0/fly1 colored)
    for one bout, written next to sam3_masks.npz. Runs `python -m viz maskvid` as
    an ISOLATED subprocess (repo root as cwd so `viz` resolves; JAX off since the
    2-D mask overlay needs no jax/mujoco) so a viz failure never fails the mask
    job. No-op-safe: logs and returns on any error.

    `session_dir` + `start_frame` are passed explicitly (this recording's video
    dir + the bout's absolute first frame) -- otherwise maskvid resolves the
    DEFAULT recording (Session0) and overlays masks on the wrong video."""
    import subprocess
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))
    bout_dir = os.path.join(out, f"bout_{bout_idx:05d}")
    cmd = [sys.executable, "-m", "viz", "maskvid", "--run", out,
           "--bout", str(bout_idx), "--n-cams", str(n_cams), "--n", str(n_frames),
           "--session-dir", str(session_dir), "--start-frame", str(int(start_frame)),
           "--out", os.path.join(bout_dir, f"maskvid_bout{bout_idx}.mp4")]
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    try:
        r = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[sam3] bout {bout_idx}: mask overlay FAILED (non-fatal):\n{r.stderr[-1200:]}")
        else:
            print(f"[sam3] bout {bout_idx}: mask overlay -> {bout_dir}/maskvid_bout{bout_idx}.mp4")
    except Exception as e:  # never let QC viz kill the mask job
        print(f"[sam3] bout {bout_idx}: mask overlay error (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Sex canonicalization by wing SONG (courtship, 2 flies).
#
# Total mask area is an unreliable sex cue: female D. melanogaster are ~15-20%
# larger AT REST, but a courting male's unilateral wing extension (song) inflates
# his projected area, so a per-bout area ratio flips depending on whether he
# sings -- and SAM3 tracks each bout independently, so which physical fly is
# fly0/fly1 is arbitrary per bout. Instead use the SONG signal: the singing
# male's projected mask area FLUCTUATES UPWARD over the bout (wing-extension
# episodes) while the female's stays steady. This canonicalizes the male to a
# fixed slot per bout (no cross-bout identity linking needed) and flags bouts
# with no clear song as ambiguous.
# ---------------------------------------------------------------------------

def _fly_area_timeseries(bm, num_animals, max_samples=200):
    """(A,T) mean mask-pixel area per fly per frame (over valid cameras); NaN
    where a fly has no mask that frame."""
    import numpy as np
    C, T = bm.num_cameras, bm.num_frames
    step = max(1, T // max_samples)
    asum = np.zeros((num_animals, T)); acnt = np.zeros((num_animals, T))
    for cam in range(C):
        idm = bm.identity_map[cam]
        if not idm:
            continue
        for f in range(0, T, step):
            for oid, data in bm.masks[cam][f].items():
                fi = idm.get(int(oid))
                if fi is None or fi >= num_animals:
                    continue
                asum[fi, f] += float(data["mask"].sum()); acnt[fi, f] += 1
    return np.where(acnt > 0, asum / np.maximum(acnt, 1), np.nan)


def sex_male_by_song(bm, num_animals, *, score_ratio_thr=2.0, min_frames=20,
                     min_cv=0.012):
    """Return (male_idx | None, info). The MALE is the fly whose projected mask
    area FLUCTUATES most over the bout -- coefficient of variation std/mean --
    since unilateral wing-extension song episodes raise his area while the
    female's stays steady. None (ambiguous) when the two flies' CVs are within
    `score_ratio_thr`, or neither exceeds `min_cv` (no clear song this bout)."""
    import numpy as np
    if num_animals != 2:
        return None, {}
    ts = _fly_area_timeseries(bm, num_animals)
    cv = []
    for a in range(num_animals):
        x = ts[a][np.isfinite(ts[a])]
        cv.append(float(np.std(x) / max(np.mean(x), 1.0)) if x.size >= min_frames else np.nan)
    cv = np.array(cv)
    info = {"song_cv": [None if not np.isfinite(s) else round(s, 4) for s in cv]}
    if not np.isfinite(cv).all() or min(cv) <= 0:
        return None, info
    ratio = float(max(cv) / min(cv))
    info["ratio"] = round(ratio, 2)
    if max(cv) < min_cv or ratio < score_ratio_thr:
        return None, info                                # no clear song -> ambiguous
    return int(np.argmax(cv)), info


def canonicalize_male_fly(bm, num_animals, *, male_slot=1, score_ratio_thr=1.8):
    """Swap identity_map so the song-identified MALE is fly `male_slot`
    (default 1). Returns (status, info): 'swapped' / 'kept' / 'ambiguous'."""
    male, info = sex_male_by_song(bm, num_animals, score_ratio_thr=score_ratio_thr)
    if male is None:
        return "ambiguous", info
    if male != male_slot:
        for cam in range(bm.num_cameras):
            if bm.identity_map[cam]:
                bm.identity_map[cam] = {o: (num_animals - 1 - fi)
                                        for o, fi in bm.identity_map[cam].items()}
        return "swapped", {**info, "male_detected_slot": male}
    return "kept", {**info, "male_detected_slot": male}


def _append_suspect_cameras_to_npz(npz_path, suspect_cameras):
    """Append a `suspect_cameras` (name-string) array to a sam3_masks.npz --
    cameras whose masks are geometric outliers the self-repair could NOT
    improve (see repair_outlier_cameras). Downstream (run_bout /
    check_bout_camera_order) can drop these views. Same cheap zip-append trick
    as append_cameras_to_npz; no-op if empty."""
    import io
    import zipfile

    import numpy as np

    if not suspect_cameras:
        return
    with zipfile.ZipFile(npz_path, mode="r") as zf:
        if "suspect_cameras.npy" in zf.namelist():
            return
    buf = io.BytesIO()
    np.save(buf, np.asarray(list(suspect_cameras)))
    with zipfile.ZipFile(npz_path, mode="a", allowZip64=True,
                         compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("suspect_cameras.npy", buf.getvalue())


def _append_sex_meta_to_npz(npz_path, sex_meta):
    """Append a `sex_meta` (0-d JSON-string array) to a sam3_masks.npz recording
    how the male/female slots were assigned: {male_slot, status ('kept'/'swapped'
    /'ambiguous'), song_cv, ratio}. Lets downstream QC surface ambiguous bouts.
    Overwrites any existing entry (re-run friendly)."""
    import io
    import json
    import zipfile

    import numpy as np

    payload = np.array(json.dumps(sex_meta))
    with zipfile.ZipFile(npz_path, mode="r") as zf:
        keep = [n for n in zf.namelist() if n != "sex_meta.npy"]
        if "sex_meta.npy" in zf.namelist():          # rewrite archive without it
            import shutil
            import tempfile
            tmpf = npz_path + ".sexmeta.tmp"
            with zipfile.ZipFile(npz_path) as zin, zipfile.ZipFile(
                    tmpf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zout:
                for n in keep:
                    zout.writestr(n, zin.read(n))
            shutil.move(tmpf, npz_path)
    buf = io.BytesIO(); np.save(buf, payload)
    with zipfile.ZipFile(npz_path, mode="a", allowZip64=True,
                         compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sex_meta.npy", buf.getvalue())


# ---------------------------------------------------------------------------
# Self-repair: fix a single camera SAM3 mis-tracked (mask on a reflection /
# shadow / wrong blob) by re-segmenting it with a BOX prompt at the fly
# location reprojected from the OTHER cameras. Runs at the end of a bout, only
# when a camera is a geometric outlier -- see the memory
# `mask-camera-order-vs-bad-centroids`. NEVER remaps the camera axis (that
# would corrupt correctly-ordered masks); it re-runs SAM3 on the flagged
# camera and keeps the result only if the reprojection residual clearly
# improves, else keeps the original and flags the camera `suspect`.
# ---------------------------------------------------------------------------

def _bout_cent_val(bm, num_animals):
    """(A,C,T,2) centroids + (A,C,T) valid from an assigned BoutMasks."""
    import numpy as np
    C, T = bm.num_cameras, bm.num_frames
    cent = np.zeros((num_animals, C, T, 2), np.float32)
    val = np.zeros((num_animals, C, T), bool)
    for fi in range(T):
        slots = bm.get_frame(fi, num_animals=num_animals)
        if slots is None:
            continue
        for f, slot in enumerate(slots):
            if slot is None:
                continue
            v = slot["valid"].numpy()
            c = slot["centroids"].numpy()
            for cam in range(C):
                if v[cam]:
                    val[f, cam, fi] = True
                    cent[f, cam, fi] = c[cam]
    return cent, val


def _loo_residual(cent, val, rt, fly, cam, frames):
    """Median leave-one-out reprojection residual (px) for `cam`: triangulate
    `fly` from the OTHER valid cameras and reproject to `cam`, vs cam's stored
    centroid. Isolates a single bad camera (unlike all-camera triangulation,
    which spreads the error). NaN if never triangulable."""
    import numpy as np
    C = cent.shape[1]
    errs = []
    for t in frames:
        if not val[fly, cam, t]:
            continue
        others = [i for i in range(C) if i != cam and val[fly, i, t]]
        if len(others) < 2:
            continue
        pts = np.zeros((C, 2))
        for i in others:
            pts[i] = cent[fly, i, t]
        rp = rt.reproject_point(rt.reconstruct_point(pts, cams_to_use=others))[cam]
        errs.append(float(np.linalg.norm(rp - cent[fly, cam, t])))
    return float(np.median(errs)) if errs else float("nan")


def _reproj_target(cent, val, rt, fly, cam):
    """(T,2) reprojected `fly` location in `cam` from the OTHER valid cameras;
    NaN where < 2 other cameras are valid."""
    import numpy as np
    C, T = cent.shape[1], cent.shape[2]
    tgt = np.full((T, 2), np.nan)
    for t in range(T):
        others = [i for i in range(C) if i != cam and val[fly, i, t]]
        if len(others) < 2:
            continue
        pts = np.zeros((C, 2))
        for i in others:
            pts[i] = cent[fly, i, t]
        tgt[t] = rt.reproject_point(rt.reconstruct_point(pts, cams_to_use=others))[cam]
    return tgt


def _pick_anchor(tgt, W, H, margin=0.1):
    """Frame whose reprojected target is in-frame and most central (best SAM3
    box-prompt anchor). None if the target is never in-frame."""
    import numpy as np
    fin = np.isfinite(tgt).all(1)
    inb = (fin & (tgt[:, 0] > W * margin) & (tgt[:, 0] < W * (1 - margin))
           & (tgt[:, 1] > H * margin) & (tgt[:, 1] < H * (1 - margin)))
    if not inb.any():
        inb = fin & (tgt[:, 0] >= 0) & (tgt[:, 0] < W) & (tgt[:, 1] >= 0) & (tgt[:, 1] < H)
        if not inb.any():
            return None
    d = np.where(inb, np.hypot(tgt[:, 0] - W / 2, tgt[:, 1] - H / 2), 1e18)
    return int(np.argmin(d))


def _resegment_camera_box(tracker, video_path, frame_start, num_frames, tgt, W, H,
                          *, box_px=(200, 160), text="insect"):
    """Re-run SAM3 on ONE camera with a BOX prompt at the reprojected target on
    the best in-frame anchor frame, propagating BOTH directions (so a fly that
    only enters this camera mid-bout is still covered). Returns (masks list of
    (H,W) bool|None per frame, centroids (T,2) NaN-filled, anchor|None)."""
    import os
    import shutil
    import tempfile

    import numpy as np

    anchor = _pick_anchor(tgt, W, H)
    if anchor is None:
        return None, None, None
    bx, by = tgt[anchor]
    bw, bh = box_px
    box = [max(0.0, (bx - bw / 2)) / W, max(0.0, (by - bh / 2)) / H,
           min(1.0, bw / W), min(1.0, bh / H)]
    masks = [None] * num_frames
    tmp = tempfile.mkdtemp(prefix="sam3_repair_")
    try:
        cam_dir = os.path.join(tmp, "cam")
        tracker._extract_bout_frames(video_path, frame_start, num_frames, cam_dir)
        P = tracker.predictor
        sid = P.handle_request({"type": "start_session", "resource_path": cam_dir,
                                "offload_video_to_cpu": True})["session_id"]
        P.handle_request({"type": "reset_session", "session_id": sid})
        resp = P.handle_request({
            "type": "add_prompt", "session_id": sid, "frame_index": anchor,
            "text": text, "bounding_boxes": [box], "bounding_box_labels": [1]})

        def store(fi, out):
            if out and out.get("out_obj_ids") is not None and len(out["out_obj_ids"]):
                masks[fi] = np.asarray(out["out_binary_masks"][0]).astype(bool)

        if isinstance(resp, dict) and resp.get("outputs"):
            store(anchor, resp["outputs"])
        for r in P.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": "both"}):
            store(r["frame_index"], r["outputs"])
        P.handle_request({"type": "close_session", "session_id": sid})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    cent = np.full((num_frames, 2), np.nan)
    for fi, m in enumerate(masks):
        if m is not None and m.any():
            ys, xs = np.where(m)
            cent[fi] = [xs.mean(), ys.mean()]
    return masks, cent, anchor


def repair_outlier_cameras(tracker, bm, repro_tool, video_paths, frame_start,
                           num_frames, num_animals, *, resid_thresh=40.0,
                           ratio=3.0, improve_frac=0.6, box_px=(200, 160),
                           text="insect", n_detect_frames=80):
    """End-of-bout self-repair. Detect cameras whose per-fly mask centroids are
    geometric outliers (a SAM3 mis-track onto a reflection/shadow/wrong blob),
    re-segment them with a BOX prompt at the fly reprojected from the good
    cameras, and KEEP the re-segment only if the residual clearly improves
    (< improve_frac x the original). Mutates bm.masks / bm.identity_map for
    accepted cameras; returns the sorted list of camera indices that remain
    SUSPECT (flag, don't touch). NEVER raises out (a repair failure logs +
    returns what it has)."""
    import numpy as np

    if bm.identity_map[0] is None:
        return []
    cent, val = _bout_cent_val(bm, num_animals)
    C, T = bm.num_cameras, bm.num_frames
    det_frames = np.arange(0, T, max(1, T // n_detect_frames))

    outliers, resid0 = set(), {}
    for f in range(num_animals):
        res = np.array([_loo_residual(cent, val, repro_tool, f, k, det_frames)
                        for k in range(C)])
        med = np.nanmedian(res)
        for k in range(C):
            if (np.isfinite(res[k]) and res[k] > resid_thresh
                    and (not np.isfinite(med) or res[k] > ratio * med)):
                outliers.add(k)
                resid0[(k, f)] = float(res[k])
    if not outliers:
        return []

    H = W = 0
    for k in range(C):
        for fi in range(T):
            if bm.masks[k][fi]:
                H, W = next(iter(bm.masks[k][fi].values()))["mask"].shape
                break
        if H:
            break
    detail = ", ".join(f"cam{k}/fly{f}={v:.0f}px" for (k, f), v in sorted(resid0.items()))
    print(f"[repair] outlier camera(s) {sorted(outliers)} (LOO residual: {detail})")

    suspect = []
    for cam in sorted(outliers):
        cam_name = os.path.splitext(os.path.basename(video_paths[cam]))[0]
        segs = {}
        for f in range(num_animals):
            tgt = _reproj_target(cent, val, repro_tool, f, cam)
            masks, ncent, anchor = _resegment_camera_box(
                tracker, video_paths[cam], frame_start, num_frames, tgt, W, H,
                box_px=box_px, text=text)
            if masks is None:
                print(f"[repair] cam{cam}({cam_name}) fly{f}: no in-frame anchor -- skip")
                continue
            segs[f] = (masks, ncent)
        if not segs:
            suspect.append(cam)
            continue
        cand_cent, cand_val = cent.copy(), val.copy()
        for f, (masks, ncent) in segs.items():
            for t in range(T):
                if np.isfinite(ncent[t]).all():
                    cand_cent[f, cam, t] = ncent[t]
                    cand_val[f, cam, t] = True
                else:
                    cand_val[f, cam, t] = False
        keep = True
        for f in segs:
            old_r = _loo_residual(cent, val, repro_tool, f, cam, det_frames)
            new_r = _loo_residual(cand_cent, cand_val, repro_tool, f, cam, det_frames)
            print(f"[repair] cam{cam}({cam_name}) fly{f}: residual {old_r:.0f}px -> {new_r:.0f}px")
            if not (np.isfinite(new_r) and (not np.isfinite(old_r) or new_r < improve_frac * old_r)):
                keep = False
        if keep:
            newframe = [{} for _ in range(T)]
            for f, (masks, ncent) in segs.items():
                for t in range(T):
                    if masks[t] is not None and masks[t].any():
                        ys, xs = np.where(masks[t])
                        newframe[t][f] = {"mask": masks[t],
                                          "centroid": np.array([xs.mean(), ys.mean()]),
                                          "score": 1.0}
            bm.masks[cam] = newframe
            bm.identity_map[cam] = {f: f for f in segs}
            print(f"[repair] cam{cam}({cam_name}): re-segment ACCEPTED (replaced masks)")
        else:
            suspect.append(cam)
            print(f"[repair] cam{cam}({cam_name}): re-segment did NOT clearly improve "
                  f"-> keeping original, flagging suspect")
    return suspect


def run_sam3_masks(*, project, session_dir, bouts_csv, out, num_animals=2,
                   limit=0, bout_ids=None, reuse_masks=True, sam3=None,
                   jarvis_root=None, manifest_name="manifest.json", lowmem=True,
                   overlay=True, overlay_cams=3, overlay_frames=300,
                   repair_outliers=True, repair_resid_thresh=40.0):
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
        manifest_name: Filename for the session manifest written under `out`.
        lowmem: If True, after building the tracker set SAM3's
              offload_output_to_cpu_for_eval so GPU memory stays bounded on long
              bouts (per-frame outputs offloaded to CPU).

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
    import torch

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

    sync_plan = ensure_sync_plan(session_dir)

    tracker = None
    tracker_load_time = None
    per_bout = []

    for b in bouts:
        bout_out = os.path.join(out, f"bout_{b['bout_idx']:05d}")
        os.makedirs(bout_out, exist_ok=True)
        npz_path = os.path.join(bout_out, pmod.MASKS_FILENAME)
        t0 = time.time()

        if reuse_masks and os.path.isfile(npz_path) and not masks_are_stale(npz_path, sync_plan):
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
                    checkpoint_path=sam3.get("checkpoint_path", None),
                    chunk_len=sam3.get("chunk_len", 1400),
                    chunk_overlap=sam3.get("chunk_overlap", 120))
                tracker_load_time = round(time.time() - t_load, 1)
                print(f"[sam3] SAM3VideoTracker loaded in {tracker_load_time}s")
                if lowmem:
                    nlm = _enable_sam3_lowmem(tracker.predictor)
                    print(f"[sam3] low-mem eval enabled on {nlm} module(s)" if nlm
                          else "[sam3] WARNING: lowmem requested but no SAM3 "
                               "module exposed offload_output_to_cpu_for_eval")

            positions_per_cam = None
            if sync_plan is not None:
                from jarvis_jax.predict.synced_reader import slot_positions
                positions_per_cam = [slot_positions(sync_plan, nm, b["start"], b["n"])[0]
                                     for nm in list(repro_tool.cameras)]
            bm = tracker.process_bout(
                video_paths, b["start"], b["n"], num_animals=num_animals,
                positions_per_cam=positions_per_cam)
            # SAM3's predictors enter a process-wide bf16 autocast
            # (self.bf16_context.__enter__() with no matching __exit__), so it
            # stays active after process_bout() returns. That leaks into the
            # reprojection geometry in assign_identities: torch.matmul/solve run
            # in bfloat16, which (a) drops the sub-pixel triangulation residuals
            # used to disambiguate fly0/fly1 below bf16 precision and (b) makes
            # r[cam].numpy() raise "unsupported ScalarType BFloat16". Force fp32.
            with torch.autocast(device_type="cuda", enabled=False):
                bm.assign_identities(repro_tool, num_animals=num_animals)
            # Sex canonicalization (courtship, 2 flies): assign_identities'
            # area-based sex cue is unreliable (a singing male's wing extension
            # inflates his area, flipping the ratio per bout, and SAM3 tracks
            # each bout independently). Override it with the wing-SONG signal so
            # the male is consistently fly `male_slot` (=1); ambiguous when no
            # clear song. This is the authoritative sex step.
            sex_status, sex_info = "n/a", {}
            if num_animals == 2:
                sex_status, sex_info = canonicalize_male_fly(bm, num_animals, male_slot=1)
                print(f"[sex] bout {b['bout_idx']}: {sex_status} "
                      f"(male=fly1; cv={sex_info.get('song_cv')} ratio={sex_info.get('ratio')})")
            # Self-repair: if a camera's masks are a geometric outlier (SAM3
            # locked onto a reflection/shadow/wrong blob), re-segment it with a
            # box prompt at the fly reprojected from the good cameras; keep only
            # if it clearly improves, else flag it `suspect`. Never fatal.
            suspect_cams = []
            if repair_outliers:
                try:
                    with torch.autocast(device_type="cuda", enabled=False):
                        suspect_idx = repair_outlier_cameras(
                            tracker, bm, repro_tool, video_paths, b["start"], b["n"],
                            num_animals, resid_thresh=repair_resid_thresh)
                    cam_list = list(repro_tool.cameras)
                    suspect_cams = [cam_list[c] for c in suspect_idx]
                except Exception as e:  # noqa: BLE001 -- repair is best-effort
                    print(f"[repair] WARNING: outlier repair failed "
                          f"({type(e).__name__}: {e}) -- saving original masks")
            pmod.save_bout_masks(bm, bout_out, num_animals)
            # Record the C-axis camera identity (see append_cameras_to_npz)
            # so this mask file is self-identifying and a future camera-order
            # scramble can be caught/corrected instead of silently corrupting
            # the pipeline. video_paths (built above from repro_tool.cameras)
            # is the same order the packed masks' C axis is in.
            append_cameras_to_npz(npz_path, list(repro_tool.cameras))
            append_sync_stamp_to_npz(npz_path, sync_plan)
            if suspect_cams:
                _append_suspect_cameras_to_npz(npz_path, suspect_cams)
                print(f"[sam3] bout {b['bout_idx']}: flagged suspect camera(s) "
                      f"{suspect_cams} (repair could not improve them)")
            if num_animals == 2:
                _append_sex_meta_to_npz(npz_path, dict(male_slot=1, status=sex_status, **sex_info))
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
        # Standard per-bout SAM-mask QC overlay (stacked cameras, fly0/fly1
        # colored) written next to sam3_masks.npz. Non-fatal.
        if overlay:
            _write_mask_overlay(out, b["bout_idx"],
                                session_dir=session_dir, start_frame=b["start"],
                                n_cams=overlay_cams, n_frames=overlay_frames)

    manifest = build_manifest(session_dir, tag, sam3, per_bout)
    manifest_path = os.path.join(out, manifest_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[sam3] wrote {len(per_bout)} bouts + {manifest_name} to {out}")
    return manifest


def run_sam3_masks_multi(*, gpus, project, session_dir, bouts_csv, out,
                         num_animals=2, limit=0, bout_ids=None,
                         reuse_masks=True, jarvis_root=None, sam3=None,
                         python=None, script=None):
    """Fan a session's bouts across multiple GPUs: one subprocess per GPU, each
    re-invoking scripts/sam3_masks.py (single-GPU) on a contiguous bout subset.
    The parent merges per-worker partial manifests into out/manifest.json.

    Falls back to the single-GPU run_sam3_masks when <=1 GPU or <=1 bout.
    """
    import json
    import subprocess
    import sys
    import tempfile

    sam3 = dict(sam3 or {})
    tag = session_tag_for(session_dir)
    csv_path = (bouts_csv if os.path.isabs(bouts_csv)
                else os.path.join(session_dir, bouts_csv))
    bouts = parse_bouts(csv_path, tag, limit=limit, bout_ids=bout_ids)
    all_ids = [b["bout_idx"] for b in bouts]

    if len(gpus) <= 1 or len(all_ids) <= 1:
        return run_sam3_masks(
            project=project, session_dir=session_dir, bouts_csv=bouts_csv,
            out=out, num_animals=num_animals, limit=limit, bout_ids=bout_ids,
            reuse_masks=reuse_masks, sam3=sam3, jarvis_root=jarvis_root)

    os.makedirs(out, exist_ok=True)
    # Pre-generate the sync plan once here in the parent so the per-GPU workers
    # below just load an already-written sync_plan.json (avoids N workers racing to
    # create it; the atomic write in ensure_sync_plan is the backstop).
    ensure_sync_plan(session_dir)
    groups = split_bouts_contiguous(all_ids, len(gpus))
    python = python or sys.executable
    if script is None:
        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "scripts", "sam3_masks.py")

    procs = []
    for gpu, grp in zip(gpus, groups):
        if not grp:
            continue
        mname = f"manifest.gpu{gpu}.json"
        env, argv = build_worker_cmd(
            python=python, script=script, gpu=gpu, bout_ids=grp,
            session_dir=session_dir, out=out, project=project,
            bouts_csv=csv_path, num_animals=num_animals,
            reuse_masks=reuse_masks, jarvis_root=jarvis_root,
            sam3_version=sam3.get("sam3_version", "sam3.1"),
            sam3_compile=sam3.get("compile", False),
            sam3_text=sam3.get("text_prompt", "insect"),
            sam3_checkpoint=sam3.get("checkpoint_path", None),
            manifest_name=mname,
            inductor_cache_dir=os.path.join(
                tempfile.gettempdir(), f"torchinductor_gpu{gpu}"))
        print(f"[sam3-multi] GPU {gpu}: {len(grp)} bouts {grp}")
        procs.append((gpu, subprocess.Popen(argv, env=env),
                      os.path.join(out, mname)))

    failures = []
    existing = []
    for gpu, p, partial in procs:
        rc = p.wait()
        if rc != 0:
            failures.append(gpu)
            print(f"[sam3-multi] GPU {gpu} worker FAILED (exit {rc})")
        elif not os.path.isfile(partial):
            # Clean exit but no manifest written = bouts silently lost; treat as
            # a failure so the merge never hands D3 a partial session unnoticed.
            failures.append(gpu)
            print(f"[sam3-multi] GPU {gpu} exited 0 but wrote no manifest "
                  f"({partial}) — treating as failed")
        else:
            existing.append(partial)
    base = build_manifest(session_dir, tag, sam3, [])
    merged = merge_manifests(existing, base=base)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(merged, f, indent=2)
    merged["failures"] = failures
    print(f"[sam3-multi] merged {merged['n_bouts']} bouts -> manifest.json "
          f"({len(failures)} GPU failures)")
    # The partial manifest (above) is written first so a re-run with
    # reuse_masks=true resumes the finished bouts. But if any worker failed we
    # raise: the masks are incomplete, and a caller (e.g. the SLURM launcher's
    # sequential job) must NOT proceed to the D3 predict stage on a partial
    # session. Failing loudly lets requeue + reuse_masks finish stage 1 first.
    if failures:
        raise RuntimeError(
            f"[sam3-multi] {len(failures)} GPU worker(s) failed: {failures}. "
            f"Partial manifest written to {out}/manifest.json; re-run "
            f"(reuse_masks=true) to finish the remaining bouts.")
    return merged
