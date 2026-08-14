"""Artifact loaders + run-path resolution + frame reading for viz views."""
import csv
import os
import numpy as np
import cv2
import stac_mjx.io_dict_to_hdf5 as ioh5
from jarvis_jax.tracking.bout_masks import load_bout_masks

def fly_dir(run_root, bout, fly):
    return os.path.join(run_root, "bouts", f"bout_{int(bout):05d}", f"fly{int(fly)}")

def load_outputs(run_root, bout, fly):
    d = ioh5.load(os.path.join(fly_dir(run_root, bout, fly), "outputs.h5"))
    return {"kp3d_mm": np.asarray(d["kp3d_mm"]), "mesh_mm": np.asarray(d["mesh_mm"]),
            "kp_names": [str(n) for n in np.asarray(d["kp_names"]).tolist()]}

def load_kp2d(run_root, bout, fly):
    with np.load(os.path.join(fly_dir(run_root, bout, fly), "kp2d.npz")) as z:
        return np.asarray(z["kp2d"]), np.asarray(z["conf"])

def load_kp3d(run_root, bout, fly):
    with np.load(os.path.join(fly_dir(run_root, bout, fly), "kp3d.npz")) as z:
        return np.asarray(z["kp3d"]), np.asarray(z["conf3d"])

def load_masks(predictions_dir, bout, fly, cameras):
    npz = os.path.join(predictions_dir, f"bout_{int(bout):05d}", "sam3_masks.npz")
    return load_bout_masks(npz, fly, expected_cameras=list(cameras))

def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path); cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, bgr = cap.read(); cap.release()
    if not ok:
        raise IndexError(f"frame {frame_idx} not readable in {video_path}")
    return bgr

def read_frames(session_dir, cameras, start, count):
    caps = [cv2.VideoCapture(os.path.join(session_dir, f"{c}.mp4")) for c in cameras]
    try:
        for cap in caps: cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        for _ in range(int(count)):
            imgs = []
            for cap in caps:
                ok, bgr = cap.read()
                imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None)
            yield imgs
    finally:
        for cap in caps: cap.release()

def sync_positions(session_dir, cameras, start_slot, count):
    """[(positions, present)] per camera for canonical slots [start_slot, +count).

    Cameras drop frames independently, so the Nth decoded frame of one mp4 is
    not necessarily the same instant as the Nth of another. `sync_plan.json`
    records, per camera, which canonical slot each decoded frame belongs to;
    this maps slots -> mp4 frame indices, with None where that camera dropped
    the slot entirely.

    No plan (or a camera absent from it) -> positional identity, i.e. exactly
    the pre-sync behaviour. Session0 has no Cam*_meta.csv at all, so it always
    takes this path; 2 of 10 Session1 recordings genuinely reindex.
    """
    plan = None
    try:
        import sys as _sys
        _pkg = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "third_party", "jarvis_jax")
        if _pkg not in _sys.path:
            _sys.path.insert(0, _pkg)
        from jarvis_jax.predict.synced_reader import load_plan, slot_positions
        plan = load_plan(session_dir)
    except Exception as e:                 # viz must never hard-fail on sync
        # ...but say so. A malformed/unreadable plan silently degrading to
        # positional is indistinguishable from a genuinely clean recording,
        # which is exactly how a wrong-schema plan can pass unnoticed.
        print(f"[viz.sync] WARNING: no usable sync plan for {session_dir} "
              f"({type(e).__name__}: {e}) -- falling back to positional reads")
        plan = None
    out = []
    for c in cameras:
        if plan is None:
            out.append(([int(start_slot + i) for i in range(int(count))],
                        [True] * int(count)))
            continue
        try:
            pos, pres = slot_positions(plan, c, int(start_slot), int(count))
        except ValueError:                 # camera not in the plan
            pos = [int(start_slot + i) for i in range(int(count))]
            pres = [True] * int(count)
        out.append((list(pos), list(pres)))
    return out


def read_frames_synced(session_dir, cameras, start_slot, count):
    """Like `read_frames`, but aligned on canonical slots rather than position.

    Yields one list per slot, entry None where that camera dropped the slot or
    the read failed. Identical output to `read_frames` when no plan exists, so
    it is safe as a drop-in.
    """
    plans = sync_positions(session_dir, cameras, start_slot, count)
    caps = [cv2.VideoCapture(os.path.join(session_dir, f"{c}.mp4")) for c in cameras]
    cur = [None] * len(cameras)
    try:
        for i in range(int(count)):
            imgs = []
            for ci, cap in enumerate(caps):
                pos, pres = plans[ci]
                target = pos[i] if pres[i] else None
                if target is None:
                    imgs.append(None)
                    continue
                if cur[ci] != target:      # seek only when not already there
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
                    cur[ci] = target
                ok, bgr = cap.read()
                cur[ci] = target + 1
                imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None)
            yield imgs
    finally:
        for cap in caps:
            cap.release()


def load_data3d_csv(csv_path):
    """Load a per-bout dense 3D-keypoint CSV (`bout_NNNNN/fly{0,1}.csv`).

    Format: 2 header rows -- row1 (1-indexed, i.e. the first line) is the
    base keypoint name repeated x4 per keypoint (`kpA,kpA,kpA,kpA,kpB,...`);
    row2 is the per-column label (`x,y,z,conf` repeated, though the on-disk
    label is literally "conf" not "confidence" -- we don't rely on row2's
    text at all, only its presence/shape). One data row per frame; col0 is
    the absolute frame index if row1's col0 header is "frame" (dense
    per-bout CSVs always have this), else frames fall back to arange(T).

    Returns (kp3d (T,K,3) float, conf (T,K) float, names (list[str] len K),
    frames (T,) int).

    NOTE: this is a deliberate extension beyond the original 3-tuple
    `(kp3d, conf, names)` sketched in the viz design docs -- the reproj-video
    view needs the absolute frame indices to align the 3D onto the raw
    session video, which is exactly what
    scripts/viz_predictions_reproject.py's `_bout_frame_range` reads col0
    for. Returning `frames` here avoids re-parsing the CSV a second time.
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header1 = next(reader)
        next(reader)  # header2 (x,y,z,conf labels) -- shape/presence only, not read
        data_rows = [row for row in reader if row and any(c.strip() for c in row)]

    has_frame_col = bool(header1) and header1[0].strip().lower() == "frame"
    name_cols = header1[1:] if has_frame_col else header1
    val_start = 1 if has_frame_col else 0

    if len(name_cols) % 4 != 0:
        raise ValueError(
            f"{csv_path}: {len(name_cols)} keypoint columns not divisible by 4 (x,y,z,conf)")
    K = len(name_cols) // 4
    names = list(name_cols[::4])  # every-4th entry = deduped base kp name

    T = len(data_rows)
    kp3d = np.full((T, K, 3), np.nan, dtype=float)
    conf = np.full((T, K), np.nan, dtype=float)
    frames = np.arange(T, dtype=int)

    def _f(x):
        x = x.strip()
        return float(x) if x else float("nan")

    for t, row in enumerate(data_rows):
        if has_frame_col:
            frames[t] = int(float(row[0]))
        vals = np.array([_f(x) for x in row[val_start:val_start + 4 * K]], dtype=float)
        vals = vals.reshape(K, 4)
        kp3d[t] = vals[:, :3]
        conf[t] = vals[:, 3]

    return kp3d, conf, names, frames

def write_video(out_path, frames_iter, fps=30, fourcc="mp4v", macro_block_size=16):
    """Write an iterable/generator of BGR uint8 frames to an H.264 mp4 that plays
    everywhere (browsers, VSCode's preview, Artifacts): libx264 + yuv420p +
    ``+faststart``. Encodes via imageio-ffmpeg's bundled ffmpeg; frames are BGR
    (cv2 convention) and converted to RGB for the encoder. Size is taken from the
    first frame; dimensions not divisible by ``macro_block_size`` are padded up
    to the nearest multiple by imageio (e.g. 1080 -> 1088 at the default 16) --
    pass ``macro_block_size=1`` when the caller needs the output's exact
    pixel dimensions preserved (verified e.g. by a downstream ffprobe check).

    ``fourcc`` is accepted for backward compatibility but ignored: the previous
    cv2 ``mp4v`` path produced MPEG-4 Part 2, which Chromium-based players (VSCode)
    refuse to decode. Falls back to cv2 ``mp4v`` ONLY if ffmpeg is unavailable
    (warns that the result may not play in VSCode/browsers).
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    frames_iter = iter(frames_iter)
    try:
        first = np.ascontiguousarray(np.asarray(next(frames_iter)))
    except StopIteration:
        raise ValueError(f"write_video: frames_iter was empty; nothing written to {out_path}")

    def _to_rgb(f):
        f = np.asarray(f)
        if f.ndim == 3 and f.shape[2] == 3:
            f = f[:, :, ::-1]  # BGR (caller/cv2 convention) -> RGB (imageio)
        return np.ascontiguousarray(f)

    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            out_path, format="FFMPEG", fps=float(fps), codec="libx264",
            pixelformat="yuv420p", macro_block_size=macro_block_size,
            output_params=["-movflags", "+faststart"])
        try:
            writer.append_data(_to_rgb(first))
            for frame in frames_iter:
                writer.append_data(_to_rgb(frame))
        finally:
            writer.close()
        return out_path
    except Exception as e:  # ffmpeg/imageio missing -> legacy cv2 (may not play in browsers)
        print(f"[write_video] warning: H.264 encode via imageio failed ({e}); "
              f"falling back to cv2 mp4v -- may not play in VSCode/browsers.")
        h, w = first.shape[0], first.shape[1]
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if not writer.isOpened():
            raise IOError(f"cv2.VideoWriter failed to open {out_path}")
        writer.write(np.asarray(first))
        for frame in frames_iter:
            writer.write(np.asarray(frame))
        writer.release()
        return out_path
