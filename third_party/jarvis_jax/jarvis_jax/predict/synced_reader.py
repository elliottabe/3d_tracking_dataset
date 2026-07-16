"""Canonical-slot frame reader: the single place jarvis_jax maps a bout's canonical
slots to per-camera mp4 positions using a sync_plan.json (or positional if absent).

plan is None or status 'clean' => positional lockstep, byte-identical to the old reads.
A slot a camera dropped yields present=False (+ a black placeholder for read_window),
which downstream treats as an invalid view.
"""
import os
import numpy as np

from jarvis_jax.predict.frame_sync import SyncPlan


def load_plan(session_dir):
    p = os.path.join(str(session_dir), "sync_plan.json")
    return SyncPlan.load(p) if os.path.exists(p) else None


def slot_positions(plan, cam_name, start_slot, T):
    """(positions, present): positions[i] = mp4 frame for output index i (canonical slot
    start_slot+i), None where the camera dropped it. plan None => positional."""
    if plan is None:
        return [int(start_slot + i) for i in range(T)], [True] * T
    if cam_name not in plan.cams:
        raise ValueError(f"camera {cam_name!r} not in sync plan (have {sorted(plan.cams)})")
    cam = plan.cams[cam_name]
    positions, present = [], []
    for i in range(T):
        s = start_slot + i
        if cam.has(s):
            positions.append(int(cam.pos(s))); present.append(True)
        else:
            positions.append(None); present.append(False)
    return positions, present


def _read_at(cap, cursor, pos):
    """Read the frame at mp4 position `pos` from cap whose next read yields `cursor`.
    Returns (frame_bgr, new_cursor). Uses cap.set only on a discontinuity."""
    import cv2
    if pos != cursor:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        cursor = pos
    ret, frame = cap.read()
    if not ret:
        return None, cursor + 1
    return frame, cursor + 1


def read_one_cam(session_dir, cam_name, plan, start_slot, T):
    """Yield (frame_rgb (H,W,3)|None, present) for each canonical slot."""
    import cv2
    path = os.path.join(str(session_dir), f"{cam_name}.mp4")
    positions, present = slot_positions(plan, cam_name, start_slot, T)
    cap = cv2.VideoCapture(path)
    cursor = None
    try:
        for i in range(T):
            if not present[i]:
                yield None, False
                continue
            pos = positions[i]
            if cursor is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos); cursor = pos
            frame, cursor = _read_at(cap, cursor, pos)
            yield (None if frame is None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (frame is not None)
    finally:
        cap.release()


def read_window(session_dir, cameras, plan, start_slot, T):
    """Yield (frames (C,H,W,3) uint8 RGB, present (C,) bool) per canonical slot. A camera
    absent at a slot gets a zero frame + present=False."""
    import cv2
    caps, poss, prss, cursors = [], [], [], []
    for c in cameras:
        caps.append(cv2.VideoCapture(os.path.join(str(session_dir), f"{c}.mp4")))
        p, pr = slot_positions(plan, c, start_slot, T)
        poss.append(p); prss.append(pr); cursors.append(None)
    # frame shape from the first capture that actually opened -- determined up front so
    # every yielded `out` is always a real (C,H,W,3) array, never None (see module docstring).
    H = W = None
    for cap in caps:
        if cap.isOpened():
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if h > 0 and w > 0:
                H, W = h, w
                break
    if H is None or W is None:
        for cap in caps:
            cap.release()
        raise FileNotFoundError(
            f"no readable camera videos in {session_dir} for cameras {list(cameras)}")
    try:
        for i in range(T):
            frames, present = [], []
            for ci in range(len(cameras)):
                if not prss[ci][i]:
                    frames.append(None); present.append(False); continue
                pos = poss[ci][i]
                if cursors[ci] is None:
                    caps[ci].set(cv2.CAP_PROP_POS_FRAMES, pos); cursors[ci] = pos
                fr, cursors[ci] = _read_at(caps[ci], cursors[ci], pos)
                if fr is None:
                    frames.append(None); present.append(False)
                else:
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    frames.append(fr); present.append(True)
            # materialize; H,W are fixed up front, so this is always a real array
            out = np.zeros((len(cameras), H, W, 3), np.uint8)
            for ci, fr in enumerate(frames):
                if fr is not None:
                    out[ci] = fr
            yield out, np.asarray(present, bool)
    finally:
        for cap in caps:
            cap.release()
