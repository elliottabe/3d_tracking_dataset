"""A miniature `pose_mvq_p3b/bouts/bout_XXXXX/` tree plus its SAM3 mask npz,
small enough for CPU tests: 2 flies, 7 cameras (the real canonical Session0
names, so tests exercise the real >= 5-camera thresholds, not a relaxed toy
count), T frames, 6 keypoints, an 80x120 frame. Geometry is exact: kp2d is
the projection of kp3d through `cam_mats`, so the reprojection gate passes
unless a test perturbs it."""
import json
import os

import numpy as np

CAMS = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
        "Cam2012857", "Cam2012861", "Cam2012862"]
KP_NAMES = ["Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_tip", "WingL_base"]
H, W = 80, 120


def cam_mats(n_cam=len(CAMS)):
    """(C,4,3) `ReprojectionTool.camera_matrices` convention (p_h @ M).

    scale=1.0 (not 2.0): at scale 2 the theta=0 camera's u = 2*x + 60 puts
    fly1 (sep_units=40 default) at u in [126, 163], entirely outside the
    80x120 frame for the WHOLE bout -- not a boundary case, a camera that
    never sees fly1 at all. scale=1.0 keeps both flies, across the full
    default drift, inside every camera with margin (checked numerically).
    """
    out = np.zeros((n_cam, 4, 3), np.float64)
    for c in range(n_cam):
        th = 2 * np.pi * c / n_cam
        out[c, :3, 0] = [1.0 * np.cos(th), -1.0 * np.sin(th), 0.0]
        out[c, :3, 1] = [0.0, 0.0, -1.0]
        out[c, 3] = [60.0, 40.0, 1.0]
    return out


def project(cm, xyz):
    from jarvis_jax.tracking.lift_mvq import project_points
    return project_points(cm, xyz)                       # (C,K,2)


def make_calib(calib_dir, cm=None):
    """Write `Cam<name>.yaml` for every camera in `CAMS` so that
    `ReprojectionTool(calib_dir).camera_matrices` reproduces `cam_mats()`.

    The YAML node is the (3,4) `projectionMatrix` -- i.e. the TRANSPOSE of the
    (4,3) `p_h @ M` matrices this module hands to `project_points` -- which is
    the convention `geometry/reprojection_tool.Camera` reads. Sorted-glob order
    of the file names is `CAMS` order (they are already sorted), so the tool's
    camera axis is the canonical one the rest of the fixture uses BY NAME.
    """
    import cv2
    cm = cam_mats() if cm is None else np.asarray(cm, np.float64)
    d = str(calib_dir)
    os.makedirs(d, exist_ok=True)
    for c, name in enumerate(CAMS):
        fs = cv2.FileStorage(os.path.join(d, f"{name}.yaml"), cv2.FILE_STORAGE_WRITE)
        fs.write("projectionMatrix", np.ascontiguousarray(cm[c].T))
        fs.release()
    return d


def make_bout(tmp_path, *, T=40, sep_units=40.0, name="bout_00004", exist=0.95,
              identity="mask", drop_frames=(), spike_frames=(), frame_start=1000,
              masks_path=None, rec="rec"):
    """Returns (bout_dir, mask_npz_path, cam_mats). fly0 sits at the origin,
    fly1 `sep_units` away in +x; both drift 0.2 units/frame.

    `frame_start`, `masks_path` and `rec` exist so a test can build SEVERAL
    bouts of the same fake run root without them colliding: absolute video
    frames are `frame_start + t` (frameset keys are absolute, so two bouts
    sharing a frame_start would overwrite each other), and the mask npz
    defaults to ONE path per tmp_path.
    """
    cm = cam_mats()
    root = tmp_path / rec / "pose_mvq_p3b" / "bouts" / name
    K, C = len(KP_NAMES), len(CAMS)
    kp3d = np.zeros((2, T, K, 3), np.float32)
    rng = np.random.default_rng(0)
    body = rng.normal(size=(K, 3)) * np.array([3.0, 1.5, 1.0])
    for f in range(2):
        base = np.array([f * sep_units, 0.0, 0.0])
        for t in range(T):
            kp3d[f, t] = body + base + np.array([0.2 * t, 0.0, 0.0])
    for t in spike_frames:                                # 8 units in one frame = 0.8 mm
        kp3d[1, t, 0] += np.array([8.0, 0.0, 0.0])
    kp2d = np.zeros((2, T, C, K, 2), np.float32)
    for f in range(2):
        for t in range(T):
            kp2d[f, t] = project(cm, kp3d[f, t])
    conf = np.full((2, T, C, K), 0.9, np.float32)
    if isinstance(identity, (list, tuple)):                # per-frame mix, e.g. some "sex"-head
        assert len(identity) == T, f"identity list must have T={T} entries, got {len(identity)}"
        identity_source = list(identity)
        identity_resolved = "mixed"
    else:
        identity_source = [identity] * T
        identity_resolved = identity
    per_frame = {
        "exist": [[exist] * T, [exist] * T],
        "collapsed": [0] * T,
        "slot_used": [[1] * T, [2] * T],
        "identity_source": identity_source,
        "assign_reason": [["typed_preferred"] * T, ["typed_preferred"] * T],
    }
    for t in drop_frames:                                  # an existence dip on fly0
        per_frame["exist"][0][t] = 0.2
    meta = {"n_frames": T, "identity_resolved": identity_resolved, "containment": True,
            "keypoint_names_written": KP_NAMES, "cameras": CAMS,
            "checkpoint": "/fake/final", "step": "final", "frame_start": int(frame_start),
            "bout": int(name.split("_")[-1]), "session_dir": str(tmp_path / "video"),
            "fly_sex": {"fly0": "female", "fly1": "male"},
            "masks_npz": str(tmp_path / "sam3_masks.npz" if masks_path is None else masks_path),
            "containment_report": {"enabled": True, "per_frame": {
                # `drop_frames` simulates an EXISTENCE dip (per_frame["exist"]
                # above), not a containment drop -- n_dropped stays all-zero
                # here so identity_gate only sees genuine containment events.
                "n_dropped": [[0] * T, [0] * T]}},
            "per_frame": per_frame}
    for f in range(2):
        d = root / f"fly{f}"; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "kp3d.npz", kp3d=kp3d[f], conf3d=conf[f].mean(1),
                 conf3d_mvq_raw=conf[f].mean(1), kp_names=np.array(KP_NAMES),
                 gates=np.asarray("{}"))
        np.savez(d / "kp2d.npz", kp2d=kp2d[f], conf=conf[f],
                 cameras=np.array(CAMS), kp_names=np.array(KP_NAMES))
    (root / "mvq_meta.json").write_text(json.dumps(meta))
    npz = make_masks(tmp_path, kp3d, cm, T, path=masks_path)
    return str(root), npz, cm


def make_masks(tmp_path, kp3d, cm, T, path=None):
    """A `BoutMaskStore`-readable npz: a filled box around each fly's
    reprojected keypoints in every camera."""
    F, _, K, _ = kp3d.shape; C = len(CAMS)
    full = np.zeros((F, C, T, H, W), bool)
    cent = np.zeros((F, C, T, 2), np.float32)
    for f in range(F):
        for t in range(T):
            uv = project(cm, kp3d[f, t])
            for c in range(C):
                x0, y0 = np.nanmin(uv[c], 0) - 4; x1, y1 = np.nanmax(uv[c], 0) + 4
                xs = slice(max(int(x0), 0), min(int(x1) + 1, W))
                ys = slice(max(int(y0), 0), min(int(y1) + 1, H))
                full[f, c, t, ys, xs] = True
                cent[f, c, t] = np.nanmean(uv[c], 0)
    p = (tmp_path / "sam3_masks.npz") if path is None else path
    os.makedirs(os.path.dirname(str(p)), exist_ok=True)
    np.savez(p, packed=np.packbits(full, axis=-1), valid=np.ones((F, C, T), bool),
             centroids=cent, shape=np.array([H, W], np.int32), cameras=np.array(CAMS))
    return str(p)
