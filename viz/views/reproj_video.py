"""reproj-video view: predicted 3D keypoints reprojected onto the raw camera
video for one bout, written as one mp4 per camera.

Reimplemented directly on viz.core (reproject/overlays/io) instead of calling
JARVIS's `create_multi_animal_videos3D` -- see scripts/viz_predictions_reproject.py
for the old JARVIS-backed reference this replaces.

Frame range comes from the per-bout dense CSV's own frame column (col0), via
`core.io.load_data3d_csv`, NOT from `scripts.run_bout.bout_start_frame`
-- that keeps this view free of the heavy jax/mujoco/egl import chain (unlike
viz/views/overlay.py, which needs `bout_start_frame` for a different reason
and imports it lazily).

Camera-by-name lookup + graceful degradation (missing camera in calibration,
missing session video, missing fly CSV, unreadable frame) follows the pattern
established in viz/views/overlay.py: warn + skip the offending unit, never
abort the whole run for one bad camera/frame/fly.
"""
import os

import cv2

from viz.core import colors as vcolors
from viz.core import io as vio
from viz.core import overlays
from viz.core import reproject


def _load_fly(pred_dir, bout, fly):
    csv_path = os.path.join(pred_dir, f"bout_{int(bout):05d}", f"fly{int(fly)}.csv")
    if not os.path.isfile(csv_path):
        return None
    kp3d, _conf, names, frames = vio.load_data3d_csv(csv_path)
    return {"kp3d": kp3d, "names": names, "frames": frames, "csv_path": csv_path}


def run(args):
    session_dir = args.session_dir
    pred_dir = args.pred_dir
    bout = int(args.bout)

    calib_dir = os.path.join(session_dir, "calibration")
    cam_mats, cam_names = reproject.camera_matrices(calib_dir)
    cam_mat_idx = {n: i for i, n in enumerate(cam_names)}
    cameras = list(args.cameras) if args.cameras else list(cam_names)

    fly_data = {}
    for fly in (0, 1):
        d = _load_fly(pred_dir, bout, fly)
        if d is None:
            print(f"[reproj-video] warning: missing fly{fly}.csv for bout {bout} "
                  f"under {os.path.join(pred_dir, f'bout_{bout:05d}')}; skipping fly{fly}")
            continue
        fly_data[fly] = d

    if not fly_data:
        bout_dir = os.path.join(pred_dir, f"bout_{bout:05d}")
        raise FileNotFoundError(
            f"no per-bout fly CSVs found for bout {bout}: neither "
            f"{os.path.join(bout_dir, 'fly0.csv')} nor {os.path.join(bout_dir, 'fly1.csv')} exist")

    # Frame range: driven by the first available fly's own frame column.
    # Both flies in a bout cover the same absolute window in practice; if
    # a fly's array runs short at a given t, that fly is simply skipped for
    # that timestep below (see `t >= len(d["kp3d"])`).
    ref_fly = min(fly_data)
    frames_ref = fly_data[ref_fly]["frames"]
    if len(frames_ref) == 0:
        raise ValueError(f"no data rows in {fly_data[ref_fly]['csv_path']}")
    start = int(frames_ref[0])
    count = len(frames_ref)
    max_frames = int(getattr(args, "max_frames", 0) or 0)
    if max_frames > 0:
        count = min(count, max_frames)

    avail_cameras = []
    for cam in cameras:
        if cam not in cam_mat_idx:
            print(f"[reproj-video] warning: camera {cam} missing from calibration "
                  f"{calib_dir}; skipping")
            continue
        video_path = os.path.join(session_dir, f"{cam}.mp4")
        if not os.path.isfile(video_path):
            print(f"[reproj-video] warning: no session video for camera {cam} at "
                  f"{video_path}; skipping")
            continue
        avail_cameras.append(cam)

    if not avail_cameras:
        raise RuntimeError(
            f"no camera videos available under {session_dir} for cameras={cameras}")

    # Masks are loaded ONCE against the FULL calibration camera-name list
    # (cam_names), never the rendered subset (avail_cameras) -- mirroring
    # viz/views/overlay.py's invariant. `load_bout_masks` (called via
    # vio.load_masks) only reorders its C axis by NAME when the npz carries a
    # `cameras` array; for legacy npz files without one, `expected_cameras`
    # is silently ignored and the returned axis is the native/full
    # calibration order. Indexing that axis by a filtered subset's position
    # (as this view used to do) would silently fetch the WRONG camera's mask
    # whenever avail_cameras is a subset or reordering of cam_names. Instead
    # we index by each camera's position in cam_names (`cam_mat_idx[cam]`,
    # the same "native index" already used to pick `cam_mats`) below.
    masks_by_fly = {}
    if getattr(args, "with_masks", False):
        for fly in fly_data:
            try:
                masks_by_fly[fly] = vio.load_masks(pred_dir, bout, fly, cam_names)
            except Exception as e:
                print(f"[reproj-video] warning: sam3_masks.npz unavailable for fly{fly} "
                      f"({e}); rendering fly{fly} without masks")

    chains_by_fly = {fly: vcolors.leg_chains(d["names"]) for fly, d in fly_data.items()}

    out_dir = args.out or os.path.join(pred_dir, "viz", f"bout_{bout:05d}")
    os.makedirs(out_dir, exist_ok=True)
    fps = int(getattr(args, "fps", 30) or 30)

    def _cam_frames(cam, cam_mat, native_idx):
        """Stream one camera's drawn BGR frames, one at a time. Reads this
        camera's mp4 alone (single-element camera list to `read_frames`, so
        each yielded `imgs` is a 1-element list) and yields each drawn frame
        immediately -- nothing beyond the current frame is held in memory,
        avoiding the ~9GB all-cameras-all-frames buffer this view used to
        build before writing anything out."""
        for t, imgs in enumerate(vio.read_frames_synced(session_dir, [cam], start, count)):
            rgb = imgs[0]
            if rgb is None:
                print(f"[reproj-video] warning: frame {start + t} unreadable for {cam}; skipping")
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            for fly, d in fly_data.items():
                if t >= len(d["kp3d"]):
                    continue
                color = vcolors.PALETTE[f"fly{fly}"]
                try:
                    uv = reproject.project(cam_mat, d["kp3d"][t])
                    overlays.draw_points(bgr, uv, color)
                    for idxs in chains_by_fly[fly].values():
                        overlays.draw_chain(bgr, uv[idxs], color)
                except Exception as e:
                    print(f"[reproj-video] warning: reprojection failed for fly{fly} "
                          f"cam {cam} t={t}: {e}")

                m = masks_by_fly.get(fly)
                if m is not None:
                    try:
                        if (t < m["valid"].shape[0] and m["valid"][t, native_idx]
                                and m["masks"][t, native_idx].any()):
                            overlays.draw_mask(bgr, m["masks"][t, native_idx], vcolors.PALETTE["mask"])
                    except Exception as e:
                        print(f"[reproj-video] warning: mask overlay failed for fly{fly} "
                              f"cam {cam} t={t}: {e}")

            yield bgr

    wrote_any = False
    for cam in avail_cameras:
        native_idx = cam_mat_idx[cam]
        cam_mat = cam_mats[native_idx]
        out_path = os.path.join(out_dir, f"reproj_bout{bout}_{cam}.mp4")
        try:
            vio.write_video(out_path, _cam_frames(cam, cam_mat, native_idx), fps=fps)
        except ValueError:
            print(f"[reproj-video] warning: no frames collected for {cam}; skipping write")
            continue
        print(f"[reproj-video] wrote {out_path}")
        wrote_any = True

    if not wrote_any:
        raise RuntimeError(
            f"no frames were rendered for bout {bout} (0 usable frames across {avail_cameras})")

    return 0
