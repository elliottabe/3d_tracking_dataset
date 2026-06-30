"""D3 session orchestration: camera geometry + bout/fly/frame -> 3D CSVs."""
import os
import numpy as np


def reorder_matrices_by_name(jax_names, jax_matrices, target_names):
    """Reorder (nc,4,3) jax_matrices (in jax_names order) to target_names order.

    Parameters
    ----------
    jax_names : list[str]
        Camera names corresponding to the rows of jax_matrices.
    jax_matrices : array-like, shape (nc, 4, 3)
        Camera projection matrices in jax_names order.
    target_names : list[str]
        Desired camera name order for the output.

    Returns
    -------
    np.ndarray, shape (nc, 4, 3)
        Matrices reordered to match target_names.

    Raises
    ------
    KeyError
        If any name in target_names is not present in jax_names.
    """
    idx = {n: i for i, n in enumerate(jax_names)}
    order = [idx[n] for n in target_names]      # KeyError if a target name is absent
    return np.asarray(jax_matrices)[order]


def camera_order_from_jarvis(project, session_dir, jarvis_root=None):
    """Authoritative camera-name order from JARVIS get_repro_tool (matches D2 npz axis).

    Parameters
    ----------
    project : str
        JARVIS project name.
    session_dir : str or Path
        Session directory (must contain a ``calibration/`` subdirectory).
    jarvis_root : str or None
        Path to the JARVIS project parent directory. Falls back to the
        ``JARVIS_ROOT`` environment variable when None.

    Returns
    -------
    list[str]
        Camera names in the order used by JARVIS / D2 sam3_masks.npz.
    """
    import sys
    import numpy as np
    root = jarvis_root or os.environ.get("JARVIS_ROOT")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    # Apply NumPy 2.0 compatibility shim before importing JARVIS so that imgaug
    # (an unmaintained dependency of JARVIS) does not trip on np.sctypes removal.
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [bool, object, bytes, str, np.void],
        }
        for fn, tgt in [("product", "prod"), ("cumproduct", "cumprod"),
                        ("round_", "round"), ("alltrue", "all"),
                        ("sometrue", "any")]:
            if not hasattr(np, fn):
                setattr(np, fn, getattr(np, tgt))
        for name, val in [("bool8", np.bool_), ("float_", np.float64),
                          ("complex_", np.complex128), ("unicode_", np.str_),
                          ("int0", np.intp), ("uint0", np.uintp)]:
            if not hasattr(np, name):
                setattr(np, name, val)
    from jarvis.config.project_manager import ProjectManager
    from jarvis.utils.reprojection import get_repro_tool
    pm = ProjectManager()
    if jarvis_root:
        pm.parent_dir = jarvis_root        # mirror D2's override (project lives at Github clone)
    assert pm.load(project), f"could not load project {project}"
    cfg = pm.get_cfg()
    session_calib = os.path.join(str(session_dir), "calibration")
    rt = get_repro_tool(cfg, session_calib if os.path.isdir(session_calib) else None)
    return list(rt.cameras)


def session_geometry(project, session_dir, jarvis_root=None):
    """Return camera names (JARVIS/D2 order) and cameraMatrices reordered to that order.

    Parameters
    ----------
    project : str
        JARVIS project name.
    session_dir : str or Path
        Session directory containing a ``calibration/`` subdirectory with
        per-camera ``Cam*.yaml`` files.
    jarvis_root : str or None
        Path to the JARVIS project parent directory (see camera_order_from_jarvis).

    Returns
    -------
    camera_names : list[str]
        Camera names in JARVIS / D2 sam3_masks.npz axis order.
    cameraMatrices : np.ndarray, shape (nc, 4, 3), dtype float32
        Projection matrices reordered to match camera_names.
    """
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    target = camera_order_from_jarvis(project, session_dir, jarvis_root)
    calib = os.path.join(str(session_dir), "calibration")
    rt = ReprojectionTool(calib)
    jax_names = list(rt.cameras)
    mats = reorder_matrices_by_name(jax_names, rt.camera_matrices.astype(np.float32), target)
    return target, mats


def _read_frame(caps, frame_idx):
    """Read one frame from each opened cv2 VideoCapture (random-access seek).

    Parameters
    ----------
    caps : list[cv2.VideoCapture]
        One open capture per camera, already positioned or seekable.
    frame_idx : int
        Absolute frame index to read from each capture.

    Returns
    -------
    np.ndarray, shape (nc, H, W, 3), dtype uint8, RGB channel order.
    """
    import cv2
    imgs = []
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame_idx}")
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(imgs)   # (nc, H, W, 3) uint8 RGB


def run_predict_session(*, session_dir, masks_dir, out, project, jarvis_root,
                        v2v_final, vitpose_ckpt, sharpen, num_animals=2, batch=8,
                        bout_ids=None, limit=0, data_root=None, num_keypoints=50):
    """Orchestrate 3-D inference for all requested bouts in one session.

    Reads D2 SAM3 masks from *masks_dir*, opens per-camera videos from
    *session_dir*, builds V3-format framesets with :func:`build_frameset`, runs
    :func:`predict_batch` in batches, and writes per-fly CSVs to *out*.

    Parameters
    ----------
    session_dir : str
        Session root containing ``Cam*.mp4`` videos and a ``calibration/``
        subdirectory.
    masks_dir : str
        D2 output directory containing ``manifest.json`` and
        ``bout_<idx:05d>/sam3_masks.npz`` files.
    out : str
        Output root; per-bout CSVs go in ``out/bout_<idx:05d>/fly{a}.csv`` and
        the merged session CSVs in ``out/data3D_fly{a}.csv``.
    project : str
        JARVIS project name (for geometry / camera order).
    jarvis_root : str or None
        Path to the JARVIS-HybridNet clone (added to sys.path for geometry +
        ``predict3D_multianimal``).
    v2v_final : str
        Path to the Orbax V2VNet checkpoint directory.
    vitpose_ckpt : str
        Path to the Orbax ViTPose checkpoint directory.
    sharpen : float
        Soft-argmax sharpening exponent passed to :func:`load_inference_model`.
    num_animals : int
        Number of fly identities expected per frame (default 2).
    batch : int
        Inference batch size (frames); padded to a device multiple internally.
    bout_ids : list[int] or None
        If given, only process bouts whose ``bout_idx`` is in this list.
    limit : int
        If >0, cap the number of bouts processed (useful for smoke-tests).
    data_root : str or None
        Root of the V3 dataset used to retrieve ``keypoint_names``.  Falls back
        to the hardcoded training-data path when None.

    Returns
    -------
    dict with keys ``bouts`` (list of processed bout_idx), ``out`` (str),
    ``camera_names`` (list[str]), and ``num_animals`` (int).
    """
    import json
    import cv2
    import jax
    from jarvis_jax.predict.infer_3d import load_inference_model, predict_batch
    from jarvis_jax.predict.session_frameset import build_frameset
    from jarvis_jax.predict.session_io import write_fly_csv, concat_fly_csvs

    os.makedirs(out, exist_ok=True)

    # Camera geometry (names + projection matrices) in JARVIS/D2 axis order.
    camera_names, cameraMatrices = session_geometry(project, session_dir, jarvis_root)

    # Joint names: 50 V3 keypoints, then vtx_0..(M-1) canonical vertices for dense pose.
    from jarvis_jax.data.v3_3d import V3FramesetDataset
    root = data_root or "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
    joint_names = list(V3FramesetDataset(root, "val").keypoint_names)
    if num_keypoints > len(joint_names):
        joint_names += [f"vtx_{i}" for i in range(num_keypoints - len(joint_names))]
    elif num_keypoints < len(joint_names):
        joint_names = joint_names[:num_keypoints]

    # Load frozen ViTPose + V2VNet inference model, replicated across devices.
    model = load_inference_model(vitpose_ckpt, v2v_final, sharpen=sharpen,
                                 num_keypoints=num_keypoints)
    nd = jax.device_count()

    # Open one VideoCapture per camera (sequential read within each bout).
    caps = [cv2.VideoCapture(os.path.join(session_dir, f"{c}.mp4"))
            for c in camera_names]

    # Enumerate bouts from the D2 manifest.
    with open(os.path.join(masks_dir, "manifest.json")) as f:
        man = json.load(f)
    bouts = man["bouts"]
    if bout_ids:
        ids = {int(b) for b in bout_ids}
        bouts = [b for b in bouts if b["bout_idx"] in ids]
    if limit and limit > 0:
        bouts = bouts[:limit]

    # predict3D_multianimal is in JARVIS tools/ — camera_order_from_jarvis
    # already added jarvis_root to sys.path; add tools/ sub-path if needed.
    import sys
    tools_dir = os.path.join(jarvis_root, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from predict3D_multianimal import LoadedBoutMasks

    per_bout_csv = {a: [] for a in range(num_animals)}

    for b in bouts:
        bi, start = b["bout_idx"], b["start"]
        lm = LoadedBoutMasks(
            os.path.join(masks_dir, f"bout_{bi:05d}", "sam3_masks.npz")
        )
        bout_dir = os.path.join(out, f"bout_{bi:05d}")
        os.makedirs(bout_dir, exist_ok=True)

        for a in range(num_animals):
            rows = []   # list of (abs_frame, kp3d|None, conf|None)
            buf = []    # list of (crops4 (nc,448,448,4), centerHM (nc,2), abs_frame)

            def flush(buf=buf, rows=rows):
                """Pad buf to device multiple, run predict_batch, slice [:B0]."""
                if not buf:
                    return
                B0 = len(buf)
                pad = (-B0) % nd
                crops = np.stack([x[0] for x in buf] + [buf[-1][0]] * pad)
                chm = np.stack([x[1] for x in buf] + [buf[-1][1]] * pad)
                # Broadcast (nc,4,3) cameraMatrices to (B, nc, 4, 3).
                cams = np.broadcast_to(
                    cameraMatrices[None], (crops.shape[0],) + cameraMatrices.shape
                ).copy()
                kp, conf, _ = predict_batch(
                    model, crops, chm.astype(np.float32), cams.astype(np.float32)
                )
                for k in range(B0):
                    rows.append((buf[k][2], np.asarray(kp[k]), np.asarray(conf[k])))
                buf.clear()

            # Sequential video read: seek once per bout, then read in order.
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            for fi in range(lm.num_frames):
                abs_frame = start + fi
                slot_list = lm.get_frame(fi, num_animals)
                if slot_list is None:
                    rows.append((abs_frame, None, None))
                    continue
                slot = slot_list[a]
                if slot is None:
                    rows.append((abs_frame, None, None))
                    continue

                # Convert torch tensors to numpy at the boundary.
                masks = slot["masks"].cpu().numpy()       # (nc, H, W) bool
                cents = slot["centroids"].cpu().numpy()   # (nc, 2)
                valid = slot["valid"].cpu().numpy()       # (nc,) bool

                # Read one RGB frame from each camera (sequential).
                frame_imgs_list = []
                for cap in caps:
                    ok, bgr = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"failed to read frame {abs_frame} from cap"
                        )
                    frame_imgs_list.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                frame_imgs = np.stack(frame_imgs_list)   # (nc, H, W, 3) uint8 RGB

                crops4, centerHM, nv = build_frameset(
                    frame_imgs, masks, cents, valid, cameraMatrices
                )
                if crops4 is None:
                    rows.append((abs_frame, None, None))
                    continue

                buf.append((crops4, centerHM, abs_frame))
                if len(buf) >= batch:
                    flush()

            flush()   # drain any remaining frames

            rows.sort(key=lambda r: r[0])
            cpath = os.path.join(bout_dir, f"fly{a}.csv")
            write_fly_csv(cpath, joint_names, rows)
            per_bout_csv[a].append(cpath)
            n_pred = sum(1 for r in rows if r[1] is not None)
            print(
                f"[predict_session] bout {bi} fly{a}: "
                f"{n_pred}/{len(rows)} predicted"
            )

    # Merge per-bout CSVs into one session-level file per fly.
    for a in range(num_animals):
        concat_fly_csvs(
            sorted(per_bout_csv[a]),
            os.path.join(out, f"data3D_fly{a}.csv"),
        )

    for cap in caps:
        cap.release()

    return {
        "bouts": [b["bout_idx"] for b in bouts],
        "out": out,
        "camera_names": camera_names,
        "num_animals": num_animals,
    }
