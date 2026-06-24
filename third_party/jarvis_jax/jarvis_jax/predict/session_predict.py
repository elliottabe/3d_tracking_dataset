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
    root = jarvis_root or os.environ.get("JARVIS_ROOT")
    if root and root not in sys.path:
        sys.path.insert(0, root)
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
