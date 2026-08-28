"""Build MuJoCo cameras that observe the model from the SAME viewpoint as the
real rig's calibrated cameras.

The rig's calibration is AFFINE, not pinhole: every camera matrix's third row
is [0, 0, 0, 1], so the projection has no depth divide and apparent size is
exactly independent of distance (measured: a 300-unit shift changes |du| by a
factor 1.0000). There is therefore no finite camera centre to recover -- an RQ
decomposition is singular -- and the correct MuJoCo counterpart is an
ORTHOGRAPHIC camera (`projection="orthographic"`), whose `fovy` is the FULL
visible height in length units (measured: fovy=2.0 shows 1.984 units).

The MuJoCo scene is in MODEL units while the calibration is in world mm, so a
per-frame similarity (scale s, rotation R, translation t) with
``X_world = s R X_model + t`` maps between them; recover it by fitting the FK'd
site positions to the same run's `kp3d_mm` (see `similarity_from_points`).
"""
from __future__ import annotations

import numpy as np


def affine_camera_rows(cam_mat_4x3):
    """(m0, o0, m1, o1) for the `ph @ M` convention used by viz.core.reproject.

    u = m0 . X + o0 ;  v = m1 . X + o1 ; the third column must be [0,0,0,1].
    """
    M = np.asarray(cam_mat_4x3, float)
    if M.shape != (4, 3):
        raise ValueError(f"expected a (4,3) camera matrix, got {M.shape}")
    third = M[:, 2]
    if not (np.allclose(third[:3], 0) and np.isclose(third[3], 1)):
        raise ValueError(
            "camera is not affine (third column != [0,0,0,1]); this builder "
            f"only handles the rig's affine calibration, got {third}")
    return M[:3, 0].copy(), float(M[3, 0]), M[:3, 1].copy(), float(M[3, 1])


def similarity_from_points(A, B):
    """Umeyama similarity (s, R, t) with ``B ~ s R A + t``."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    U, S, Vt = np.linalg.svd(A0.T @ B0 / len(A))
    dsign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, dsign]) @ U.T
    s = float((S * np.array([1, 1, dsign])).sum() / ((A0 ** 2).sum() / len(A)))
    return s, R, cb - s * R @ ca


def mat_to_quat(Rc):
    """3x3 rotation -> MuJoCo (w, x, y, z)."""
    q = np.empty(4)
    tr = np.trace(Rc)
    if tr > 0:
        k = 0.5 / np.sqrt(1.0 + tr)
        q[:] = (0.25 / k, (Rc[2, 1] - Rc[1, 2]) * k,
                (Rc[0, 2] - Rc[2, 0]) * k, (Rc[1, 0] - Rc[0, 1]) * k)
    else:
        i = int(np.argmax(np.diag(Rc)))
        j, k_ = (i + 1) % 3, (i + 2) % 3
        r = np.sqrt(1.0 + Rc[i, i] - Rc[j, j] - Rc[k_, k_])
        v = np.zeros(3); v[i] = 0.5 * r
        v[j] = (Rc[j, i] + Rc[i, j]) / (2 * r)
        v[k_] = (Rc[k_, i] + Rc[i, k_]) / (2 * r)
        q[0] = (Rc[k_, j] - Rc[j, k_]) / (2 * r)
        q[1:] = v
    return q / np.linalg.norm(q)


def mujoco_camera_from_affine(cam_mat_4x3, img_wh, s, R, t, anchor_model,
                              back_off=None):
    """An orthographic MuJoCo camera in MODEL space matching a real camera.

    Args:
        cam_mat_4x3: the rig camera, `ph @ M` convention.
        img_wh: (width, height) of the real image in px.
        s, R, t: model->world similarity, ``X_world = s R X_model + t``.
        anchor_model: a 3-vector in model space to centre the view on.
        back_off: distance to pull the camera back along its own +z. Only
            affects clipping (an orthographic view has no perspective), so it
            just needs to clear the scene.

    Returns (pos, quat, fovy) for `projection="orthographic"`.
    """
    W, H = float(img_wh[0]), float(img_wh[1])
    m0, o0, m1, o1 = affine_camera_rows(cam_mat_4x3)
    # push the camera rows through the similarity into model space
    m0m = s * (R.T @ m0); o0m = float(m0 @ t + o0)
    m1m = s * (R.T @ m1); o1m = float(m1 @ t + o1)

    right = m0m / np.linalg.norm(m0m)
    up = -m1m / np.linalg.norm(m1m)          # image v grows DOWN
    zc = np.cross(right, up)                 # MuJoCo cameras look along -z
    zc /= np.linalg.norm(zc)
    # re-orthogonalise: the affine rows need not be exactly perpendicular
    up = np.cross(zc, right); up /= np.linalg.norm(up)

    fovy = H / np.linalg.norm(m1m)           # FULL visible height, model units

    # Put the optical axis through the pixel centre: solve u(X)=W/2, v(X)=H/2
    # for X = anchor + a*right + b*up.
    a0 = np.array([m0m @ right, m0m @ up])
    a1 = np.array([m1m @ right, m1m @ up])
    rhs = np.array([W / 2.0 - (m0m @ anchor_model + o0m),
                    H / 2.0 - (m1m @ anchor_model + o1m)])
    ab = np.linalg.solve(np.vstack([a0, a1]), rhs)
    centre = np.asarray(anchor_model, float) + ab[0] * right + ab[1] * up

    if back_off is None:
        back_off = 10.0 * fovy
    pos = centre + back_off * zc
    quat = mat_to_quat(np.column_stack([right, up, zc]))
    return pos, quat, float(fovy)


def project_with_mujoco_camera(X_model, pos, quat, fovy, img_wh):
    """Where MuJoCo will draw `X_model`, in pixels -- for verifying the build."""
    from viz.core.mjcam import mat_to_quat  # noqa: F401  (self-doc)
    W, H = float(img_wh[0]), float(img_wh[1])
    w, x, y, z = quat
    Rc = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    d = (np.asarray(X_model, float) - np.asarray(pos, float)) @ Rc   # into camera axes
    ppu = H / fovy                                                    # px per model unit
    return np.column_stack([W / 2.0 + d[:, 0] * ppu, H / 2.0 - d[:, 1] * ppu])
