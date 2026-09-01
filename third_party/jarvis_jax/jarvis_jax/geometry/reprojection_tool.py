"""NumPy port of JARVIS ReprojectionTool + Camera.

Port of:
  third_party/JARVIS-HybridNet/jarvis/dataset/utils.py
  Classes: Camera, ReprojectionTool

Dependencies: numpy, cv2 (for reading OpenCV YAML calibration files).
No torch or jax required.

Calibration directory layout (discovered from real data):
  <calib_dir>/
    Cam<id>.yaml   (one file per camera, OpenCV YAML format)
  Each YAML contains a 'projectionMatrix' node: a 3×4 matrix (rows=3, cols=4).

Public API
----------
ReprojectionTool(calib_dir)
    .num_cameras          : int
    .camera_matrices      : ndarray (num_cam, 4, 3) float32
                            Each 3×4 projection matrix transposed to (4,3)
                            so that homogeneous 3D point x@M gives (3,).
    .reproject_point(p3d) : (num_cam, 2) float64
                            DLT projection + perspective divide.
    .reconstruct_point(points2d, cams_to_use=None) : (3,) float64
                            DLT triangulation via SVD of the ray-constraint
                            matrix; dehomogenize the last right-singular vector.
    .reproject_points(p3d) : (..., num_cam, 2) float64
                            BATCHED reproject_point over any leading shape.
                            BIT-IDENTICAL to stacking reproject_point (same
                            float64 matrices, same 4-term dot order via
                            np.einsum -- NOT a BLAS matmul, which reassociates
                            and drifts ~1e-13). See reproject_points.
    .reconstruct_points(points2d, cams) : (N, 3) float64
                            BATCHED reconstruct_point for N systems that all
                            use the SAME NUMBER of cameras. Bit-identical to
                            the per-point loop (numpy's stacked SVD calls the
                            same LAPACK routine per matrix).
"""

import os
import glob

import cv2
import numpy as np


class Camera:
    """Load a single camera's 3×4 projection matrix from an OpenCV YAML file."""

    def __init__(self, name: str, calib_path: str):
        self.name = name
        # 3×4 projection matrix (shape (3, 4), float64)
        self.cameraMatrix = self._get_mat_from_file(calib_path, "projectionMatrix")

    @staticmethod
    def _get_mat_from_file(filepath: str, node_name: str) -> np.ndarray:
        """Read a matrix node from an OpenCV FileStorage YAML file."""
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        mat = fs.getNode(node_name).mat()
        fs.release()
        if mat is None or mat.size == 0:
            raise ValueError(
                f"Node '{node_name}' not found or empty in '{filepath}'"
            )
        return mat  # shape (3, 4), dtype float64


class ReprojectionTool:
    """NumPy DLT reprojection / triangulation tool.

    Parameters
    ----------
    calib_dir : str
        Directory containing per-camera calibration YAML files
        (``Cam*.yaml``).  Files are discovered via glob and sorted
        lexicographically so that camera indices are stable.
    """

    def __init__(self, calib_dir: str):
        calib_files = sorted(glob.glob(os.path.join(calib_dir, "Cam*.yaml")))
        if not calib_files:
            raise FileNotFoundError(
                f"No 'Cam*.yaml' calibration files found in '{calib_dir}'"
            )

        self.cameras: dict[str, Camera] = {}
        for path in calib_files:
            name = os.path.splitext(os.path.basename(path))[0]  # e.g. "Cam2012630"
            self.cameras[name] = Camera(name, path)

        self._camera_list = list(self.cameras.values())
        self.num_cameras: int = len(self._camera_list)

        # Build (num_cam, 4, 3) array: each 3×4 matrix transposed → (4,3)
        # so that  p_h @ M  gives the homogeneous image point (3,).
        self.camera_matrices: np.ndarray = np.zeros(
            (self.num_cameras, 4, 3), dtype=np.float32
        )
        for i, cam in enumerate(self._camera_list):
            self.camera_matrices[i] = cam.cameraMatrix.T.astype(np.float32)

        # float64 (num_cam, 3, 4) stack for the batched reproject_points /
        # reconstruct_points below. Deliberately NOT `camera_matrices`, which is
        # float32: reproject_point does its dot in the cameras' native float64,
        # so reusing the float32 stack would move every reprojection by ~1e-2 px
        # and silently change every QC number that goes through it.
        self._cam_mats_f64: np.ndarray = np.stack(
            [cam.cameraMatrix for cam in self._camera_list]).astype(np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reproject_point(self, p3d: np.ndarray) -> np.ndarray:
        """Project a single 3D point onto all cameras.

        Parameters
        ----------
        p3d : (3,) array_like
            3D point in world coordinates.

        Returns
        -------
        pts2d : ndarray (num_cam, 2) float64
            Pixel coordinates for each camera (perspective-divided).
        """
        p3d = np.asarray(p3d, dtype=np.float64)
        p3d_h = np.concatenate([p3d, [1.0]])  # homogeneous (4,)

        pts2d = np.zeros((self.num_cameras, 2), dtype=np.float64)
        for i, cam in enumerate(self._camera_list):
            # cam.cameraMatrix is (3,4); dot with (4,) → (3,)
            proj = cam.cameraMatrix.dot(p3d_h)
            # perspective divide: normalise by the third coordinate
            pts2d[i] = (proj / proj[2])[:2]
        return pts2d

    def reconstruct_point(
        self,
        points2d: np.ndarray,
        cams_to_use: list[int] | None = None,
    ) -> np.ndarray:
        """Triangulate a 3D point from 2D observations via DLT + SVD.

        Parameters
        ----------
        points2d : (num_cam, 2) array_like
            Observed 2D pixel coordinates, one row per camera.
        cams_to_use : list of int, optional
            Indices of cameras to include.  Default: all cameras.

        Returns
        -------
        X : ndarray (3,) float64
            Triangulated 3D point.  Returns [0,0,0] if fewer than 2 cameras.
        """
        if cams_to_use is None:
            cams_to_use = list(range(self.num_cameras))

        if len(cams_to_use) < 2:
            return np.zeros(3, dtype=np.float64)

        points2d = np.asarray(points2d, dtype=np.float64)  # (num_cam, 2)

        # Collect projection matrices and 2D observations for used cameras
        # points2d layout: (num_cam, 2)  → column i = camera cams_to_use[i]
        # Matches the original: pointsToUse[:, i] = points[:, cam]
        # where points was (2, num_cam).  Here points2d is (num_cam, 2) so
        # we index as points2d[cam, :].
        cam_mats = []
        pts = []
        for cam_idx in cams_to_use:
            cam_mats.append(self._camera_list[cam_idx].cameraMatrix)  # (3,4)
            pts.append(points2d[cam_idx])  # (2,)

        n = len(cam_mats)
        # Build DLT constraint matrix A  (2n × 4)
        # For camera i with projection matrix P (3×4) and observation (u,v):
        #   [u * P[2] - P[0]]
        #   [v * P[2] - P[1]]
        A = np.zeros((2 * n, 4), dtype=np.float64)
        for i in range(n):
            P = cam_mats[i]   # (3, 4)
            uv = pts[i]       # (2,)
            # uv.reshape(2,1) * P[2].reshape(1,4) - P[0:2]  → (2,4)
            A[2 * i : 2 * i + 2] = (
                uv.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]
            )

        _, _, Vh = np.linalg.svd(A)
        # Last row of Vh is the right-singular vector for the smallest singular value
        X_h = Vh[-1]          # (4,) homogeneous 3D point
        X_h = X_h / X_h[3]   # dehomogenize
        return X_h[:3]

    # ------------------------------------------------------------------
    # Batched forms of the two primitives above.
    #
    # Both are BIT-IDENTICAL to looping the scalar version -- verified on this
    # rig's real calibration and asserted in
    # tests/test_reprojection_tool_batched.py -- because the QC report is a
    # continuous time series across runs: an optimisation that moved these by
    # one ulp would show up as a (tiny, unexplained) drift in every bout's
    # qc.json.  The per-point loops were the pipeline's single biggest QC cost
    # (7M scalar reproject_point calls per bout in jarvis_jax.tracking.qc).
    # ------------------------------------------------------------------

    def reproject_points(self, p3d: np.ndarray) -> np.ndarray:
        """Project a batch of 3D points onto all cameras.

        Parameters
        ----------
        p3d : (..., 3) array_like
            3D points in world coordinates (any leading shape).

        Returns
        -------
        pts2d : ndarray (..., num_cam, 2) float64
            Pixel coordinates, perspective-divided.  Equals
            ``np.stack([self.reproject_point(p) for p in p3d])`` exactly
            (non-finite input propagates the same way).

        Uses ``np.einsum`` and NOT ``@``/``np.tensordot``: einsum evaluates the
        4-term dot in the same order as ``M.dot(p_h)``, while the BLAS-backed
        products reassociate and disagree by ~1e-13 px.
        """
        p3d = np.asarray(p3d, dtype=np.float64)
        ph = np.concatenate([p3d, np.ones((*p3d.shape[:-1], 1))], axis=-1)
        proj = np.einsum('...k,cjk->...cj', ph, self._cam_mats_f64)
        return proj[..., :2] / proj[..., 2:3]

    def reconstruct_points(
        self,
        points2d: np.ndarray,
        cams: np.ndarray,
    ) -> np.ndarray:
        """Triangulate N points, each from the same NUMBER of cameras.

        Parameters
        ----------
        points2d : (N, n, 2) array_like
            Observed 2-D pixel coordinates; ``points2d[i, k]`` is point i as
            seen by camera ``cams[i, k]``.
        cams : (N, n) int array_like
            Camera indices per system, in the order the DLT rows should be
            stacked (``reconstruct_point``'s ``cams_to_use`` order).

        Returns
        -------
        X : ndarray (N, 3) float64
            Same values as ``[reconstruct_point(obs_i, cams_to_use=cams[i])]``.
            ``n < 2`` returns zeros, matching ``reconstruct_point``.

        A non-finite observation makes LAPACK fail to converge and raises
        ``numpy.linalg.LinAlgError`` -- exactly as the scalar version does on
        the same input, just for the whole batch at once.
        """
        points2d = np.asarray(points2d, dtype=np.float64)
        cams = np.asarray(cams, dtype=int)
        N, n = cams.shape
        if n < 2:
            return np.zeros((N, 3), dtype=np.float64)
        P = self._cam_mats_f64[cams]                      # (N, n, 3, 4)
        # [u*P[2] - P[0]; v*P[2] - P[1]] per camera, same expression as the loop
        rows = points2d[..., None] * P[:, :, 2:3, :] - P[:, :, 0:2, :]  # (N,n,2,4)
        A = rows.reshape(N, 2 * n, 4)
        Vh = np.linalg.svd(A)[2]
        X_h = Vh[:, -1, :]
        return X_h[:, :3] / X_h[:, 3:4]
