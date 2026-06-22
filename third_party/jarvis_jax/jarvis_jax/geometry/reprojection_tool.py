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
