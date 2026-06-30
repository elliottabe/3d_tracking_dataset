"""Workstream E metrics on a predict_full output.

(1) 3-D accuracy: predicted vs STAC-GT 3-D, split into the 50 keypoints and the
    200 vertices (median / mean Euclidean error, raw-mm frame).
(2) Multi-view consistency: convert per-camera 2-D predictions to full-image
    coords, triangulate each joint across cameras, reproject, and report the
    residual (px).  Low = the views agree on one 3-D point.
"""
import argparse
import numpy as np


def _dlt_triangulate(P34, uv):
    """P34: (C,3,4) projection mats; uv: (C,2). Returns (3,) via SVD."""
    A = np.concatenate([uv[:, 0:1] * P34[:, 2] - P34[:, 0],
                        uv[:, 1:2] * P34[:, 2] - P34[:, 1]], axis=0)
    _, _, Vh = np.linalg.svd(A)
    X = Vh[-1]
    return X[:3] / X[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predict_full .npz")
    a = ap.parse_args()
    z = np.load(a.pred, allow_pickle=True)
    pred3d, gt3d, vis = z["pred3d"], z["gt3d"], z["vis"]
    pred2d, cHM, cM = z["pred2d"], z["centerHM"], z["cameraMatrices"]
    n, J = pred3d.shape[:2]

    # ---- (1) 3-D accuracy ----
    d = np.linalg.norm(pred3d - gt3d, axis=2)            # (n, J) mm
    kp_m = vis[:, :50].astype(bool)
    print("==== 3-D accuracy (raw-mm frame) ====")
    print(f"  keypoints (50): median {np.median(d[:, :50][kp_m]):.2f}  mean {d[:, :50][kp_m].mean():.2f}")
    print(f"  vertices (200): median {np.median(d[:, 50:]):.2f}  mean {d[:, 50:].mean():.2f}")

    # ---- (2) multi-view consistency (triangulate per-cam 2-D, reproject) ----
    P34 = np.transpose(cM, (0, 1, 3, 2))                 # (n,C,3,4)  from (n,C,4,3)
    full = pred2d + (cHM[:, :, None, :] - 224.0)          # crop->full-image px
    resid = []
    step = max(1, n // 300)                              # subsample framesets for speed
    for i in range(0, n, step):
        for j in range(J):
            uv = full[i, :, j, :]; P = P34[i]
            X = _dlt_triangulate(P, uv)
            ph = np.einsum("cij,j->ci", P, np.concatenate([X, [1.0]]))  # (C,3)
            rp = ph[:, :2] / ph[:, 2:3]
            resid.append((j, float(np.median(np.linalg.norm(rp - uv, axis=1)))))
    resid = np.array(resid)
    kpr = resid[resid[:, 0] < 50, 1]; vtr = resid[resid[:, 0] >= 50, 1]
    print("==== multi-view consistency (reprojection residual, px) ====")
    print(f"  keypoints: median {np.median(kpr):.2f}")
    print(f"  vertices:  median {np.median(vtr):.2f}")


if __name__ == "__main__":
    main()
