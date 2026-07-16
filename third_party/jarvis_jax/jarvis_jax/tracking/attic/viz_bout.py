"""Per-bout dense-pose-on-video overlay (run after predict_session writes CSVs).

Projects each fly's predicted 3-D joints onto one camera for a few frames and
saves a PNG.  Body/leg vertices are drawn cyan, wing vertices red (wing = the
trailing vtx indices, per the canonical-mesh FPS subset).  cv2-only (no JAX),
so it runs cheaply at the end of each array task.
"""
import argparse, csv, glob, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _load_csv(path):
    r = list(csv.reader(open(path)))[2:]
    fr = [int(x[0]) for x in r]
    import numpy as np
    a = np.array([[np.nan if v == "nan" else float(v) for v in row[1:]] for row in r]).reshape(len(r), -1, 4)
    return np.array(fr), a


def main():
    import numpy as np, cv2
    cv2.setNumThreads(0)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax")
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="bout dir with fly{a}.csv")
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--mesh", required=True, help="canonical mesh npz (for wing vtx mask)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-animals", type=int, default=2)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--n-kp", type=int, default=50)
    a = ap.parse_args()

    flies = []
    for k in range(a.num_animals):
        p = os.path.join(a.pred_dir, f"fly{k}.csv")
        if os.path.exists(p):
            flies.append(_load_csv(p))
    if not flies:
        print("no fly CSVs"); return
    nj = flies[0][1].shape[1]
    nv = nj - a.n_kp                                   # number of vertices
    # wing mask over the vertex block, from the mesh FPS subset segment names
    z = np.load(a.mesh, allow_pickle=True)
    fps = z[f"fps_{nv}"]
    id2n = {int(s): (n.decode() if isinstance(n, bytes) else n)
            for s, n in zip(z["seg_ids"], z["seg_names"])}
    iswing = np.array(["wing" in id2n[int(s)] for s in z["vertex_segment"][fps]])

    rt = ReprojectionTool(os.path.join(a.session_dir, "calibration"))
    camname = list(rt.cameras.keys())[0]; cam = rt.cameras[camname]
    cap = cv2.VideoCapture(os.path.join(a.session_dir, f"{camname}.mp4"))

    def proj(p3):
        ph = np.concatenate([p3, np.ones((len(p3), 1))], 1) @ cam.cameraMatrix.T
        return ph[:, :2] / ph[:, 2:3]

    fr0 = flies[0][0]
    idxs = [int(x) for x in np.linspace(5, len(fr0) - 5, a.n_frames)]
    fig, ax = plt.subplots(1, a.n_frames, figsize=(4.2 * a.n_frames, 4.4))
    ax = np.atleast_1d(ax)
    for col, ix in enumerate(idxs):
        absf = int(fr0[ix]); cap.set(cv2.CAP_PROP_POS_FRAMES, absf); ok, bgr = cap.read()
        if not ok:
            ax[col].axis("off"); continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pts = []
        for mk, (fr, A) in zip(["o", "x"], flies):
            v = A[ix, a.n_kp:, :3]
            if np.isnan(v).all():
                continue
            uv = proj(v); pts.append(uv)
            ax[col].scatter(uv[~iswing, 0], uv[~iswing, 1], s=9, c="cyan", marker=mk, linewidths=0.5)
            ax[col].scatter(uv[iswing, 0], uv[iswing, 1], s=11, c="red", marker=mk, linewidths=0.6)
        allp = np.vstack(pts) if pts else np.array([[img.shape[1] / 2, img.shape[0] / 2]])
        x0, y0 = allp.min(0) - 30; x1, y1 = allp.max(0) + 30
        ax[col].set_xlim(max(0, x0), min(img.shape[1], x1)); ax[col].set_ylim(min(img.shape[0], y1), max(0, y0))
        ax[col].imshow(img); ax[col].set_title(f"frame {absf}"); ax[col].axis("off")
    fig.suptitle(f"{os.path.basename(a.pred_dir)} dense pose ({camname}) — body/leg=cyan, wing=red; fly0=o fly1=x")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=110); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
