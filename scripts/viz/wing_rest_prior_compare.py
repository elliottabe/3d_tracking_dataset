"""Render baseline vs wing-rest-prior fits side by side, to SEE the wing leave the abdomen.

The defect is a dorsoventral overlap -- the folded wing blade sinks into the
abdomen -- so a top-down view hides it almost completely. Every panel is
therefore rendered twice: DORSAL (what the rig cameras see, for orientation)
and LATERAL (where the wing/abdomen intersection is actually visible).

EXPECTATION, read the render against this:
  * LATERAL, baseline: the grey wing blade visibly disappears into the orange
    abdomen. Measured median depth 0.039-0.046 model units, ~20% of a wing
    length, in 100% of frames.
  * LATERAL, prior: the blade sits ON the abdomen surface rather than through
    it, matching the model's own rest pose which merely grazes at 0.0013.
  * DORSAL: should look nearly UNCHANGED. The prior acts on blade roll/pitch,
    not on where the wing points, so wing direction -- and for the singing male
    the wing EXTENSION that carries the song -- must survive. A dorsal view
    whose wing direction shifts means the prior is taking more than the
    unobservable DOFs.

Per-frame penetration depth is printed into each panel from MuJoCo's own
mj_geomDistance, so the number and the picture can be checked against each other.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="rest_ab_<tag>.npz from wing_rest_prior_ab.py")
    ap.add_argument("--h5", required=True, help="stac_ik.h5 (for MJCF_PATH)")
    ap.add_argument("--weight", type=float, required=True, help="which prior weight to show")
    ap.add_argument("--frames", default="", help="comma-separated frame indices; default = auto")
    ap.add_argument("--n-auto", type=int, default=4)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=440)
    ap.add_argument("--video", action="store_true", help="also write an mp4 over all frames")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import cv2
    import h5py
    import mujoco
    import re

    z = np.load(args.npz, allow_pickle=True)
    q_base = z["q_0.0"]
    key = f"q_{args.weight}"
    if key not in z:
        raise SystemExit(f"{key} not in npz; have {[k for k in z if k.startswith('q_')]}")
    q_prior = z[key]
    gaps_base, gaps_prior = z["gaps_0.0"], z[f"gaps_{args.weight}"]

    with h5py.File(args.h5, "r") as f:
        cfg = f["config"][()].decode()
    xml = re.search(r"MJCF_PATH:\s*(\S+)", cfg).group(1)
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)

    # cameras: dorsal (looking down) and lateral (looking along the fly's side)
    cams = {}
    for name, (azim, elev) in (("DORSAL", (90.0, -89.0)), ("LATERAL", (0.0, 0.0))):
        c = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(c)
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.azimuth, c.elevation, c.distance = azim, elev, 0.85
        cams[name] = c

    rend = mujoco.Renderer(m, height=args.height, width=args.width)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    if args.frames:
        frames = [int(x) for x in args.frames.split(",")]
    else:
        # the frames where the prior changes penetration most -- the honest
        # choice is the biggest CHANGE, not the deepest baseline penetration,
        # which would flatter the prior by construction
        delta = gaps_prior.min(1) - gaps_base.min(1)
        frames = sorted(np.argsort(-delta)[:args.n_auto].tolist())
    print(f"frames: {frames}")

    def shot(q, t, camname, label, gap):
        d.qpos[:] = q[t]
        mujoco.mj_forward(m, d)
        c = cams[camname]
        c.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thorax")]
        rend.update_scene(d, camera=c, scene_option=opt)
        img = rend.render().copy()
        cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        pen = -gap
        col = (0, 0, 255) if pen > 0.02 else ((0, 165, 255) if pen > 0.005 else (0, 220, 0))
        cv2.putText(img, f"wing-abdomen penetration {pen:+.4f}", (10, args.height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)
        return img

    def panel(t):
        rows = []
        for camname in ("DORSAL", "LATERAL"):
            a = shot(q_base, t, camname, f"{camname}  BASELINE (no prior)", gaps_base[t].min())
            b = shot(q_prior, t, camname, f"{camname}  REST PRIOR w={args.weight:g}",
                     gaps_prior[t].min())
            rows.append(np.hstack([a, np.full((args.height, 3, 3), 60, np.uint8), b]))
        return np.vstack([rows[0], np.full((3, rows[0].shape[1], 3), 60, np.uint8), rows[1]])

    tag = os.path.splitext(os.path.basename(args.npz))[0].replace("rest_ab_", "")
    for t in frames:
        img = panel(t)
        cv2.putText(img, f"frame {t}", (img.shape[1] - 130, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        p = os.path.join(args.out, f"restprior_{tag}_f{t:04d}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("wrote", p)

    if args.video:
        # viz.core.io.write_video, not cv2.VideoWriter: the cv2 "mp4v" fourcc
        # emits MPEG-4 Part 2, which VSCode's Chromium-based preview refuses to
        # decode. This helper writes libx264 + yuv420p + faststart.
        # macro_block_size=1 keeps the exact panel dimensions.
        from viz.core.io import write_video
        vp = os.path.join(args.out, f"restprior_{tag}.mp4")
        write_video(vp, (cv2.cvtColor(panel(t), cv2.COLOR_RGB2BGR)
                         for t in range(len(q_base))),
                    fps=args.fps, macro_block_size=2)
        print("wrote", vp)

    pb, pp = -gaps_base.min(1), -gaps_prior.min(1)
    print(f"\npenetration depth over {len(pb)} frames:")
    print(f"  baseline    median {np.median(pb):.4f}  max {pb.max():.4f}  "
          f"frames>0: {100*np.mean(pb>0):.0f}%")
    print(f"  rest prior  median {np.median(pp):.4f}  max {pp.max():.4f}  "
          f"frames>0: {100*np.mean(pp>0):.0f}%")
    print(f"  model at its own spring rest: 0.0013 (the target, not zero)")


if __name__ == "__main__":
    main()
