"""Render the real `viz sidebyside` for a BASELINE and a PRIOR fit, stacked.

The user's question is "how different is this from the raw data", so the
comparison has to include the raw data: the sidebyside left panel is the actual
camera frame + SAM mask + ViTPose 2D skeleton, and the right panel is the fitted
model through that same camera's own (orthographic/telecentric) calibration.
Stacking the baseline row above the prior row keeps the left panels identical,
so any difference between the rows is the fit and nothing else.

HOW: `viz sidebyside --right rigcam` reads three things from the fly dir --
`stac_ik.h5:qpos`, `qpos_refined.npz:qpos`, and `outputs.h5:kp3d_mm` (the FITTED
site positions, from which it recovers the model->world similarity). Patching
qpos alone would leave kp3d_mm describing the OLD pose and the recovered
similarity would misalign the model against the video. So kp3d_mm is recomputed
here by forward kinematics at the new qpos, mapped into mm with the per-frame
`bridge_s/R/t` stored alongside.

SELF-CHECK: the same FK+bridge math is first run on the BASELINE qpos and
compared against the stored kp3d_mm. If it cannot reproduce what is already on
disk, the transform is being applied wrongly and the patched render would be
quietly misaligned, so the script refuses to continue.

`mesh_mm` is NOT recomputed -- it is only used by `--right reproj`, and this
renders `--right rigcam`.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def fk_sites_mm(xml, offsets, kp_names, qpos, bridge_s, bridge_R, bridge_t):
    """Fitted marker sites at `qpos`, expressed in the same mm frame as kp3d_mm."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    sn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    for i, n in enumerate(kp_names):
        s = f"tracking[{n}]"
        if s in sn:
            m.site_pos[sn.index(s)] = offsets[i]
    idx = [sn.index(f"tracking[{n}]") if f"tracking[{n}]" in sn else -1 for n in kp_names]
    out = np.full((len(qpos), len(kp_names), 3), np.nan)
    for t in range(len(qpos)):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        p = np.stack([d.site_xpos[i] if i >= 0 else np.full(3, np.nan) for i in idx])
        out[t] = bridge_s[t] * (bridge_R[t] @ p.T).T + bridge_t[t]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly-dir", required=True, help="source bout fly dir")
    ap.add_argument("--npz", default=None, help="rest_ab_*.npz with q_0.0 and q_<weight>")
    ap.add_argument("--weight", type=float, default=None)
    ap.add_argument("--variant", action="append", default=[],
                    help="LABEL=/path/to/stac_ik.h5 (repeatable). Each variant supplies its "
                         "OWN qpos AND offsets -- offsets differ between fits, and they move "
                         "the fitted marker positions, so they cannot be taken from the source.")
    ap.add_argument("--bout", type=int, required=True)
    ap.add_argument("--fly", type=int, required=True)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--camera", default="track1")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--work", required=True, help="scratch dir for the patched trees")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import cv2
    import h5py
    from viz.config import resolve_body_model_xml
    from viz.core.io import write_video

    src = args.fly_dir
    variants = {}          # label -> (qpos, offsets|None, bridge|None)
    if args.variant:
        for spec in args.variant:
            lab, _, path = spec.partition("=")
            with h5py.File(path, "r") as f:
                q_v, off_v = f["qpos"][()], f["offsets"][()]
            # Each fit has its OWN model->world bridge (qpos_refined.npz, written
            # by the polish stage). Using the source's bridge for a variant whose
            # 3D differs would misalign that variant against the video, which is
            # the whole point of the comparison.
            sib = os.path.join(os.path.dirname(path), "qpos_refined.npz")
            br = None
            if os.path.exists(sib):
                z = np.load(sib)
                if all(k in z for k in ("bridge_s", "bridge_R", "bridge_t")):
                    br = (z["bridge_s"], z["bridge_R"], z["bridge_t"])
            variants[lab] = (q_v, off_v, br)
    else:
        z = np.load(args.npz, allow_pickle=True)
        variants["BASELINE (no prior)"] = (z["q_0.0"], None, None)
        variants[f"REST PRIOR w={args.weight:g}"] = (z[f"q_{args.weight}"], None, None)
    if len(variants) != 2:
        raise SystemExit(f"need exactly 2 variants to stack, got {len(variants)}")

    with h5py.File(os.path.join(src, "stac_ik.h5"), "r") as f:
        q_full = f["qpos"][()]
        offsets = f["offsets"][()]
        kp_names = [x.decode() for x in f["kp_names"][()]]
        cfgtxt = f["config"][()].decode()
    xml = resolve_body_model_xml(re.search(r"MJCF_PATH:\s*(\S+)", cfgtxt).group(1))
    qref = dict(np.load(os.path.join(src, "qpos_refined.npz")))
    with h5py.File(os.path.join(src, "outputs.h5"), "r") as f:
        kp3d_mm0 = f["kp3d_mm"][()]

    bs, bR, bt = qref["bridge_s"], qref["bridge_R"], qref["bridge_t"]

    # --- self-check: reproduce the stored kp3d_mm from the baseline qpos ---
    chk = fk_sites_mm(xml, offsets, kp_names, q_full[:40], bs[:40], bR[:40], bt[:40])
    err = np.nanmedian(np.linalg.norm(chk - kp3d_mm0[:40], axis=-1))
    span = np.nanmax(kp3d_mm0[:40]) - np.nanmin(kp3d_mm0[:40])
    print(f"self-check: FK+bridge reproduces stored kp3d_mm to {err:.4f} mm "
          f"({100*err/span:.2f}% of the {span:.2f} mm bounding span)")
    if err > 0.05 * span:
        raise SystemExit("FK+bridge does NOT reproduce the stored kp3d_mm; refusing to "
                         "patch, the render would be silently misaligned.")

    rows = []
    for vi, (label, (q_ab, off_ab, br_ab)) in enumerate(variants.items()):
        slug = "".join(c if c.isalnum() else "_" for c in label)[:28] or f"v{vi}"
        off_use = offsets if off_ab is None else off_ab
        bs_v, bR_v, bt_v = (bs, bR, bt) if br_ab is None else br_ab
        run = os.path.join(args.work, slug)
        fly_dst = os.path.join(run, "bouts", f"bout_{args.bout:05d}", f"fly{args.fly}")
        os.makedirs(fly_dst, exist_ok=True)
        for fn in os.listdir(src):
            s, dst = os.path.join(src, fn), os.path.join(fly_dst, fn)
            # qpos_wingfit.npz is deliberately NOT carried across. Each arm's
            # pose is the qpos_refined.npz patched below; symlinking the wing
            # fit into both arms let `viz sidebyside` pick it for BOTH and the
            # comparison silently showed nothing.
            if fn == "qpos_wingfit.npz":
                continue
            if os.path.isfile(s) and not os.path.exists(dst):
                (shutil.copy2 if fn in ("stac_ik.h5", "outputs.h5", "qpos_refined.npz")
                 else os.symlink)(s, dst)

        q = q_full.copy()
        n = min(len(q_ab), len(q))
        q[:n] = q_ab[:n]
        kp3d = kp3d_mm0.copy()
        kp3d[:n] = fk_sites_mm(xml, off_use, kp_names, q[:n],
                               bs_v[:n], bR_v[:n], bt_v[:n])

        with h5py.File(os.path.join(fly_dst, "stac_ik.h5"), "r+") as f:
            f["qpos"][...] = q
            f["offsets"][...] = off_use
        with h5py.File(os.path.join(fly_dst, "outputs.h5"), "r+") as f:
            f["qpos"][...] = q
            f["kp3d_mm"][...] = kp3d
            # This copy now holds the PATCHED pose, so its provenance stamp must
            # say so. Leaving the source run's `pose_source` in place would make
            # the file claim a wing-fit pose it no longer contains, and
            # viz.core.io.load_qpos trusts that stamp to decide what to draw.
            if "pose_source" in f:
                del f["pose_source"]
            f["pose_source"] = np.asarray("none").astype("S")
        d2 = dict(qref)
        d2["qpos"] = q
        if br_ab is not None:
            d2["bridge_s"], d2["bridge_R"], d2["bridge_t"] = br_ab
        np.savez(os.path.join(fly_dst, "qpos_refined.npz"), **d2)

        out_mp4 = os.path.join(args.work, f"sbs_{slug}.mp4")
        cmd = [sys.executable, "-m", "viz", "sidebyside", "--run", run,
               "--bout", str(args.bout), "--fly", str(args.fly), "--n", str(args.n),
               "--camera", args.camera, "--conf", str(args.conf),
               "--session-dir", args.session_dir,
               "--predictions-dir", args.predictions_dir,
               "--start-frame", str(args.start_frame), "--fps", str(args.fps),
               "--right", "rigcam",
               # Each arm IS the qpos_refined.npz patched above; pin the render
               # to it rather than let 'auto' choose.
               "--pose", "refined",
               "--out", out_mp4]
        print("+", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           env={**os.environ, "MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu"})
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            raise SystemExit(f"sidebyside failed for {slug}")
        rows.append((label, out_mp4))

    # --- stack the two rows into one video ---
    caps = [(lab, cv2.VideoCapture(p)) for lab, p in rows]
    frames = []
    while True:
        imgs = []
        for lab, c in caps:
            ok, im = c.read()
            if not ok:
                break
            cv2.rectangle(im, (0, 0), (430, 30), (0, 0, 0), -1)
            cv2.putText(im, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            imgs.append(im)
        if len(imgs) != len(caps):
            break
        w = min(i.shape[1] for i in imgs)
        imgs = [i[:, :w] for i in imgs]
        frames.append(np.vstack([imgs[0],
                                 np.full((3, w, 3), 60, np.uint8), imgs[1]]))
    for _, c in caps:
        c.release()
    if not frames:
        raise SystemExit("no frames read back from the two sidebyside renders")

    tag = (os.path.basename(args.npz).replace("rest_ab_", "").replace(".npz", "")
           if args.npz else "variants")
    vp = os.path.join(args.out, f"sbs_prior_{tag}.mp4")
    # macro_block_size=2: libx264+yuv420p REQUIRES even dimensions, and an odd
    # size makes the encoder die with a broken pipe and silently fall back to
    # cv2 mp4v, which VSCode cannot play.
    write_video(vp, frames, fps=args.fps, macro_block_size=2)
    png = os.path.join(args.out, f"sbs_prior_{tag}.png")
    cv2.imwrite(png, frames[len(frames) // 2])
    print(f"wrote {vp}\nwrote {png}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
