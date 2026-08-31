"""Draw the STAC tracking sites on the MuJoCo mesh, by name.

Answers "where does the model think each landmark sits on the body?" -- and in
particular whether a wing landmark sits on the wing MARGIN or along its long
axis, which is what decides whether blade roll is observable at all. Measured on
bout_00001 fly0: the detector's V12 sits 0.0070 from the wing's long axis where
the model's site sits at 0.0357, so one degree of roll displaces it by 0.00062
against a marker residual of ~0.011 -- roll is unresolvable to about +/-18 deg.

Two marker sets are drawn wherever `--fly-dir` is given:
  STAR  the XML's own site_pos -- where the body model says the landmark is
  CROSS the FITTED offsets from that bout's stac_ik.h5 -- where STAC moved it
        to match the detector
A large star-to-cross gap on a wing landmark is the model and the detector
disagreeing about which anatomical point the name refers to.

Cameras are ORTHOGRAPHIC, matching the rig's telecentric lenses, so positions
can be read off the image without perspective foreshortening; site pixels come
from viz.core.mjcam.project_with_mujoco_camera, the same projector the rigcam
sidebyside panel is verified with.

Pose defaults to the model's SPRING REST (wings folded back and flat, wing_yaw
+85.94 deg = its joint stop), which is the canonical anatomical pose; pass
`--frame N` to use a fitted frame from the bout instead.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

WING_JOINTS = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
               "wing_yaw_right", "wing_roll_right", "wing_pitch_right")


def cam_quat(view):
    """(pos_dir, quat) for an orthographic camera. MuJoCo looks along -z_cam, +y_cam up."""
    from viz.core.mjcam import mat_to_quat
    if view == "dorsal":       # look down -z, fly's +y (left) up in image
        R = np.eye(3)
        d = np.array([0.0, 0.0, 1.0])
    elif view == "lateral":
        # The fly's long axis is world X (head at +x), so a SIDE view looks
        # along -Y, not -X -- looking along -X gives a head-on view.
        # cols = (right, up, back); MuJoCo views along -back.
        R = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        d = np.array([0.0, 1.0, 0.0])
    else:                      # "front": look along -x, world +z up
        R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        d = np.array([1.0, 0.0, 0.0])
    return d, mat_to_quat(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=None, help="body model XML (default: from --fly-dir config)")
    ap.add_argument("--fly-dir", default=None, help="bout fly dir, to also draw FITTED offsets")
    ap.add_argument("--frame", type=int, default=None, help="use this fitted frame's pose")
    ap.add_argument("--groups", default="wing",
                    help="'wing', 'all', or comma-separated keypoint name substrings")
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import cv2
    import h5py
    import mujoco
    from viz.core.mjcam import project_with_mujoco_camera

    offsets = kp_names = qpos = None
    xml = args.xml
    if args.fly_dir:
        with h5py.File(os.path.join(args.fly_dir, "stac_ik.h5"), "r") as f:
            offsets = f["offsets"][()]
            kp_names = [x.decode() for x in f["kp_names"][()]]
            cfg = f["config"][()].decode()
            if args.frame is not None:
                qpos = f["qpos"][()][args.frame]
        xml = xml or re.search(r"MJCF_PATH:\s*(\S+)", cfg).group(1)
    if xml is None:
        raise SystemExit("need --xml or --fly-dir")

    spec = mujoco.MjSpec.from_file(xml)
    spec.visual.global_.offwidth = args.size
    spec.visual.global_.offheight = args.size
    cam = spec.worldbody.add_camera()
    cam.name = "ortho"
    cam.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    cam.fovy = 1.0
    m = spec.compile()
    d = mujoco.MjData(m)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ortho")

    sn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    track = [(i, n[len("tracking["):-1]) for i, n in enumerate(sn)
             if n and n.startswith("tracking[")]
    if args.groups == "wing":
        sel = [(i, n) for i, n in track if "Wing" in n]
    elif args.groups == "all":
        sel = track
    else:
        pats = args.groups.split(",")
        sel = [(i, n) for i, n in track if any(p in n for p in pats)]
    if not sel:
        raise SystemExit(f"no tracking sites matched --groups {args.groups}")
    print(f"{len(sel)} sites: {[n for _, n in sel]}")

    xml_pos = {n: m.site_pos[i].copy() for i, n in sel}
    if offsets is not None:
        for i, n in sel:
            if n in kp_names:
                m.site_pos[i] = offsets[kp_names.index(n)]
    fit_pos = {n: m.site_pos[i].copy() for i, n in sel}

    # pose
    mujoco.mj_resetData(m, d)
    if qpos is not None:
        d.qpos[:] = qpos
        pose_label = f"fitted frame {args.frame}"
    else:
        d.qpos[:] = m.qpos0
        for j in WING_JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid >= 0:
                a = int(m.jnt_qposadr[jid])
                d.qpos[a] = m.qpos_spring[a]
        pose_label = "model SPRING REST (wings folded)"
    mujoco.mj_forward(m, d)

    # site world positions for both marker sets, at this pose
    def world_of(local_map):
        out = {}
        for i, n in sel:
            bid = m.site_bodyid[i]
            R = d.xmat[bid].reshape(3, 3)
            out[n] = d.xpos[bid] + R @ local_map[n]
        return out

    W_xml, W_fit = world_of(xml_pos), world_of(fit_pos)

    rend = mujoco.Renderer(m, height=args.size, width=args.size)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    centre = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thorax")].copy()
    tips = [n for _, n in sel if "_V1" in n]
    tip_centre = (np.mean([W_fit[n] for n in tips], axis=0) if tips else centre)

    panels = []
    for view, fovy, focus in (("dorsal", 0.75, centre), ("lateral", 0.75, centre),
                              ("dorsal", 0.20, tip_centre)):
        dirv, quat = cam_quat(view)
        pos = focus + dirv * 5.0
        m.cam_pos[cid] = pos
        m.cam_quat[cid] = quat
        m.cam_fovy[cid] = fovy   # orthographic: fovy = FULL visible height in
                                 # length units; the projection itself is set on
                                 # the spec at compile time (mjPROJ_ORTHOGRAPHIC)
        mujoco.mj_forward(m, d)          # cam_xpos/xmat derive here -- must re-forward
        rend.update_scene(d, camera=cid, scene_option=opt)
        img = np.ascontiguousarray(rend.render()[:, :, ::-1])   # RGB -> BGR

        for label, WP, marker, col in (("XML site", W_xml, "star", (80, 80, 230)),
                                       ("FITTED offset", W_fit, "cross", (60, 200, 255))):
            if offsets is None and label == "FITTED offset":
                continue
            names = [n for _, n in sel]
            px = project_with_mujoco_camera(np.stack([WP[n] for n in names]),
                                            pos, quat, fovy, (args.size, args.size))
            for n, (u, v) in zip(names, px):
                if not (0 <= u < args.size and 0 <= v < args.size):
                    continue
                u, v = int(u), int(v)
                if marker == "star":
                    cv2.drawMarker(img, (u, v), col, cv2.MARKER_STAR, 18, 2)
                else:
                    cv2.drawMarker(img, (u, v), col, cv2.MARKER_CROSS, 16, 2)
                if fovy < 0.5:      # only label the zoomed panel, else it is a mess
                    cv2.putText(img, n, (u + 10, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, col, 1, cv2.LINE_AA)
        tag = f"{view.upper()}  fovy={fovy:g}" + ("  [ZOOM]" if fovy < 0.5 else "")
        cv2.putText(img, tag, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        panels.append(img)

    grid = np.hstack([np.hstack([p, np.full((args.size, 3, 3), 60, np.uint8)])
                      for p in panels])[:, :-3]
    bar = np.full((44, grid.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, f"tracking sites on the MuJoCo mesh  |  {pose_label}  |  "
                     f"STAR = XML site_pos, CROSS = fitted offset", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    out = os.path.join(args.out, f"tracking_sites_{args.groups.replace(',', '-')}.png")
    cv2.imwrite(out, np.vstack([bar, grid]))
    print("wrote", out)

    if offsets is not None:
        print(f"\n{'site':14}{'|XML - FITTED| (model units)':>32}")
        for _, n in sel:
            print(f"  {n:12s}{np.linalg.norm(xml_pos[n] - fit_pos[n]):>30.4f}")


if __name__ == "__main__":
    main()
