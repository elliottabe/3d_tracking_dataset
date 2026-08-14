#!/usr/bin/env python3
"""Act 3 -- body scale, then root alignment (the solver's first two real stages).

This act renders SNAPSHOTS `stage_ik.py` recorded from an actual STAC solve
(`06_stages.npz`) -- it does not invent motion. Two measured facts drive the
whole timeline (do not restate the pre-measurement storyboard):

1. **Body scale is applied to the KEYPOINTS, not the mesh.** Scale comes from
   `preprocess_keypoints_for_ik.compute_shared_scale` (Umeyama trunk scale;
   `shared_scale = 0.1261` for this clip). The model geometry is never
   touched -- `qpos_scaled` is bit-identical to `qpos_default` in
   `06_stages.npz`. So this act animates the KEYPOINT CLOUD shrinking onto a
   fixed-size mesh, never the mesh growing/shrinking.
2. **`root_optimization` does NOT rotate.** `configs/anatomy/v1.yaml` sets
   `TRUNK_OPTIMIZATION_KEYPOINTS: {}`, so `root_optimization`'s keypoint-fit
   loss is uniformly zero and only the manual root-translation it also
   performs has any effect. Measured on `06_stages.npz`: `qpos_root[3:7]`
   (the free-joint quaternion) is bit-identical to `qpos_default[3:7]`
   (`[1,0,0,0]` both), and `qpos_root[7:]` (all joint DOF) is bit-identical
   to `qpos_default[7:]` too. Only `qpos_root[:3]` (the free-joint
   translation) differs. So this act is scale + TRANSLATION only -- no
   rotation. Orienting is Act 4's `pose_optimization` (34.73 deg measured
   there), not this one.

Because those two stages are genuinely translation-only here, `_slerp_qpos`
SLERPs a quaternion that never actually changes over this act. It is used
anyway (rather than a plain lerp of `qpos[:3]`/`qpos[7:]` only) so the same
helper is correct for Act 4, where `pose_optimization` produces a real
34.73 deg rotation and a lerped quaternion would denormalise and visibly
tumble.

WORLD-COORDINATE STORY (measured, not staged): the raw triangulated keypoints
(`04_kp3d_filt.npz`) sit in an arena-relative mm frame far from the model's
own origin (frame 450 mean ~[13.15, 1.99, 1.19]), while the model's rest qpos
places its root at the world origin. So at f=0 the mesh and the (unscaled)
cloud are ALREADY far apart -- that is the real triangulated position, not a
staged offset. Scaling the cloud by `shared_scale` (~0.126) shrinks it toward
the origin (frame-450 scaled mean ~[1.66, 0.25, 0.15]) which happens to land
close to where `root_optimization` then translates the mesh
(`qpos_root[:3]` = [1.65, 0.25, 0.26]) -- because `root_optimization` sets
the root translation to the (scaled) root keypoint itself. The timeline
below follows this real geometry: cloud shrinks toward the mesh first
(rescale), then the mesh translates the rest of the way to meet it
(root_optimization).

CAMERA: reuses the exact lookat/spread/distance construction
`stage_ik.py:qc_stages`'s wide diagnostic camera uses (never the model's
`hero` camera, which frames the mesh only and loses the far-away raw cloud
entirely), generalised to recompute every frame (mesh_ctr and the cloud both
move over the act). Azimuth/elevation are FIXED (not orbiting) -- a moving
viewer azimuth would risk being misread, frame-to-frame, as the mesh itself
rotating, which is exactly what this act must not depict. Only lookat and
distance adapt, so both the mesh and the cloud stay in frame as they move;
the mesh's own qpos quaternion is bit-identical at every frame in this act
(asserted at runtime, see `render_act3`). Keypoint sphere radius is scaled
with the camera's current `spread` (not a fixed absolute size) purely so the
cloud stays legible whether the shot is wide (f=0, raw cloud ~13 mm from the
mesh) or tight (f=599, cloud+mesh co-located) -- this changes how big we DRAW
markers, never the mesh geometry.

Timeline (600 frames, 30 fps):
  f   0-119  mesh fades in at qpos_default (alpha ramp), cloud already fully
             visible at its RAW (unscaled) position/size -- both in frame,
             mesh visibly offset from the cloud, cloud the wrong SIZE for it.
  f 120-299  cloud shrinks from raw scale (factor 1.0) to `shared_scale`
             (0.1261) about the world origin -- an exact `kp3d * factor`
             lerp of the interpolation factor, since scaling is linear; the
             scale factor is burned in on screen. Mesh does not move.
  f 300-539  qpos interpolates qpos_default -> qpos_root (`_slerp_qpos`):
             translation only. Cloud stays at shared_scale (fixed). Residual
             ticks down 1.710 -> 0.118 mm.
  f 540-599  hold: mesh and cloud co-located, residual 0.118 mm.

EXPECTATION: f=0 mesh+cloud both visible, mesh offset from cloud, cloud
wrong SIZE; f=299 cloud matches model scale, still displaced; f=539 cloud
and mesh co-located, residual has fallen 13.404 -> 0.118 mm. The mesh does
NOT change size and does NOT rotate at any point.
FALSIFICATION: a visibly rotating mesh means the act is animating something
the solver did not do; a cloud leaving frame means the wrong (hero) camera
was used.

Colours come from viz/core/colors.py via draw.py / stage_ik.py's `_GROUP_RGB`
-- never invented here.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))
sys.path.insert(0, str(_REPO / "stac-mjx"))

from scripts.viz.ik_explainer import clip_io, draw               # noqa: E402
from scripts.viz.ik_explainer.stage_ik import (                  # noqa: E402
    _add_sphere, _tracking_site_map, _GROUP_RGB,
)
from viz.core.colors import PALETTE, keypoint_groups              # noqa: E402

# --- canvas / timeline ------------------------------------------------------
CANVAS_W, CANVAS_H = 1920, 1080
N_OUT = 600
FADE_START, FADE_END = 0, 119            # mesh alpha ramp (inclusive)
SCALE_START, SCALE_END = 120, 299        # cloud shrinks to shared_scale
ROOT_START, ROOT_END = 300, 539          # qpos_default -> qpos_root
HOLD_START = 540

XML_PATH = _REPO / "models" / "fruitfly_v1" / "fruitfly_v1_free.xml"

# --- viewer (presentation) camera -------------------------------------------
# Same lookat/spread/distance construction as stage_ik.py:qc_stages's wide
# diagnostic camera (azimuth/elevation base values match it, 120/-20), with
# azimuth animated for a ~40 deg sweep across the act -- see module docstring.
# Fixed (not orbiting): a moving VIEWER azimuth risks being misread, in a
# frame-by-frame diff, as the MESH rotating -- exactly the thing this act
# must not depict (root_optimization does not rotate; see module docstring).
# lookat/distance still adapt every frame (mesh_ctr and the cloud genuinely
# move), which is not an orbit, just keeping both in frame.
AZ_START, AZ_SWEEP = 120.0, 0.0
ELEV = -20.0

# Mesh fade-in floor: f=0 must already show a (translucent) mesh, per the
# task's own falsification check ("f=0: mesh AND cloud both visible") -- a
# fade that starts at literal alpha=0 would make f=0 mesh-invisible.
MESH_ALPHA_MIN = 0.45


def _smoothstep(p):
    p = np.clip(p, 0.0, 1.0)
    return 3 * p ** 2 - 2 * p ** 3


def _slerp_qpos(qa, qb, t):
    """Interpolate a MuJoCo free-joint qpos: `qpos[:3]` (translation) lerp,
    `qpos[3:7]` (w,x,y,z quaternion) SLERP, `qpos[7:]` (hinge/joint DOF) lerp.

    A plain lerp of the quaternion denormalises and visibly tumbles; SLERP
    keeps it a unit quaternion throughout. In THIS act the quaternion never
    actually changes (qa[3:7] == qb[3:7] == identity, per the module
    docstring's measured fact), so SLERP is a no-op here -- it is used anyway
    so this helper is directly reusable for Act 4's real 34.73 deg rotation.
    """
    qa, qb = np.asarray(qa, np.float64), np.asarray(qb, np.float64)
    pos = (1 - t) * qa[:3] + t * qb[:3]
    quat_a_xyzw = np.roll(qa[3:7], -1)     # (w,x,y,z) -> (x,y,z,w) for scipy
    quat_b_xyzw = np.roll(qb[3:7], -1)
    rots = Rotation.from_quat(np.stack([quat_a_xyzw, quat_b_xyzw]))
    slerp = Slerp([0.0, 1.0], rots)
    quat_t_xyzw = slerp(t).as_quat()
    quat_t_wxyz = np.roll(quat_t_xyzw, 1)  # back to (w,x,y,z)
    joints = (1 - t) * qa[7:] + t * qb[7:]
    return np.concatenate([pos, quat_t_wxyz, joints])


def _wide_camera(mesh_ctr, cloud_pts, model_extent, azimuth):
    """The wide diagnostic camera: same lookat/spread/distance formula
    stage_ik.py's qc_stages() uses for its wide-camera row (never the
    model's `hero` camera, which frames the mesh only), generalised to be
    recomputed every frame since both mesh_ctr and the cloud move here.
    """
    finite = cloud_pts[np.all(np.isfinite(cloud_pts), axis=-1)]
    cloud_ctr = finite.mean(axis=0)
    lookat = (mesh_ctr + cloud_ctr) / 2.0
    spread = max(
        float(np.max(np.linalg.norm(finite - lookat, axis=-1))),
        float(np.linalg.norm(mesh_ctr - lookat)),
        model_extent * 0.5,
    )
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = spread * 2.6
    cam.azimuth, cam.elevation = azimuth, ELEV
    return cam, spread


def _stage_at(f, residual_default, residual_scaled, residual_root,
              qpos_default, qpos_root, shared_scale):
    """Return (stage_label, residual_mm, scale_factor, qpos, mesh_alpha) for
    output frame `f`. Pure function of the recorded stage snapshots -- no
    numbers invented here, only interpolated between measured stage values.
    """
    if f <= FADE_END:
        p = _smoothstep((f - FADE_START) / float(FADE_END - FADE_START))
        # Floor at MESH_ALPHA_MIN rather than 0: the self-review requires
        # f=0 to already show the mesh (translucent is fine, invisible is
        # not), so this is a build from translucent -> fully opaque, not
        # literally transparent -> opaque.
        alpha = MESH_ALPHA_MIN + (1.0 - MESH_ALPHA_MIN) * p
        return "default (rest pose)", residual_default, 1.0, qpos_default, alpha
    if f <= SCALE_END:
        p = _smoothstep((f - SCALE_START) / float(SCALE_END - SCALE_START))
        resid = (1 - p) * residual_default + p * residual_scaled
        factor = (1 - p) * 1.0 + p * shared_scale
        return "rescale", resid, factor, qpos_default, 1.0
    if f <= ROOT_END:
        p = _smoothstep((f - ROOT_START) / float(ROOT_END - ROOT_START))
        resid = (1 - p) * residual_scaled + p * residual_root
        qpos = _slerp_qpos(qpos_default, qpos_root, p)
        return "root_optimization", resid, shared_scale, qpos, 1.0
    return "root_optimization (held)", residual_root, shared_scale, qpos_root, 1.0


def render_act3(clip: str = clip_io.CLIP_DEFAULT) -> Path:
    dirs = clip_io.out_dirs(clip)

    with np.load(dirs["predictions"] / "06_stages.npz", allow_pickle=True) as z:
        qpos_default = np.asarray(z["qpos_default"], np.float64)
        qpos_root = np.asarray(z["qpos_root"], np.float64)
        residual_mm = np.asarray(z["residual_mm"], np.float64)
        stage_names = [str(s) for s in z["stage_names"]]
        kp_names = [str(n) for n in z["kp_names"]]
        frame_for_stills = int(z["frame_for_stills"])
        shared_scale = float(z["shared_scale"])

    assert stage_names[:3] == ["default", "scaled", "root"], (
        f"unexpected 06_stages.npz stage_names order: {stage_names}")
    residual_default, residual_scaled, residual_root = (
        float(residual_mm[0]), float(residual_mm[1]), float(residual_mm[2]))

    # Measured-fact guard: root_optimization must not have rotated or moved
    # any joint DOF (module docstring point 2). If a future re-run of
    # stage_ik.py ever produces a rotating root_optimization, this act's
    # entire "scale + translation only" premise would be wrong -- fail loudly
    # rather than silently render the old (incorrect) story.
    if not np.allclose(qpos_default[3:], qpos_root[3:], atol=1e-9):
        raise ValueError(
            "qpos_default[3:] != qpos_root[3:] -- root_optimization rotated "
            "and/or moved joint DOF in this solve. Act 3's premise (scale + "
            "translation only, no rotation) no longer matches the recorded "
            "snapshots; refusing to render a mesh-rotation animation that "
            "contradicts the measured facts.")

    with np.load(dirs["predictions"] / "04_kp3d_filt.npz", allow_pickle=True) as z:
        kp3d = np.asarray(z["kp3d"], np.float64)
        kp3d_names = [str(n) for n in z["kp_names"]]
    if kp3d_names != kp_names:
        raise ValueError(
            "04_kp3d_filt.npz kp_names != 06_stages.npz kp_names -- refusing "
            "to mix keypoint orders (CLAUDE.md's keypoint-order bug class).")

    kp3d_raw_frame = kp3d[frame_for_stills]   # (50,3) mm, MODEL order, RAW (unscaled)
    print(f"[act3] frame_for_stills={frame_for_stills}, shared_scale={shared_scale:.4f}")
    print(f"[act3] residuals default/scaled/root = "
          f"{residual_default:.3f}/{residual_scaled:.3f}/{residual_root:.3f} mm")
    print(f"[act3] qpos_root - qpos_default (translation only, mm): "
          f"{(qpos_root[:3] - qpos_default[:3])}")

    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    grey = PALETTE["mesh"][0] / 255.0   # PALETTE["mesh"] is (200,200,200): BGR==RGB here
    mj_model.geom_rgba[:, :3] = grey

    site_map = _tracking_site_map(mj_model, kp_names)
    missing = [n for n in kp_names if n not in site_map]
    if missing:
        raise ValueError(f"tracking[...] site missing for keypoints: {missing}")
    body_site_idxs = np.asarray([site_map[n] for n in kp_names])
    groups = keypoint_groups(kp_names)
    group_of = {i: g for g, idxs in groups.items() for i in idxs}

    out_dir = dirs["frames"] / "act3_align"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for f in range(N_OUT):
        stage_label, resid, factor, qpos, mesh_alpha = _stage_at(
            f, residual_default, residual_scaled, residual_root,
            qpos_default, qpos_root, shared_scale,
        )
        mj_model.geom_rgba[:, 3] = mesh_alpha

        d = mujoco.MjData(mj_model)
        d.qpos[:] = qpos
        mujoco.mj_forward(mj_model, d)

        mesh_ctr = np.asarray(d.site_xpos[body_site_idxs]).mean(axis=0)
        cloud_pts = kp3d_raw_frame * factor   # linear scale about world origin

        azimuth = AZ_START + AZ_SWEEP * (f / float(N_OUT - 1))
        cam, spread = _wide_camera(mesh_ctr, cloud_pts, mj_model.stat.extent, azimuth)
        marker_r = max(spread * 0.03, mj_model.stat.extent * 0.006)

        with mujoco.Renderer(mj_model, height=CANVAS_H, width=CANVAS_W) as renderer:
            renderer.update_scene(d, camera=cam)
            scn = renderer.scene
            for i, p3 in enumerate(cloud_pts):
                if np.all(np.isfinite(p3)):
                    rgb = _GROUP_RGB.get(group_of.get(i), (1.0, 1.0, 1.0))
                    _add_sphere(scn, p3, np.array((*rgb, 1.0), np.float32), marker_r)
            img_rgb = np.ascontiguousarray(renderer.render())

        canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        canvas = draw.stage_title(canvas, "ACT 3", "Body scale, then root alignment")
        canvas = draw.label(canvas, f"stage: {stage_label}", (48, 150),
                             scale=0.6, color=(255, 255, 255))
        canvas = draw.label(canvas, f"residual: {resid:.3f} mm", (48, 182),
                             scale=0.6, color=(190, 255, 190))
        shown_factor = shared_scale if f > SCALE_END else factor
        canvas = draw.label(
            canvas, f"keypoint scale factor: {shown_factor:.4f} "
                    f"(Umeyama trunk fit, target shared_scale={shared_scale:.4f})",
            (48, 210), scale=0.5, color=(190, 190, 190))
        canvas = draw.label(
            canvas, "mesh size is fixed -- only the keypoint cloud is rescaled; "
                    "root_optimization translates the mesh, it does not rotate it",
            (48, 238), scale=0.45, color=(150, 150, 150))
        canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}", (48, CANVAS_H - 24),
                             scale=0.45, color=(150, 150, 150))

        cv2.imwrite(str(out_dir / f"f{f:05d}.png"), canvas)

    dt = time.time() - t0
    print(f"[act3] wrote {N_OUT} frames to {out_dir} in {dt:.1f} s "
          f"({dt / N_OUT:.3f} s/frame)")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    args = ap.parse_args()
    render_act3(args.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
