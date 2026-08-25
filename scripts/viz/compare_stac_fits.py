"""Before/after comparison of STAC fits: are the keypoints sitting on the body?

Renders the same frame from one or more `stac_ik.h5` outputs side by side, with
three overlays so the failure mode is visible rather than inferred:

  WHITE   observed keypoints (what was tracked, after the global scale)
  CYAN    fitted marker sites (where STAC put the markers)
  ORANGE  anatomical sites (where the XML puts them on the mesh)

If white and cyan agree but both sit away from the orange/mesh, the IK converged
but the marker offsets absorbed a size or proportion error -- markers float off
the body. That is invisible to residual and NaN checks.

The arena is attached for its headlight: the fly-only XML has no lights and
renders black. The fly is placed above the floor, not at the pipeline's -0.125
floor-alignment target, which would bury the abdomen.

Usage:
    python scripts/viz/compare_stac_fits.py \
        --fit baseline=/path/a/stac_ik.h5 --fit fixed=/path/b/stac_ik.h5 \
        --xml ../fruitfly_body_models/fruitfly_v2_3_ik/fruitfly_v2_3_ik.xml \
        --anatomy configs/anatomy/v2_3.yaml \
        --out /path/compare.png --frame 400
"""
from __future__ import annotations

import os

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

WHITE = [1.0, 1.0, 1.0, 1.0]
CYAN = [0.0, 0.9, 1.0, 1.0]
ORANGE = [1.0, 0.55, 0.0, 1.0]


def build_for(xml: Path, arena: Path, spawn_z: float, pre_h5=None):
    """Compose arena+fly, applying per-segment calibration when that arm used it."""
    fly = mujoco.MjSpec.from_file(str(xml))
    if pre_h5:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from utils.segment_calibration import read_segment_scales
        seg = read_segment_scales(pre_h5)
        if seg:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'stac-mjx'))
            from stac_mjx.rescale import rescale_per_segment
            rescale_per_segment(fly, seg)
            print(f'  (morphed {len(seg)} segments for this panel)')
    ar = mujoco.MjSpec.from_file(str(arena))
    ar.worldbody.add_frame(pos=[0, 0, spawn_z], quat=[1, 0, 0, 0]) \
        .attach_body(fly.body('thorax'), '', suffix='_fly')
    m = ar.compile()
    return m, mujoco.MjData(m)


def add_sphere(scene, pos, rgba, size):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0, 0]), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit', action='append', required=True,
                    metavar='LABEL=PATH', help='repeatable: label=stac_ik.h5')
    ap.add_argument('--seg', action='append', default=[],
                    metavar='LABEL=PREPROCESSED_H5',
                    help='repeatable: for arms that used per-segment '
                         'calibration, the preprocessed h5 holding '
                         'info/segment_scales. That arm is then rendered '
                         'against its own MORPHED model, so its anatomical '
                         'overlay is not misrepresented.')
    ap.add_argument('--xml', required=True, type=Path)
    ap.add_argument('--anatomy', required=True, type=Path)
    ap.add_argument('--arena', type=Path, default=None)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--frame', type=int, default=200)
    ap.add_argument('--spawn-z', type=float, default=0.35)
    ap.add_argument('--distance', type=float, default=0.55)
    ap.add_argument('--azimuth', type=float, default=90.0)
    ap.add_argument('--elevation', type=float, default=-8.0)
    ap.add_argument('--marker', type=float, default=0.0035)
    ap.add_argument('--height', type=int, default=820)
    ap.add_argument('--width', type=int, default=1040)
    args = ap.parse_args()

    arena = args.arena or args.xml.parent / 'floor.xml'
    segmap = dict(s.split('=', 1) for s in args.seg)

    cfg = OmegaConf.load(args.anatomy)
    init = {k: np.array([float(x) for x in str(v).split()])
            for k, v in cfg.model.KEYPOINT_INITIAL_OFFSETS.items()}
    pairs = cfg.model.KEYPOINT_MODEL_PAIRS

    panels = []
    for spec in args.fit:
        label, path = spec.split('=', 1)
        m, d = build_for(args.xml, arena, args.spawn_z, segmap.get(label))
        with h5py.File(path) as f:
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in np.asarray(f['kp_names'])]
            qpos = np.asarray(f['qpos'][args.frame])
            obs = np.asarray(f['kp_data'][args.frame]).reshape(-1, 3)
            fitted = np.asarray(f['marker_sites'][args.frame])

        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)

        # anatomical sites: XML local offsets on the fitted pose
        anat = []
        for n in names:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                    f'{pairs[n]}_fly')
            anat.append(d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ init[n])
        anat = np.array(anat)

        # the fit's world frame is the fly's own; shift overlays onto the render
        shift = anat.mean(0) - fitted.mean(0)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth, cam.elevation = args.azimuth, args.elevation
        cam.distance = args.distance
        cam.lookat[:] = anat.mean(0)

        with mujoco.Renderer(m, height=args.height, width=args.width) as r:
            r.update_scene(d, camera=cam)
            for p in anat:
                add_sphere(r.scene, p, ORANGE, args.marker * 0.8)
            for p in fitted + shift:
                add_sphere(r.scene, p, CYAN, args.marker)
            for p in obs + shift:
                add_sphere(r.scene, p, WHITE, args.marker * 0.7)
            img = r.render()
        if img.max() == 0:
            raise SystemExit(f'ERROR: {label} rendered black')
        gap = float(np.linalg.norm(fitted - (anat - shift), axis=1).mean())
        print(f'{label}: mean |fitted site - anatomical site| = {gap:.4f}')
        panels.append(img)

    out = np.concatenate(panels, axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, out)
    print(f'wrote {args.out} {out.shape}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
