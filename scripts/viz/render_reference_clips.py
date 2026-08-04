"""Render clips from a packed reference-clip HDF5 to mp4, for visual QC.

Replays each clip's ``qpos`` through the MuJoCo model and renders offscreen.
This is the check that catches what numeric QC cannot: whether the fly actually
looks like a fly (limb assignment, orientation, ground contact).

Needs an EGL/GPU context (set MUJOCO_GL=egl, which this module does on import).

Usage:
    python scripts/viz/render_reference_clips.py \
        --h5 /path/Fruitfly_v2_3_walk_1000hz_interp_padded.h5 \
        --xml models/fruitfly_v2_3_ik/fruitfly_v2_3_ik.xml \
        --clips longest,shortest,median --out-dir /path/renders

``--clips`` accepts integer indices and/or the keywords longest / shortest /
median. ``--stride`` subsamples frames (the data is 1000 Hz; the default 10
gives a 100 fps source downsampled to a watchable 30 fps output).
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


def resolve_clips(spec: str, clip_lengths: np.ndarray) -> list[int]:
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok == 'longest':
            out.append(int(np.argmax(clip_lengths)))
        elif tok == 'shortest':
            out.append(int(np.argmin(clip_lengths)))
        elif tok == 'median':
            out.append(int(np.argsort(clip_lengths)[len(clip_lengths) // 2]))
        else:
            out.append(int(tok))
    seen, uniq = set(), []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def build_model(fly_xml: Path, arena_xml: Path | None, spawn_z: float):
    """Compose arena + fly, matching how postprocess_stac_data.py builds its model.

    The fly-only XML has no lights or skybox, so rendering it alone yields a
    fully black frame. The arena supplies the headlight, skybox and floor, and
    the -0.125 spawn offset is the same `floor_alignment.target_z` the poses were
    aligned to, so the fly sits on the floor.
    """
    if arena_xml is None:
        return mujoco.MjModel.from_xml_path(str(fly_xml)), None
    fly = mujoco.MjSpec.from_file(str(fly_xml))
    arena = mujoco.MjSpec.from_file(str(arena_xml))
    frame = arena.worldbody.add_frame(pos=[0, 0, spawn_z], quat=[1, 0, 0, 0])
    frame.attach_body(fly.body('thorax'), '', suffix='_fly')
    return arena.compile(), 'thorax_fly'


def render_clip(model, qpos, height, width, camera, stride, track_body,
                distance, elevation, azimuth):
    data = mujoco.MjData(model)
    cam = None
    if camera is None and track_body is not None:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body)
        if bid >= 0:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.distance = distance
            cam.elevation = elevation
            cam.azimuth = azimuth
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for t in range(0, qpos.shape[0], stride):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            if camera is not None:
                renderer.update_scene(data, camera=camera)
            elif cam is not None:
                cam.lookat[:] = data.xpos[bid]   # follow the fly
                renderer.update_scene(data, camera=cam)
            else:
                renderer.update_scene(data)
            frames.append(renderer.render())
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--h5', required=True, type=Path)
    ap.add_argument('--xml', required=True, type=Path)
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--clips', default='longest,median,shortest')
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--camera', default=None,
                    help='camera name; omit to track the fly with a free camera')
    ap.add_argument('--max-frames', type=int, default=600,
                    help='cap rendered frames per clip (after striding)')
    ap.add_argument('--arena', type=Path, default=None,
                    help='arena/floor XML to attach the fly into. Strongly '
                         'recommended: the fly-only model has no lights, so '
                         'rendering without it produces black frames.')
    ap.add_argument('--spawn-z', type=float, default=-0.125,
                    help='fly spawn offset in the arena; must match '
                         'postprocessing.floor_alignment.target_z')
    ap.add_argument('--distance', type=float, default=0.6)
    ap.add_argument('--elevation', type=float, default=-20.0)
    ap.add_argument('--azimuth', type=float, default=135.0)
    args = ap.parse_args()

    model, track_body = build_model(args.xml, args.arena, args.spawn_z)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5) as f:
        cl = np.asarray(f['clip_lengths'])
        nq = f['qpos'].shape[2]
        if nq != model.nq:
            raise SystemExit(
                f'ERROR: h5 qpos has {nq} columns but the model has nq={model.nq}. '
                f'Wrong --xml for this dataset?')
        idxs = resolve_clips(args.clips, cl)
        print(f'{args.h5.name}: {len(cl)} clips, model nq={model.nq}')
        for i in idxs:
            L = int(cl[i])
            q = np.asarray(f['qpos'][i][:L])   # true length only, no padding
            frames = render_clip(model, q, args.height, args.width,
                                 args.camera, args.stride, track_body,
                                 args.distance, args.elevation, args.azimuth)
            frames = frames[:args.max_frames]
            arr = np.asarray(frames)
            if arr.max() == 0:
                raise SystemExit(
                    f'ERROR: clip {i} rendered entirely black. The model has no '
                    f'lighting — pass --arena pointing at the floor/arena XML.')
            out = args.out_dir / f'clip{i:03d}_len{L}.mp4'
            imageio.mimwrite(out, frames, fps=args.fps, codec='libx264',
                             pixelformat='yuv420p',
                             output_params=['-crf', '23'])
            print(f'  clip {i}: {L} frames -> {len(frames)} rendered -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
