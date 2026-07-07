"""fit-check view: STAC-fit verification frames (mesh + data keypoints + fitted
model markers + a red error tendon per keypoint whose length IS the residual).

Ported verbatim from scripts/render_fit_check.py's main() -- only the
argparse wrapper was dropped; the render call and still-frame sampling are
unchanged. Use these frames to confirm keypoints land on the model marker
sites even where the *visual mesh* shows a gap (the kinematic-only
calibration morph lengthens the joint/site but leaves the base-size mesh, so
a lengthened femur can look detached at the femur-tibia joint while the
site/keypoint still coincide).

Env / import ordering (load-bearing, matches the source script): MUJOCO_GL
and PYOPENGL_PLATFORM must be set to "egl" BEFORE mujoco is imported, so they
are set at module import time here, same as the reference script. But
`stac_mjx.viz.viz_stac` itself pulls in jax + mujoco (heavy), so it is
imported lazily inside run() -- by then the env vars set at module top are
already in effect -- keeping `import viz.views.fit_check` itself cheap under
JAX_PLATFORMS=cpu (matches the kp_qc/legskel/overlay views' lazy-import
convention).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path

import numpy as np
import imageio.v2 as imageio


def run(args) -> int:
    ik_h5 = Path(args.ik_h5)
    if not ik_h5.exists():
        raise FileNotFoundError(f"fit-check: ik_h5 does not exist: {ik_h5}")

    # MuJoCo's renderer.update_scene(..., camera=camera) breaks on None; the
    # CLI's --camera defaults to None so coerce to the fruitfly model's
    # "track1" camera here (same default the source script's argparse used).
    camera = args.camera if args.camera is not None else "track1"

    out = Path(args.out) if args.out else (ik_h5.parent / "fit_check")
    out.mkdir(parents=True, exist_ok=True)

    tag = f"{camera}_{args.start}{'_noerr' if args.no_error else ''}"
    mp4 = out / f"fit_check_{tag}.mp4"

    from stac_mjx.viz import viz_stac  # heavy (jax + mujoco); lazy on purpose

    _, frames = viz_stac(
        str(ik_h5), None, args.n, str(mp4),
        start_frame=args.start, camera=camera, height=900, width=1400,
        base_path=Path(args.body_model_dir), show_marker_error=not args.no_error,
    )
    for i in np.linspace(0, len(frames) - 1, args.n_stills).astype(int):
        png = out / f"fit_check_{tag}_f{args.start + int(i)}.png"
        imageio.imwrite(png, frames[int(i)])
        print(f"wrote {png}")
    print(f"wrote {mp4}")
    return 0
