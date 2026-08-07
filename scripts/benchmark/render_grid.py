"""Fixed-frame MuJoCo render grid for benchmark bouts (visual A/B).

Same frames every run -> renders are directly comparable across variants.
Render pattern follows scripts/viz/compare_stac_fits.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def render_bout(qpos: np.ndarray, frames: list[int], mjcf_path: str,
                out_png: Path, size: tuple[int, int] = (320, 320)) -> None:
    import imageio.v2 as imageio
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90.0, -20.0, 1.0
    panels = []
    with mujoco.Renderer(m, height=size[1], width=size[0]) as r:
        for f in frames:
            f = int(np.clip(f, 0, qpos.shape[0] - 1))
            d.qpos[:] = qpos[f]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = d.qpos[:3]
            r.update_scene(d, camera=cam)
            panels.append(r.render())
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_png, np.concatenate(panels, axis=1))


def main(argv=None) -> None:
    import h5py
    from scripts.benchmark.manifest import bout_fly_dir, entries, load_manifest
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=Path("configs/benchmark/bouts.yaml"))
    ap.add_argument("--root", type=str, default="SOURCE",
                    help="variant root, or SOURCE for the original outputs")
    ap.add_argument("--mjcf", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    manifest = load_manifest(args.manifest)
    root = None if args.root == "SOURCE" else Path(args.root)
    for e in entries(manifest):
        for fly in e["flies"]:
            bd = bout_fly_dir({**e, "fly": fly}, root=root)
            if not (bd / "outputs.h5").exists():
                print(f"skip (no outputs.h5): {bd}")
                continue
            with h5py.File(bd / "outputs.h5", "r") as f:
                qpos = f["qpos"][:]
            frames = e.get("render_frames") or [0, qpos.shape[0] // 2,
                                                qpos.shape[0] - 1]
            render_bout(qpos, frames, args.mjcf,
                        args.out_dir / f"{e['run_key']}_bout_{e['bout']}_fly{fly}.png")


if __name__ == "__main__":
    main()
