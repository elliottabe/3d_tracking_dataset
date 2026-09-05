#!/usr/bin/env python
"""What does ONE mvq window cost on a GPU? (P4a spec §7.)

The mask-free front end's whole throughput budget rests on one number: the
per-window cost of a batched, single-pass `MVQRunner.infer`. Spec §7 assumes
40 ms/window (coarse 20-40 min + fine 60-90 min per recording, inside the
3 GPU-hour acceptance) and sets the deviation rule: **if the measured cost is
above 60 ms/window the fine pass runs at stride 2** with linear interpolation
of the 3D output (a fly moves ~0.04 mm between frames at 800 fps).

EXPECTATION (written before running, so the measurement can disagree with it).
The bout-28 figure script measured 0.18 s per window UNBATCHED and running
BOTH the prompted and the unprompted forward, i.e. ~90 ms per window per
pass with a batch of 8. Batched 32-wide with one pass, the per-window cost
should land in the tens of ms -- under the 60 ms deviation line, and near the
40 ms the budget assumes. A number ABOVE 60 ms means the fine pass is a
stride-2 pass and §7's table is optimistic by that factor; a number far BELOW
40 ms (say under 10) would be suspicious, not good news: it would mean the
timed region is not actually waiting for the device, and the first thing to
check is that `infer` still returns numpy (it does -- `assemble` calls
`np.asarray` on the model outputs, which blocks until they are ready).

Windows are built at real frames of the recording (one synced frame set read
through `predict.synced_reader.read_window`) around a real fly centre, jittered
by up to 3 mm so the 32 windows are not literally one image repeated. The cost
does not depend on the content -- every window is the same 7x448x448x3 tensor
whatever it contains -- so the centre only has to be plausible; what the real
frames buy is that the crop path, the decode and the dtypes are the real ones.

    scripts/slurm/submit_task.sh --time 0:30:00 --mem 32 mvq_wincost \\
      "export PYTHONPATH=third_party/jarvis_jax:. && \\
       python scripts/benchmark/mvq_window_cost.py --run <P3a final> \\
         --session-dir <session> --frame 446975"

Writes `figures/2026-09-mvq/p4_maskfree/window_cost.json`.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax"))
sys.path.insert(0, ROOT)

import jax  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from jarvis_jax.predict.synced_reader import load_plan, read_window  # noqa: E402
from jarvis_jax.tracking.lift_mvq import MVQRunner  # noqa: E402

DEF_SESSION = ("/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/"
               "courtship/Session0/2025_10_20_13_20_04")
DEF_RECORDING_CFG = os.path.join(ROOT, "configs", "recording", "session0.yaml")
# fly0's mvq keypoint centroid on bout 28 frame 446975 (world units, 0.1 mm
# per unit) from that bout's smoke run -- a real fly position on this frame.
DEF_CENTRE = (219.1, 14.7, 13.8)
STRIDE1_MS = 60.0          # spec §7: above this, the fine pass runs at stride 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a final/ dir, or (with --step) the run dir")
    ap.add_argument("--step", default=None, help="load ckpt/<step> (or 'latest') instead of final/")
    ap.add_argument("--attn_impl", default=None, help="override the run's own attn_impl ('xla' on CPU)")
    ap.add_argument("--session-dir", default=DEF_SESSION)
    ap.add_argument("--recording-cfg", default=DEF_RECORDING_CFG,
                    help="hydra recording config defining the CANONICAL camera order")
    ap.add_argument("--calib-dir", default=None, help="default <session-dir>/calibration")
    ap.add_argument("--frame", type=int, default=446975, help="absolute (canonical slot) frame")
    ap.add_argument("--centre", type=float, nargs=3, default=list(DEF_CENTRE),
                    help="world centre of the windows, in units (0.1 mm)")
    ap.add_argument("--jitter-mm", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "2026-09-mvq", "p4_maskfree"))
    a = ap.parse_args()

    cameras = [str(c) for c in OmegaConf.load(a.recording_cfg).cameras]
    runner = MVQRunner(a.run, step=a.step, attn_impl=a.attn_impl,
                       calib_dir=a.calib_dir or os.path.join(a.session_dir, "calibration"),
                       cameras=cameras, batch=a.batch)
    print(f"[wincost] {jax.default_backend()} {jax.devices()}; checkpoint {a.run} "
          f"step {runner.step_label}; attn_impl "
          f"{a.attn_impl or runner.meta['model'].get('attn_impl')}", flush=True)

    frames, present = next(iter(read_window(a.session_dir, cameras, load_plan(a.session_dir),
                                            int(a.frame), 1)))
    print(f"[wincost] frame {a.frame}: {frames.shape} {frames.dtype}, "
          f"cameras present {int(np.sum(present))}/{len(cameras)}", flush=True)
    rng = np.random.default_rng(0)
    centres = (np.asarray(a.centre, np.float64)[None]
               + rng.uniform(-a.jitter_mm * 10.0, a.jitter_mm * 10.0, size=(a.batch, 3)))

    t = time.perf_counter()
    w = runner.windows(frames, present, centres)
    build_s = time.perf_counter() - t
    print(f"[wincost] built {a.batch} windows in {build_s * 1e3:.0f} ms "
          f"({build_s / a.batch * 1e3:.2f} ms/window, CPU cropping)", flush=True)

    for _ in range(a.warmup):
        runner.infer(w)
    ts = []
    for _ in range(a.iters):
        t = time.perf_counter()
        out = runner.infer(w)                    # returns numpy -> blocks on the device
        ts.append(time.perf_counter() - t)
    ts = np.asarray(ts)
    per = float(ts.mean()) / a.batch * 1e3
    per_total = per + build_s / a.batch * 1e3
    res = {
        "checkpoint": os.path.abspath(a.run), "step": runner.step_label,
        "attn_impl": a.attn_impl or runner.meta["model"].get("attn_impl"),
        # `device_kind` is the CARD ("NVIDIA L40S" vs "NVIDIA A40"): the cost
        # differs by ~1.85x between them, which is enough to flip §7's
        # stride decision, so the number is meaningless without it.
        "backend": jax.default_backend(), "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "crops_dtype": str(w["crops"].dtype), "batch": a.batch,
        "session_dir": a.session_dir, "frame": int(a.frame),
        "centre_units": [float(x) for x in a.centre], "jitter_mm": a.jitter_mm,
        "iters": a.iters, "warmup": a.warmup,
        "batch_ms_mean": float(ts.mean() * 1e3), "batch_ms_std": float(ts.std() * 1e3),
        "batch_ms_min": float(ts.min() * 1e3),
        "ms_per_window_infer": per,
        "ms_per_window_build": build_s / a.batch * 1e3,
        "ms_per_window_total": per_total,
        "stride1_threshold_ms": STRIDE1_MS,
        "fine_pass_stride": 1 if per_total <= STRIDE1_MS else 2,
        "exist_shape": list(np.shape(out["exist"])),
    }
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "window_cost.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[wincost] {per:.1f} ms/window (infer) + {res['ms_per_window_build']:.1f} "
          f"(window build) = {per_total:.1f} ms/window total; batch of {a.batch} took "
          f"{res['batch_ms_mean']:.0f}+-{res['batch_ms_std']:.0f} ms over {a.iters} iters",
          flush=True)
    print(f"[wincost] spec §7 decision: fine pass stride "
          f"{res['fine_pass_stride']} (rule: stride 1 if <= {STRIDE1_MS:.0f} ms/window)",
          flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
