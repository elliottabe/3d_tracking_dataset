#!/usr/bin/env python3
"""Measure ONE STAC IK bout-fly solve: peak device memory, wall, GPU utilisation.

THE QUESTION THIS ANSWERS. STAC IK is 57% of the per-bout pipeline (643 s of
~1120 s). Two ways to speed it up were on the table:

  (a) vmap/concatenate several bout-flies into one solve, and
  (b) run several INDEPENDENT bout solves per GPU as separate processes.

(b) is strictly better if it fits, because it sidesteps the two things that
make (a) hard: ragged clip lengths (bouts run ~290-2000 frames, and the
pipeline splits further on NaN gaps -- see run_bout._solve_segments_into), and
ragged convergence (the female needs MORE CG iterations on a SHORTER clip, so a
lockstep joint solve runs every item to the worst case). (b) needs no solver
change and no clip-boundary smoothness fix. Its only constraint is memory.

So the decisive number is: how much device memory does ONE solve actually use,
and does one solve saturate the GPU? That is what this measures.

METHOD NOTE THAT MAKES OR BREAKS THE NUMBER. JAX preallocates ~75% of the
device by default, so `nvidia-smi` would report the PREALLOCATION, not the
solve -- a meaningless number that looks authoritative. This forces
XLA_PYTHON_CLIENT_PREALLOCATE=false and reads
`device.memory_stats()['peak_bytes_in_use']`, which is the true high-water
mark. GPU utilisation is sampled in a background thread across the solve,
because a single instantaneous sample is exactly how this session already
mistook a healthy run for a hang.

Sweeping --frames also answers the ragged-length half of the question: if peak
memory is linear in T with a small intercept, concurrency is predictable and
short clips are cheap to co-schedule.

    python scripts/analysis/stac_ik_memory_probe.py --fly 1 --frames 2007 500
"""
import argparse
import json
import os
import subprocess
import threading
import time

# MUST precede any jax import: otherwise the 75% preallocation hides the real
# high-water mark and every number below is the allocator's, not the solve's.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

SCRATCH = "/gscratch/portia/eabe/data/Johnson_lab/scratch/stacik_task9/pose"


class GpuSampler(threading.Thread):
    """Sample utilisation + nvidia-smi memory every `period` s until stopped."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        # NOT `self._stop`: threading.Thread has its own private _stop()
        # method, and shadowing it with an Event makes Thread's own teardown
        # call the Event -> TypeError, which destroyed a COMPLETED 466 s
        # measurement on job 39502485.
        self.period, self._halt = period, threading.Event()
        self.util, self.mem = [], []

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                first = out.splitlines()[0]
                u, m = (x.strip() for x in first.split(","))
                self.util.append(int(u)); self.mem.append(int(m))
            except Exception:
                pass
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set(); self.join(timeout=15)

    def summary(self):
        def pct(v, q):
            if not v:
                return None
            s = sorted(v); return s[min(len(s) - 1, int(q * len(s)))]
        return {"n_samples": len(self.util),
                "util_mean": (sum(self.util) / len(self.util)) if self.util else None,
                "util_p50": pct(self.util, 0.5), "util_p90": pct(self.util, 0.9),
                "util_max": max(self.util) if self.util else None,
                "smi_mem_max_MiB": max(self.mem) if self.mem else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", type=int, default=1, choices=(0, 1))
    ap.add_argument("--frames", type=int, nargs="*", default=[2007],
                    help="clip lengths to solve; sweep to get the T-scaling")
    ap.add_argument("--scratch", default=SCRATCH)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np
    from hydra import initialize_config_dir, compose
    import jax

    # this file is scripts/analysis/<name>.py -- THREE levels up to the repo
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_dir = os.path.join(repo, "configs")
    import stac_mjx  # noqa: F401  (registers the multirun_save_dir resolver)
    from omegaconf import OmegaConf
    # `basename` is registered as an IMPORT SIDE EFFECT of scripts/run_bout.py,
    # which this probe does not import. Without it, composition succeeds and
    # then blows up later, deep inside run_stac, on
    # recording.predictions_dir -- because omegaconf resolves interpolations
    # LAZILY, so a probe that composes the config and reads a couple of keys
    # looks fine while the tree is still broken (job 39502091).
    OmegaConf.register_new_resolver(
        "basename", lambda q: os.path.basename(os.path.normpath(str(q))),
        replace=True)
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="pipeline", overrides=["paths=hyak"])
    # Force the WHOLE tree to resolve now, so a bad interpolation fails here
    # with a clear message instead of 6 minutes of imports later.
    OmegaConf.to_container(cfg, resolve=True)

    from jarvis_jax.tracking.stac import ik_only_bout

    bout = os.path.join(a.scratch, "bouts", "bout_00028", f"fly{a.fly}")
    z = np.load(os.path.join(bout, "kp3d.npz"))
    kp3d_full = z["kp3d"]
    with open(os.path.join(a.scratch, "scale.json")) as f:
        _sc = json.load(f)
    # Per-fly scale when present -- scale.json carries BOTH a global `scale`
    # and `scale_by_fly` (fly0 0.011617, fly1 0.011744), and using the global
    # one would time a solve the pipeline never runs. Body scale feeds marker
    # offsets, and a wrong one has already been shown in this repo to be
    # absorbed silently (the 38x scale defect), so it is worth getting right
    # even in a probe.
    scale = float((_sc.get("scale_by_fly") or {}).get(str(a.fly), _sc.get("scale", 1.0)))
    kp_names = list(cfg.model.KP_NAMES)

    dev = jax.local_devices()[0]
    plat = f"{dev.platform}:{dev.device_kind}"
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                               "--format=csv,noheader"], capture_output=True,
                              text=True).stdout.strip().splitlines()[0]
    print(f"device: {plat}  |  {gpu_name}", flush=True)
    print(f"kp3d {kp3d_full.shape}  scale {scale}  fly{a.fly}", flush=True)

    results = []
    for T in a.frames:
        kp = kp3d_full[:T]
        # NaN frames would be split into separate segments by the real
        # pipeline; report how many so a clean T is not silently compared
        # against a gappy one.
        finite = int(np.isfinite(kp.reshape(len(kp), -1)).all(1).sum())
        out_dir = os.path.join(a.scratch, "_memprobe")
        os.makedirs(out_dir, exist_ok=True)
        # `offsets_path` is resolved RELATIVE TO save_path by
        # stac_mjx.run_stac, so a fresh output dir must carry the offsets it
        # is told to load. Symlink rather than copy, and rather than writing
        # our probe outputs into the perf run's own scratch alongside its
        # results.
        link = os.path.join(out_dir, "offsets.h5")
        if not os.path.exists(link):
            os.symlink(os.path.join(a.scratch, "offsets.h5"), link)

        try:
            dev.memory_stats()   # reset baseline by reading it
        except Exception:
            pass
        sampler = GpuSampler(); sampler.start()
        t0 = time.time()
        ik_only_bout(cfg, kp, kp_names, offsets_path="offsets.h5",
                     out_h5=f"memprobe_T{T}.h5", save_path=out_dir, scale=scale)
        wall = time.time() - t0
        # The solve is done and its numbers exist; from here on nothing in the
        # instrumentation is allowed to lose them.
        ms = {}
        try:
            ms = dev.memory_stats() or {}
        except Exception as e:
            print(f"  [warn] memory_stats failed: {e}", flush=True)
        try:
            sampler.stop()
        except Exception as e:
            print(f"  [warn] gpu sampler teardown failed: {e}", flush=True)
        peak = ms.get("peak_bytes_in_use")
        limit = ms.get("bytes_limit")
        row = {"T": T, "finite_frames": finite, "wall_s": round(wall, 2),
               "peak_bytes": peak, "bytes_limit": limit,
               "peak_GiB": round(peak / 2**30, 3) if peak else None,
               "limit_GiB": round(limit / 2**30, 1) if limit else None,
               "gpu": gpu_name, **sampler.summary()}
        if peak and limit:
            row["fits_concurrently"] = int(limit // peak)
        results.append(row)
        print(json.dumps(row), flush=True)
        if a.out:      # checkpoint after EVERY T, not once at the end
            try:
                with open(a.out, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"  [warn] could not checkpoint {a.out}: {e}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"{'T':>6}{'finite':>8}{'wall_s':>9}{'peak_GiB':>10}{'util_mean':>11}"
          f"{'util_p90':>10}{'n_concurrent':>13}")
    for r in results:
        print(f"{r['T']:>6}{r['finite_frames']:>8}{r['wall_s']:>9.1f}"
              f"{(r['peak_GiB'] or 0):>10.3f}{(r['util_mean'] or 0):>11.1f}"
              f"{(r['util_p90'] or 0):>10}{r.get('fits_concurrently','?'):>13}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
