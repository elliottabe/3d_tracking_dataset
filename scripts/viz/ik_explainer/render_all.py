#!/usr/bin/env python3
"""Task-32 Change 2: render the four ik_explainer acts CONCURRENTLY, then
assemble.

Each act (`acts/act{1..4}_*.py`) reads only from `<clip>/ik_explainer/
predictions/` and writes only to its own `<clip>/ik_explainer/frames/<act>/`
directory -- there is no shared state between them, so running all four as
independent subprocesses is safe. This is a pure orchestration layer: it does
not change what any act computes or draws, and it does not replace the
existing single-act entry points (`python acts/act1_views.py --clip ...`
etc. keep working exactly as before -- this script is additive).

GPU ASSIGNMENT: Acts 3/4 (`act3_align.py`/`act4_solve.py`) render through
MuJoCo EGL and are the only two that touch a GPU. When two GPUs both look
idle (`nvidia-smi`: low memory + low utilization), they are pinned to
DIFFERENT physical devices via `CUDA_VISIBLE_DEVICES`/`MUJOCO_EGL_DEVICE_ID`
so they don't contend for the same EGL context. Another user's job may
already occupy a GPU: if only one GPU looks idle, both acts 3/4 share it
(rather than crash or refuse to run); if `nvidia-smi` is unavailable or every
GPU looks busy, neither act's environment is touched and both fall back to
whatever the default device is. Acts 1/2 are CPU-only (OpenCV) and always run
alongside, unmodified.

FAILURE HANDLING: each act's stdout+stderr is captured to
`<clip>/ik_explainer/frames/render_all_logs/<act>.log` AND echoed to this
process's own stdout (prefixed `[<act>]`) once that act finishes, so a
failure in one act is never silently swallowed. If ANY act exits non-zero,
this driver prints every failing act's captured output, exits 1, and does
NOT invoke the assembler -- a partial frame set (e.g. an act that died
mid-render) must never reach `assemble.py`'s own per-act frame-count
assertion by way of this driver skipping straight past a failure. On success,
`assemble.py` is invoked as a final step (its own exact-frame-count checks
remain the last-resort backstop regardless of how the frames got there).

Usage:
    python scripts/viz/ik_explainer/render_all.py --clip <clip>
    python scripts/viz/ik_explainer/render_all.py --clip <clip> --skip-assemble
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from scripts.viz.ik_explainer import clip_io  # noqa: E402

ACTS_DIR = Path(__file__).resolve().parent / "acts"

# (log/display name, script filename, needs_gpu)
ACT_SPECS = [
    ("act1_views", "act1_views.py", False),
    ("act2_triangulate", "act2_triangulate.py", False),
    ("act3_align", "act3_align.py", True),
    ("act4_solve", "act4_solve.py", True),
]

# nvidia-smi thresholds for "this GPU looks idle" -- deliberately loose (an
# idle EGL/CUDA context can sit at a few hundred MiB even with nothing
# rendering) so a genuinely free card isn't misclassified as busy.
_BUSY_UTIL_PCT = 20.0
_BUSY_MEM_MIB = 2048.0


def _free_gpu_indices() -> list:
    """Query `nvidia-smi` for GPUs that look idle. Returns [] if nvidia-smi
    is unavailable/unparseable OR if every GPU looks busy -- callers must
    treat an empty list as "no safe assignment to make", not "no GPUs
    exist", and fall back to leaving the environment untouched."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception as e:
        print(f"[render_all] nvidia-smi query failed ({e!r}) -- proceeding "
              f"without a GPU assignment (acts 3/4 use the default device)")
        return []
    free = []
    for line in out.strip().splitlines():
        try:
            idx_s, mem_s, util_s = (p.strip() for p in line.split(","))
            idx, mem, util = int(idx_s), float(mem_s), float(util_s)
        except ValueError:
            continue
        if mem <= _BUSY_MEM_MIB and util <= _BUSY_UTIL_PCT:
            free.append(idx)
    return free


def gpu_assignment_for_acts_3_4() -> dict:
    """-> {"act3_align": gpu_idx_or_None, "act4_solve": gpu_idx_or_None}.

    Two idle GPUs -> one each (no EGL contention). One idle GPU -> both
    share it (another user's job may hold the other -- sharing, not
    crashing, is the required fallback). Zero idle / no nvidia-smi -> leave
    both untouched, deferring to whatever CUDA_VISIBLE_DEVICES/
    MUJOCO_EGL_DEVICE_ID this process already had (if any)."""
    free = _free_gpu_indices()
    if len(free) >= 2:
        print(f"[render_all] GPUs {free[0]} and {free[1]} both look idle -- "
              f"act3_align -> GPU{free[0]}, act4_solve -> GPU{free[1]}")
        return {"act3_align": free[0], "act4_solve": free[1]}
    if len(free) == 1:
        print(f"[render_all] only GPU{free[0]} looks idle (the other may be "
              f"occupied by another job) -- act3_align and act4_solve will "
              f"share GPU{free[0]}")
        return {"act3_align": free[0], "act4_solve": free[0]}
    print("[render_all] no GPU reported idle (or nvidia-smi unavailable) -- "
          "act3_align/act4_solve will use the default device, unmodified")
    return {"act3_align": None, "act4_solve": None}


def _run_one(name: str, script: str, gpu, clip: str, log_dir: Path) -> dict:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    script_path = ACTS_DIR / script
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(script_path), "--clip", clip],
        cwd=str(_REPO), env=env, capture_output=True, text=True,
    )
    dt = time.time() - t0
    log_path = log_dir / f"{name}.log"
    log_path.write_text(
        f"$ {' '.join(proc.args)}\n"
        f"(gpu={gpu}, elapsed={dt:.1f}s, returncode={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n")
    return {"name": name, "returncode": proc.returncode, "elapsed": dt,
            "stdout": proc.stdout, "stderr": proc.stderr, "log_path": log_path,
            "gpu": gpu}


def render_all(clip: str, skip_assemble: bool = False) -> int:
    dirs = clip_io.out_dirs(clip)
    log_dir = dirs["frames"] / "render_all_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_for = gpu_assignment_for_acts_3_4()

    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=len(ACT_SPECS)) as pool:
        futures = {
            pool.submit(_run_one, name, script, gpu_for.get(name), clip, log_dir): name
            for name, script, _needs_gpu in ACT_SPECS
        }
        for fut in futures:
            r = fut.result()
            results[r["name"]] = r
    wall = time.time() - t0

    failed = []
    for name, _script, _gpu in ACT_SPECS:
        r = results[name]
        status = "OK" if r["returncode"] == 0 else f"FAILED (exit {r['returncode']})"
        print(f"[render_all] {name}: {status} in {r['elapsed']:.1f}s "
              f"(gpu={r['gpu']}, log={r['log_path']})")
        # Always surface each act's captured output -- a failure must never
        # be silently swallowed, and even a successful act's own prints
        # (frame counts, timings, measured facts) are useful signal here.
        if r["stdout"].strip():
            print(f"[{name} stdout]\n{r['stdout']}")
        if r["stderr"].strip():
            print(f"[{name} stderr]\n{r['stderr']}")
        if r["returncode"] != 0:
            failed.append(name)

    print(f"[render_all] all {len(ACT_SPECS)} acts finished in {wall:.1f}s wall-clock "
          f"(parallel; see per-act elapsed above for each act's own render time)")

    if failed:
        print(f"[render_all] FAILED: {failed} -- refusing to assemble a "
              f"partial frame set. See the logs above / under {log_dir}.",
              file=sys.stderr)
        return 1

    if skip_assemble:
        print("[render_all] --skip-assemble set: not invoking assemble.py")
        return 0

    assemble_script = Path(__file__).resolve().parent / "assemble.py"
    print(f"[render_all] all acts OK -- invoking {assemble_script}")
    proc = subprocess.run([sys.executable, str(assemble_script), "--clip", clip],
                          cwd=str(_REPO))
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    ap.add_argument("--skip-assemble", action="store_true",
                     help="render the four acts but do not invoke assemble.py "
                          "(useful for isolating render timing from encode timing)")
    args = ap.parse_args()
    return render_all(args.clip, skip_assemble=args.skip_assemble)


if __name__ == "__main__":
    raise SystemExit(main())
