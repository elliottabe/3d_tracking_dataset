"""D2: SAM3 mask+identity front-end driver (PyTorch SAM3 reused; no JAX).

Pure helpers (torch-free) + run_sam3_masks (lazily imports the heavy JARVIS
PyTorch bits). Produces per-bout sam3_masks.npz + a session manifest for the
JAX D3 pipeline to consume.
"""
import csv
import math
import os


def session_tag_for(session_dir: str) -> str:
    """'.../Session0/2025_10_20_13_20_04' -> 'Session0/2025_10_20_13_20_04'."""
    return "/".join(session_dir.rstrip("/").split("/")[-2:])


def parse_bouts(csv_path, session_tag, *, limit: int = 0, bout_ids=None):
    """Rows of the bouts CSV as {bout_idx, start, end, n}. Filtered to
    fly_id == session_tag ONLY when the CSV has a fly_id column; a fly_id-less
    CSV (e.g. Session1 good_bouts.csv) is already per-recording so all rows are
    kept. limit>0 keeps the first N; bout_ids (iterable) keeps only those bout_idx."""
    ids = set(int(b) for b in bout_ids) if bout_ids else None
    out = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        has_fly_id = reader.fieldnames is not None and "fly_id" in reader.fieldnames
        for r in reader:
            # Only filter by fly_id when the column exists; a fly_id-less CSV
            # (e.g. Session1 good_bouts.csv) is already per-recording -> keep all.
            if has_fly_id and session_tag and r.get("fly_id") != session_tag:
                continue
            bi = int(r["bout_idx"])
            if ids is not None and bi not in ids:
                continue
            s, e = int(r["start_frame"]), int(r["end_frame"])
            out.append({"bout_idx": bi, "start": s, "end": e, "n": e - s + 1})
    out.sort(key=lambda b: b["bout_idx"])
    if limit and limit > 0:
        out = out[:limit]
    return out


def video_paths_for(session_dir, camera_names):
    """Ordered <session_dir>/<cam>.mp4 for each camera name (raises if missing)."""
    paths = []
    for cam in camera_names:
        p = os.path.join(session_dir, f"{cam}.mp4")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing video for camera {cam}: {p}")
        paths.append(p)
    return paths


def bout_stats(loaded_masks, num_animals: int) -> dict:
    """Summary stats from a LoadedBoutMasks-like object (.valid (A,C,F))."""
    import numpy as np
    valid = np.asarray(loaded_masks.valid)            # (A,C,F)
    A, C, F = valid.shape
    per_fly = [float(valid[a].mean()) for a in range(A)]
    cams_valid_per_frame = valid.any(axis=0).sum(axis=0) if A else np.zeros(F)
    # mean over frames of (#cams with >=1 valid fly)
    mean_cams = float(valid.any(axis=0).sum(axis=0).mean()) if F else 0.0
    return {
        "num_animals_saved": int(loaded_masks.num_animals_saved),
        "num_cameras": int(C),
        "num_frames": int(F),
        "per_fly_valid_frac": per_fly,
        "mean_cams_valid_per_frame": mean_cams,
    }


def build_manifest(session_dir, session_tag, sam3_settings, per_bout) -> dict:
    return {
        "session_dir": session_dir,
        "session_tag": session_tag,
        "sam3_settings": dict(sam3_settings),
        "n_bouts": len(per_bout),
        "bouts": list(per_bout),
    }


def resolve_gpus(spec, *, env=None, device_count=None):
    """Resolve the sam3.gpus config value to a list of GPU ids.

    Order: explicit non-empty list `spec` -> CUDA_VISIBLE_DEVICES in `env`
    -> range(device_count). `env`/`device_count` are injected for testability
    (production passes os.environ and torch.cuda.device_count()).
    """
    if spec is not None:
        try:
            ids = [int(x) for x in list(spec)]
        except TypeError:
            ids = []
        if ids:
            return ids
    if env is None:
        env = os.environ
    cvd = env.get("CUDA_VISIBLE_DEVICES", "") or ""
    if cvd.strip():
        return [int(x) for x in cvd.split(",") if x.strip()]
    if device_count is None:
        import torch
        device_count = torch.cuda.device_count()
    return list(range(int(device_count)))


def split_bouts_contiguous(bout_ids, n_gpus):
    """Split bout_ids into n_gpus contiguous blocks (ceil per block). The result
    always has exactly n_gpus entries; trailing blocks are [] when there are
    fewer bouts than GPUs."""
    n = len(bout_ids)
    per = math.ceil(n / n_gpus) if n_gpus > 0 else n
    groups = []
    for gi in range(n_gpus):
        s = gi * per
        e = min(s + per, n)
        groups.append(list(bout_ids[s:e]) if s < n else [])
    return groups


def _ov(v):
    """Format a value as a Hydra override RHS (None -> 'null')."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def build_worker_cmd(*, python, script, gpu, bout_ids, session_dir, out, project,
                     bouts_csv, num_animals, reuse_masks, jarvis_root,
                     sam3_version, sam3_compile, sam3_text, sam3_checkpoint,
                     manifest_name, inductor_cache_dir, base_env=None):
    """Build the (env, argv) for one single-GPU SAM3 worker subprocess.

    Overrides every sam3.* field explicitly so the worker does not depend on the
    parent's Hydra groups. `sam3.gpus=[0]` is the recursion guard: with
    CUDA_VISIBLE_DEVICES=<gpu> the worker sees one device as cuda:0 and runs the
    single-GPU path on sam3.sam3_gpu=0.
    """
    bout_csv = ",".join(str(b) for b in bout_ids)
    argv = [
        python, script,
        "sam3=default",
        "sam3.gpus=[0]",
        "sam3.sam3_gpu=0",
        # Quote the comma value so Hydra parses it as a STRING (the entrypoint
        # does str(bout_ids).split(",")). Unquoted "1,2,3" -> Hydra "Ambiguous
        # value"; bracketed [1,2,3] -> a list that str().split(",") mangles.
        f"sam3.bout_ids='{bout_csv}'",
        "sam3.limit=0",
        f"sam3.session_dir={session_dir}",
        f"sam3.out={out}",
        f"sam3.bouts_csv={bouts_csv}",
        f"sam3.project={project}",
        f"sam3.num_animals={num_animals}",
        f"sam3.reuse_masks={_ov(bool(reuse_masks))}",
        f"sam3.jarvis_root={_ov(jarvis_root)}",
        f"sam3.sam3_version={sam3_version}",
        f"sam3.sam3_compile={_ov(bool(sam3_compile))}",
        f"sam3.sam3_text={sam3_text}",
        f"sam3.sam3_checkpoint={_ov(sam3_checkpoint)}",
        f"sam3.manifest_name={manifest_name}",
    ]
    env = dict(base_env if base_env is not None else os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env, argv


def merge_manifests(partial_paths, *, base):
    """Merge per-worker partial manifests into one. Concatenates each partial's
    `bouts`, dedupes by bout_idx, sorts by bout_idx; copies `base` for the
    session-level fields and sets n_bouts/bouts."""
    import json
    bouts = []
    seen = set()
    for p in partial_paths:
        with open(p) as f:
            m = json.load(f)
        for b in m.get("bouts", []):
            bi = b.get("bout_idx")
            if bi in seen:
                continue
            seen.add(bi)
            bouts.append(b)
    bouts.sort(key=lambda b: (b.get("bout_idx") is None, b.get("bout_idx")))
    merged = dict(base)
    merged["n_bouts"] = len(bouts)
    merged["bouts"] = bouts
    return merged


def run_sam3_masks(*, project, session_dir, bouts_csv, out, num_animals=2,
                   limit=0, bout_ids=None, reuse_masks=True, sam3=None,
                   jarvis_root=None, manifest_name="manifest.json"):
    """Run SAM3 video tracking + identity over a session's bouts, writing a
    per-bout sam3_masks.npz + a session manifest.

    Reuses the JARVIS PyTorch SAM3 stack (SAM3VideoTracker, save_bout_masks,
    LoadedBoutMasks from JARVIS-HybridNet). All heavy imports are lazy so the
    module stays torch-free at import time.

    Args:
        project: JARVIS project name (passed to ProjectManager.load).
        session_dir: Absolute path to the recording session directory.
        bouts_csv: Path (absolute or relative to session_dir) of the bouts CSV.
        out: Output root directory; per-bout subdirs are created here.
        num_animals: Number of animals to track (default 2).
        limit: If >0, process only the first N bouts (by bout_idx order).
        bout_ids: If given, only process bouts with these bout_idx values.
        reuse_masks: If True and the npz already exists, skip re-running SAM3.
        sam3: Dict of SAM3VideoTracker kwargs (sam3_version, gpu_id, compile,
              text_prompt, checkpoint_path). Defaults filled in if missing.
        jarvis_root: Absolute path to the JARVIS-HybridNet root that contains
              the `projects/` directory. If None, defaults to the
              JARVIS_ROOT env var, then falls back to the third_party copy
              (which has no projects — pass an explicit path or set the env
              var when the third_party copy is used).

    Returns:
        The manifest dict (also written to <out>/manifest.json).
    """
    import json
    import sys
    import time

    sam3 = dict(sam3 or {})

    # --- Lazy heavy imports: PyTorch / JARVIS ---
    # The code root (third_party/JARVIS-HybridNet) provides importable JARVIS
    # modules.  The *projects* root may differ — some installs keep projects in
    # a separate checkout.  We split these two concerns:
    #
    # code_root: used only for sys.path so that `import jarvis.*` resolves.
    #   Derived from this file's location (4 dirs up = third_party/,
    #   then sibling JARVIS-HybridNet/).
    # projects_root: the JARVIS-HybridNet directory that owns `projects/`.
    #   Resolved from the `jarvis_root` kwarg, then the JARVIS_ROOT env var,
    #   then falls back to code_root (works when both coincide).
    code_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "JARVIS-HybridNet")
    print(f"[sam3] code_root (imports): {code_root}")
    assert os.path.isdir(code_root), f"JARVIS-HybridNet not found at {code_root}"

    if jarvis_root is None:
        jarvis_root = os.environ.get("JARVIS_ROOT", code_root)
    print(f"[sam3] jarvis_root (projects): {jarvis_root}")
    assert os.path.isdir(jarvis_root), (
        f"jarvis_root not found: {jarvis_root} — set JARVIS_ROOT or pass jarvis_root=")

    for p in (code_root, os.path.join(code_root, "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from jarvis.config.project_manager import ProjectManager
    from jarvis.utils.reprojection import get_repro_tool
    from jarvis.prediction.sam3_video_tracker import SAM3VideoTracker
    import predict3D_multianimal as pmod  # save_bout_masks, LoadedBoutMasks, MASKS_FILENAME
    import torch

    pm = ProjectManager()
    # Override parent_dir so ProjectManager looks for projects in jarvis_root.
    if os.path.realpath(pm.parent_dir) != os.path.realpath(jarvis_root):
        pm.parent_dir = jarvis_root
        print(f"[sam3] ProjectManager.parent_dir overridden -> {jarvis_root}")
    assert pm.load(project), f"Could not load JARVIS project '{project}'"
    cfg = pm.get_cfg()

    session_calib = os.path.join(session_dir, "calibration")
    repro_tool = get_repro_tool(
        cfg, session_calib if os.path.isdir(session_calib) else None)
    assert repro_tool is not None, "ReprojectionTool not available"

    tag = session_tag_for(session_dir)
    csv_path = (bouts_csv if os.path.isabs(bouts_csv)
                else os.path.join(session_dir, bouts_csv))
    bouts = parse_bouts(csv_path, tag, limit=limit, bout_ids=bout_ids)
    video_paths = video_paths_for(session_dir, list(repro_tool.cameras))
    os.makedirs(out, exist_ok=True)

    tracker = None
    tracker_load_time = None
    per_bout = []

    for b in bouts:
        bout_out = os.path.join(out, f"bout_{b['bout_idx']:05d}")
        os.makedirs(bout_out, exist_ok=True)
        npz_path = os.path.join(bout_out, pmod.MASKS_FILENAME)
        t0 = time.time()

        if reuse_masks and os.path.isfile(npz_path):
            lm = pmod.LoadedBoutMasks(npz_path)
            print(f"[sam3] bout {b['bout_idx']}: reusing existing {npz_path}")
        else:
            if tracker is None:
                t_load = time.time()
                tracker = SAM3VideoTracker(
                    gpu_id=sam3.get("gpu_id", 0),
                    text_prompt=sam3.get("text_prompt", "insect"),
                    sam3_version=sam3.get("sam3_version", "sam3.1"),
                    compile=sam3.get("compile", False),
                    checkpoint_path=sam3.get("checkpoint_path", None))
                tracker_load_time = round(time.time() - t_load, 1)
                print(f"[sam3] SAM3VideoTracker loaded in {tracker_load_time}s")

            bm = tracker.process_bout(
                video_paths, b["start"], b["n"], num_animals=num_animals)
            # SAM3's predictors enter a process-wide bf16 autocast
            # (self.bf16_context.__enter__() with no matching __exit__), so it
            # stays active after process_bout() returns. That leaks into the
            # reprojection geometry in assign_identities: torch.matmul/solve run
            # in bfloat16, which (a) drops the sub-pixel triangulation residuals
            # used to disambiguate fly0/fly1 below bf16 precision and (b) makes
            # r[cam].numpy() raise "unsupported ScalarType BFloat16". Force fp32.
            with torch.autocast(device_type="cuda", enabled=False):
                bm.assign_identities(repro_tool, num_animals=num_animals)
            pmod.save_bout_masks(bm, bout_out, num_animals)
            lm = pmod.LoadedBoutMasks(npz_path)

        st = bout_stats(lm, num_animals)
        st.update(
            bout_idx=b["bout_idx"],
            start=b["start"],
            end=b["end"],
            npz=npz_path,
            seconds=round(time.time() - t0, 1),
        )
        if tracker_load_time is not None and len(per_bout) == 0:
            st["sam3_load_seconds"] = tracker_load_time
        per_bout.append(st)
        print(f"[sam3] bout {b['bout_idx']}: {st['num_frames']} frames, "
              f"mean cams valid {st['mean_cams_valid_per_frame']:.2f}, "
              f"{st['seconds']}s -> {npz_path}")

    manifest = build_manifest(session_dir, tag, sam3, per_bout)
    manifest_path = os.path.join(out, manifest_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[sam3] wrote {len(per_bout)} bouts + {manifest_name} to {out}")
    return manifest


def run_sam3_masks_multi(*, gpus, project, session_dir, bouts_csv, out,
                         num_animals=2, limit=0, bout_ids=None,
                         reuse_masks=True, jarvis_root=None, sam3=None,
                         python=None, script=None):
    """Fan a session's bouts across multiple GPUs: one subprocess per GPU, each
    re-invoking scripts/sam3_masks.py (single-GPU) on a contiguous bout subset.
    The parent merges per-worker partial manifests into out/manifest.json.

    Falls back to the single-GPU run_sam3_masks when <=1 GPU or <=1 bout.
    """
    import json
    import subprocess
    import sys
    import tempfile

    sam3 = dict(sam3 or {})
    tag = session_tag_for(session_dir)
    csv_path = (bouts_csv if os.path.isabs(bouts_csv)
                else os.path.join(session_dir, bouts_csv))
    bouts = parse_bouts(csv_path, tag, limit=limit, bout_ids=bout_ids)
    all_ids = [b["bout_idx"] for b in bouts]

    if len(gpus) <= 1 or len(all_ids) <= 1:
        return run_sam3_masks(
            project=project, session_dir=session_dir, bouts_csv=bouts_csv,
            out=out, num_animals=num_animals, limit=limit, bout_ids=bout_ids,
            reuse_masks=reuse_masks, sam3=sam3, jarvis_root=jarvis_root)

    os.makedirs(out, exist_ok=True)
    groups = split_bouts_contiguous(all_ids, len(gpus))
    python = python or sys.executable
    if script is None:
        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "scripts", "sam3_masks.py")

    procs = []
    for gpu, grp in zip(gpus, groups):
        if not grp:
            continue
        mname = f"manifest.gpu{gpu}.json"
        env, argv = build_worker_cmd(
            python=python, script=script, gpu=gpu, bout_ids=grp,
            session_dir=session_dir, out=out, project=project,
            bouts_csv=csv_path, num_animals=num_animals,
            reuse_masks=reuse_masks, jarvis_root=jarvis_root,
            sam3_version=sam3.get("sam3_version", "sam3.1"),
            sam3_compile=sam3.get("compile", False),
            sam3_text=sam3.get("text_prompt", "insect"),
            sam3_checkpoint=sam3.get("checkpoint_path", None),
            manifest_name=mname,
            inductor_cache_dir=os.path.join(
                tempfile.gettempdir(), f"torchinductor_gpu{gpu}"))
        print(f"[sam3-multi] GPU {gpu}: {len(grp)} bouts {grp}")
        procs.append((gpu, subprocess.Popen(argv, env=env),
                      os.path.join(out, mname)))

    failures = []
    existing = []
    for gpu, p, partial in procs:
        rc = p.wait()
        if rc != 0:
            failures.append(gpu)
            print(f"[sam3-multi] GPU {gpu} worker FAILED (exit {rc})")
        elif not os.path.isfile(partial):
            # Clean exit but no manifest written = bouts silently lost; treat as
            # a failure so the merge never hands D3 a partial session unnoticed.
            failures.append(gpu)
            print(f"[sam3-multi] GPU {gpu} exited 0 but wrote no manifest "
                  f"({partial}) — treating as failed")
        else:
            existing.append(partial)
    base = build_manifest(session_dir, tag, sam3, [])
    merged = merge_manifests(existing, base=base)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(merged, f, indent=2)
    merged["failures"] = failures
    print(f"[sam3-multi] merged {merged['n_bouts']} bouts -> manifest.json "
          f"({len(failures)} GPU failures)")
    # The partial manifest (above) is written first so a re-run with
    # reuse_masks=true resumes the finished bouts. But if any worker failed we
    # raise: the masks are incomplete, and a caller (e.g. the SLURM launcher's
    # sequential job) must NOT proceed to the D3 predict stage on a partial
    # session. Failing loudly lets requeue + reuse_masks finish stage 1 first.
    if failures:
        raise RuntimeError(
            f"[sam3-multi] {len(failures)} GPU worker(s) failed: {failures}. "
            f"Partial manifest written to {out}/manifest.json; re-run "
            f"(reuse_masks=true) to finish the remaining bouts.")
    return merged
