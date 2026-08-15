"""Build variant run_roots from frozen inputs, emit run commands, collect scorecards.

A variant re-runs scripts/run_bout.py per bout with stage-skipping doing the
freezing: kp2d/kp3d/kp3d_filt pre-exist in the variant root (hardlinked from
frozen/), so stages A/B/B2 skip; scale.json / segment_scales.json / offsets.h5
are deliberately ABSENT so they recompute under the variant's config (that is
the Track 1 treatment). Stage C (STAC) onward runs fresh.

GPU note: emitted commands must run on a compute node (sbatch ckpt-g2 or an
interactive GPU shell) -- never a login node.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

import numpy as np

from scripts.benchmark.manifest import bout_fly_dir, entries, load_manifest
from scripts.benchmark.metrics import (
    compute_bout_metrics, joint_bounds, joint_limit_violation_rate)
from scripts.benchmark.scorecard import (
    build_scorecard, cohort_of, split_series, write_scorecard)

FROZEN_INPUTS = ("kp2d.npz", "kp3d.npz", "kp3d_filt.npz")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def build_variant_root(manifest: dict, dest_root: Path, variant: str,
                       frozen_inputs: tuple[str, ...] = FROZEN_INPUTS) -> Path:
    """`frozen_inputs` selects which stage artifacts the variant inherits.

    Default: all three (Track 1, STAC-level treatments). A treatment that
    changes an EARLIER stage must freeze only that stage's inputs -- e.g.
    a triangulation change freezes just kp2d.npz so Stage B/B2 recompute;
    linking kp3d would make stage-skipping serve the baseline triangulation
    and the A/B would compare a run against itself.
    """
    dest_root = Path(dest_root)
    frozen = dest_root / "frozen"
    vroot = dest_root / "variants" / variant
    for e in entries(manifest):
        for fly in e["flies"]:
            rel = (Path(e["run_key"]) / "bouts"
                   / f"bout_{int(e['bout']):05d}" / f"fly{fly}")
            for name in frozen_inputs:
                src = frozen / rel / name
                if src.exists():
                    _link_or_copy(src, vroot / rel / name)
        sex = frozen / e["run_key"] / "bouts" / f"bout_{int(e['bout']):05d}" / "sex.json"
        if sex.exists():
            _link_or_copy(sex, vroot / e["run_key"] / "bouts"
                          / f"bout_{int(e['bout']):05d}" / "sex.json")
    return vroot


def variant_commands(manifest: dict, variant_root: Path,
                     overrides: list[str]) -> list[str]:
    cmds = []
    # Group entries by run_key to merge bouts and avoid concurrent artifact races
    by_run_key = {}
    for e in entries(manifest):
        rk = e["run_key"]
        if rk not in by_run_key:
            by_run_key[rk] = []
        by_run_key[rk].append(e)

    for run_key, es in by_run_key.items():
        # Use first entry for recording_cfg, session_dir, etc.
        e = es[0]
        # Sort bouts numerically to ensure deterministic order
        bouts = sorted([int(x["bout"]) for x in es])
        if len(bouts) == 1:
            bout_str = str(bouts[0])
        else:
            # Multiple bouts: use escaped single quotes for hydra sweep syntax
            bout_str = f"\\'{','.join(map(str, bouts))}\\'"

        parts = ["python", "scripts/run_bout.py", "paths=hyak",
                 f"recording={e['recording_cfg']}",
                 f"recording.session_dir={e['session_dir']}",
                 f"outputs.out={Path(variant_root) / run_key}",
                 f"bout_ids={bout_str}", *overrides]
        cmds.append(" ".join(shlex.quote(p) if " " in p else p for p in parts))
    return cmds


# fly_id_review.py's DEFAULT_MALE_FLY: bouts that predate sexing
# canonicalization (no per-bout sex.json yet) default to fly1=male,
# fly0=female -- the same convention id_review.json already records for
# every unreviewed bout it scans (source="default"). Falling back to it
# here (instead of raising) lets the benchmark cohort split run on bouts
# whose recording hasn't been through canonicalize_session_sex.py; treat
# those cohort labels as provisional pending a real review pass.
_DEFAULT_MALE_FLY = 1


def _male_fly(bout_dir_parent: Path) -> int:
    sex = bout_dir_parent / "sex.json"
    if sex.is_file():
        return int(json.loads(sex.read_text()).get("male_fly"))
    return _DEFAULT_MALE_FLY


def collect(manifest: dict, root_for_outputs: Path | None,
            mjcf_path: str | None, out_json: Path) -> dict:
    lb = ub = None
    if mjcf_path:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        lb, ub = joint_bounds(model)
    thr = float(manifest.get("proximity_threshold_bl", 2.0))
    per_bout = []
    for e in entries(manifest):
        for fly in e["flies"]:
            bd = bout_fly_dir({**e, "fly": fly}, root=root_for_outputs)
            partner = None
            if e["assay"] == "courtship" and len(e["flies"]) > 1:
                partner = bout_fly_dir({**e, "fly": 1 - fly},
                                       root=root_for_outputs)
            calib = Path(e["session_dir"]) / "calibration"
            res = compute_bout_metrics(bd, partner_dir=partner,
                                       calib_dir=calib if calib.is_dir() else None)
            if lb is not None:
                import h5py
                with h5py.File(bd / "outputs.h5", "r") as f:
                    res["scalars"]["jl_violation_rate"] = \
                        joint_limit_violation_rate(f["qpos"][:], lb, ub)
            prox = res["series"].get("proximity_bl")
            splits = {k: split_series(v, prox, thr)
                      for k, v in res["series"].items() if k != "proximity_bl"}
            per_bout.append({
                "cohort": cohort_of(e, fly, _male_fly(bd.parent)),
                "run_key": e["run_key"], "bout": e["bout"], "fly": fly,
                "tags": e.get("tags", []), "scalars": res["scalars"],
                "splits": splits})
    sc = build_scorecard(per_bout)
    write_scorecard(Path(out_json), sc)
    return sc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["build", "commands", "collect", "baseline"])
    ap.add_argument("--manifest", type=Path,
                    default=Path("configs/benchmark/bouts.yaml"))
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--freeze", action="append", default=None,
                    metavar="NAME.npz",
                    help="stage input(s) to inherit from frozen/ (repeatable). "
                         f"Default: all of {FROZEN_INPUTS}. A treatment that "
                         "changes an earlier stage must freeze only that "
                         "stage's inputs, e.g. --freeze kp2d.npz for a "
                         "triangulation change.")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--mjcf", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    manifest = load_manifest(args.manifest)
    dest_root = Path(manifest["benchmark_root"])
    if args.mode in ("build", "commands"):
        frozen_inputs = FROZEN_INPUTS if args.freeze is None else tuple(args.freeze)
        vroot = build_variant_root(manifest, dest_root, args.variant,
                                   frozen_inputs=frozen_inputs)
        if args.mode == "commands":
            for c in variant_commands(manifest, vroot, args.override):
                print(c)
    elif args.mode == "collect":
        vroot = dest_root / "variants" / args.variant
        collect(manifest, vroot, args.mjcf,
                args.out or vroot / "scorecard.json")
    else:  # baseline: metrics straight off the SOURCE dirs, no re-run
        collect(manifest, None, args.mjcf,
                args.out or dest_root / "baseline_scorecard.json")


if __name__ == "__main__":
    main()
