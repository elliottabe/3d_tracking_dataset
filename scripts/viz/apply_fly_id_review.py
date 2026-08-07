"""Apply reviewed fly identities: physically swap fly0/fly1 bout dirs.

Reads <root>/id_review.json (written by scripts/viz/fly_id_review.py). For
every bout whose reviewed male currently lives in fly0/, swaps the two fly
dirs so the whole tree is uniformly fly0 = female, fly1 = male, and updates
the bout's sex.json (male_fly: 1, applied_swap: true).

DRY-RUN BY DEFAULT -- prints the full plan. Pass --apply to execute.
Refuses to run while any bout is pending or unsure; --allow-pending treats
pending bouts as confirmed-original (unsure always blocks).

Touches ONLY pose/bouts/bout_*/fly{0,1} dirs, sex.json, and the manifest.

Usage:
    python scripts/viz/apply_fly_id_review.py                 # dry-run
    python scripts/viz/apply_fly_id_review.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.viz.fly_id_review import (
    DEFAULT_ROOT,
    MANIFEST_NAME,
    bout_dir_from_key,
    load_manifest,
    read_sex_json,
    save_manifest,
)

TMP_NAME = ".fly_swap_tmp"


def validate(manifest: dict, allow_pending: bool) -> list[str]:
    """Blockers that must be resolved before --apply may run."""
    errors = []
    for key, e in sorted(manifest["bouts"].items()):
        if e["status"] == "unsure":
            errors.append(f"{key}: unsure (resolve in the GUI first)")
        elif e["status"] == "pending" and not allow_pending:
            errors.append(f"{key}: pending (review it, or pass --allow-pending)")
    return errors


def plan_swaps(manifest: dict, allow_pending: bool = False) -> list[str]:
    """Bout keys whose reviewed male currently lives in fly0/ (need a swap)."""
    ok_status = {"confirmed", "swapped"} | ({"pending"} if allow_pending else set())
    return [key for key, e in sorted(manifest["bouts"].items())
            if e["reviewed_male_fly"] == 0
            and not e.get("applied")
            and e["status"] in ok_status
            and not e.get("warning")]


def swap_bout(root: Path, key: str, entry: dict) -> None:
    """Swap fly0/ <-> fly1/ for one bout and rewrite its sex.json."""
    bout_dir = bout_dir_from_key(root, key)
    tmp = bout_dir / TMP_NAME
    if tmp.exists():
        raise RuntimeError(
            f"{key}: stale {TMP_NAME} from an interrupted swap -- inspect manually")
    (bout_dir / "fly0").rename(tmp)
    (bout_dir / "fly1").rename(bout_dir / "fly0")
    tmp.rename(bout_dir / "fly1")
    existing = read_sex_json(bout_dir) or {}
    sex = dict(existing)
    sex.update({
        "male_fly": 1,
        "original_male_fly": existing.get("original_male_fly",
                                          entry["original_male_fly"]),
        "applied_swap": True,
        "confidence": "user",
        "method": "manual-gui",
    })
    tmp_json = bout_dir / "sex.json.tmp"
    tmp_json.write_text(json.dumps(sex, indent=2))
    os.replace(tmp_json, bout_dir / "sex.json")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true",
                        help="execute the swaps (default: dry-run)")
    parser.add_argument("--allow-pending", action="store_true",
                        help="treat pending bouts as confirmed-original")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    manifest = load_manifest(root)
    if manifest is None:
        raise SystemExit(f"no {MANIFEST_NAME} under {root} -- run fly_id_review.py first")

    errors = validate(manifest, args.allow_pending)
    if errors:
        print(f"REFUSING to apply -- {len(errors)} unresolved bout(s):")
        for line in errors:
            print(f"  {line}")
        raise SystemExit(1)

    swaps = plan_swaps(manifest, args.allow_pending)
    warned = sorted(k for k, e in manifest["bouts"].items() if e.get("warning"))
    print(f"{len(manifest['bouts'])} bouts total; {len(swaps)} need a fly0<->fly1 swap; "
          f"{len(warned)} excluded (warnings)")
    for key in swaps:
        print(f"  swap: {key}")
    for key in warned:
        e = manifest["bouts"][key]
        flag = " (WOULD NEED SWAP -- fix manually)" if e["reviewed_male_fly"] == 0 else ""
        print(f"  excluded: {key} [{e['warning']}]{flag}")
    if not args.apply:
        print("dry-run only -- pass --apply to execute")
        return

    for key in swaps:
        entry = manifest["bouts"][key]
        swap_bout(root, key, entry)
        entry["applied"] = True
        entry["reviewed_male_fly"] = 1  # male now lives in fly1
        save_manifest(root, manifest)   # crash-safe: persist after every bout
        print(f"  swapped: {key}")
    print(f"done: {len(swaps)} bout(s) swapped; tree is now fly0=female, fly1=male")


if __name__ == "__main__":
    main()
