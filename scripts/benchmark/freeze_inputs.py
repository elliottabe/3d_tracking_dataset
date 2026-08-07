"""Freeze benchmark 2D/3D inputs so every variant re-runs from identical data."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scripts.benchmark.manifest import bout_fly_dir, entries

REQUIRED = ("kp2d.npz", "kp3d.npz")
OPTIONAL = ("kp3d_filt.npz",)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(manifest: dict, dest_root: Path) -> dict:
    dest_root = Path(dest_root)
    frozen = dest_root / "frozen"
    sums: dict[str, str] = {}

    def copy(src: Path, rel: Path):
        dst = frozen / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sums[str(rel)] = _sha256(dst)

    for e in entries(manifest):
        for fly in e["flies"]:
            src_dir = bout_fly_dir({**e, "fly": fly})
            rel_dir = (Path(e["run_key"]) / "bouts"
                       / f"bout_{int(e['bout']):05d}" / f"fly{fly}")
            for name in REQUIRED:
                if not (src_dir / name).is_file():
                    raise FileNotFoundError(f"{e['run_key']} bout {e['bout']} "
                                            f"fly{fly}: missing {name}")
                copy(src_dir / name, rel_dir / name)
            for name in OPTIONAL:
                if (src_dir / name).is_file():
                    copy(src_dir / name, rel_dir / name)
        if e["assay"] == "courtship":
            sex = src_dir.parent / "sex.json"          # bout-level
            if sex.is_file():
                copy(sex, rel_dir.parent / "sex.json")
    frozen.mkdir(parents=True, exist_ok=True)
    (frozen / "checksums.json").write_text(json.dumps(sums, indent=2,
                                                      sort_keys=True))
    return sums


def verify(dest_root: Path) -> list[str]:
    frozen = Path(dest_root) / "frozen"
    sums = json.loads((frozen / "checksums.json").read_text())
    return [rel for rel, digest in sums.items()
            if not (frozen / rel).is_file() or _sha256(frozen / rel) != digest]
