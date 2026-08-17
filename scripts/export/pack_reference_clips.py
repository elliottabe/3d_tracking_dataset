"""Pack a combined per-bout IK h5 into a batched reference-clip dataset.

Replaces cells 75-77 of fly_mimic/notebooks/Walking_data_cleaning.ipynb.

Output layout (consumed by fly_mimic's ReferenceClips / HDF5ReferenceClips):
    qpos         (N, T_max, nq)
    qvel         (N, T_max, nv)
    xpos         (N, T_max, nbody, 3)
    xquat        (N, T_max, nbody, 4)
    kp_data      (N, T_max, 150)
    clip_lengths (N,) int32   -- TRUE unpadded lengths (see below)
    qpos_names   group of nq scalar strings, one per COLUMN

Padding repeats the final frame, matching the reference file.

Deliberate deviation: the v1 reference file stores the PADDED length in
clip_lengths for every clip, and fly_mimic compensates with
unpadded_clip_lengths() sniffing repeated trailing frames. We store the true
lengths instead; the file still loads, and use_unpadded_clip_length becomes
unnecessary. The root attr clip_lengths_are_true records this.

Usage:
    python scripts/export/pack_reference_clips.py \
        --input  .../ik_output_combined_v2_3_free_running_interpolated.h5 \
        --output .../Fruitfly_v2_3_walk_1000hz_interp_padded.h5 \
        --anatomy v2_3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from utils.stac_data_utils import sorted_bout_keys

ARRAY_KEYS = ['qpos', 'qvel', 'xpos', 'xquat', 'kp_data']


def pack_clips(bouts: list[dict], names_qpos: list[str]) -> dict:
    """Stack per-bout arrays into (N, T_max, ...) with final-frame padding."""
    if not bouts:
        raise ValueError('no bouts to pack')

    nq = bouts[0]['qpos'].shape[1]
    if len(names_qpos) != nq:
        raise ValueError(
            f'qpos_names has {len(names_qpos)} entries but qpos has {nq} '
            f'columns; fly_mimic requires one name per column')

    lengths = [int(b['qpos'].shape[0]) for b in bouts]
    t_max = max(lengths)

    out: dict = {}
    for key in ARRAY_KEYS:
        stacked = []
        for b in bouts:
            arr = np.asarray(b[key])
            n_pad = t_max - arr.shape[0]
            if n_pad:
                pad = np.tile(arr[-1:], (n_pad,) + (1,) * (arr.ndim - 1))
                arr = np.concatenate([arr, pad], axis=0)
            stacked.append(arr)
        out[key] = np.stack(stacked, axis=0).astype(np.float32)

    out['clip_lengths'] = np.asarray(lengths, dtype=np.int32)
    out['qpos_names'] = list(names_qpos)
    return out


def _decode(values) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def load_bouts(path: Path) -> tuple[list[dict], list[str]]:
    """Read bout_NNN groups in sorted order, plus info/names_qpos."""
    with h5py.File(path, 'r') as f:
        # NUMERIC order: mixed zero-pad widths (bout_999/bout_1000) sort
        # lexicographically out of order vs the index-ordered info arrays.
        keys = sorted_bout_keys(k for k in f if k.startswith('bout'))
        bouts = [{k: np.asarray(f[bk][k]) for k in ARRAY_KEYS} for bk in keys]
        info = f['info']
        raw = info['names_qpos']
        if isinstance(raw, h5py.Group):
            names = _decode([raw[str(i)][()] for i in range(len(raw))])
        else:
            names = _decode(np.asarray(raw))
    return bouts, names


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[2],
            text=True).strip()
    except Exception:
        return 'unknown'


def write_h5(out: dict, path: Path, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        for key in ARRAY_KEYS + ['clip_lengths']:
            f.create_dataset(key, data=out[key], compression='gzip',
                             compression_opts=5)
        grp = f.create_group('qpos_names')
        for i, name in enumerate(out['qpos_names']):
            grp[str(i)] = name
        for k, v in attrs.items():
            f.attrs[k] = v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--anatomy', default='v2_3')
    ap.add_argument('--source-hz', type=float, default=800.0)
    ap.add_argument('--target-hz', type=float, default=1000.0)
    args = ap.parse_args()

    bouts, names = load_bouts(args.input)
    out = pack_clips(bouts, names)

    write_h5(out, args.output, {
        'source_file': str(args.input),
        'anatomy': args.anatomy,
        'source_hz': args.source_hz,
        'target_hz': args.target_hz,
        'git_sha': _git_sha(),
        'clip_lengths_are_true': True,
    })

    print(f'wrote {args.output}')
    print(f'  clips={out["qpos"].shape[0]} T_max={out["qpos"].shape[1]} '
          f'nq={out["qpos"].shape[2]} nbody={out["xpos"].shape[2]}')
    print(f'  clip_lengths min={out["clip_lengths"].min()} '
          f'max={out["clip_lengths"].max()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
