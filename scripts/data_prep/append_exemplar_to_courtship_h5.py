"""Replace the Session0/2025_10_20_13_20_04 exemplar bout in a copy of the
Session0 combined-both h5 with the full-length (2006 frames) per-fly IK h5
data. The notebook's ``load_and_merge_courtship_h5`` then merges this file
with the main h5 normally, and the exemplar carries the full frame range.

Run once to produce a new Session0 combined h5; the notebook then loads that
file in place of the original and no in-notebook splice is needed.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np


SRC_H5 = Path('/data2/users/eabe/datasets/Johnson_lab/courtship/Session0/'
              '2025_10_20_13_20_04/v1/ik_output_combined_v1_courtship_both.h5')
DST_H5 = SRC_H5.with_name(
    'ik_output_combined_v1_courtship_both_full_exemplar.h5'
)

EX_FLY0 = Path('/data2/users/eabe/datasets/Johnson_lab/courtship/Session0/'
               '2025_10_20_13_20_04/Predictions_3D_V4_phase4_enumfix/v1/'
               'ik_output_combined_v1_courtship_fly0.h5')
EX_FLY1 = EX_FLY0.with_name(EX_FLY0.name.replace('fly0', 'fly1'))

# Start frame of the bout pair to replace (Session0 h5 entries [30]/[31]).
TARGET_START_FRAME = 446306


def _delete_datasets(group: h5py.Group) -> None:
    """Recursively empty a group (remove all child Datasets and Groups)."""
    for name in list(group.keys()):
        del group[name]


def _copy_datasets(src_group: h5py.Group, dst_group: h5py.Group) -> None:
    for name, obj in src_group.items():
        if isinstance(obj, h5py.Dataset):
            dst_group.create_dataset(name, data=obj[...])
        elif isinstance(obj, h5py.Group):
            sub = dst_group.create_group(name)
            _copy_datasets(obj, sub)


def _find_pair_indices(info: h5py.Group, start_frame: int) -> tuple[int, int]:
    """Locate the (fly0, fly1) info-indices matching ``start_frame``."""
    starts = info['start_frames']
    sf = info['source_flies']
    keys = sorted((k for k in starts.keys() if k.isdigit()), key=int)
    i0 = i1 = None
    for k in keys:
        if int(starts[k][...]) != start_frame:
            continue
        src = sf[k][...]
        if hasattr(src, 'item'):
            src = src.item()
        src = src.decode() if isinstance(src, bytes) else str(src)
        if src == 'fly0' and i0 is None:
            i0 = int(k)
        elif src == 'fly1' and i1 is None:
            i1 = int(k)
    if i0 is None or i1 is None:
        raise KeyError(f'no (fly0, fly1) pair at start_frame={start_frame}')
    return i0, i1


def _write_scalar(group: h5py.Group, key: str, value) -> None:
    if key in group:
        del group[key]
    group.create_dataset(key, data=value)


def main() -> None:
    assert SRC_H5.exists(), SRC_H5
    assert EX_FLY0.exists(), EX_FLY0
    assert EX_FLY1.exists(), EX_FLY1

    if DST_H5.exists():
        DST_H5.unlink()
    print(f'[copy] {SRC_H5.name}\n   ->  {DST_H5.name}')
    shutil.copy2(SRC_H5, DST_H5)

    with h5py.File(DST_H5, 'a') as dst, \
         h5py.File(EX_FLY0, 'r') as f0, \
         h5py.File(EX_FLY1, 'r') as f1:
        i0, i1 = _find_pair_indices(dst['info'], TARGET_START_FRAME)
        bout_keys = sorted(k for k in dst.keys() if k.startswith('bout_'))
        k0, k1 = bout_keys[i0], bout_keys[i1]
        print(f'[replace] pair at start_frame={TARGET_START_FRAME}: '
              f'info_idx=({i0},{i1})  bouts=({k0},{k1})')

        # 1. Replace bout groups in place.
        for key, src_file in ((k0, f0), (k1, f1)):
            _delete_datasets(dst[key])
            _copy_datasets(src_file['bout_000'], dst[key])
            print(f'  rewrote {key} from {Path(src_file.filename).name}  '
                  f'T={src_file["bout_000/kp_data"].shape[0]}')

        # 2. Update per-bout info fields (end_frames, clip_lengths) to match.
        info = dst['info']
        new_end   = int(f0['info/end_frames/0'][...])
        new_clip  = int(f0['info/clip_lengths/0'][...])
        for k in (str(i0), str(i1)):
            _write_scalar(info['end_frames'],   k, np.int64(new_end))
            _write_scalar(info['clip_lengths'], k, np.int64(new_clip))
        print(f'  updated info/end_frames={new_end}  clip_lengths={new_clip}')

    print(f'[done] output: {DST_H5}')


if __name__ == '__main__':
    main()
