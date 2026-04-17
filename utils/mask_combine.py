"""Cross-folder SAM3 mask concatenation.

Used by the combine step of the 3D-tracking pipeline to merge every per-folder
``sam3_aligned.h5`` (written by ``utils.fly_detection.aggregate_per_bout_predictions``)
into a single combined masks file that lives next to ``ik_output_combined*.h5``.

Per-folder files are already sex-swap-aligned so ``[A=0]=male, [A=1]=female``;
concatenation along the frame axis preserves that convention. Derived heading /
relative-angle arrays are concatenated in lock-step.
"""

from pathlib import Path
from typing import List

import h5py
import numpy as np


def concatenate_per_folder_masks(folder_files: List[Path],
                                 out_path: Path) -> None:
    """Stream-concatenate per-folder sam3_aligned.h5 files along the frame axis.

    Writes ``out_path`` with:
        /mask_packed, /centroids, /valid — concatenated across folders
        /folder_boundaries [N_folders, 2] — [start_row, end_row_inclusive]
        /folder_paths — UTF-8 source paths in the same order as /folder_boundaries
        /bout_boundaries — concatenated with folder offsets added to each row
        /bout_frames — concatenated as-is (absolute frame numbers)
        /sex_swaps — concatenated across folders
        /derived/* — concatenated along frame axis

    Root attrs (``H``, ``W``, ``W_packed``, ``fps``, ``layout``) are copied from
    the first input; all inputs must share the same camera count, H, and W.
    """
    if not folder_files:
        return

    # Copy masks in frame-axis chunks so memory stays bounded regardless of
    # how many frames are in a single source file.
    FRAME_CHUNK = 128

    folder_starts: List[int] = []
    folder_ends: List[int] = []
    bout_boundaries_chunks: List[np.ndarray] = []
    bout_frames_chunks: List[np.ndarray] = []
    sex_swaps_chunks: List[np.ndarray] = []
    derived_chunks: dict[str, List[np.ndarray]] = {}
    row_offset = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as fo:
        mask_ds = cent_ds = valid_ds = None

        for src in folder_files:
            with h5py.File(src, 'r') as fi:
                pack = fi['mask_packed']
                cent = fi['centroids']
                vld = fi['valid']
                A, C, F, H, W_pack = pack.shape

                if mask_ds is None:
                    mask_ds = fo.create_dataset(
                        'mask_packed', shape=(A, C, 0, H, W_pack),
                        maxshape=(A, C, None, H, W_pack), dtype='uint8',
                        compression='gzip', compression_opts=4,
                        chunks=(A, C, max(1, min(64, F)), H, W_pack),
                    )
                    cent_ds = fo.create_dataset(
                        'centroids', shape=(A, C, 0, 2),
                        maxshape=(A, C, None, 2), dtype='float32',
                    )
                    valid_ds = fo.create_dataset(
                        'valid', shape=(A, C, 0),
                        maxshape=(A, C, None), dtype='bool',
                    )
                    for k, v in fi.attrs.items():
                        fo.attrs[k] = v
                else:
                    expected = (mask_ds.shape[0], mask_ds.shape[1],
                                mask_ds.shape[3], mask_ds.shape[4])
                    if (A, C, H, W_pack) != expected:
                        raise ValueError(
                            f"Shape mismatch between {folder_files[0]} and {src}: "
                            f"expected (A,C,H,W_pack)={expected}, "
                            f"got ({A},{C},{H},{W_pack})"
                        )

                new_size = row_offset + F
                mask_ds.resize((A, C, new_size, H, W_pack))
                cent_ds.resize((A, C, new_size, 2))
                valid_ds.resize((A, C, new_size))
                # Stream masks in frame-axis chunks to bound peak memory.
                for cs in range(0, F, FRAME_CHUNK):
                    ce = min(cs + FRAME_CHUNK, F)
                    mask_ds[:, :, row_offset + cs:row_offset + ce, :, :] = \
                        pack[:, :, cs:ce, :, :]
                cent_ds[:, :, row_offset:new_size, :] = cent[:]
                valid_ds[:, :, row_offset:new_size] = vld[:]

                folder_starts.append(row_offset)
                folder_ends.append(new_size - 1)

                if 'bout_boundaries' in fi:
                    bb = fi['bout_boundaries'][:].astype(np.int64)
                    bb = bb + row_offset
                    bout_boundaries_chunks.append(bb)
                if 'bout_frames' in fi:
                    bout_frames_chunks.append(fi['bout_frames'][:].astype(np.int64))
                if 'sex_swaps' in fi:
                    sex_swaps_chunks.append(fi['sex_swaps'][:].astype(bool))

                if 'derived' in fi:
                    d_grp = fi['derived']
                    for k in d_grp.keys():
                        derived_chunks.setdefault(k, []).append(d_grp[k][:])

                row_offset = new_size

        fo.create_dataset(
            'folder_boundaries',
            data=np.stack([np.asarray(folder_starts, dtype=np.int64),
                           np.asarray(folder_ends, dtype=np.int64)], axis=1),
        )
        fo.create_dataset(
            'folder_paths',
            data=np.asarray([str(p) for p in folder_files],
                            dtype=h5py.string_dtype(encoding='utf-8')),
        )
        if bout_boundaries_chunks:
            fo.create_dataset('bout_boundaries',
                              data=np.concatenate(bout_boundaries_chunks, axis=0))
        if bout_frames_chunks:
            fo.create_dataset('bout_frames',
                              data=np.concatenate(bout_frames_chunks, axis=0))
        if sex_swaps_chunks:
            fo.create_dataset('sex_swaps',
                              data=np.concatenate(sex_swaps_chunks, axis=0))

        if derived_chunks:
            g_out = fo.create_group('derived')
            for k, chunks in derived_chunks.items():
                g_out.create_dataset(k, data=np.concatenate(chunks, axis=0),
                                     compression='gzip')
            with h5py.File(folder_files[0], 'r') as fi:
                if 'derived' in fi:
                    for ak, av in fi['derived'].attrs.items():
                        g_out.attrs[ak] = av
