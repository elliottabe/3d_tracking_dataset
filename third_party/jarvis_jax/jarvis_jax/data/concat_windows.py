"""One window-dataset surface over SEVERAL v12-format roots (mvq-v2 spec §4).

v2 trains on the human v12 export PLUS a pseudo-label export written in the
same format (`data/pseudo_export.py`), and later a single-fly export and an
empty-window negatives export. They are separate roots -- separate
calibrations, images, manifests and annotation ids -- so they cannot be
merged into one `V12WindowDataset`; this class stacks them index-wise instead
and forwards every per-window question to the sub-dataset that owns it.

Everything downstream keeps working because it only ever asks the dataset
questions BY INDEX: `window_batches` (`len`, `__getitem__`, `epoch`),
`train_mvq._balanced_weights` (`windows`, `manifest`, `is_female`) and
`train_mvq._cohorts` (`is_female`, `n_flies`, `calib_group`, `fly_centroids`).
Per-window provenance (`source`, `weight`, `role`) resolves inside each
sub-dataset, so a pseudo root's 0.3 weight reaches `sample_weight` unchanged.

What it refuses, rather than silently mixing:
  * different `keypoint_names` (the keypoint axis would mean two different
    things in one batch -- CLAUDE.md's keypoint-order history);
  * different window lengths T, or different `pair_deltas` (the `crops` axis
    would not stack, and copy-paste pairs donors by spacing);
  * a recording that two roots describe DIFFERENTLY in any manifest field a
    loader or sampler reads (`_MERGE_CHECKED`) -- the merge is per field, so a
    field only one root defines is simply taken, but a real disagreement would
    silently change what a window means depending on merge order.

`calib_group` is checked by CONTENT, not name (Calibration witness test,
2026-09-06): the pseudo/negatives exports name a recording's calibration
group after the recording id, while the human root uses letters, so the same
recording legitimately carries two different `calib_group` NAMES across
roots. A name mismatch loads both `<root>/calibrations/<group>` dirs via
`ReprojectionTool` and compares `camera_matrices` BY CAMERA NAME
(`np.allclose(atol=1e-6)`): identical content (a serialisation difference,
e.g. `A` vs the recording id both meaning the same physical calibration) is
accepted silently and recorded in `calib_alias`; genuinely different content
raises ValueError naming the recording, both roots/groups and the max
|diff| -- UNLESS `allow_calib_mismatch=True`, which prints one WARNING per
recording and records it in `calib_mismatches` instead of raising. Either
way each sub-dataset keeps triangulating its own samples with its OWN
calibration dir (`V12WindowDataset.calib_group`/`_rt`); this class never
merges calibration files, only compares them.
"""
from __future__ import annotations

import bisect
import os

import numpy as np

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

# manifest[rec] fields some loader/sampler path READS, and its reader. A
# disagreement between two roots about one of these is a conflict, not a merge:
# last-one-wins would decide, invisibly, which root's description of the
# recording every window of BOTH roots is interpreted with. Fields not listed
# here (bookkeeping like `split`, `sex_source`, `checkpoint`) are not read by
# any window/sampler code path and merge last-one-wins.
_MERGE_CHECKED = {
    # Special-cased in __init__ (compared by CONTENT, not name) -- see module
    # docstring "Calibration witness test". Still listed here so it reads as a
    # checked field; the generic name-equality branch never fires for it.
    "calib_group": "V12WindowDataset.calib_group/_rt -- which calibration triangulates it",
    "sex": "v5_3d._resolve_sex fallback, and unlabelled_sex's 'mixed' branch",
    "fly_sex": "v5_3d._resolve_sex per-fly fallback, and unlabelled_sex",
    "n_flies": "unlabelled_sex -- is an animal present but unlabelled?",
    "behavior": "train_mvq._balanced_weights -- the sampler's balance category",
    "source": "V12WindowDataset._fs_field -> source() -- real vs pseudo",
    "weight": "V12WindowDataset._fs_field -> weight() -> the sample's sample_weight",
    "role": "V12WindowDataset._fs_field -> role()",
}


def _calib_matrices_by_name(calib_dir):
    """{camera name: (4,3) float64 affine matrix} for one `calibrations/<group>`
    dir, keyed the way `ReprojectionTool` itself pairs them (`.cameras` and
    `.camera_matrices` are built from the same iteration, so zipping them is
    safe) -- comparing BY NAME is what makes this correct across two roots
    whose camera files needn't glob-sort the same way."""
    rt = ReprojectionTool(calib_dir)
    return {name: np.asarray(mat, np.float64) for name, mat in zip(rt.cameras, rt.camera_matrices)}


def _max_calib_diff(dir_a, dir_b):
    """Max |diff| between two calibration dirs' camera matrices, matched BY
    NAME. Raises ValueError (unconditionally -- this is a structural problem,
    not a content-drift one `allow_calib_mismatch` is meant to paper over) if
    the two dirs do not calibrate the same camera set."""
    mats_a, mats_b = _calib_matrices_by_name(dir_a), _calib_matrices_by_name(dir_b)
    if set(mats_a) != set(mats_b):
        raise ValueError(f"calibration dirs {dir_a!r} and {dir_b!r} calibrate different camera "
                         f"sets: {sorted(mats_a)} vs {sorted(mats_b)}")
    return max(float(np.max(np.abs(mats_a[name] - mats_b[name]))) for name in mats_a)


class ConcatWindowDataset:
    def __init__(self, datasets, names=None, allow_calib_mismatch=False):
        self.datasets = list(datasets)
        if not self.datasets:
            raise ValueError("ConcatWindowDataset needs at least one dataset")
        self.names = list(names) if names is not None else [str(getattr(d, "root", i))
                                                            for i, d in enumerate(self.datasets)]
        if len(self.names) != len(self.datasets):
            raise ValueError(f"names has {len(self.names)} entries for {len(self.datasets)} datasets")
        first = self.datasets[0]
        for d, nm in zip(self.datasets[1:], self.names[1:]):
            if list(d.keypoint_names) != list(first.keypoint_names):
                bad = next((a, b) for a, b in zip(first.keypoint_names, d.keypoint_names) if a != b) \
                    if len(d.keypoint_names) == len(first.keypoint_names) else None
                raise ValueError(
                    f"keypoint_names differ between roots {first.root!r} ({self.names[0]}) and "
                    f"{d.root!r} ({nm}): {len(first.keypoint_names)} vs {len(d.keypoint_names)} names"
                    + (f", first mismatch {bad[0]!r} vs {bad[1]!r}" if bad else "")
                    + " -- one keypoint axis cannot mean two orders")
            if int(d.T) != int(first.T):
                raise ValueError(f"window length T differs between roots {first.root!r} (T={first.T}) "
                                 f"and {d.root!r} (T={d.T}): the crops axis would not stack")
            if tuple(d.pair_deltas) != tuple(first.pair_deltas):
                raise ValueError(f"pair_deltas differ between roots {first.root!r} "
                                 f"({tuple(first.pair_deltas)}) and {d.root!r} ({tuple(d.pair_deltas)}): "
                                 f"the two roots would contribute windows of different spacings")
        self.keypoint_names = list(first.keypoint_names)
        self.T = int(first.T)
        self.allow_calib_mismatch = bool(allow_calib_mismatch)
        # rec -> {root name: that root's OWN calib_group name}, populated only
        # when two roots named the same recording's calibration DIFFERENTLY but
        # the calibration CONTENT matched (a serialisation difference, e.g. a
        # letter group vs the recording-id group the pseudo exports use) --
        # "for logging"; nothing downstream reads it.
        self.calib_alias = {}
        # [{"recording", "roots": [{"name","root","calib_group"}, ...], "max_diff"}]
        # one entry per recording, only when `allow_calib_mismatch=True` let a
        # genuine content disagreement through instead of raising.
        self.calib_mismatches = []
        # Merged per-recording manifest, PER FIELD (`_balanced_weights` reads
        # manifest[rec]["behavior"], `_resolve_sex` reads sex/fly_sex, ...). A
        # recording two roots describe differently in any field a loader reads is
        # a real conflict, not a merge-order question; a field only one root
        # defines is simply taken.
        self.manifest = {}
        owner = {}                      # rec -> {field: (root name, root path)}
        _calib_warned = set()           # rec -- so a 3rd+ root shares ONE warning/entry
        for d, nm in zip(self.datasets, self.names):
            for rec, meta in d.manifest.items():
                merged = self.manifest.setdefault(rec, {})
                who = owner.setdefault(rec, {})
                for field, value in meta.items():
                    if field == "calib_group" and "calib_group" in merged and merged[field] != value:
                        # NAMES differ -- the generic equality check above would
                        # raise here; compare CONTENT instead (module docstring).
                        prev_name, prev_root = who["calib_group"]
                        prev_group = merged["calib_group"]
                        prev_dir = os.path.join(prev_root, "calibrations", str(prev_group))
                        this_dir = os.path.join(d.root, "calibrations", str(value))
                        max_diff = _max_calib_diff(prev_dir, this_dir)
                        if max_diff <= 1e-6:
                            alias = self.calib_alias.setdefault(rec, {})
                            alias[prev_name] = prev_group
                            alias[nm] = value
                        elif rec not in _calib_warned:
                            _calib_warned.add(rec)
                            msg = (f"recording {rec!r}: manifest field 'calib_group' is "
                                  f"{prev_group!r} in root {prev_name} ({prev_root}) and {value!r} "
                                  f"in root {nm} ({d.root}). The calibration CONTENT differs by max "
                                  f"|diff| {max_diff:.6g} (not a serialisation rounding difference) "
                                  f"-- which one is right is undetermined here, so this refuses to "
                                  f"silently pick one. Pass allow_calib_mismatch=True to proceed "
                                  f"with each root using its OWN calibration per sample.")
                            if not self.allow_calib_mismatch:
                                raise ValueError(msg)
                            print(f"WARNING: {msg}", flush=True)
                            self.calib_mismatches.append({
                                "recording": rec,
                                "roots": [{"name": prev_name, "root": prev_root,
                                          "calib_group": prev_group},
                                         {"name": nm, "root": d.root, "calib_group": value}],
                                "max_diff": max_diff,
                            })
                        # canonical merged value/owner stays the first-seen one --
                        # every later root is compared against the SAME baseline.
                        continue
                    if field in _MERGE_CHECKED and field in merged and merged[field] != value:
                        pn, pr = who[field]
                        raise ValueError(
                            f"recording {rec!r}: manifest field {field!r} is {merged[field]!r} in "
                            f"root {pn} ({pr}) and {value!r} in root {nm} ({d.root}). It is read by "
                            f"{_MERGE_CHECKED[field]}, so the two roots would describe the same "
                            f"recording differently depending on merge order")
                    merged[field] = value
                    who[field] = (nm, d.root)
        self._cum = []
        n = 0
        for d in self.datasets:
            n += len(d)
            self._cum.append(n)
        self.windows = [w for d in self.datasets for w in d.windows]
        self.win_delta = [d.delta(k) for d in self.datasets for k in range(len(d))]
        self._epoch = int(getattr(first, "epoch", 0))

    # ------------------------------------------------------------------ index
    def __len__(self):
        return self._cum[-1] if self._cum else 0

    def which(self, i):
        """(sub-dataset index, index WITHIN that dataset) for global index i."""
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(f"index {i} out of range for {len(self)} windows")
        d = bisect.bisect_right(self._cum, i)
        return d, i - (self._cum[d - 1] if d else 0)

    def _at(self, i):
        d, k = self.which(i)
        return self.datasets[d], k

    def __getitem__(self, i):
        ds, k = self._at(i)
        return ds[k]

    # --------------------------------------------------------------- delegates
    def name(self, i):
        """Which root window i came from (the `names` this concat was built with)."""
        return self.names[self.which(i)[0]]

    def camera_names(self, i):
        ds, k = self._at(i); return ds.camera_names(k)

    def calib_group(self, i):
        ds, k = self._at(i); return ds.calib_group(k)

    def is_female(self, i):
        ds, k = self._at(i); return ds.is_female(k)

    def n_flies(self, i):
        ds, k = self._at(i); return ds.n_flies(k)

    def fly_centroids(self, i):
        ds, k = self._at(i); return ds.fly_centroids(k)

    def unlabelled_sex(self, i):
        ds, k = self._at(i); return ds.unlabelled_sex(k)

    def window_fly_sex(self, i):
        ds, k = self._at(i); return ds.window_fly_sex(k)

    def source(self, i):
        ds, k = self._at(i); return ds.source(k)

    def weight(self, i):
        ds, k = self._at(i); return ds.weight(k)

    def role(self, i):
        ds, k = self._at(i); return ds.role(k)

    def delta(self, i):
        ds, k = self._at(i); return ds.delta(k)

    def paste_window(self, i, rng):
        ds, k = self._at(i); return ds.paste_window(k, rng)

    def window_index(self, rec, fly, f0, delta=None):
        """Global index of ONE window BY KEY. Every sub-dataset is asked; a key
        present in more than one root is ambiguous and raises ValueError (which
        root's pixels did the caller mean?)."""
        hits = []
        for d, (ds, off) in enumerate(zip(self.datasets, [0] + self._cum[:-1])):
            try:
                hits.append(off + ds.window_index(rec, fly, f0, delta))
            except KeyError:
                continue
        if not hits:
            raise KeyError((rec, int(fly), int(f0), delta))
        if len(hits) > 1:
            raise ValueError(f"{(rec, int(fly), int(f0))} exists in {len(hits)} of the concatenated "
                             f"roots ({[self.name(i) for i in hits]}) -- index the sub-dataset directly")
        return hits[0]

    # ----------------------------------------------------------------- epoch
    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        """`window_batches` sets `ds.epoch = seed` once per epoch, and each
        sub-dataset seeds its own per-sample jitter/copy-paste RNG on it -- so
        the write has to reach all of them or half the concat would repeat the
        same augmentation every epoch."""
        self._epoch = int(value)
        for d in self.datasets:
            d.epoch = int(value)
