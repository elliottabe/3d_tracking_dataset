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
  * different window lengths T (the `crops` axis would not stack);
  * the same recording described with two different `calib_group`s by two
    roots (the same fly would triangulate two ways, and the merged
    `manifest[rec]` could only hold one of them).
"""
from __future__ import annotations

import bisect


class ConcatWindowDataset:
    def __init__(self, datasets, names=None):
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
        self.keypoint_names = list(first.keypoint_names)
        self.T = int(first.T)
        # Merged per-recording manifest (`_balanced_weights` reads
        # manifest[rec]["behavior"]). A recording two roots describe differently
        # is a real conflict, not a merge order question.
        self.manifest = {}
        owner = {}
        for d, nm in zip(self.datasets, self.names):
            for rec, meta in d.manifest.items():
                prev = self.manifest.get(rec)
                if prev is not None and prev.get("calib_group") != meta.get("calib_group"):
                    raise ValueError(
                        f"recording {rec!r} has calib_group {prev.get('calib_group')!r} in "
                        f"{owner[rec]} and {meta.get('calib_group')!r} in {nm}: the two roots "
                        f"disagree about how to triangulate the same fly")
                self.manifest[rec] = meta
                owner[rec] = nm
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
