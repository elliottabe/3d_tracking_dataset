"""Content-keyed identity for the red_data image corpora.

WHY THIS EXISTS. `split_v5` splits by RECORDING NAME. That is only a valid
split key if a name identifies footage -- and in this dataset it does not.
The same physical capture was ingested more than once under different session
names, so a name-keyed split cannot see that it is putting the same pixels on
both sides. Measured 2026-09-02 over all 39,536 upstream jpgs (md5, no
sampling): 20,125 distinct contents, 19,411 redundant copies, and SIX
recording-name pairs/groups that are literally the same footage:

    2026_01_13_18_47_45 == 2026_03_22_12_07_40         (2,709 frames)
    2026_03_09_14_39_40 == 2026_04_07_11_33_33 == 2026_04_08_14_59_45
    2026_06_11_13_58_43 == 2026_06_11_13_58_45
                        == 2026_06_15_12_12_33 == 2026_06_15_12_12_34
    2026_05_27_11_56_05 == 2026_05_27_11_57_05         (105 frames, ALL)
    2026_06_18_19_23_03 == 2026_06_19_11_09_36         (161 frames, ALL)
    2026_06_09_15_38_35 == 2026_06_10_15_05_02         (7 frames)

Three of those groups are the SAME footage labelled once per fly and then
filed under a name one second apart (`courtship_28_34_male` /
`_female` -> `..._12_12_33` / `..._12_12_34`). Three of the four
`VAL_RECORDINGS` in build_v5_dataset.py sit in such a group with their twin
in train, which is why 29.6% of val images were byte-identical to a train
image.

THE UNIT. Duplicates are exact at whole-frameset granularity: every
cross-recording duplicate group carries ONE frame number and ONE camera, and
all 7 cameras of a duplicated frame duplicate together (1,225/1,225 groups,
zero partial). So the correct atom for a split is

    capture = (alias component, frame number)

which subsumes content identity (a content group never spans two captures),
the frameset (all 7 cameras of a frame), and the two-flies-one-capture rule
`split_v5` already enforces. Assign captures, and train/val content overlap
is zero by construction -- `content_overlap` still asserts it, because
"by construction" is what the name-keyed split also believed.

FRAME NUMBERS ARE NOT A CLOCK. Do not merge recordings on frame-number
proximity: counters free-run per session, so unrelated recordings collide
(2026_01_29_14_09_33 female and 2026_06_01_15_34_04 grooming male reach
min |dframe| = 4 while sharing zero bytes). Byte identity is the only
evidence used here.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

_FRAME_RE = re.compile(r"Frame_(\d+)")


def frame_no(path: str) -> int:
    """Frame number from a `.../Cam<serial>/Frame_<n>.jpg` path."""
    m = _FRAME_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"no Frame_<n> in {path!r}")
    return int(m.group(1))


def md5_file(path: str, _buf: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(_buf):
            h.update(chunk)
    return h.hexdigest()


def hash_paths(paths, workers: int = 64) -> dict[str, str]:
    """md5 every path. IO-latency-bound on GPFS, so thread-parallel: measured
    5.7 files/s serial vs 488 files/s at 192 threads (39,536 files in 81 s).
    A size-only prefilter was measured and rejected -- jpeg sizes collide, it
    ruled out only 11% of files."""
    paths = list(paths)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        digests = list(ex.map(md5_file, paths))
    return dict(zip(paths, digests))


def content_groups(path_hash: dict[str, str]) -> dict[str, list[str]]:
    """md5 -> every path holding those bytes."""
    out: dict[str, list[str]] = defaultdict(list)
    for p, h in path_hash.items():
        out[h].append(p)
    return {h: sorted(ps) for h, ps in out.items()}


class _DSU:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def alias_components(path_hash: dict[str, str], recording_of) -> dict[str, str]:
    """recording -> component id, where two recordings share a component iff
    some image is byte-identical between them (transitively).

    The component id is the lexicographically smallest member, so it is stable
    across runs and readable in a manifest. Recordings with no alias map to
    themselves, which makes this a drop-in replacement for "the recording" as
    a holdout unit."""
    dsu = _DSU()
    for h, paths in content_groups(path_hash).items():
        recs = sorted({recording_of(p) for p in paths})
        for r in recs:
            dsu.find(r)
        for r in recs[1:]:
            dsu.union(recs[0], r)
    members: dict[str, list[str]] = defaultdict(list)
    for r in dsu.p:
        members[dsu.find(r)].append(r)
    out: dict[str, str] = {}
    for group in members.values():
        cid = min(group)
        for r in group:
            out[r] = cid
    return out


def capture_of(recording: str, frame: int, aliases: dict[str, str]) -> str:
    """The split atom: `<component>/Frame_<n>`. Every image byte-identical to
    another maps to the SAME capture, so assigning captures to sides cannot
    leak content."""
    return f"{aliases.get(recording, recording)}/Frame_{frame:06d}"


def content_overlap(train_hashes, val_hashes) -> set[str]:
    """The assertion the old build was missing. Non-empty means val pixels are
    in train, whatever the recording names say."""
    return set(train_hashes) & set(val_hashes)


def assert_disjoint(train_hashes, val_hashes, *, what: str = "split") -> None:
    ov = content_overlap(train_hashes, val_hashes)
    if ov:
        raise ValueError(
            f"{what} leaks: {len(ov)} image content hashes appear in BOTH "
            f"train and val (e.g. {sorted(ov)[:3]}). A name-keyed split "
            f"cannot see this -- key on content, see content_index.py.")
