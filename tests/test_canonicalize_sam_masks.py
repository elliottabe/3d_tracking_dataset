"""Tests for scripts/canonicalize_sam_masks.py.

The failure this file exists to prevent: a fly-slot swap that is PERFORMED but
not CORRECT. A half-applied swap (packed moved, centroids left behind) is
silently self-consistent to every downstream metric -- masks and centroids each
look fine on their own -- and inverts every male/female conclusion in the
dataset. So the tests check the three fly-indexed arrays move TOGETHER, that
the swap round-trips exactly, that non-fly members (notably `shape`, whose
leading dimension is also 2 but means [H, W]) are never touched, and that the
male slot is resolved BY MEASUREMENT against the target file rather than by
replaying an index from another file's slot space.
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from scripts.canonicalize_sam_masks import (
    FLY_AXIS_MEMBERS,
    Refusal,
    canonical_cameras,
    canonicalize_bout_npz,
    read_members,
    resolve_pose_fly_to_slot,
    review_male_pose_fly,
    swap_fly_axis,
)

H, W, C, T = 8, 16, 3, 40   # T >= min_frames (20) so every camera casts a vote
CAMS = ["Cam0001", "Cam0002", "Cam0003"]


# ---------------------------------------------------------------------------
# synthetic fixtures: slot s is tagged with the value (s + 1) in EVERY
# fly-indexed array, so a half-applied swap breaks the tag agreement.
# ---------------------------------------------------------------------------

def make_mask_npz(path, *, slot_xy, cameras=CAMS, extra=None, tag=True):
    """A 2-fly sam3_masks.npz. slot_xy[s] = (x, y) centroid for slot s (all cams,
    all frames). Every fly-indexed array carries slot s's tag = s + 1."""
    packed = np.zeros((2, len(cameras), T, H, W // 8), np.uint8)
    centroids = np.zeros((2, len(cameras), T, 2), np.float32)
    valid = np.ones((2, len(cameras), T), bool)
    for s in (0, 1):
        if tag:
            packed[s] = s + 1
        centroids[s, :, :, 0] = slot_xy[s][0]
        centroids[s, :, :, 1] = slot_xy[s][1]
    d = dict(packed=packed, centroids=centroids, valid=valid,
             cameras=np.asarray(cameras), shape=np.asarray([H, W], np.int32),
             version=np.asarray(1, np.int32))
    d.update(extra or {})
    np.savez_compressed(path, **d)
    return path


def make_pose_bout(bout_dir, *, fly_xy, cameras=CAMS):
    """pose bout dir with fly0/fly1 kp2d.npz; fly k's keypoints sit at fly_xy[k]."""
    for k in (0, 1):
        d = bout_dir / f"fly{k}"
        d.mkdir(parents=True, exist_ok=True)
        kp = np.zeros((T, len(cameras), 4, 2), np.float32)
        kp[..., 0] = fly_xy[k][0]
        kp[..., 1] = fly_xy[k][1]
        np.savez(d / "kp2d.npz", kp2d=kp, conf=np.ones((T, len(cameras), 4), np.float32))
    return bout_dir


def entry(**kw):
    e = {"reviewed_male_fly": 1, "status": "confirmed", "original_male_fly": 1,
         "applied": False, "reviewed_at": "2026-08-29T22:48:41+00:00"}
    e.update(kw)
    return e


# ---------------------------------------------------------------------------
# swap_fly_axis
# ---------------------------------------------------------------------------

def test_swap_reverses_every_fly_indexed_array(tmp_path):
    p = make_mask_npz(tmp_path / "m.npz", slot_xy={0: (10, 10), 1: (900, 900)},
                      extra={"in_frame": np.stack([np.zeros((C, T), np.int8),
                                                   np.ones((C, T), np.int8)])})
    m = read_members(p)
    out = swap_fly_axis(m)
    for k in ("packed", "centroids", "valid", "in_frame"):
        assert k in FLY_AXIS_MEMBERS
        np.testing.assert_array_equal(out[k][0], m[k][1])
        np.testing.assert_array_equal(out[k][1], m[k][0])


def test_swap_leaves_shape_alone_even_though_its_leading_dim_is_2(tmp_path):
    """`shape` is [H, W], not [fly0, fly1]. Reversing it silently transposes
    every mask unpack downstream."""
    p = make_mask_npz(tmp_path / "m.npz", slot_xy={0: (10, 10), 1: (900, 900)})
    m = read_members(p)
    out = swap_fly_axis(m)
    np.testing.assert_array_equal(out["shape"], [H, W])
    np.testing.assert_array_equal(out["cameras"], m["cameras"])
    np.testing.assert_array_equal(out["version"], m["version"])


def test_swap_twice_is_the_identity(tmp_path):
    p = make_mask_npz(tmp_path / "m.npz", slot_xy={0: (10, 10), 1: (900, 900)})
    m = read_members(p)
    back = swap_fly_axis(swap_fly_axis(m))
    for k in m:
        np.testing.assert_array_equal(back[k], m[k])


def test_swap_refuses_an_unclassified_member(tmp_path):
    """A member the writer added later is neither known-fly nor known-non-fly.
    Guessing either way is how a fly axis gets left behind."""
    p = make_mask_npz(tmp_path / "m.npz", slot_xy={0: (10, 10), 1: (900, 900)},
                      extra={"per_fly_scores": np.zeros((2, C, T), np.float32)})
    with pytest.raises(Refusal, match="per_fly_scores"):
        swap_fly_axis(read_members(p))


def test_the_three_arrays_move_together(tmp_path):
    """MUTATION GUARD: fails if `packed` is swapped but `centroids` is not.

    slot s is tagged s+1 in packed and sits at x = 100*(s+1) in centroids, so
    the two agree only while they move as a unit.
    """
    p = make_mask_npz(tmp_path / "m.npz", slot_xy={0: (100, 0), 1: (200, 0)})
    out = swap_fly_axis(read_members(p))
    for s in (0, 1):
        tag = int(np.unique(out["packed"][s])[0])          # 1 or 2
        x = float(out["centroids"][s, 0, 0, 0])            # 100 or 200
        assert x == pytest.approx(100.0 * tag), (
            f"slot {s}: packed carries tag {tag} but centroids say x={x} -- "
            "packed and centroids were not swapped together")


# ---------------------------------------------------------------------------
# review_male_pose_fly  (refuse rather than guess)
# ---------------------------------------------------------------------------

def test_review_returns_the_reviewed_fly():
    assert review_male_pose_fly("k", entry(reviewed_male_fly=0)) == 0
    assert review_male_pose_fly("k", entry(reviewed_male_fly=1)) == 1


@pytest.mark.parametrize("bad", [None, 2, -1, "1", True])
def test_review_refuses_a_non_binary_male(bad):
    with pytest.raises(Refusal, match="reviewed_male_fly"):
        review_male_pose_fly("k", entry(reviewed_male_fly=bad))


@pytest.mark.parametrize("status", ["unsure", "bad", "pending", "unknown"])
def test_review_refuses_an_unconfirmed_status(status):
    with pytest.raises(Refusal, match="status"):
        review_male_pose_fly("k", entry(status=status))


def test_review_accepts_swapped_status():
    """'swapped' means the reviewer overruled the tracker -- the very case this
    whole exercise exists for. It must NOT be refused."""
    assert review_male_pose_fly("k", entry(status="swapped", reviewed_male_fly=0,
                                           original_male_fly=1)) == 0


def test_review_refuses_a_missing_entry():
    with pytest.raises(Refusal, match="no review entry"):
        review_male_pose_fly("k", None)


# ---------------------------------------------------------------------------
# resolve_pose_fly_to_slot -- measure, never replay an index
# ---------------------------------------------------------------------------

def test_resolve_maps_pose_flies_onto_the_slots_they_sit_on(tmp_path):
    p = make_mask_npz(tmp_path / "m.npz", slot_xy={0: (100, 100), 1: (900, 100)})
    b = make_pose_bout(tmp_path / "bout_00001", fly_xy={0: (100, 100), 1: (900, 100)})
    got = resolve_pose_fly_to_slot(p, b, CAMS, sep_px=200.0)
    assert got[0][0] == 0 and got[1][0] == 1


def test_resolve_detects_a_source_whose_slots_are_reversed(tmp_path):
    """The measured Session0 case: the same bout's two mask sets disagree on
    which slot is which fly. Replaying the review's index would mislabel it."""
    p = make_mask_npz(tmp_path / "m.npz", slot_xy={0: (900, 100), 1: (100, 100)})
    b = make_pose_bout(tmp_path / "bout_00001", fly_xy={0: (100, 100), 1: (900, 100)})
    got = resolve_pose_fly_to_slot(p, b, CAMS, sep_px=200.0)
    assert got[0][0] == 1 and got[1][0] == 0


def test_resolve_uses_camera_names_not_positions(tmp_path):
    """kp2d's camera axis is the canonical (calibration-glob) order; the npz
    stores its own order. Indexing the two by position plots one camera's
    keypoints against another camera's masks."""
    permuted = [CAMS[2], CAMS[0], CAMS[1]]
    packed = np.zeros((2, 3, T, H, W // 8), np.uint8)
    centroids = np.zeros((2, 3, T, 2), np.float32)
    valid = np.ones((2, 3, T), bool)
    # per-camera positions differ, so a wrong camera mapping scrambles the vote
    for ci, cam in enumerate(permuted):
        base = 1000 * CAMS.index(cam)
        centroids[0, ci, :, 0] = base + 100
        centroids[1, ci, :, 0] = base + 900
    np.savez_compressed(tmp_path / "m.npz", packed=packed, centroids=centroids,
                        valid=valid, cameras=np.asarray(permuted),
                        shape=np.asarray([H, W], np.int32),
                        version=np.asarray(1, np.int32))
    b = tmp_path / "bout_00001"
    for k in (0, 1):
        d = b / f"fly{k}"
        d.mkdir(parents=True)
        kp = np.zeros((T, 3, 4, 2), np.float32)
        for ci in range(3):                       # CANONICAL camera order
            kp[:, ci, :, 0] = 1000 * ci + (100 if k == 0 else 900)
        np.savez(d / "kp2d.npz", kp2d=kp, conf=np.ones((T, 3, 4), np.float32))
    got = resolve_pose_fly_to_slot(tmp_path / "m.npz", b, CAMS, sep_px=200.0)
    assert got[0] == (0, 1.0, T * 3) and got[1] == (1, 1.0, T * 3)


def test_resolve_refuses_a_collapse(tmp_path):
    """Both pose flies on the SAME animal makes the two slot assignments cost
    the same, so no frame is decisive and there is nothing to vote on."""
    p = make_mask_npz(tmp_path / "m.npz", slot_xy={0: (100, 100), 1: (900, 100)})
    b = make_pose_bout(tmp_path / "bout_00001", fly_xy={0: (100, 100), 1: (100, 100)})
    with pytest.raises(Refusal, match="collapse"):
        resolve_pose_fly_to_slot(p, b, CAMS, sep_px=200.0)


def test_resolve_refuses_too_few_separated_frames(tmp_path):
    p = make_mask_npz(tmp_path / "m.npz", slot_xy={0: (100, 100), 1: (140, 100)})
    b = make_pose_bout(tmp_path / "bout_00001", fly_xy={0: (100, 100), 1: (140, 100)})
    with pytest.raises(Refusal, match="separated frames"):
        resolve_pose_fly_to_slot(p, b, CAMS, sep_px=200.0)


def test_resolve_refuses_a_missing_kp2d(tmp_path):
    p = make_mask_npz(tmp_path / "m.npz", slot_xy={0: (100, 100), 1: (900, 100)})
    b = tmp_path / "bout_00001"
    (b / "fly0").mkdir(parents=True)
    with pytest.raises(Refusal, match="kp2d"):
        resolve_pose_fly_to_slot(p, b, CAMS, sep_px=200.0)


# ---------------------------------------------------------------------------
# canonicalize_bout_npz -- provenance, idempotence, source untouched
# ---------------------------------------------------------------------------

PROV = {"review_file": "id_review_reviewed_20260829.json",
        "review_sha256": "abc123", "review_reviewed_at": "2026-08-29T22:48:41+00:00",
        "review_status": "swapped", "reviewed_male_pose_fly": 0}


def sex_meta_of(path):
    with np.load(path, allow_pickle=True) as d:
        return json.loads(str(d["sex_meta"]))


def test_canonicalize_swaps_when_the_male_is_in_slot0(tmp_path):
    src = make_mask_npz(tmp_path / "src.npz", slot_xy={0: (100, 0), 1: (900, 0)})
    dst = tmp_path / "out" / "sam3_masks.npz"
    res = canonicalize_bout_npz(src, dst, male_slot_src=0, provenance=PROV)
    assert res["status"] == "swapped" and res["applied_swap"] is True
    with np.load(dst, allow_pickle=True) as d:
        # the male (tag 1, x=100) must now be slot 1
        assert int(np.unique(d["packed"][1])[0]) == 1
        assert float(d["centroids"][1, 0, 0, 0]) == pytest.approx(100.0)
    sm = sex_meta_of(dst)
    assert sm["male_slot"] == 1 and sm["original_male_slot"] == 0
    assert sm["method"] == "human_id_review"
    assert sm["review_file"] == "id_review_reviewed_20260829.json"


def test_canonicalize_passes_through_when_the_male_is_already_slot1(tmp_path):
    src = make_mask_npz(tmp_path / "src.npz", slot_xy={0: (100, 0), 1: (900, 0)})
    dst = tmp_path / "out" / "sam3_masks.npz"
    res = canonicalize_bout_npz(src, dst, male_slot_src=1, provenance=PROV)
    assert res["status"] == "kept" and res["applied_swap"] is False
    a, b = read_members(src), read_members(dst)
    for k in FLY_AXIS_MEMBERS:
        if k in a:
            np.testing.assert_array_equal(a[k], b[k])
    assert sex_meta_of(dst)["original_male_slot"] == 1


def test_canonicalize_never_writes_to_the_source(tmp_path):
    src = make_mask_npz(tmp_path / "src.npz", slot_xy={0: (100, 0), 1: (900, 0)})
    before = (src.stat().st_mtime_ns, src.read_bytes())
    canonicalize_bout_npz(src, tmp_path / "out" / "sam3_masks.npz",
                          male_slot_src=0, provenance=PROV)
    assert (src.stat().st_mtime_ns, src.read_bytes()) == before


def test_canonicalize_refuses_to_write_onto_the_source(tmp_path):
    src = make_mask_npz(tmp_path / "src.npz", slot_xy={0: (100, 0), 1: (900, 0)})
    with pytest.raises(Refusal, match="in place"):
        canonicalize_bout_npz(src, src, male_slot_src=0, provenance=PROV)


def test_canonicalize_is_idempotent(tmp_path):
    """Re-canonicalizing the OUTPUT must not swap back. The male slot is
    re-measured against the file being read, so the second pass sees slot 1."""
    src = make_mask_npz(tmp_path / "src.npz", slot_xy={0: (100, 0), 1: (900, 0)})
    one = tmp_path / "one" / "sam3_masks.npz"
    canonicalize_bout_npz(src, one, male_slot_src=0, provenance=PROV)
    b = make_pose_bout(tmp_path / "bout_00001", fly_xy={0: (900, 0), 1: (100, 0)})
    got = resolve_pose_fly_to_slot(one, b, CAMS, sep_px=200.0)
    assert got[1][0] == 1                       # pose fly1 (the male) is slot 1
    two = tmp_path / "two" / "sam3_masks.npz"
    res = canonicalize_bout_npz(one, two, male_slot_src=got[1][0], provenance=PROV)
    assert res["status"] == "kept"
    a, c = read_members(one), read_members(two)
    for k in FLY_AXIS_MEMBERS:
        if k in a:
            np.testing.assert_array_equal(a[k], c[k])


def test_canonicalize_preserves_every_member(tmp_path):
    src = make_mask_npz(tmp_path / "src.npz", slot_xy={0: (100, 0), 1: (900, 0)},
                        extra={"sync": np.asarray(["clean", "1249995", "1"])})
    dst = tmp_path / "out" / "sam3_masks.npz"
    canonicalize_bout_npz(src, dst, male_slot_src=0, provenance=PROV)
    with zipfile.ZipFile(src) as za, zipfile.ZipFile(dst) as zb:
        assert set(za.namelist()) | {"sex_meta.npy"} == set(zb.namelist())
    with np.load(dst, allow_pickle=True) as d:
        assert [str(x) for x in d["sync"]] == ["clean", "1249995", "1"]


def test_canonicalize_refuses_a_single_fly_npz(tmp_path):
    packed = np.zeros((1, C, T, H, W // 8), np.uint8)
    np.savez_compressed(tmp_path / "m.npz", packed=packed,
                        centroids=np.zeros((1, C, T, 2), np.float32),
                        valid=np.ones((1, C, T), bool), cameras=np.asarray(CAMS),
                        shape=np.asarray([H, W], np.int32),
                        version=np.asarray(1, np.int32))
    with pytest.raises(Refusal, match="2 flies"):
        canonicalize_bout_npz(tmp_path / "m.npz", tmp_path / "o.npz",
                              male_slot_src=0, provenance=PROV)


# ---------------------------------------------------------------------------
# canonical_cameras
# ---------------------------------------------------------------------------

def test_canonical_cameras_is_the_sorted_calibration_glob(tmp_path):
    for n in ("Cam2012862", "Cam2012630", "Cam2012853"):
        (tmp_path / f"{n}.yaml").write_text("")
        (tmp_path / f"{n}_dlt.csv").write_text("")
    assert canonical_cameras(tmp_path) == ["Cam2012630", "Cam2012853", "Cam2012862"]


def test_canonical_cameras_refuses_an_empty_dir(tmp_path):
    with pytest.raises(Refusal, match="calibration"):
        canonical_cameras(tmp_path)


def test_one_bad_camera_cannot_outvote_the_rest(tmp_path):
    """Per-camera majority, not pooled frames.

    cam5's masks sit on the wrong blob for 400 frames; cams 1-4 are right for
    40 frames each. Weighting by frame count would hand the decision to the
    single broken view -- the documented per-camera SAM3 failure mode. The four
    good cameras must win 4-1.
    """
    cams = [f"Cam000{i}" for i in range(1, 6)]
    TT, good, nc = 400, 40, 5
    cent = np.zeros((2, nc, TT, 2), np.float32)
    val = np.zeros((2, nc, TT), bool)
    for ci in range(nc):
        flip = ci == nc - 1                  # last camera has the slots reversed
        cent[0, ci, :, 0] = 900 if flip else 100
        cent[1, ci, :, 0] = 100 if flip else 900
        val[:, ci, : (TT if flip else good)] = True
    np.savez_compressed(tmp_path / "m.npz",
                        packed=np.zeros((2, nc, TT, H, W // 8), np.uint8),
                        centroids=cent, valid=val, cameras=np.asarray(cams),
                        shape=np.asarray([H, W], np.int32),
                        version=np.asarray(1, np.int32))
    b = tmp_path / "bout_00001"
    for k in (0, 1):
        d = b / f"fly{k}"
        d.mkdir(parents=True)
        kp = np.zeros((TT, nc, 4, 2), np.float32)
        kp[..., 0] = 100 if k == 0 else 900
        np.savez(d / "kp2d.npz", kp2d=kp, conf=np.ones((TT, nc, 4), np.float32))
    got = resolve_pose_fly_to_slot(tmp_path / "m.npz", b, cams, sep_px=200.0)
    assert got[0][0] == 0 and got[1][0] == 1
    assert got[0][1] == pytest.approx(3 / 5)          # 4-1 -> |sum| / n_cameras


def test_resolve_refuses_an_even_camera_split(tmp_path):
    cent = np.zeros((2, 2, T, 2), np.float32)
    for ci in range(2):
        flip = ci == 1
        cent[0, ci, :, 0] = 900 if flip else 100
        cent[1, ci, :, 0] = 100 if flip else 900
    np.savez_compressed(tmp_path / "m.npz",
                        packed=np.zeros((2, 2, T, H, W // 8), np.uint8),
                        centroids=cent, valid=np.ones((2, 2, T), bool),
                        cameras=np.asarray(CAMS[:2]),
                        shape=np.asarray([H, W], np.int32),
                        version=np.asarray(1, np.int32))
    b = tmp_path / "bout_00001"
    for k in (0, 1):
        d = b / f"fly{k}"
        d.mkdir(parents=True)
        kp = np.zeros((T, 2, 4, 2), np.float32)
        kp[..., 0] = 100 if k == 0 else 900
        np.savez(d / "kp2d.npz", kp2d=kp, conf=np.ones((T, 2, 4), np.float32))
    with pytest.raises(Refusal, match="split evenly"):
        resolve_pose_fly_to_slot(tmp_path / "m.npz", b, CAMS[:2], sep_px=200.0)


def test_confidence_is_read_off_the_agreeing_cameras_only(tmp_path):
    """A dissenting camera's own shakiness must not be averaged into the
    winners' confidence. Three cameras agree at 0.80 frame-by-frame; a fourth
    dissents at a near-coin-flip 0.51. Pooling all four gives 0.73 and would
    refuse a call that three cameras actually make firmly."""
    cams = [f"Cam000{i}" for i in range(1, 5)]
    TT = 100
    cross = {0: 20, 1: 20, 2: 20, 3: 51}      # frames with the two flies crossed
    cent = np.zeros((2, 4, TT, 2), np.float32)
    cent[0, :, :, 0] = 100
    cent[1, :, :, 0] = 900
    np.savez_compressed(tmp_path / "m.npz",
                        packed=np.zeros((2, 4, TT, H, W // 8), np.uint8),
                        centroids=cent, valid=np.ones((2, 4, TT), bool),
                        cameras=np.asarray(cams),
                        shape=np.asarray([H, W], np.int32),
                        version=np.asarray(1, np.int32))
    b = tmp_path / "bout_00001"
    for k in (0, 1):
        d = b / f"fly{k}"
        d.mkdir(parents=True)
        kp = np.zeros((TT, 4, 4, 2), np.float32)
        for ci in range(4):
            kp[:, ci, :, 0] = 100 if k == 0 else 900
            kp[: cross[ci], ci, :, 0] = 900 if k == 0 else 100
        np.savez(d / "kp2d.npz", kp2d=kp, conf=np.ones((TT, 4, 4), np.float32))
    got = resolve_pose_fly_to_slot(tmp_path / "m.npz", b, cams, sep_px=200.0)
    assert got[0][0] == 0 and got[1][0] == 1
    assert got[0][1] == pytest.approx(0.5)            # 3-1 -> |sum| / n_cameras
