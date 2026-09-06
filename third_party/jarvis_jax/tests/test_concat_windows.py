"""ConcatWindowDataset: one loader surface over several v12-format roots
(mvq-v2 trains on the human v12 root PLUS a pseudo-label export, spec §4)."""
import json
import os

import cv2
import numpy as np
import pytest
from mvq_fixtures import CAMS, REC, make_v12_root


def _root(base, **kw):
    """`make_v12_root` writes into `<base>/v12` and does not create `base`."""
    base.mkdir(parents=True, exist_ok=True)
    return make_v12_root(base, **kw)


def _rename_calib_group(root, old, new):
    """Rename a v12 root's calibration dir AND its manifest `calib_group` --
    the pseudo/negatives exports name a recording's group after the
    recording id while the human root uses letters (Calibration witness
    test, 2026-09-06), so a NAME change with the SAME files is the normal
    case, not a corruption."""
    os.rename(os.path.join(root, "calibrations", old), os.path.join(root, "calibrations", new))
    man = json.load(open(os.path.join(root, "manifest.json")))
    man["recordings"][REC]["calib_group"] = new
    json.dump(man, open(os.path.join(root, "manifest.json"), "w"))


def _perturb_calib_matrix(root, group, cam, delta=5.0):
    """Add `delta` to one camera's projection matrix (0,0) entry -- several
    px of reprojection, the size of a genuine calibration disagreement, not
    the ~1e-15 serialisation noise a re-export produces."""
    path = os.path.join(root, "calibrations", group, f"{cam}.yaml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    mat = fs.getNode("projectionMatrix").mat()
    fs.release()
    mat = mat.copy(); mat[0, 0] += delta
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("projectionMatrix", mat)
    fs.release()


def _pseudo(root, weight=0.3):
    """Mark a whole root as a pseudo export (root-level manifest defaults, the
    shape `data/pseudo_export.py` writes), so the two roots of a concat are
    distinguishable and `source`/`weight` assertions can actually fail."""
    p = os.path.join(root, "manifest.json")
    man = json.load(open(p))
    man["source"] = "pseudo"; man["weight"] = float(weight)
    json.dump(man, open(p, "w"))
    return root


def test_concat_indexes_and_delegates(tmp_path):
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS, window_batches
    ra, rb = _root(tmp_path / "a"), _pseudo(_root(tmp_path / "b"))
    a = V12WindowDataset(ra, "train", T=1, train=False)
    b = V12WindowDataset(rb, "train", T=1, train=False)
    c = ConcatWindowDataset([a, b], names=["real", "pseudo"])
    assert len(c) == len(a) + len(b)
    assert c.which(len(a)) == (1, 0) and c.keypoint_names == a.keypoint_names
    np.testing.assert_array_equal(c[len(a)]["crops"], b[0]["crops"])
    # provenance must come from the OWNING root, so the two roots have to differ
    # here or this assertion cannot fail (both roots "real" would be vacuous)
    assert c.source(len(a)) == b.source(0) == "pseudo" and c.source(0) == a.source(0) == "real"
    assert c.weight(len(a)) == 0.3 and c.weight(0) == 1.0
    assert c.camera_names(0) == a.camera_names(0)
    c.epoch = 7
    assert a.epoch == 7 and b.epoch == 7
    batch = next(window_batches(c, 2, shuffle=False, num_workers=1, drop_last=False))
    assert set(batch) == set(WINDOW_KEYS) and batch["crops"].shape[0] == 2


def test_every_accessor_answers_for_the_owning_sub_dataset(tmp_path):
    """Each per-index accessor must be the SUB-dataset's answer for the LOCAL
    index -- an off-by-one offset here would label a pseudo window "real" (and
    weight it 1.0) while reading the human root's pixels."""
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ra, rb = _root(tmp_path / "a"), _root(tmp_path / "b", n_frames=4)
    man = json.load(open(os.path.join(rb, "manifest.json")))
    man["source"] = "pseudo"; man["weight"] = 0.3; man["role"] = "partner"
    json.dump(man, open(os.path.join(rb, "manifest.json"), "w"))
    a = V12WindowDataset(ra, "train", T=1, train=False)
    b = V12WindowDataset(rb, "train", T=1, train=False)
    c = ConcatWindowDataset([a, b], names=["real", "pseudo"])
    assert len(a) and len(b) and len(c) == len(a) + len(b)
    for i in range(len(c)):
        d, k = c.which(i)
        sub = (a, b)[d]
        assert c.source(i) == sub.source(k) and c.weight(i) == sub.weight(k)
        assert c.role(i) == sub.role(k) and c.delta(i) == sub.delta(k)
        assert c.is_female(i) == sub.is_female(k) and c.n_flies(i) == sub.n_flies(k)
        assert c.calib_group(i) == sub.calib_group(k) and c.camera_names(i) == sub.camera_names(k)
        assert c.unlabelled_sex(i) == sub.unlabelled_sex(k)
        assert c.name(i) == ("real", "pseudo")[d]
        np.testing.assert_array_equal(c.fly_centroids(i), sub.fly_centroids(k))
        assert c.windows[i] == sub.windows[k]
    # the two roots really are distinguishable: only the second is pseudo
    assert {c.source(i) for i in range(len(a))} == {"real"}
    assert {c.source(len(a) + i) for i in range(len(b))} == {"pseudo"}
    assert {c.weight(len(a) + i) for i in range(len(b))} == {0.3}
    assert float(c[len(a)]["sample_weight"]) == pytest.approx(0.3)


def test_concat_supports_the_sampler_and_cohort_surface(tmp_path):
    """`_balanced_weights`/`_cohorts` read `len`, `windows`, `manifest`,
    `is_female`, `n_flies`, `calib_group`, `fly_centroids` -- all of which must
    work on the concat, or a mixed-root run cannot build its sampler."""
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.train.train_mvq import _balanced_weights, _cohorts
    a = V12WindowDataset(_root(tmp_path / "a"), "train", T=1, train=False)
    # root b's fly0 is annotated MALE in every frame, so the two roots differ in
    # host sex: a female cohort / female weight mass that ignored `which(i)` would
    # come out the same for both halves and the assertions below could not fail.
    b = V12WindowDataset(_root(tmp_path / "b", n_frames=4,
                               fly0_sex_by_frame={f: "male" for f in range(4)}),
                         "train", T=1, train=False)
    c = ConcatWindowDataset([a, b])
    expect_f = [a.is_female(i) for i in range(len(a))] + [b.is_female(i) for i in range(len(b))]
    assert any(expect_f) and not all(expect_f)          # the check is not vacuous
    assert not any(expect_f[len(a):])                   # every root-b window is male-host
    w = _balanced_weights(c, 0.5, 1.0)
    assert w.shape == (len(c),) and abs(float(w.sum()) - 1.0) < 1e-9 and (w > 0).all()
    coh = _cohorts(c)
    np.testing.assert_array_equal(coh["female"], np.array(expect_f))
    np.testing.assert_array_equal(coh["single_fly"],
                                  np.array([c.n_flies(i) == 1 for i in range(len(c))]))
    assert coh["group_A"].all() and set(c.manifest) == {REC}   # one recording, one group
    # female_weight really moves mass onto root a's female-host windows
    is_f = np.array(expect_f)
    w3 = _balanced_weights(c, 0.5, 3.0)
    assert float(w3[is_f].sum()) > float(w[is_f].sum())


def test_concat_refuses_mismatched_keypoint_orders(tmp_path):
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ra, rb = _root(tmp_path / "a"), _root(tmp_path / "b")
    names = json.load(open(os.path.join(rb, "annotations", "keypoint_names.json")))
    rev = list(reversed(names))
    json.dump(rev, open(os.path.join(rb, "annotations", "keypoint_names.json"), "w"))
    for split in ("train", "val"):
        p = os.path.join(rb, "annotations", f"instances_{split}.json")
        coco = json.load(open(p)); coco["keypoint_names"] = rev; json.dump(coco, open(p, "w"))
    with pytest.raises(ValueError, match="keypoint"):
        ConcatWindowDataset([V12WindowDataset(ra, "train", T=1, train=False),
                             V12WindowDataset(rb, "train", T=1, train=False)])


def test_concat_accepts_a_renamed_but_content_identical_calib_group(tmp_path):
    """Both roots carry the same recording, but the calibration group is named
    DIFFERENTLY (the pseudo/negatives exports name it after the recording id,
    the human root uses a letter) -- if the calibration CONTENT is the same,
    that's a serialisation difference, not a real disagreement, and must be
    accepted with an alias recorded, not raise (Calibration witness test,
    2026-09-06)."""
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ra, rb = _root(tmp_path / "a"), _root(tmp_path / "b")
    _rename_calib_group(rb, "A", "B")     # same files, renamed dir + manifest field
    a = V12WindowDataset(ra, "train", T=1, train=False)
    b = V12WindowDataset(rb, "train", T=1, train=False)
    assert a.calib_group(0) == "A" and b.calib_group(0) == "B"
    ok = ConcatWindowDataset([a, b], names=["real", "pseudo"])
    assert ok.calib_alias[REC] == {"real": "A", "pseudo": "B"}
    assert ok.calib_mismatches == []
    # the merged manifest keeps the first-seen group name; each sub-dataset
    # still triangulates with its OWN calibration dir (never merged)
    assert ok.manifest[REC]["calib_group"] == "A"
    assert ok.calib_group(0) == "A" and ok.calib_group(len(a)) == "B"
    # the same recording with the SAME group in both roots is the normal case
    # too, and needs no alias at all
    _rename_calib_group(rb, "B", "A")
    ok2 = ConcatWindowDataset([a, V12WindowDataset(rb, "train", T=1, train=False)])
    assert ok2.manifest[REC]["calib_group"] == "A" and ok2.calib_alias == {}


def test_concat_refuses_two_roots_whose_calibration_content_actually_differs(tmp_path):
    """A renamed group whose matrices genuinely differ (not a serialisation
    difference) must still raise BY DEFAULT -- which calibration is right is
    not this class's call to make silently."""
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ra, rb = _root(tmp_path / "a"), _root(tmp_path / "b")
    _rename_calib_group(rb, "A", "B")
    _perturb_calib_matrix(rb, "B", CAMS[0], delta=5.0)
    a = V12WindowDataset(ra, "train", T=1, train=False)
    b = V12WindowDataset(rb, "train", T=1, train=False)
    with pytest.raises(ValueError, match="calibration CONTENT differs") as e:
        ConcatWindowDataset([a, b], names=["real", "pseudo"])
    msg = str(e.value)
    assert REC in msg and "real" in msg and "pseudo" in msg and "max |diff|" in msg


def test_concat_allow_calib_mismatch_override_warns_and_records(tmp_path, capsys):
    """`allow_calib_mismatch=True` proceeds instead of raising: prints one
    WARNING per recording and records the disagreement in
    `calib_mismatches` -- each sub-dataset still uses its OWN calibration
    per sample (never merged)."""
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ra, rb = _root(tmp_path / "a"), _root(tmp_path / "b")
    _rename_calib_group(rb, "A", "B")
    _perturb_calib_matrix(rb, "B", CAMS[0], delta=5.0)
    a = V12WindowDataset(ra, "train", T=1, train=False)
    b = V12WindowDataset(rb, "train", T=1, train=False)
    ok = ConcatWindowDataset([a, b], names=["real", "pseudo"], allow_calib_mismatch=True)
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1 and REC in out and "calibration CONTENT differs" in out
    assert len(ok.calib_mismatches) == 1
    m = ok.calib_mismatches[0]
    assert m["recording"] == REC and m["max_diff"] > 1.0
    assert {r["name"] for r in m["roots"]} == {"real", "pseudo"}
    assert {r["calib_group"] for r in m["roots"]} == {"A", "B"}
    # each root still uses its OWN calibration for its own samples
    assert ok.calib_group(0) == "A" and ok.calib_group(len(a)) == "B"


def test_concat_records_conflicting_manifest_fields_without_raising(tmp_path):
    """Round 2 (2026-09-06): NOTHING reads `ConcatWindowDataset.manifest[rec]`
    except `calib_group` -- `_balanced_weights` (and its `behavior` read) runs
    PER SUB-DATASET on that root's OWN manifest (`train_mvq._mix_weights`),
    same for `source`/`weight`/`role`/`sex`/`fly_sex` via their per-index
    delegates -- so a disagreeing `behavior`/`sex`/... is BY DESIGN (the real
    human root really does say 'climbing+courtship' for one recording the
    pseudo export calls 'courtship') and must be RECORDED, never refused.
    A field only one root defines is merged without complaint, same as
    before."""
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ra, rb = _root(tmp_path / "a"), _root(tmp_path / "b")
    p = os.path.join(rb, "manifest.json")
    man = json.load(open(p))
    man["recordings"][REC]["behavior"] = "climbing"          # root a says "courtship"
    json.dump(man, open(p, "w"))
    a = V12WindowDataset(ra, "train", T=1, train=False)
    b = V12WindowDataset(rb, "train", T=1, train=False)
    assert a.manifest[REC]["behavior"] == "courtship" and b.manifest[REC]["behavior"] == "climbing"
    ok = ConcatWindowDataset([a, b], names=["real", "pseudo"])
    # first root's value wins in the merged (unread) manifest; the disagreement
    # is recorded by root name, not silently dropped
    assert ok.manifest[REC]["behavior"] == "courtship"
    assert ok.manifest_disagreements[REC]["behavior"] == {"real": "courtship", "pseudo": "climbing"}
    # the SUB-datasets are untouched -- each root's own manifest still says
    # what it always said, so `_mix_weights`' per-root `_balanced_weights` call
    # (the actual reader) still sees the right category for its own windows
    assert a.manifest[REC]["behavior"] == "courtship" and b.manifest[REC]["behavior"] == "climbing"

    # sex disagreeing too -- also recorded, never raised
    man["recordings"][REC]["behavior"] = "courtship"
    man["recordings"][REC]["sex"] = "female"                 # root a says "mixed"
    json.dump(man, open(p, "w"))
    ok2 = ConcatWindowDataset([a, V12WindowDataset(rb, "train", T=1, train=False)],
                             names=["real", "pseudo"])
    assert ok2.manifest[REC]["sex"] == "mixed"
    assert ok2.manifest_disagreements[REC]["sex"] == {"real": "mixed", "pseudo": "female"}

    # empty/None on one side is NEVER a disagreement -- the negatives root's
    # fly_sex is always {} (no per-fly identity); the non-empty side wins with
    # no record at all
    man["recordings"][REC]["sex"] = "mixed"
    man["recordings"][REC]["fly_sex"] = {}
    json.dump(man, open(p, "w"))
    ok3 = ConcatWindowDataset([a, V12WindowDataset(rb, "train", T=1, train=False)],
                             names=["real", "pseudo"])
    assert ok3.manifest[REC]["fly_sex"] == a.manifest[REC]["fly_sex"] != {}
    assert "fly_sex" not in ok3.manifest_disagreements.get(REC, {})
    assert "sex" not in ok3.manifest_disagreements.get(REC, {}) and REC not in ok3.calib_alias

    # a field only ONE root defines is taken, not a conflict; an unread
    # bookkeeping field may differ freely
    man["recordings"][REC].pop("fly_sex")
    man["recordings"][REC]["checkpoint"] = "/fake/final"     # not read by any window path
    json.dump(man, open(p, "w"))
    ok4 = ConcatWindowDataset([a, V12WindowDataset(rb, "train", T=1, train=False)],
                             names=["real", "pseudo"])
    assert ok4.manifest[REC]["fly_sex"] == a.manifest[REC]["fly_sex"]
    assert ok4.manifest[REC]["checkpoint"] == "/fake/final"
    assert a.manifest[REC].get("checkpoint") is None         # the sub-dataset's dict is not mutated
    assert ok4.manifest_disagreements.get(REC, {}) == {}


def test_concat_requires_the_same_window_length_and_spacings(tmp_path):
    """A T=1 root and a T=2 root cannot be stacked into one batch: `crops` is
    (T, C, 448, 448, 3) and `window_batches` stacks them. Different
    `pair_deltas` would also mean the two roots contribute different spacings
    (and copy-paste pairs donors BY spacing)."""
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ra, rb = _root(tmp_path / "a"), _root(tmp_path / "b")
    with pytest.raises(ValueError, match="T"):
        ConcatWindowDataset([V12WindowDataset(ra, "train", T=1, train=False),
                             V12WindowDataset(rb, "train", T=2, train=False)])
    with pytest.raises(ValueError, match="pair_deltas"):
        ConcatWindowDataset([V12WindowDataset(ra, "train", T=2, pair_deltas=(1, 4), train=False),
                             V12WindowDataset(rb, "train", T=2, pair_deltas=(1,), train=False)])
    with pytest.raises(ValueError, match="at least one"):
        ConcatWindowDataset([])
