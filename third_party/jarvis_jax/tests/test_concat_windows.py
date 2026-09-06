"""ConcatWindowDataset: one loader surface over several v12-format roots
(mvq-v2 trains on the human v12 root PLUS a pseudo-label export, spec §4)."""
import json
import os

import numpy as np
import pytest
from mvq_fixtures import REC, make_v12_root


def _root(base, **kw):
    """`make_v12_root` writes into `<base>/v12` and does not create `base`."""
    base.mkdir(parents=True, exist_ok=True)
    return make_v12_root(base, **kw)


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


def test_concat_refuses_two_roots_that_disagree_about_a_calibration(tmp_path):
    """Both roots carry the same recording name but different `calib_group`:
    the same fly would be triangulated two ways, and `manifest[rec]` (read by
    `_balanced_weights`) could only hold one of them."""
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ra, rb = _root(tmp_path / "a"), _root(tmp_path / "b")
    os.rename(os.path.join(rb, "calibrations", "A"), os.path.join(rb, "calibrations", "B"))
    man = json.load(open(os.path.join(rb, "manifest.json")))
    man["recordings"][REC]["calib_group"] = "B"
    json.dump(man, open(os.path.join(rb, "manifest.json"), "w"))
    a = V12WindowDataset(ra, "train", T=1, train=False)
    b = V12WindowDataset(rb, "train", T=1, train=False)
    assert a.calib_group(0) == "A" and b.calib_group(0) == "B"
    with pytest.raises(ValueError, match="calib_group"):
        ConcatWindowDataset([a, b])
    # the same recording with the SAME group in both roots is fine (a pseudo
    # export of a recording the human root also labels is the normal case)
    man["recordings"][REC]["calib_group"] = "A"
    json.dump(man, open(os.path.join(rb, "manifest.json"), "w"))
    os.rename(os.path.join(rb, "calibrations", "B"), os.path.join(rb, "calibrations", "A"))
    ok = ConcatWindowDataset([a, V12WindowDataset(rb, "train", T=1, train=False)])
    assert ok.manifest[REC]["calib_group"] == "A"


def test_concat_refuses_a_conflicting_manifest_field(tmp_path):
    """`calib_group` is not the only field that matters: `_balanced_weights`
    reads `behavior`, and last-one-wins there would silently put BOTH roots'
    windows in one sampler category or the other depending on merge order.
    Every field a loader/sampler path reads is conflict-checked; a field only
    one root defines is merged without complaint."""
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
    with pytest.raises(ValueError, match="behavior") as e:
        ConcatWindowDataset([a, b], names=["real", "pseudo"])
    assert REC in str(e.value) and "real" in str(e.value) and "pseudo" in str(e.value)
    # sex is read too (the _resolve_sex fallback chain)
    man["recordings"][REC]["behavior"] = "courtship"
    man["recordings"][REC]["sex"] = "female"                 # root a says "mixed"
    json.dump(man, open(p, "w"))
    with pytest.raises(ValueError, match="'sex'"):
        ConcatWindowDataset([a, V12WindowDataset(rb, "train", T=1, train=False)])
    # a field only ONE root defines is taken, not a conflict; an unread
    # bookkeeping field may differ freely
    man["recordings"][REC].pop("sex")
    man["recordings"][REC]["checkpoint"] = "/fake/final"     # not read by any window path
    json.dump(man, open(p, "w"))
    ok = ConcatWindowDataset([a, V12WindowDataset(rb, "train", T=1, train=False)])
    assert ok.manifest[REC]["sex"] == "mixed" and ok.manifest[REC]["checkpoint"] == "/fake/final"
    assert a.manifest[REC].get("checkpoint") is None         # the sub-dataset's dict is not mutated


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
