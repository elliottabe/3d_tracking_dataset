import numpy as np
from jarvis_jax.tracking import sexing

KP = ["Scutellum", "Abd_tip", "WingL_base", "WingL_V13", "WingR_base", "WingR_V13"]
IDX = {n: i for i, n in enumerate(KP)}


def _make(T, wing_osc):
    """(T,K,3) kp3d + (T,K) conf. Body axis +x. Left wing angle = 45 +/- wing_osc deg
    (oscillating); right wing static folded near the body axis."""
    K = len(KP)
    kp = np.zeros((T, K, 3), float)
    conf = np.ones((T, K), float)
    kp[:, IDX["Scutellum"]] = [0, 0, 0]
    kp[:, IDX["Abd_tip"]] = [1, 0, 0]
    kp[:, IDX["WingR_base"]] = [0.5, 0, 0]
    kp[:, IDX["WingR_V13"]] = [1.4, 0.02, 0]          # ~1.3 deg from +x, static
    t = np.linspace(0, 4 * np.pi, T)
    ang = np.radians(45.0 + wing_osc * np.sin(t))
    kp[:, IDX["WingL_base"]] = [0.5, 0, 0]
    kp[:, IDX["WingL_V13"], 0] = 0.5 + np.cos(ang)    # wing vec = [cos(ang), sin(ang), 0]
    kp[:, IDX["WingL_V13"], 1] = np.sin(ang)
    return kp, conf


def test_wing_song_cv_oscillating_gt_static():
    kp_m, c = _make(200, wing_osc=40)     # singing male
    kp_f, _ = _make(200, wing_osc=1)      # static female
    cv_m = sexing.wing_song_cv(kp_m, c, IDX)
    cv_f = sexing.wing_song_cv(kp_f, c, IDX)
    assert cv_m > cv_f
    assert cv_m > 0.05


def test_wing_song_cv_min_frames_and_low_conf():
    kp, c = _make(10, wing_osc=40)
    assert np.isnan(sexing.wing_song_cv(kp, c, IDX, min_frames=20))   # too few frames
    kp, c = _make(200, wing_osc=40)
    c[:, IDX["WingL_V13"]] = 0.0
    c[:, IDX["WingR_V13"]] = 0.0                                       # both tips low-conf
    assert np.isnan(sexing.wing_song_cv(kp, c, IDX, conf_min=0.2))


def test_sex_decision_clear_high():
    d = sexing.sex_bout_from_pose(0.10, 0.40)         # ratio 4.0
    assert d["male_fly"] == 1 and d["confidence"] == "high" and d["method"] == "pose_wing_cv"


def test_sex_decision_clear_medium():
    d = sexing.sex_bout_from_pose(0.20, 0.36)         # ratio 1.8
    assert d["male_fly"] == 1 and d["confidence"] == "medium"


def test_sex_decision_male_fly0():
    d = sexing.sex_bout_from_pose(0.40, 0.10)         # fly0 higher
    assert d["male_fly"] == 0 and d["confidence"] == "high"


def test_sex_decision_weak_with_mask_fallback():
    d = sexing.sex_bout_from_pose(0.15, 0.16, mask_sex_meta={"song_cv": [0.1, 0.2]})  # ratio ~1.07
    assert d["male_fly"] == 1 and d["confidence"] == "low" and d["method"] == "mask_fallback"


def test_sex_decision_weak_no_mask_unknown():
    d = sexing.sex_bout_from_pose(0.15, 0.16, mask_sex_meta=None)
    assert d["male_fly"] is None and d["confidence"] == "unknown" and d["method"] == "unresolved"


import json
import os


def _write_bout(bout_dir, male_fly, T=200):
    """Create bout_dir/fly0,fly1 each with kp3d.npz (male fly has oscillating wing)
    plus an 'orig_flyN' marker file so tests can tell which physical dir moved."""
    for fly in (0, 1):
        d = os.path.join(bout_dir, f"fly{fly}")
        os.makedirs(d)
        osc = 40 if fly == male_fly else 1
        kp, c = _make(T, wing_osc=osc)
        np.savez(os.path.join(d, "kp3d.npz"), kp3d=kp, conf3d=c)
        open(os.path.join(d, f"orig_fly{fly}"), "w").close()
    return bout_dir


def test_canonicalize_swaps_male_to_fly1(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=0)     # male currently fly0
    res = sexing.canonicalize_bout(bd, KP)
    assert res["applied_swap"] is True
    assert res["male_fly"] == 1 and res["original_male_fly"] == 0
    assert os.path.exists(os.path.join(bd, "fly1", "orig_fly0"))   # was fly0, now fly1
    assert os.path.exists(os.path.join(bd, "fly0", "orig_fly1"))
    assert json.load(open(os.path.join(bd, "sex.json")))["male_fly"] == 1


def test_canonicalize_no_swap_when_male_already_fly1(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=1)
    res = sexing.canonicalize_bout(bd, KP)
    assert res["applied_swap"] is False and res["male_fly"] == 1
    assert os.path.exists(os.path.join(bd, "fly1", "orig_fly1"))


def test_canonicalize_idempotent(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=0)
    sexing.canonicalize_bout(bd, KP)                 # swaps
    res2 = sexing.canonicalize_bout(bd, KP)          # male now fly1 -> no-op
    assert res2["applied_swap"] is False and res2["male_fly"] == 1


def test_canonicalize_dry_run_moves_nothing(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=0)
    res = sexing.canonicalize_bout(bd, KP, dry_run=True)
    assert res["applied_swap"] is True and res["original_male_fly"] == 0
    assert os.path.exists(os.path.join(bd, "fly0", "orig_fly0"))    # NOT moved
    assert not os.path.exists(os.path.join(bd, "sex.json"))         # NOT written


def test_canonicalize_missing_kp3d_unknown(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=0)
    os.remove(os.path.join(bd, "fly1", "kp3d.npz"))
    res = sexing.canonicalize_bout(bd, KP)
    assert res["confidence"] == "unknown" and res["applied_swap"] is False


def test_swap_recovers_partial_after_step1(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=1)
    f0 = os.path.join(bd, "fly0")
    tmp = os.path.join(bd, ".fly_swap_tmp")
    os.rename(f0, tmp)                    # simulate crash after step 1 (fly0->tmp done)
    sexing._swap_fly_dirs(bd)            # recover + complete
    assert os.path.exists(os.path.join(bd, "fly0", "orig_fly1"))
    assert os.path.exists(os.path.join(bd, "fly1", "orig_fly0"))
    assert not os.path.exists(tmp)


def test_swap_recovers_partial_after_step2(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=1)
    f0 = os.path.join(bd, "fly0")
    f1 = os.path.join(bd, "fly1")
    tmp = os.path.join(bd, ".fly_swap_tmp")
    os.rename(f0, tmp)                    # step 1: fly0 -> tmp
    os.rename(f1, f0)                     # step 2: fly1 -> fly0  (fly1 now missing)
    sexing._swap_fly_dirs(bd)            # recover: tmp -> fly1
    assert os.path.exists(os.path.join(bd, "fly1", "orig_fly0"))
    assert os.path.exists(os.path.join(bd, "fly0", "orig_fly1"))
    assert not os.path.exists(tmp)


# ---------------------------------------------------------------------------
# Human ID review as the authority (scripts/canonicalize_sam_masks.py bakes the
# review into the mask npz's sex_meta; a review manifest can also be passed
# straight in). The heuristic runs only when neither is available.
# ---------------------------------------------------------------------------

HUMAN_META = {"male_slot": 1, "status": "kept", "method": "human_id_review",
              "review_file": "id_review_reviewed_20260829.json",
              "review_reviewed_at": "2026-08-29T22:48:41+00:00",
              "original_male_slot": 1}


def test_review_from_mask_meta_reads_the_human_slot():
    d = sexing.review_from_mask_meta(HUMAN_META)
    assert d["male_fly"] == 1 and d["confidence"] == "user"
    assert d["method"] == "human_id_review_masks"
    assert d["review_file"] == "id_review_reviewed_20260829.json"


def test_review_from_mask_meta_ignores_the_area_vote():
    """sam3_driver's own mask-area vote writes sex_meta too. It is a heuristic,
    not a human decision, and must not be promoted to 'user' authority."""
    assert sexing.review_from_mask_meta(
        {"male_slot": 1, "status": "kept", "method": "mask_area_vote",
         "agreement": 0.43}) is None
    assert sexing.review_from_mask_meta(None) is None


def test_review_from_mask_meta_rejects_a_nonbinary_slot():
    assert sexing.review_from_mask_meta({**HUMAN_META, "male_slot": 2}) is None
    assert sexing.review_from_mask_meta({**HUMAN_META, "male_slot": None}) is None


def test_review_from_manifest_entry():
    e = {"reviewed_male_fly": 0, "status": "swapped", "original_male_fly": 1,
         "reviewed_at": "2026-08-29T22:49:04+00:00"}
    d = sexing.review_from_manifest_entry(e)
    assert d["male_fly"] == 0 and d["method"] == "human_id_review_manifest"
    assert d["confidence"] == "user"


def test_review_from_manifest_entry_rejects_unreviewed_or_unsure():
    assert sexing.review_from_manifest_entry(None) is None
    for status in ("unsure", "bad", "pending"):
        assert sexing.review_from_manifest_entry(
            {"reviewed_male_fly": 1, "status": status}) is None
    assert sexing.review_from_manifest_entry(
        {"reviewed_male_fly": 2, "status": "confirmed"}) is None


def test_canonicalize_prefers_the_review_over_the_wing_cv(tmp_path):
    """The wing-CV heuristic says fly0 (it is the one singing in this fixture).
    The human says fly1. The human must win, and sex.json must say so."""
    bd = _write_bout(str(tmp_path / "bout"), male_fly=0)
    res = sexing.canonicalize_bout(bd, KP, mask_sex_meta=HUMAN_META)
    assert res["applied_swap"] is False              # already fly1 per the human
    assert res["male_fly"] == 1
    assert res["method"] == "human_id_review_masks"
    assert res["authority"] == "human_review"
    assert res["heuristic_male_fly"] == 0            # the disagreement is recorded
    assert res["heuristic_agrees"] is False
    assert os.path.exists(os.path.join(bd, "fly0", "orig_fly0"))    # NOT moved
    j = json.load(open(os.path.join(bd, "sex.json")))
    assert j["male_fly"] == 1 and j["authority"] == "human_review"


def test_canonicalize_applies_a_review_that_needs_a_swap(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=1)
    res = sexing.canonicalize_bout(
        bd, KP, mask_sex_meta={**HUMAN_META, "male_slot": 0})
    assert res["applied_swap"] is True and res["male_fly"] == 1
    assert os.path.exists(os.path.join(bd, "fly1", "orig_fly0"))


def test_canonicalize_uses_a_manifest_entry_when_the_mask_has_no_review(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=1)
    res = sexing.canonicalize_bout(
        bd, KP, review_entry={"reviewed_male_fly": 0, "status": "swapped"})
    assert res["method"] == "human_id_review_manifest"
    assert res["applied_swap"] is True and res["male_fly"] == 1


def test_the_mask_review_outranks_the_manifest_entry(tmp_path):
    """Both are human, but only the mask's is expressed in the SLOT space the
    pose fly dirs are built from. A manifest entry names a fly index in the
    tree it was reviewed against, which a re-run does not reproduce."""
    bd = _write_bout(str(tmp_path / "bout"), male_fly=1)
    res = sexing.canonicalize_bout(
        bd, KP, mask_sex_meta=HUMAN_META,
        review_entry={"reviewed_male_fly": 0, "status": "swapped"})
    assert res["method"] == "human_id_review_masks"
    assert res["applied_swap"] is False


def test_canonicalize_falls_back_to_the_heuristic_with_no_review(tmp_path):
    bd = _write_bout(str(tmp_path / "bout"), male_fly=0)
    res = sexing.canonicalize_bout(bd, KP, mask_sex_meta={
        "male_slot": 1, "method": "mask_area_vote", "status": "kept"})
    assert res["method"] == "pose_wing_cv" and res["authority"] == "heuristic"
    assert res["applied_swap"] is True


def test_review_key_for_bout_dir():
    key = sexing.review_key_for(
        "/data/processed/courtship/Session1/2026_04_02_12_11_50/pose/bouts/bout_00004")
    assert key == "Session1/2026_04_02_12_11_50/bout_00004"


def test_load_review_accepts_both_manifest_shapes(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"convention": {"female": 0, "male": 1},
                             "bouts": {"S/r/bout_00001": {"reviewed_male_fly": 1}}}))
    assert "S/r/bout_00001" in sexing.load_review(str(p))
    p.write_text(json.dumps({"S/r/bout_00002": {"reviewed_male_fly": 1}}))
    assert "S/r/bout_00002" in sexing.load_review(str(p))
