# tests/test_lift_mvq.py
"""`jarvis_jax.tracking.lift_mvq.MVQRunner` -- the shared frames -> windows ->
pipeline-format runner (spec 2026-09-04-mvq-maskfree-frontend-design §4.2).

Three properties, each of which is a live bug class in this pipeline:

  * `windows()` must reproduce `V12WindowDataset._build`'s geometry EXACTLY
    (same `crop_origin`, same `t_local = M c + t - origin`, same pixels). A
    runner that crops differently from the training loader is a silent domain
    shift no residual check would name.
  * `to_pipeline()` must permute the keypoint axis BY NAME. The mvq model
    speaks the v12 detector order and the pipeline speaks model/XML order;
    indexing one with the other is the keypoint-order trap in CLAUDE.md
    (a "collapsed right wing vein" that was really a middle-left leg).
  * the Stage-B `gates` signature it stamps into kp3d.npz must be the string
    `scripts/run_bout.py::stage_b_gate_signature` computes for an mvq run, or
    every mvq-produced bout is refused (or, worse, waved through with
    `allow_stale_kp3d`, which disables the check for the DLT gates too).
"""
import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mvq_fixtures import CAMS, REC, make_v12_root

RUN_BOUT = Path(__file__).resolve().parents[3] / "scripts" / "run_bout.py"


def _tiny_final(tmp_path):
    """Train the tiny model for 1 step so a real `final/` exists (the
    smoke-test recipe, `tests/test_train_mvq_smoke.py`). Module-scoped below:
    the 448-crop forward is the slow part of this file on CPU."""
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7,
                     n_instances=4, n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1,
                     dec_heads=4, mlp_ratio=2.0, refine_passes=1, patch_rgb=3, fourier_bands=2,
                     backbone="tiny", backbone_depth=1, backbone_heads=4, remat=False)
    tcfg = MVQTrainConfig(total_steps=1, batch_size=2, warmup_steps=1, eval_every=1,
                          save_every=1, log_every=1, num_workers=1, pretrained=False,
                          window_lengths=(1,), smoke=True)
    run = tmp_path / "run"
    run_training(str(root), out_dir=str(run / "final"), ckpt_dir=str(run / "ckpt"),
                 mcfg=mcfg, tcfg=tcfg, aug=MVAugParams(enabled=False), weights=LossWeights())
    return str(root), str(run / "final")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _tiny_final(tmp_path_factory.mktemp("lift_mvq"))


def _runner(tiny, **kw):
    from jarvis_jax.tracking.lift_mvq import MVQRunner
    root, final = tiny
    kw.setdefault("batch", 2)
    return MVQRunner(final, calib_dir=os.path.join(root, "calibrations", "A"),
                     cameras=CAMS, **kw)


def test_runner_windows_match_dataset_geometry(tiny):
    """The runner's window IS the training loader's window: same crop origin,
    same local offset of the centre's projection, same pixels."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root, final = tiny
    ds = V12WindowDataset(root, "val", T=1, train=False)
    i = ds.windows.index((REC, 0, 1))
    s = ds[i]
    r = _runner(tiny)
    frames = np.stack([ds._decode(ds._img[img]) for img, _ in ds._frame_infos(REC, 1)])
    assert frames.shape[0] == len(CAMS)
    w = r.windows(frames, np.ones(7, bool), s["center3D"][None])
    np.testing.assert_array_equal(w["origin"][0], s["crop_origin"])
    np.testing.assert_allclose(w["t_local"][0, 0], s["t_local"][0], atol=1e-4)
    np.testing.assert_array_equal(w["crops"][0, 0], s["crops"][0])
    np.testing.assert_allclose(w["M"][0], s["M"], atol=0)


def test_runner_infer_and_read_typed_shapes(tiny):
    r = _runner(tiny)
    frames = np.zeros((7, 448, 1936, 3), np.uint8)
    out = r.infer(r.windows(frames, np.ones(7, bool), np.zeros((1, 3), np.float32)))
    assert out["kp3d"].shape == (1, 4, 50, 3) and out["kp2d"].shape == (1, 4, 7, 50, 2)
    assert out["exist"].shape == (1, 4) and out["vis"].shape == (1, 4, 7, 50)
    assert out["sex_prob"].shape == (1, 4) and out["conf_raw"].shape == (1, 4, 50)
    res = r.read_typed(out, 0, want_sex=0)
    assert res is None or (res["slot"] == 1 and res["kp3d"].shape == (50, 3)
                           and res["kp2d"].shape == (7, 50, 2))
    male = r.read_typed(out, 0, want_sex=1)
    assert male is None or male["slot"] == 2


def _fake_out(r, exists, sex_prob=None, xyz_norm=None):
    """A hand-built `infer` result with chosen per-slot existence -- lets the
    slot rules be tested without a forward whose existences are whatever the
    1-step tiny model happens to emit."""
    I, K, C = r.I, r.K, len(r.cameras)
    xyz = np.zeros((1, I, K, 3), np.float32)
    if xyz_norm is not None:                        # distance of each slot from the ROI origin
        xyz[0, :, :, 0] = np.asarray(xyz_norm, np.float32)[:, None]
    return {"exist": np.asarray(exists, np.float32)[None],
            "xyz": xyz,
            "kp3d": np.zeros((1, I, K, 3), np.float32),
            "kp2d": np.zeros((1, I, C, K, 2), np.float32),
            "vis": np.full((1, I, C, K), 0.5, np.float32),
            "conf_raw": np.zeros((1, I, K), np.float32),
            "sex_prob": np.asarray(sex_prob if sex_prob is not None else [0.5] * I,
                                   np.float32)[None]}


def test_exist_thresh_is_honoured_by_both_slot_rules(tiny):
    """The runner's `exist_thresh` must gate the POLICY too, not just
    `read_typed`. A slot at 0.6 is above the module default (0.5) and below a
    runner configured at 0.7: if `policy_slot` still filtered on the constant
    it would hand back the very slot `read_typed` just refused."""
    lax = _runner(tiny)                                    # exist_thresh defaults to 0.5
    strict = _runner(tiny, exist_thresh=0.7)
    out = _fake_out(lax, [0.0, 0.6, 0.2, 0.0], xyz_norm=[9.0, 1.0, 2.0, 9.0])
    assert lax.read_typed(out, 0, want_sex=0)["slot"] == 1
    assert lax.policy_slot(out, 0, prompted=False, has_mask=False) == 1
    assert strict.read_typed(out, 0, want_sex=0) is None
    assert strict.policy_slot(out, 0, prompted=False, has_mask=False) is None
    # and above the strict threshold both rules agree again
    hi = _fake_out(lax, [0.0, 0.8, 0.2, 0.0], xyz_norm=[9.0, 1.0, 2.0, 9.0])
    assert strict.read_typed(hi, 0, want_sex=0)["slot"] == 1
    assert strict.policy_slot(hi, 0, prompted=False, has_mask=False) == 1


def test_read_typed_single_fly_takes_whichever_typed_slot_exists(tiny):
    """Spec §4.2: a single-fly recording asks for SEX_UNKNOWN and gets
    whichever typed slot exists -- the sex is REPORTED, not assumed -- the
    higher existence when both clear the threshold, None when neither does."""
    from jarvis_jax.train.matching import SEX_UNKNOWN
    r = _runner(tiny)
    male_only = r.read_typed(_fake_out(r, [0.0, 0.1, 0.9, 0.0], sex_prob=[.5, .8, .05, .5]),
                             0, want_sex=SEX_UNKNOWN)
    assert male_only["slot"] == 2 and male_only["sex_prob"] == pytest.approx(0.05)
    female_only = r.read_typed(_fake_out(r, [0.0, 0.9, 0.2, 0.0], sex_prob=[.5, .95, .1, .5]),
                               0, want_sex=SEX_UNKNOWN)
    assert female_only["slot"] == 1 and female_only["sex_prob"] == pytest.approx(0.95)
    both_male = r.read_typed(_fake_out(r, [0.0, 0.8, 0.9, 0.0]), 0, want_sex=SEX_UNKNOWN)
    assert both_male["slot"] == 2 and both_male["exist"] == pytest.approx(0.9)
    both_female = r.read_typed(_fake_out(r, [0.0, 0.95, 0.6, 0.0]), 0, want_sex=SEX_UNKNOWN)
    assert both_female["slot"] == 1 and both_female["exist"] == pytest.approx(0.95)
    assert r.read_typed(_fake_out(r, [0.9, 0.2, 0.1, 0.9]), 0, want_sex=SEX_UNKNOWN) is None


def test_concat_windows_batches_frames_and_refuses_mixed_prompts(tiny):
    """A batch spans FRAMES (different images), so the callers build one
    window per frame and concatenate. Concatenating a prompted window with an
    unprompted one must refuse rather than silently drop the prompt."""
    from jarvis_jax.tracking.lift_mvq import concat_windows
    r = _runner(tiny)
    frames = np.zeros((7, 448, 1936, 3), np.uint8)
    a = r.windows(frames, np.ones(7, bool), np.zeros((1, 3), np.float32))
    b = r.windows(frames, np.ones(7, bool), np.full((1, 3), 30.0, np.float32))
    w = concat_windows([a, b])
    assert w["crops"].shape[0] == 2
    np.testing.assert_array_equal(w["origin"][1], b["origin"][0])
    np.testing.assert_allclose(w["t_local"][1], b["t_local"][0])
    np.testing.assert_allclose(w["centres"][1], b["centres"][0])
    p = r.windows(frames, np.ones(7, bool), np.zeros((1, 3), np.float32),
                  prompt_mask=lambda bb, c: np.zeros((448, 1936), bool))
    assert p["prompt_mask"].shape == (1, 1, 7, 448, 448)
    with pytest.raises(ValueError, match="prompt_mask"):
        concat_windows([a, p])


def test_infer_refuses_prompt_on_without_a_mask(tiny):
    """`prompt_on` with no `prompt_mask` in the window would run the prompted
    branch on an all-zero mask -- slot 0 pointing at nothing."""
    r = _runner(tiny)
    w = r.windows(np.zeros((7, 448, 1936, 3), np.uint8), np.ones(7, bool),
                  np.zeros((1, 3), np.float32))
    with pytest.raises(ValueError, match="prompt"):
        r.infer(w, prompt_on=True)


def test_to_pipeline_permutes_by_name_and_signs_gates(tiny):
    r = _runner(tiny)
    model_names = list(reversed(r.kp_names))          # any permutation of the same names
    T, C, K = 3, 7, 50
    kp3d = np.random.default_rng(0).normal(size=(T, K, 3)).astype(np.float32)
    kp2d = np.zeros((T, C, K, 2), np.float32)
    vis = np.full((T, C, K), 0.9, np.float32)
    raw = np.full((T, K), 0.05, np.float32)
    p = r.to_pipeline(kp3d, kp2d, vis, raw, model_names)
    iL, iR = r.kp_names.index("EyeL"), r.kp_names.index("EyeR")
    jL, jR = model_names.index("EyeL"), model_names.index("EyeR")
    np.testing.assert_allclose(p["kp3d"][:, jL], kp3d[:, iL])
    np.testing.assert_allclose(p["kp3d"][:, jR], kp3d[:, iR])
    # the eye-spacing invariant is permutation-invariant: it must not move
    np.testing.assert_allclose(np.linalg.norm(p["kp3d"][:, jL] - p["kp3d"][:, jR], axis=-1),
                               np.linalg.norm(kp3d[:, iL] - kp3d[:, iR], axis=-1), atol=1e-6)
    assert list(p["kp_names"]) == model_names and p["conf3d"].shape == (T, K)
    assert abs(p["conf3d"].mean() - 0.9) < 1e-6
    assert p["conf3d_mvq_raw"].shape == (T, K) and abs(p["conf3d_mvq_raw"].mean() - 0.05) < 1e-6
    assert p["gates"]["lifter"] == "mvq" and len(p["gates"]["sha256"]) == 16
    # `conf3d` averages over CAMERAS, so a wrongly shaped `vis` must refuse
    # rather than average over time and return a plausible (T,K)-shaped lie
    with pytest.raises(ValueError, match="vis"):
        r.to_pipeline(kp3d, kp2d, vis[:, 0], raw, model_names)
    with pytest.raises(ValueError, match="kp2d"):
        r.to_pipeline(kp3d, kp2d[:, :3], vis, raw, model_names)


def _stage_b_gate_signature():
    """`stage_b_gate_signature` exec'd out of run_bout.py without importing it
    (it pulls mujoco/hydra/stac at module level) -- the same AST trick
    `tests/test_run_bout_pipeline_structure.py` uses."""
    fn = next(n for n in ast.parse(RUN_BOUT.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "stage_b_gate_signature")
    g = {"json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<rb>", "exec"), g)
    return g["stage_b_gate_signature"]


def test_gates_signature_round_trips_through_run_bout(tiny):
    """A kp3d.npz written by the runner must be ACCEPTED by run_bout.py's
    Stage-B staleness check when `pipeline.lifter: mvq`, without
    `allow_stale_kp3d` (which would disable the check for everything)."""
    from omegaconf import OmegaConf
    root, final = tiny
    r = _runner(tiny)
    sig = _stage_b_gate_signature()
    cfg = OmegaConf.create({
        "detector": {"conf_thresh": 0.3},
        "pipeline": {"lifter": "mvq"},
        "mvq": {"checkpoint": final, "step": None, "exist_thresh": 0.5},
    })
    assert sig(cfg) == r.gates_string()
    assert json.loads(sig(cfg))["lifter"] == "mvq"
    # a different threshold must move it (that is the whole point)
    other = OmegaConf.merge(cfg, OmegaConf.create({"mvq": {"exist_thresh": 0.7}}))
    assert sig(other) != sig(cfg)
    # and the DLT branch must be untouched by the new key
    dlt = OmegaConf.create({"detector": {"conf_thresh": 0.3}, "masks": {},
                            "wing_collapse": {"enabled": False},
                            "rigid_repair": {"enabled": False}})
    assert "lifter" not in sig(dlt)
    # "latest" is not a checkpoint identity: it names a different step every
    # time the training job saves, so a signature built from it would compare
    # equal to a kp3d.npz produced by DIFFERENT weights. Refuse it by name.
    latest = OmegaConf.merge(cfg, OmegaConf.create({"mvq": {"step": "latest"}}))
    with pytest.raises(ValueError, match="concrete int"):
        sig(latest)
    from jarvis_jax.tracking.lift_mvq import mvq_gate_signature
    with pytest.raises(ValueError, match="concrete int"):
        mvq_gate_signature(final, step="latest")
