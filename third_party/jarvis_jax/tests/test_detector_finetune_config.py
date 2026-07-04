import os
import sys

import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

CFG = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"
REPO_ROOT = os.path.dirname(CFG)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from scripts.run_pseudolabel_finetune import resolve_label_source  # noqa: E402


def test_detector_finetune_config_resolves():
    os.environ.setdefault("USER", "eabe")
    import stac_mjx  # noqa: F401  (registers the multirun_save_dir resolver)
    with initialize_config_dir(version_base=None, config_dir=CFG):
        c = compose(config_name="detector_finetune")
        d = OmegaConf.to_container(c, resolve=True)
    assert d["gate"]["consensus_px"] == 25.0
    assert d["val_recordings"]
    # regression: unquoted 2026_05_27_11_56_05 parses as a YAML int
    # (underscore digit-separators) unless quoted in the config -- these must
    # stay strings for V3Dataset(..., recordings=...) matching.
    assert all(isinstance(r, str) for r in d["val_recordings"])
    assert "2026_05_27_11_56_05" in d["val_recordings"]
    assert d["out_dir"].endswith("v4_kp_silbootstrap/final")
    assert d["real_root"].endswith("red_data_unified_V3")   # resolves via paths.red_data_v3_root


# ---------------------------------------------------------------------------
# resolve_label_source: per-recording `label_sources` entry resolution
# (Task 7 fix -- multi-recording session-generic driver).
# ---------------------------------------------------------------------------

def _make_session_dir(tmp_path, name="2026_04_02_11_52_43", n_predictions=1,
                      with_bouts_csv=True, cam_names=("Cam2012631", "Cam2012630")):
    session_dir = tmp_path / name
    calib_dir = session_dir / "calibration"
    calib_dir.mkdir(parents=True)
    for cam in cam_names:
        (calib_dir / f"{cam}.yaml").write_text("")   # resolver only globs filenames
    for i in range(n_predictions):
        (session_dir / f"Predictions_3D_{123 + i}").mkdir()
    bouts_csv = None
    if with_bouts_csv:
        bouts_csv = session_dir / "courtship_bouts_unified_summary.csv"
        bouts_csv.write_text(
            "fly_id,bout_idx,start_frame,end_frame\n"
            f"{name}/{os.path.basename(str(session_dir))},0,10,20\n")
    return session_dir, bouts_csv


def test_resolve_label_source_defaults_resolve_cleanly(tmp_path):
    session_dir, bouts_csv = _make_session_dir(tmp_path, n_predictions=1)
    run_root = tmp_path / "run_root"
    resolved = resolve_label_source({
        "session_dir": str(session_dir), "run_root": str(run_root)})

    assert resolved["session_dir"] == str(session_dir)
    assert resolved["run_root"] == str(run_root)
    assert resolved["calib_dir"] == str(session_dir / "calibration")
    assert resolved["cameras"] == ["Cam2012630", "Cam2012631"]   # sorted, not insertion order
    assert resolved["predictions_dir"] == str(session_dir / "Predictions_3D_123")
    assert resolved["bouts_csv"] == str(bouts_csv)


def test_resolve_label_source_explicit_overrides_win(tmp_path):
    session_dir, _ = _make_session_dir(tmp_path, n_predictions=1, with_bouts_csv=False)
    run_root = tmp_path / "run_root"
    other_csv = tmp_path / "good_bouts.csv"
    other_csv.write_text("bout_idx,start_frame,end_frame\n0,5,15\n")
    other_preds = tmp_path / "custom_preds"
    other_preds.mkdir()

    resolved = resolve_label_source({
        "session_dir": str(session_dir), "run_root": str(run_root),
        "bouts_csv": str(other_csv), "predictions_dir": str(other_preds),
        "cameras": ["Cam2012630"]})

    assert resolved["bouts_csv"] == str(other_csv)
    assert resolved["predictions_dir"] == str(other_preds)
    assert resolved["cameras"] == ["Cam2012630"]


def test_resolve_label_source_ambiguous_predictions_dir_errors_clearly(tmp_path):
    session_dir, _ = _make_session_dir(tmp_path, n_predictions=2)
    run_root = tmp_path / "run_root"
    with pytest.raises(ValueError) as exc:
        resolve_label_source({"session_dir": str(session_dir), "run_root": str(run_root)})
    msg = str(exc.value)
    assert "Predictions_3D_123" in msg and "Predictions_3D_124" in msg
    assert "predictions_dir=" in msg


def test_resolve_label_source_missing_bouts_csv_errors_clearly(tmp_path):
    # Session1-like: no courtship_bouts_unified_summary.csv on disk.
    session_dir, _ = _make_session_dir(tmp_path, n_predictions=1, with_bouts_csv=False)
    run_root = tmp_path / "run_root"
    with pytest.raises(ValueError) as exc:
        resolve_label_source({"session_dir": str(session_dir), "run_root": str(run_root)})
    assert "bouts_csv" in str(exc.value)


def test_resolve_label_source_requires_session_dir_and_run_root(tmp_path):
    with pytest.raises(ValueError):
        resolve_label_source({"run_root": str(tmp_path / "run_root")})
    with pytest.raises(ValueError):
        resolve_label_source({"session_dir": str(tmp_path)})
