"""convert_dict_to_path must not scaffold directories for every paths key.

batch_run_stac overrides paths.base_dir to the Predictions_3D_* folder, so
input-root keys like out_root (${base_dir}/courtship), processed_root,
vit_runs_root and red_data_v3_root resolve INSIDE it -- and the old blanket
mkdir side effect scaffolded empty courtship/, processed/, jax_vitpose_runs/
and red_data/ dirs in every predictions folder. Only the run-output dirs
(save_dir, log_dir, ckpt_dir, fig_dir) need pre-creation; artifact writers
create their own parents (io_dict_to_hdf5.save, batch drivers).
"""
from pathlib import Path

from utils.path_utils import convert_dict_to_path


def test_only_run_output_dirs_are_created(tmp_path):
    d = {
        "user": "eabe",
        "base_dir": str(tmp_path / "predfolder"),
        "out_root": str(tmp_path / "predfolder" / "courtship"),
        "processed_root": str(tmp_path / "predfolder" / "processed"),
        "vit_runs_root": str(tmp_path / "predfolder" / "jax_vitpose_runs"),
        "red_data_v3_root": str(tmp_path / "predfolder" / "red_data" / "red_data_unified_V3"),
        "save_dir": str(tmp_path / "predfolder" / "analysis"),
        "log_dir": str(tmp_path / "predfolder" / "analysis" / "logs"),
        "ckpt_dir": str(tmp_path / "predfolder" / "analysis" / "ckpt"),
        "fig_dir": str(tmp_path / "predfolder" / "analysis" / "figures"),
    }
    out = convert_dict_to_path(dict(d))

    # all values (except user) still converted to Path
    assert isinstance(out["out_root"], Path)
    assert out["user"] == "eabe"

    # run-output dirs created...
    for k in ("save_dir", "log_dir", "ckpt_dir", "fig_dir"):
        assert Path(d[k]).is_dir(), k
    # ...input roots NOT scaffolded
    for k in ("out_root", "processed_root", "vit_runs_root", "red_data_v3_root"):
        assert not Path(d[k]).exists(), f"{k} should not be scaffolded"
