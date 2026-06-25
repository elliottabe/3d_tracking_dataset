import importlib.util
import subprocess
import sys
from pathlib import Path

# Load the repo-root launcher module by absolute path (it lives outside the package).
_REPO = Path(__file__).resolve().parents[3]
_LAUNCHER = _REPO / "scripts" / "slurm_predict_session.py"
_spec = importlib.util.spec_from_file_location("slurm_predict_session", _LAUNCHER)
slps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(slps)


def _sample_script():
    return slps.build_script(
        job_name="predsess_run4",
        partition="ckpt-g2", account="portia",
        nodelist_line="#SBATCH --nodelist=g[3090-3137]",
        exclude_line="#SBATCH --exclude=g[3107,3115,3109]",
        requeue_line="#SBATCH --requeue",
        gpus=8, cpus=32, mem=128, time_limit="3-00:00:00",
        conda_env="3d_tracking", mail_user="eabe@uw.edu",
        pkg_dir=Path("/repo/third_party/jarvis_jax"),
        log_dir="/runs/predict_session/run4",
        run_name="run4", paths="hyak",
        session_dir="/data/Session0/vid",
        masks_dir="/runs/predict_session/run4/sam3_masks",
        out="/runs/predict_session/run4",
        overrides_str=" predict_session.batch=8",
    )


def test_sbatch_directives_from_slurm_group():
    s = _sample_script()
    assert "#SBATCH --partition=ckpt-g2" in s
    assert "#SBATCH --account=portia" in s
    assert "#SBATCH --gpus=8" in s
    assert "#SBATCH --cpus-per-task=32" in s
    assert "#SBATCH --mem=128G" in s
    assert "#SBATCH --time=3-00:00:00" in s
    assert "#SBATCH --requeue" in s
    assert "#SBATCH --nodelist=g[3090-3137]" in s
    assert "#SBATCH --exclude=g[3107,3115,3109]" in s
    assert "#SBATCH -o /runs/predict_session/run4/slurm-%j.out" in s


def test_two_stages_in_order():
    s = _sample_script()
    i1 = s.index("scripts/sam3_masks.py")
    i2 = s.index("scripts/predict_session.py")
    assert i1 < i2, "SAM3 stage must precede the predict stage"


def test_per_stage_ld_library_path():
    s = _sample_script()
    assert 'export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"' in s
    cu13 = 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"'
    assert cu13 in s
    # cu13 export is before the SAM3 stage; LD_LIBRARY_PATH is unset before the predict stage
    assert s.index(cu13) < s.index("scripts/sam3_masks.py")
    assert s.index("unset LD_LIBRARY_PATH") < s.index("scripts/predict_session.py")
    assert s.index("scripts/sam3_masks.py") < s.index("unset LD_LIBRARY_PATH")


def test_stage1_sam3_overrides():
    s = _sample_script()
    assert "sam3.session_dir=/data/Session0/vid" in s
    assert "sam3.out=/runs/predict_session/run4/sam3_masks" in s
    assert "sam3.sam3_compile=false" in s


def test_stage2_predict_overrides_and_masks_coupling():
    s = _sample_script()
    assert "run_id=run4" in s
    assert "model=hybridnet" in s
    assert "predict_session.session_dir=/data/Session0/vid" in s
    assert "predict_session.masks_dir=/runs/predict_session/run4/sam3_masks" in s
    assert "predict_session.out=/runs/predict_session/run4" in s
    # masks_dir handed to stage 2 == sam3.out handed to stage 1
    assert "sam3.out=/runs/predict_session/run4/sam3_masks" in s


def test_passthrough_appended_to_both_stages():
    s = _sample_script()
    # the override string is appended to both commands
    assert s.count("predict_session.batch=8") == 2


def test_dry_run_emits_full_script():
    """End-to-end: --dry-run composes the real configs and prints a submittable script."""
    repo = Path(__file__).resolve().parents[3]
    out = subprocess.run(
        [sys.executable, "scripts/slurm_predict_session.py",
         "--run-name", "run4", "--dry-run"],
        cwd=repo, capture_output=True, text=True,
    )
    assert out.returncode == 0, f"launcher failed:\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
    s = out.stdout
    # real slurm/ckpt_g2 + paths/hyak values flowed through
    assert "#SBATCH --partition=ckpt-g2" in s
    assert "#SBATCH --gpus=8" in s
    assert "#SBATCH --requeue" in s
    # both stages present, in order, with the coupling + env discipline
    assert s.index("scripts/sam3_masks.py") < s.index("scripts/predict_session.py")
    assert "sam3.sam3_compile=false" in s
    assert "run_id=run4" in s
    # masks now live in the processed tree, as a sibling of predictions
    assert "processed/courtship/Session0/2025_10_20_13_20_04/sam3_masks" in s
    assert "processed/courtship/Session0/2025_10_20_13_20_04/predictions" in s
    assert s.index("unset LD_LIBRARY_PATH") < s.index("scripts/predict_session.py")
