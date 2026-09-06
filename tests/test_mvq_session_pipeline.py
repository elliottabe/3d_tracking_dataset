"""Dry-run string tests for scripts/slurm/mvq_session_pipeline.sh.

Same shape as tests/test_slurm_dir_array.py and
third_party/jarvis_jax/tests/test_slurm_courtship_array.py: exercise the
launcher's discovery + command construction, never call `sbatch`.

The per-recording submitter (scripts/slurm_bout_array.py) is replaced with a
stub via MVQ_BOUT_ARRAY_CMD so these run against a fake processed tree without
composing the real Hydra config, whose recording group points at absolute
cluster paths. What is under test here is the DRIVER: which recordings it
discovers, which it skips, the bouts-CSV fallback, and the dependency graph.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "slurm" / "mvq_session_pipeline.sh"

TS_A = "2026_04_02_12_11_50"      # 2 bouts, unified CSV
TS_B = "2026_04_02_14_54_28"      # 3 bouts, only a per-fly CSV -> fly_id-less copy
TS_C = "2026_04_02_15_12_14"      # no masks -> skipped

UNIFIED = "fly_id,bout_idx,start_frame,end_frame\nSession1/x,1,10,20\n"
PERFLY = "fly_id,bout_idx,start_frame,end_frame\nSession1/x_fly0,1,10,20\n"


@pytest.fixture
def tree(tmp_path):
    """A fake <video session dir> + <processed root> pair."""
    session = tmp_path / "Video_recordings" / "courtship" / "Session1"
    processed = tmp_path / "processed" / "courtship"
    for ts, n_bouts, csv in ((TS_A, 2, UNIFIED), (TS_B, 3, None), (TS_C, 0, UNIFIED)):
        rec = session / ts
        rec.mkdir(parents=True)
        if csv is not None:
            (rec / "courtship_bouts_unified_summary.csv").write_text(csv)
        else:
            (rec / "courtship_bouts_fly0_summary.csv").write_text(PERFLY)
        for b in range(1, n_bouts + 1):
            bd = processed / "Session1" / ts / "sam3_masks" / f"bout_{b:05d}"
            bd.mkdir(parents=True)
            (bd / "sam3_masks.npz").write_bytes(b"")
    # a recording dir with a bout_* dir but NO sam3_masks.npz must also be
    # skipped: on the mvq route the masks are the input, not work to queue.
    (processed / "Session1" / TS_C / "sam3_masks" / "bout_00001").mkdir(parents=True)
    return session, processed


@pytest.fixture
def stub(tmp_path):
    """Stands in for scripts/slurm_bout_array.py; prints its argv, submits nothing."""
    p = tmp_path / "stub_bout_array.sh"
    p.write_text("#!/bin/bash\necho \"STUB_SUBMITTER $*\"\n")
    p.chmod(0o755)
    return p


def run_driver(tree, stub, *extra):
    session, processed = tree
    env = dict(os.environ)
    env["MVQ_BOUT_ARRAY_CMD"] = str(stub)
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run(
        ["bash", str(DRIVER), "--session", str(session), "--processed", str(processed),
         "--run-name", "pose_mvq_test", "--dry-run", *extra],
        capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout + r.stderr


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_discovers_recordings_with_masks_and_skips_the_one_without(tree, stub):
    out = run_driver(tree, stub)
    assert f"=== Session1 / {TS_A} : 2 bouts with masks" in out
    assert f"=== Session1 / {TS_B} : 3 bouts with masks" in out
    assert f"skip Session1/{TS_C} (no bout_*/sam3_masks.npz" in out
    assert "submitted 2 recording chain(s)" in out


def test_recording_config_defaults_from_session_folder_name(tree, stub):
    out = run_driver(tree, stub)
    # Session1 -> recording=session1, targeted by session_dir (never by
    # recording.timestamp: an unquoted 2026_04_02_… is a Hydra INT).
    assert "recording=session1" in out
    assert f"recording.session_dir={tree[0]}/{TS_A}" in out
    assert "recording.timestamp" not in out


def test_lifter_mvq_array_by_default(tree, stub):
    out = run_driver(tree, stub)
    assert "--lifter mvq --mvq-lift array" in out
    assert "--mvq-lift skip" not in out


def test_outputs_out_is_the_per_recording_run_root(tree, stub):
    out = run_driver(tree, stub)
    _session, processed = tree
    assert f"outputs.out={processed}/Session1/{TS_A}/pose_mvq_test" in out


# ---------------------------------------------------------------------------
# bouts CSV
# ---------------------------------------------------------------------------

def test_flyidless_csv_only_when_the_unified_one_is_missing(tree, stub):
    out = run_driver(tree, stub)
    _session, processed = tree
    csv_b = f"{processed}/Session1/{TS_B}/pose_mvq_test/bouts_unified_summary.csv"
    assert f"recording.bouts_csv={csv_b}" in out
    assert "fly_id-less copy of courtship_bouts_fly0_summary.csv" in out
    # TS_A has a good unified CSV -> no override at all (the recording config's
    # own default is right, and a stale override would pin a wrong bout set).
    a_line = [ln for ln in out.splitlines() if f"recording.session_dir={tree[0]}/{TS_A}" in ln]
    assert a_line and "bouts_csv" not in a_line[0]


def test_dry_run_writes_nothing(tree, stub):
    run_driver(tree, stub)
    _session, processed = tree
    assert not (processed / "Session1" / TS_B / "pose_mvq_test").exists()


# ---------------------------------------------------------------------------
# dependency graph
# ---------------------------------------------------------------------------

def test_dependency_graph_per_recording_then_collect(tree, stub):
    out = run_driver(tree, stub)
    for ts in (TS_A, TS_B):
        assert f"lift[{ts}] -> precompute[{ts}] -> ik[{ts}] -> aggregate[{ts}]" in out
    assert "aggregate[*] -> collect" in out
    assert f"lift[{TS_C}]" not in out


def test_collect_job_depends_on_every_aggregate_and_is_cpu_only(tree, stub):
    out = run_driver(tree, stub)
    _session, processed = tree
    assert "--- collect script (dry-run) ---" in out
    assert "#SBATCH --dependency=afterok:<aggregate_JOBID>:<aggregate_JOBID>" in out
    assert "#SBATCH --gpus=0" in out
    assert "#SBATCH --constraint" not in out          # CPU job: no GPU constraint
    assert "scripts/session_collect.py --session-name Session1" in out
    assert f"--processed {processed} --run-name pose_mvq_test" in out


def test_no_collect_skips_the_collect_job(tree, stub):
    out = run_driver(tree, stub, "--no-collect")
    assert "collect: skipped (--no-collect)" in out
    assert "aggregate[*] -> collect" not in out
    assert "session_collect.py" not in out


def test_dry_run_submits_nothing(tree, stub):
    out = run_driver(tree, stub)
    assert "Dependency graph (dry-run, nothing submitted):" in out
    assert "Submitted collect:" not in out


# ---------------------------------------------------------------------------
# --only / --local-gpus
# ---------------------------------------------------------------------------

def test_only_restricts_to_one_recording(tree, stub):
    out = run_driver(tree, stub, "--only", TS_B)
    assert f"=== Session1 / {TS_B}" in out
    assert f"=== Session1 / {TS_A}" not in out
    assert f"lift[{TS_B}] -> precompute[{TS_B}]" in out
    assert f"lift[{TS_A}]" not in out
    assert "submitted 1 recording chain(s)" in out


def test_local_gpus_switches_the_lift_to_skip(tree, stub):
    out = run_driver(tree, stub, "--local-gpus", "2")
    assert "--lifter mvq --mvq-lift skip" in out
    assert "--mvq-lift array" not in out
    assert "local lift: 2 worker(s), checkpoint " in out
    assert "(dry-run) would lift locally on 2 GPU(s): 1 2" in out
    # the lift is no longer a queued job, so the graph says so
    assert "jobs: lift=local" in out


def test_local_gpus_above_the_cap_is_refused(tree, stub):
    session, processed = tree
    env = dict(os.environ)
    env["MVQ_BOUT_ARRAY_CMD"] = str(stub)
    r = subprocess.run(
        ["bash", str(DRIVER), "--session", str(session), "--processed", str(processed),
         "--dry-run", "--local-gpus", "8"],
        capture_output=True, text=True, env=env)
    assert r.returncode == 2
    assert "refusing --local-gpus 8 > 4" in r.stderr


def test_max_local_gpus_raises_the_cap(tree, stub):
    out = run_driver(tree, stub, "--local-gpus", "8", "--max-local-gpus", "8")
    assert "local lift: 8 worker(s)" in out


# ---------------------------------------------------------------------------
# failure propagation
# ---------------------------------------------------------------------------

def test_failed_chain_submission_exits_nonzero_but_submits_the_others(tree, stub, tmp_path):
    """A submitter that fails for ONE recording must not cost the others their
    queue slot -- but the driver must still exit non-zero so a wrapper notices."""
    bad = tmp_path / "stub_fail_one.sh"
    bad.write_text(
        "#!/bin/bash\n"
        f"case \"$*\" in *{TS_A}*) echo 'boom' >&2; exit 1 ;; esac\n"
        "echo \"STUB_SUBMITTER $*\"\n")
    bad.chmod(0o755)
    session, processed = tree
    env = dict(os.environ)
    env["MVQ_BOUT_ARRAY_CMD"] = str(bad)
    r = subprocess.run(
        ["bash", str(DRIVER), "--session", str(session), "--processed", str(processed),
         "--run-name", "pose_mvq_test", "--dry-run"],
        capture_output=True, text=True, env=env)
    assert r.returncode == 1
    out = r.stdout + r.stderr
    assert f"ERROR: chain submission for Session1/{TS_A} failed" in out
    assert f"lift[{TS_B}] -> precompute[{TS_B}]" in out       # the other one survived
    assert "submitted 1 recording chain(s)" in out
