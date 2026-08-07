"""Tests for scripts/benchmark/freeze_inputs.py (synthetic source trees)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.freeze_inputs import freeze, verify


def make_source_bout(root: Path, bout: int, flies=(0,), with_filt=True,
                     sex_json=None) -> Path:
    bdir = root / "bouts" / f"bout_{bout:05d}"
    for f in flies:
        d = bdir / f"fly{f}"
        d.mkdir(parents=True)
        np.savez(d / "kp2d.npz", kp2d=np.zeros((4, 2, 3, 2)), conf=np.ones((4, 2, 3)))
        np.savez(d / "kp3d.npz", kp3d=np.zeros((4, 3, 3)), conf3d=np.ones((4, 3)))
        if with_filt:
            np.savez(d / "kp3d_filt.npz", kp3d=np.zeros((4, 3, 3)),
                     conf3d=np.ones((4, 3)))
    if sex_json is not None:
        (bdir / "sex.json").write_text(json.dumps(sex_json))
    return bdir


def manifest_for(src: Path, run_key="courtship_r", assay="courtship",
                 flies=(0, 1), bout=1):
    return {"benchmark_root": "", "proximity_threshold_bl": 2.0, "bouts": [{
        "run_key": run_key, "layout": "canonical", "source_root": str(src),
        "recording_cfg": "session1", "session_dir": "/nope", "assay": assay,
        "bout": bout, "flies": list(flies), "tags": [], "render_frames": []}]}


def test_freeze_copies_inputs_and_checksums(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 1, flies=(0, 1), sex_json={"male_fly": 1})
    sums = freeze(manifest_for(src), dest)
    base = dest / "frozen" / "courtship_r" / "bouts" / "bout_00001"
    for f in (0, 1):
        for name in ("kp2d.npz", "kp3d.npz", "kp3d_filt.npz"):
            assert (base / f"fly{f}" / name).exists()
    assert (base / "sex.json").exists()
    assert (dest / "frozen" / "checksums.json").exists()
    assert any(k.endswith("kp2d.npz") for k in sums)
    assert verify(dest) == []


def test_freeze_missing_artifact_raises(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    bdir = make_source_bout(src, 1, flies=(0,))
    (bdir / "fly0" / "kp3d.npz").unlink()
    with pytest.raises(FileNotFoundError, match="kp3d.npz"):
        freeze(manifest_for(src, assay="free_running", flies=(0,)), dest)


def test_freeze_optional_filt_and_sex(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 2, flies=(0,), with_filt=False)   # no filt, no sex.json
    freeze(manifest_for(src, run_key="fr", assay="free_running",
                        flies=(0,), bout=2), dest)
    base = dest / "frozen" / "fr" / "bouts" / "bout_00002"
    assert not (base / "fly0" / "kp3d_filt.npz").exists()


def test_verify_detects_tamper(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 1, flies=(0,))
    freeze(manifest_for(src, run_key="fr", assay="free_running", flies=(0,)), dest)
    victim = dest / "frozen" / "fr" / "bouts" / "bout_00001" / "fly0" / "kp3d.npz"
    victim.write_bytes(b"corrupt")
    bad = verify(dest)
    assert len(bad) == 1 and bad[0].endswith("kp3d.npz")


def make_source_bout_with_qc(root: Path, bout: int, n_frames=10, reproj_px=None,
                              soft_iou=None, flies=(0,)) -> Path:
    """Helper to create a bout with qc.json metadata."""
    bdir = make_source_bout(root, bout, flies=flies)
    for f in flies:
        qc_data = {"n_frames": n_frames}
        if reproj_px is not None:
            qc_data["per_camera_reproj_px"] = {"median": reproj_px}
        if soft_iou is not None:
            qc_data["silhouette_iou"] = {"soft_median": soft_iou}
        (bdir / f"fly{f}" / "qc.json").write_text(json.dumps(qc_data))
    return bdir


def test_select_bouts_csv_contract(tmp_path, capsys):
    """Verify CSV header matches contract and rows are sorted by reproj_px descending."""
    import csv
    import io
    from scripts.benchmark.select_bouts import main
    root = tmp_path / "free_running"
    root.mkdir()

    # Create 2 bouts with different reproj_px values
    # Expected glob pattern: *_bouts/bouts/bout_*
    make_source_bout_with_qc(root / "test_bouts", bout=1, reproj_px=5.0, soft_iou=0.8)
    make_source_bout_with_qc(root / "test_bouts", bout=2, reproj_px=15.0, soft_iou=0.9)

    main(["--freerun-root", str(root)])
    captured = capsys.readouterr().out

    # Use csv module to parse output properly
    reader = csv.reader(io.StringIO(captured))
    rows = list(reader)

    # Verify header contract
    expected_header = ["run_key", "bout", "fly", "assay", "n_frames", "reproj_px",
                       "soft_iou", "proximity_median", "suggestion"]
    assert rows[0] == expected_header, f"Got {rows[0]}"

    # Verify rows are sorted by reproj_px descending
    assert len(rows) == 3  # header + 2 data rows
    repr1 = float(rows[1][5])
    repr2 = float(rows[2][5])
    assert repr1 > repr2, f"Not sorted descending: {repr1} vs {repr2}"


def test_select_bouts_top_n(tmp_path, capsys):
    """Verify --top N truncates to N data rows."""
    import csv
    import io
    from scripts.benchmark.select_bouts import main
    root = tmp_path / "free_running"
    root.mkdir()

    # Create 3 bouts
    make_source_bout_with_qc(root / "test_bouts", bout=1, reproj_px=5.0)
    make_source_bout_with_qc(root / "test_bouts", bout=2, reproj_px=15.0)
    make_source_bout_with_qc(root / "test_bouts", bout=3, reproj_px=10.0)

    main(["--freerun-root", str(root), "--top", "1"])
    captured = capsys.readouterr().out

    reader = csv.reader(io.StringIO(captured))
    rows = list(reader)
    assert len(rows) == 2, f"Expected 2 rows (header + 1 data), got {len(rows)}"
    assert float(rows[1][5]) == 15.0, f"Top row should have highest reproj_px"


def test_select_bouts_suggestion(tmp_path, capsys):
    """Verify suggestion column logic: close_interaction, hard, clean."""
    import csv
    import io
    from scripts.benchmark.select_bouts import main
    root = tmp_path / "free_running"
    root.mkdir()

    # Bout with reproj > 12.0 -> "hard"
    make_source_bout_with_qc(root / "test_bouts", bout=1, reproj_px=20.0)
    # Bout with reproj <= 12.0 -> "clean"
    make_source_bout_with_qc(root / "test_bouts", bout=2, reproj_px=5.0)

    main(["--freerun-root", str(root)])
    captured = capsys.readouterr().out

    reader = csv.reader(io.StringIO(captured))
    rows = list(reader)
    # First data row: reproj=20.0 -> "hard"
    assert rows[1][-1] == "hard", f"reproj=20.0 should be 'hard', got {rows[1][-1]}"
    # Second data row: reproj=5.0 -> "clean"
    assert rows[2][-1] == "clean", f"reproj=5.0 should be 'clean', got {rows[2][-1]}"
