"""End-to-end CPU smoke test for ``jarvis_jax.scripts.train_centerdetect``:
a tiny synthetic v5-shaped root, 1 epoch, random-init model -- proves the
dataset -> oversampling -> train step -> per-epoch two-peak eval ->
checkpoint -> metrics.json wiring is correct, without needing a GPU or real
data. Not a localization/accuracy gate (random init, 1 epoch)."""
import json
import os

import numpy as np
from PIL import Image

from jarvis_jax.scripts.train_centerdetect import run_training


def _make_fixture(tmp_path):
    root = tmp_path
    (root / "annotations").mkdir()
    images, annotations = [], []
    rng = np.random.RandomState(0)
    # 3 one-fly images + 1 two-fly image, train; 1 two-fly image, val.
    specs = {
        "train": [("recA", 1), ("recB", 1), ("recC", 1), ("recD", 2)],
        "val": [("recE", 2)],
    }
    img_id = 0
    ann_id = 0
    manifest = {"version": 1, "calib_groups": {}, "recordings": {}}
    for split, recs in specs.items():
        coco_images, coco_anns = [], []
        for rec, n_flies in recs:
            (root / "images" / rec / "cam1").mkdir(parents=True, exist_ok=True)
            arr = rng.randint(0, 256, (60, 120, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / "images" / rec / "cam1" / "Frame_0.jpg")
            coco_images.append({"id": img_id, "width": 120, "height": 60,
                               "recording": rec, "file_name": f"{rec}/cam1/Frame_0.jpg"})
            fly_sex = {}
            for fly_id in range(n_flies):
                cx = 20.0 + fly_id * 60.0
                coco_anns.append({"id": ann_id, "image_id": img_id,
                                  "bbox": [cx - 5.0, 20.0, 10.0, 10.0],
                                  "sex": "unknown" if n_flies == 2 else "male",
                                  "fly_id": fly_id, "src_ann_id": ann_id})
                ann_id += 1
                if n_flies == 2:
                    fly_sex[f"fly{fly_id}"] = "male" if fly_id == 0 else "female"
            manifest["recordings"][rec] = {"sex": "male" if n_flies == 1 else "unknown",
                                           "split": split}
            if fly_sex:
                manifest["recordings"][rec]["fly_sex"] = fly_sex
            img_id += 1
        coco = {"keypoint_names": [], "skeleton": [], "categories": [],
                "images": coco_images, "annotations": coco_anns, "framesets": []}
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco))
    (root / "manifest.json").write_text(json.dumps(manifest))
    return str(root)


def test_run_training_one_epoch_smoke(tmp_path):
    root = _make_fixture(tmp_path)
    run_dir = tmp_path / "run"
    metrics = run_training(root, str(run_dir), epochs=1, batch_size=2,
                           num_workers=1, seed=0)
    assert len(metrics) == 1
    m = metrics[0]
    assert m["epoch"] == 1
    assert np.isfinite(m["train_loss"])
    assert m["n_two_fly_val"] == 1
    assert 0.0 <= m["two_peak_rate"] <= 1.0

    assert os.path.exists(os.path.join(run_dir, "metrics.json"))
    assert os.path.exists(os.path.join(run_dir, "summary.json"))
    assert os.path.isdir(os.path.join(run_dir, "ckpt", "epoch_001"))

    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["best_epoch"]["epoch"] == 1


def test_run_training_with_copy_paste_smoke(tmp_path):
    """Wiring smoke test for the optional copy-paste-synthesis path (masks
    are absent in this fixture -- `_load_mask` degrades to an all-zero/
    all-transparent sprite rather than raising, so this proves
    `run_training(copy_paste_p=...)` runs end to end and reports the new
    false-positive metrics, not that the pasted content looks like a fly)."""
    root = _make_fixture(tmp_path)
    run_dir = tmp_path / "run_cp"
    metrics = run_training(root, str(run_dir), epochs=1, batch_size=2,
                           num_workers=1, seed=0, copy_paste_p=1.0)
    assert len(metrics) == 1
    m = metrics[0]
    assert "fp_rate_ratio_ge_0.5" in m
    assert m["n_single_fly_val"] == 0   # this fixture's val split is all two-fly
    assert np.isnan(m["fp_conf1_mean"])  # no single-fly val rows -- nan, not a crash
