"""ensure_sync_plan atomic-write / concurrency regression (multi-GPU + array).

Multiple SAM3 workers (one per GPU, or one per bout in the SLURM array) can call
ensure_sync_plan concurrently. The write must be atomic: the visible
sync_plan.json is always complete, no partial/truncated file, no leftover temps.
"""
import glob
import json
import os
import threading

from jarvis_jax.predict import sam3_driver as sd


def _write_meta(path, n=120, delta=1250000):
    with open(path, "w") as f:
        f.write("frame_id,timestamp,timestamp_sys,ptp_offset\n")
        t = 1000
        for i in range(n):
            f.write(f"{i},{t},{t},0\n")
            t += delta


def _make_recording(d, cams=4):
    for c in range(cams):
        _write_meta(os.path.join(d, f"Cam201200{c}_meta.csv"))


def test_ensure_sync_plan_concurrent_no_partial_file(tmp_path):
    _make_recording(str(tmp_path))
    results = []
    lock = threading.Lock()

    def _call():
        r = sd.ensure_sync_plan(str(tmp_path))
        with lock:
            results.append(r)

    threads = [threading.Thread(target=_call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    sp = os.path.join(str(tmp_path), "sync_plan.json")
    # final file exists and is COMPLETE valid JSON with the expected keys
    with open(sp) as f:
        d = json.load(f)                      # raises if truncated/corrupt
    assert d["status"] == "clean"
    assert "cameras" in d and "delta_ns" in d and len(d["cameras"]) == 4
    # no leftover temp files from the atomic write
    assert glob.glob(os.path.join(str(tmp_path), ".sync_plan.*.tmp")) == []
    # every concurrent caller got a usable plan back (never None / never crashed)
    assert len(results) == 8
    assert all(r is not None and r.status == "clean" for r in results)


def test_ensure_sync_plan_no_meta_returns_none_and_writes_nothing(tmp_path):
    # no Cam*_meta.csv -> positional fallback, no plan written
    assert sd.ensure_sync_plan(str(tmp_path)) is None
    assert not os.path.exists(os.path.join(str(tmp_path), "sync_plan.json"))
