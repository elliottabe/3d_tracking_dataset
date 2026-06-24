import csv, math
import numpy as np
from jarvis_jax.predict.session_io import write_fly_csv, concat_fly_csvs

NAMES = [f"j{i}" for i in range(50)]


def _read(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_write_fly_csv_schema(tmp_path):
    rows = [(100, np.arange(150).reshape(50, 3).astype(float), np.full(50, 0.5)),
            (101, None, None)]   # None -> NaN row
    p = tmp_path / "fly0.csv"
    write_fly_csv(str(p), NAMES, rows)
    r = _read(str(p))
    assert r[0][0] == "frame" and r[0][1:5] == ["j0", "j0", "j0", "j0"]   # name x4
    assert r[1][1:5] == ["x", "y", "z", "confidence"]
    assert len(r[0]) == 1 + 50 * 4
    assert r[2][0] == "100" and float(r[2][1]) == 0.0 and float(r[2][3]) == 2.0
    assert r[3][0] == "101" and math.isnan(float(r[3][1]))   # NaN row


def test_concat_fly_csvs_frame_sorted(tmp_path):
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    write_fly_csv(str(a), NAMES, [(5, np.zeros((50, 3)), np.zeros(50))])
    write_fly_csv(str(b), NAMES, [(2, np.ones((50, 3)), np.ones(50))])
    out = tmp_path / "data3D_fly0.csv"
    concat_fly_csvs([str(a), str(b)], str(out))
    r = _read(str(out))
    assert len(r) == 2 + 2          # 2 header rows + 2 data rows
    assert r[2][0] == "2" and r[3][0] == "5"   # frame-sorted
