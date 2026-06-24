"""CSV writers for D3 per-fly 3D keypoint output (Predictions_3D schema)."""
import csv
import numpy as np


def _header_rows(joint_names):
    h1 = ["frame"]
    h2 = ["frame"]
    for n in joint_names:
        h1 += [n, n, n, n]
        h2 += ["x", "y", "z", "confidence"]
    return h1, h2


def write_fly_csv(path, joint_names, rows):
    """rows: list of (frame:int, kp (50,3) | None, conf (50,) | None).
    None kp/conf -> a NaN data row (frame predicted-but-invalid)."""
    h1, h2 = _header_rows(joint_names)
    nj = len(joint_names)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(h1)
        w.writerow(h2)
        for frame, kp, conf in rows:
            if kp is None:
                w.writerow([frame] + ["nan"] * (nj * 4))
                continue
            kp = np.asarray(kp); conf = np.asarray(conf)
            row = [frame]
            for j in range(nj):
                row += [float(kp[j, 0]), float(kp[j, 1]), float(kp[j, 2]), float(conf[j])]
            w.writerow(row)


def concat_fly_csvs(bout_csv_paths, out_path):
    """Merge per-bout fly CSVs into one session CSV (header once, rows frame-sorted)."""
    header, data = None, []
    for p in bout_csv_paths:
        with open(p, newline="") as f:
            r = list(csv.reader(f))
        if header is None:
            header = r[:2]
        data.extend(r[2:])
    data.sort(key=lambda row: int(row[0]))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        for hr in (header or []):
            w.writerow(hr)
        w.writerows(data)
