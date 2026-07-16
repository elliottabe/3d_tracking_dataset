"""Analyze Workstream-D robustness: does kp+dense IK degrade less under occlusion?

For each method (kp, dense), measure qpos drift of the occluded fits relative to
that method's own full-data (occ 0) fit:
    drift = mean |qpos(occ) - qpos(occ0)|   (root 6-DoF and joint angles separately)
A smaller drift = more occlusion-robust.  Compares kp-only vs kp+200-vertices.
"""
import argparse, glob, os
import numpy as np
import stac_mjx.io_dict_to_hdf5 as ioh5


def load_qpos(d, prefix, occ):
    # match Fruitfly_ik_<prefix>...occ<NN>.h5 (prefix 'kp' or 'dense', any src tag)
    hits = [p for p in glob.glob(os.path.join(d, f"Fruitfly_ik_{prefix}*occ{int(occ*100)}.h5"))
            if "_fit" not in os.path.basename(p)]
    return np.asarray(ioh5.load(hits[0])["qpos"]) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    print(f"{'metric':<22}{'kp-only':>12}{'kp+dense':>12}{'gain':>10}")
    for occ in (0.3, 0.5):
        for label, sl in [("root 6dof (occ%d)" % (occ*100), slice(0, 7)),
                          ("joint ang (occ%d)" % (occ*100), slice(7, None))]:
            row = {}
            for mode in ("kp", "dense"):
                q0 = load_qpos(a.dir, mode, 0.0); qo = load_qpos(a.dir, mode, occ)
                _ = mode
                if q0 is None or qo is None:
                    row[mode] = np.nan; continue
                row[mode] = float(np.mean(np.abs(qo[:, sl] - q0[:, sl])))
            gain = (row["kp"] - row["dense"]) / row["kp"] * 100 if row.get("kp") else np.nan
            print(f"{label:<22}{row.get('kp',np.nan):>12.4f}{row.get('dense',np.nan):>12.4f}{gain:>9.1f}%")
    print("\n(gain = % reduction in occlusion-induced drift from adding dense vertices)")


if __name__ == "__main__":
    main()
