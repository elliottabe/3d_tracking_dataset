#!/usr/bin/env python
"""Figure gate 1 for the mvq lifter.

EXPECTATION (write before looking): after 1k+ steps the GREEN reprojected 3D
and the CYAN 2D head both sit on the fly in every valid camera, within a
few px of the WHITE human labels on easy male frames. If green is offset
by the same vector in all cameras, center3D/t_local bookkeeping is wrong
(crop origin or local offset); if green is right in some cameras and
rotated/mirrored in others, the per-camera geometry tokens or the camera
order by name is wrong; if cyan is fine and green is not, the 3D path is
broken independently of the encoder. On female wall/contact frames the
expectation is looser: points stay on HER body, never on the male.
"""
import argparse, json, os, sys
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax")); sys.path.insert(0, ROOT)
from jarvis_jax.models.mvq.checkpoint import load_mvq_model
from jarvis_jax.models.mvq.geometry import project_local
from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS
from jarvis_jax.train.train_mvq import normalize_crops, MM_PER_UNIT


def _policy_instance(xyz, exist_logit, prompted: bool):
    """Same policy train_mvq.evaluate uses: prompted -> instance 0 (the
    prompt targets that query slot); unprompted -> among instances the model
    itself claims exist (sigmoid(exist_logit) >= 0.5), the one whose
    predicted centroid (mean xyz over T,K, ROI-local so the ROI origin is
    (0,0,0)) sits closest to the ROI centre. None (a miss) if unprompted and
    no instance clears the threshold."""
    if prompted:
        return 0
    exist = 1 / (1 + np.exp(-np.asarray(exist_logit))) > 0.5
    cand = np.where(exist)[0]
    if cand.size == 0:
        return None
    return int(cand[np.argmin(np.linalg.norm(xyz[cand].mean(axis=(1, 2)), axis=-1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="a final/ dir (default), or (with --step) the RUN dir (parent of final/ and ckpt/)")
    ap.add_argument("--step", default=None, help="load ckpt/<step> (or 'latest') instead of final/")
    ap.add_argument("--attn_impl", default=None, help="override the run's own attn_impl (e.g. 'xla' on CPU)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cases", default="female,two_fly,worst"); ap.add_argument("--out", required=True)
    ap.add_argument("--prompted", action="store_true")
    a = ap.parse_args()
    root = a.root or "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902"
    step = int(a.step) if (a.step is not None and a.step != "latest") else a.step
    model, meta = load_mvq_model(a.run, step=step, attn_impl=a.attn_impl)
    ds = V12WindowDataset(root, a.split, T=1, train=False)
    names = ds.keypoint_names
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for i in range(len(ds)):
        s = ds[i]; b = {k: jnp.asarray(v)[None] for k, v in s.items()}
        prompted = a.prompted and bool(s["prompt_mask"].any())
        on = jnp.array([prompted])
        out = model(normalize_crops(b["crops"]), b["cam_valid"], b["M"], b["t_local"], b["prompt_mask"], prompt_on=on)
        xyz = np.asarray(out["xyz"][0]); gt = s["kp3d_local"][0, 0]; has = s["has3d"][0, 0]
        # ORACLE instance (nearest GT -- not available at real inference time, diagnostic only)
        d = np.linalg.norm(xyz[:, 0] - gt[None], axis=-1); inst = int(np.argmin((d * has).sum(1)))
        # POLICY instance (what a real inference call would actually pick, see _policy_instance)
        inst_policy = _policy_instance(xyz, out["exist_logit"][0], prompted)
        mpjpe_policy = (float(d[inst_policy][has].mean()) if inst_policy is not None and has.any()
                       else np.nan)
        uv3 = np.asarray(project_local(jnp.asarray(xyz[inst, 0]), b["M"][0], b["t_local"][0, 0]))   # (K,C,2)
        uv2 = np.asarray(out["uv"][0, inst, 0])                                                     # (C,K,2)
        vis = s["vis2d"][0, 0]                                                                     # (C,K)
        re = np.linalg.norm(uv3.transpose(1, 0, 2) - s["kp2d"][0, 0], axis=-1)
        rows.append(dict(i=i, mpjpe_units=float(d[inst][has].mean()) if has.any() else np.nan,
                         mpjpe_policy_units=mpjpe_policy, policy_miss=inst_policy is None,
                         reproj_px=float(re[vis].mean()) if vis.any() else np.nan,
                         exist=[float(x) for x in 1 / (1 + np.exp(-np.asarray(out["exist_logit"][0])))],
                         female=bool(ds.is_female(i)), two_fly=ds.n_flies(i) > 1, group=ds.calib_group(i),
                         cam_names=ds.camera_names(i), sample=s, uv3=uv3, uv2=uv2, inst=inst))
    finite = lambda vals: [v for v in vals if np.isfinite(v)]
    oracle_mean = float(np.mean(finite([r["mpjpe_units"] for r in rows]))) if rows else float("nan")
    policy_vals = finite([r["mpjpe_policy_units"] for r in rows])
    policy_mean = float(np.mean(policy_vals)) if policy_vals else float("nan")
    miss_frac = float(np.mean([r["policy_miss"] for r in rows])) if rows else float("nan")
    print(f"mpjpe3d oracle={oracle_mean:.3f} policy={policy_mean:.3f} units "
         f"(policy_miss_frac={miss_frac:.3f}, prompted={a.prompted})")
    json.dump([{k: v for k, v in r.items() if k not in ("sample", "uv3", "uv2")} for r in rows],
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for case in a.cases.split(","):
        if case == "female": sel = [r for r in rows if r["female"]]
        elif case == "two_fly": sel = [r for r in rows if r["two_fly"]]
        else: sel = sorted(rows, key=lambda r: -np.nan_to_num(r["reproj_px"], nan=1e9))
        sel = sel[: a.n]
        if not sel:
            print(f"[{case}] no samples"); continue
        # Grid width is the MAX camera count across selected rows (a row with fewer
        # cameras than another blanks its extra columns below) -- rows are not assumed
        # to all share one sample's camera count.
        C = max(r["sample"]["crops"].shape[1] for r in sel)
        fig, axes = plt.subplots(len(sel), C, figsize=(2.2 * C, 2.2 * len(sel)), squeeze=False)
        for r_i, r in enumerate(sel):
            s = r["sample"]; cam_names = r["cam_names"]; Cr = s["crops"].shape[1]
            for c in range(C):
                ax = axes[r_i, c]
                if c >= Cr:
                    ax.axis("off"); continue
                ax.imshow(s["crops"][0, c]); ax.set_xticks([]); ax.set_yticks([])
                cam = cam_names[c]
                if not s["cam_valid"][0, c]:
                    ax.set_title(f"{cam} absent", fontsize=7); continue
                vis = s["vis2d"][0, 0, c]; g2 = s["kp2d"][0, 0, c]
                ax.scatter(g2[vis, 0], g2[vis, 1], s=6, c="white", label="human 2D")
                ax.scatter(r["uv2"][c, :, 0], r["uv2"][c, :, 1], s=6, c="cyan", label="model 2D head")
                ax.scatter(r["uv3"][:, c, 0], r["uv3"][:, c, 1], s=6, c="lime", label="model 3D reprojected")
                e = np.linalg.norm(r["uv3"][:, c] - g2, axis=-1)[vis]
                ax.set_title(f"{cam} {e.mean():.1f}px" if e.size else cam, fontsize=7)
            axes[r_i, 0].set_ylabel(f"#{r['i']} {'F' if r['female'] else 'M'} grp{r['group']}\n"
                                    f"{r['mpjpe_units']*MM_PER_UNIT:.2f}mm", fontsize=7)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=6, loc="upper left",
                  bbox_to_anchor=(0.0, 1.0), bbox_transform=fig.transFigure)
        fig.suptitle(f"mvq {os.path.basename(os.path.dirname(a.run.rstrip('/')))} — {case} — white=human, cyan=2D head, green=reprojected 3D")
        fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(a.out, f"gate1_{case}.png"), dpi=130); plt.close(fig)
        print("wrote", os.path.join(a.out, f"gate1_{case}.png"))


if __name__ == "__main__":
    main()
