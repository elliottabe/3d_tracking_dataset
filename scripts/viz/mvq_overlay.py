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
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax")); sys.path.insert(0, ROOT)
from jarvis_jax.models.mvq import MVQConfig, MVQModel, assemble
from jarvis_jax.models.mvq.geometry import project_local
from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS
from jarvis_jax.train.train_mvq import normalize_crops, MM_PER_UNIT


def load_model(final_dir):
    """Restore mvq_run.json + the Orbax EMA state into a fresh MVQModel.

    The checkpoint was saved data-parallel-replicated across however many
    GPUs the training job used (commonly 4); its stored array metadata
    records that device count. A bare jax.ShapeDtypeStruct target (no
    sharding) makes Orbax try to honor the CHECKPOINT's own sharding, which
    raises "Topology mismatch detected" the moment the current process's
    visible device count differs from the training run's -- e.g. restoring
    on a single GPU for a quick figure. Pin the target to an explicit
    REPLICATED sharding over whatever devices are visible now instead (same
    pattern as convert/build_checkpoint.py::load_vitpose), which restores
    correctly on 1, 4, or any other device count.
    """
    meta = json.load(open(os.path.join(final_dir, "mvq_run.json")))
    cfg = MVQConfig(**meta["model"])
    model = nnx.eval_shape(lambda: MVQModel(cfg, rngs=nnx.Rngs(0)))
    gdef, state = nnx.split(model)
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    target = jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl), state)
    restored = ocp.StandardCheckpointer().restore(final_dir, target=target)
    model = nnx.merge(gdef, restored); model.eval()
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cases", default="female,two_fly,worst"); ap.add_argument("--out", required=True)
    ap.add_argument("--prompted", action="store_true")
    a = ap.parse_args()
    root = a.root or "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902"
    model, meta = load_model(a.run)
    ds = V12WindowDataset(root, a.split, T=1, train=False)
    names = ds.keypoint_names
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for i in range(len(ds)):
        s = ds[i]; b = {k: jnp.asarray(v)[None] for k, v in s.items()}
        on = jnp.array([a.prompted and bool(s["prompt_mask"].any())])
        out = model(normalize_crops(b["crops"]), b["cam_valid"], b["M"], b["t_local"], b["prompt_mask"], prompt_on=on)
        xyz = np.asarray(out["xyz"][0]); gt = s["kp3d_local"][0, 0]; has = s["has3d"][0, 0]
        d = np.linalg.norm(xyz[:, 0] - gt[None], axis=-1); inst = int(np.argmin((d * has).sum(1)))
        uv3 = np.asarray(project_local(jnp.asarray(xyz[inst, 0]), b["M"][0], b["t_local"][0, 0]))   # (K,C,2)
        uv2 = np.asarray(out["uv"][0, inst, 0])                                                     # (C,K,2)
        vis = s["vis2d"][0, 0]                                                                     # (C,K)
        re = np.linalg.norm(uv3.transpose(1, 0, 2) - s["kp2d"][0, 0], axis=-1)
        rows.append(dict(i=i, mpjpe_units=float(d[inst][has].mean()) if has.any() else np.nan,
                         reproj_px=float(re[vis].mean()) if vis.any() else np.nan,
                         exist=[float(x) for x in 1 / (1 + np.exp(-np.asarray(out["exist_logit"][0])))],
                         female=bool(ds.is_female(i)), two_fly=ds.n_flies(i) > 1, group=ds.calib_group(i),
                         cam_names=ds.camera_names(i), sample=s, uv3=uv3, uv2=uv2, inst=inst))
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
