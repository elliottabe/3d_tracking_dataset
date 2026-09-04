#!/usr/bin/env python
"""Figure gate 1 for the mvq lifter: which SLOT lands on which animal.

EXPECTATION (write before looking): every slot the model claims exists
(sigmoid(exist_logit) >= 0.5) is drawn in its own colour on top of the WHITE
human labels. On a single-fly window exactly one typed slot should exist and
its points should sit on the fly within a few px. On a contact/mating pair
each existing slot's points must stay on ONE animal -- the female slot
(cyan) on the female, the male slot (orange) on the male -- and never split
across both bodies; a slot whose points straddle the two flies is the
step-14000 mixing failure this gate exists to see, and shows up as a low
`mask_containment` in summary.json for that sample.

Geometry failure signatures, unchanged: if a slot's reprojected 3D is offset
by the same vector in all cameras, center3D/t_local bookkeeping is wrong
(crop origin or local offset); if it is right in some cameras and
rotated/mirrored in others, the per-camera geometry tokens or the camera
order by name is wrong; if the 2D head ('+') is fine and the reprojected 3D
is not, the 3D path is broken independently of the encoder.

LEGEND (slot colours, from viz.core.colors.PALETTE where they fit):
  white x  human 2D labels (the host fly)
  magenta  slot 0 -- the prompted fly      (PALETTE["abdomen"])
  cyan     slot 1 -- the FEMALE            (PALETTE["fly0"])
  orange   slot 2 -- the MALE              (PALETTE["fly1"])
  yellow   slot 3 -- other/same-sex second (PALETTE["thorax"])
A LEGACY checkpoint with a slot count other than 4 (e.g. the 3-slot 30k run)
has untyped slots; the same colours are reused positionally and the row label
says `legacy` so nobody reads "cyan == female" into it. Filled dots are that
slot's reprojected 3D; '+' is the 2D head of the ORACLE slot only.
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
from jarvis_jax.models.mvq.policy import EXIST_THRESH, mask_containment, policy_instance
from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS
from jarvis_jax.train.matching import N_SLOTS
from jarvis_jax.train.train_mvq import normalize_crops, MM_PER_UNIT, CONTACT_UNITS
from viz.core.colors import PALETTE

_RGB = lambda key: tuple(c / 255.0 for c in reversed(PALETTE[key]))     # PALETTE is BGR
# slot -> (colour, meaning) for the P3a typed-slot table (spec §3)
SLOT_COLOR = [_RGB("abdomen"), _RGB("fly0"), _RGB("fly1"), _RGB("thorax")]
SLOT_NAME = ["slot0 prompted", "slot1 female", "slot2 male", "slot3 other"]


def _slot_color(sl):
    return SLOT_COLOR[sl % len(SLOT_COLOR)]


def _slot_label(sl, n_instances):
    """Legend text for a slot. Only a 4-slot model has the P3a typed meaning
    (spec §3); any other count is a legacy untyped decoder, so the label says
    so rather than asserting "cyan == female" about an untyped slot."""
    return SLOT_NAME[sl] if n_instances == N_SLOTS else f"slot{sl} (untyped, legacy)"


def _is_contact(ds, i):
    """True when window i has >=2 labelled flies whose frame-0 centroids
    (`ds.fly_centroids`) are within CONTACT_UNITS world units -- the SAME
    constant `train_mvq._cohorts`'s `contact_pair` val cohort uses (imported,
    not re-hardcoded, and decoupled from `mv_copy_paste.CopyPasteParams` --
    the augmentation's `contact_sep` and this diagnostic threshold are
    allowed to move independently). Amended 2026-09-04: real mounting pairs
    in the labelled data sit at ~24-30 units, not <=15 (see p3a-notes.md)."""
    c = ds.fly_centroids(i)
    if c.shape[0] < 2 or not np.isfinite(c[:2]).all():
        return False
    return float(np.linalg.norm(c[0] - c[1])) < CONTACT_UNITS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="a final/ dir (default), or (with --step) the RUN dir (parent of final/ and ckpt/)")
    ap.add_argument("--step", default=None, help="load ckpt/<step> (or 'latest') instead of final/")
    ap.add_argument("--attn_impl", default=None, help="override the run's own attn_impl (e.g. 'xla' on CPU)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cases", default="female,two_fly,contact_pair,worst"); ap.add_argument("--out", required=True)
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
        I = xyz.shape[0]
        # ORACLE instance (nearest GT -- not available at real inference time, diagnostic only)
        d = np.linalg.norm(xyz[:, 0] - gt[None], axis=-1); inst = int(np.argmin((d * has).sum(1)))
        exist_probs = 1 / (1 + np.exp(-np.asarray(out["exist_logit"][0])))                          # (I,)
        # POLICY instance -- the ONE shared implementation (models/mvq/policy.py), so this
        # figure and train_mvq.evaluate's numbers cannot disagree about what was picked
        inst_policy = policy_instance(exist_probs, xyz, prompted=prompted,
                                      has_mask=bool(s["prompt_mask"].any()))
        mpjpe_policy = (float(d[inst_policy][has].mean()) if inst_policy is not None and has.any()
                       else np.nan)
        # every slot the model claims exists, plus the oracle slot, gets drawn
        drawn = sorted({int(sl) for sl in np.where(exist_probs >= EXIST_THRESH)[0]} | {inst})
        uv3 = {sl: np.asarray(project_local(jnp.asarray(xyz[sl, 0]), b["M"][0], b["t_local"][0, 0]))
              for sl in drawn}                                                                      # (K,C,2) each
        uv2 = np.asarray(out["uv"][0, inst, 0])                                                     # (C,K,2)
        vis = s["vis2d"][0, 0]                                                                     # (C,K)
        re = np.linalg.norm(uv3[inst].transpose(1, 0, 2) - s["kp2d"][0, 0], axis=-1)
        # `sex_logit` is absent on a pre-P3a (legacy) checkpoint, and meaningless when the
        # sex head could not be restored (load_mvq_model reports that in _unrestored_leaves)
        sex_head_ok = ("sex_logit" in out
                       and not any("heads/sex" in u for u in meta.get("_unrestored_leaves", [])))
        sex_prob = (float(1 / (1 + np.exp(-np.asarray(out["sex_logit"][0, inst])))) if sex_head_ok
                   else float("nan"))
        # mask containment of the POLICY instance -- the SAME definition train_mvq.evaluate
        # reports (shared models/mvq/policy.mask_containment), so this summary.json is
        # directly comparable to a run's val numbers and can serve as the baseline row.
        contain = (mask_containment(xyz[inst_policy, 0], s["M"], s["t_local"][0],
                                    s["prompt_mask"][0], s["cam_valid"][0], vis)
                   if inst_policy is not None else float("nan"))
        rows.append(dict(i=i, mpjpe_units=float(d[inst][has].mean()) if has.any() else np.nan,
                         mpjpe_policy_units=mpjpe_policy, policy_miss=inst_policy is None,
                         reproj_px=float(re[vis].mean()) if vis.any() else np.nan,
                         exist=[float(x) for x in exist_probs], mask_containment=contain,
                         female=bool(ds.is_female(i)), two_fly=ds.n_flies(i) > 1, group=ds.calib_group(i),
                         contact=_is_contact(ds, i), slot=inst, slot_policy=inst_policy,
                         drawn=drawn, n_instances=I, sex_prob=sex_prob,
                         cam_names=ds.camera_names(i), sample=s, uv3=uv3, uv2=uv2, inst=inst))
    finite = lambda vals: [v for v in vals if np.isfinite(v)]
    oracle_mean = float(np.mean(finite([r["mpjpe_units"] for r in rows]))) if rows else float("nan")
    policy_vals = finite([r["mpjpe_policy_units"] for r in rows])
    policy_mean = float(np.mean(policy_vals)) if policy_vals else float("nan")
    miss_frac = float(np.mean([r["policy_miss"] for r in rows])) if rows else float("nan")
    contain_vals = finite([r["mask_containment"] for r in rows])
    contain_mean = float(np.mean(contain_vals)) if contain_vals else float("nan")
    print(f"mpjpe3d oracle={oracle_mean:.3f} policy={policy_mean:.3f} units "
         f"(policy_miss_frac={miss_frac:.3f}, prompted={a.prompted})")
    print(f"mask_containment mean={contain_mean:.4f} over {len(contain_vals)}/{len(rows)} scorable "
         f"windows (policy instance's reprojected visible keypoints inside the host prompt_mask)")
    json.dump({"run": a.run, "step": a.step, "split": a.split, "prompted": bool(a.prompted),
              "n_windows": len(rows), "mpjpe3d_oracle_units": oracle_mean,
              "mpjpe3d_policy_units": policy_mean, "policy_miss_frac": miss_frac,
              "mask_containment_mean": contain_mean, "n_mask_containment_scored": len(contain_vals),
              "unrestored_leaves": meta.get("_unrestored_leaves", []),
              "samples": [{k: v for k, v in r.items() if k not in ("sample", "uv3", "uv2")}
                          for r in rows]},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for case in a.cases.split(","):
        if case == "female": sel = [r for r in rows if r["female"]]
        elif case == "two_fly": sel = [r for r in rows if r["two_fly"]]
        elif case == "contact_pair": sel = [r for r in rows if r["contact"]]
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
                ax.scatter(g2[vis, 0], g2[vis, 1], s=14, c="white", marker="x", linewidths=0.6,
                          label="human 2D")
                # every EXISTING slot (plus the oracle), each in its own colour: this is
                # what tells you whether a slot's points stay on ONE animal
                for sl in r["drawn"]:
                    ax.scatter(r["uv3"][sl][:, c, 0], r["uv3"][sl][:, c, 1], s=5,
                              c=[_slot_color(sl)], label=_slot_label(sl, r["n_instances"]))
                ax.scatter(r["uv2"][c, :, 0], r["uv2"][c, :, 1], s=9, marker="+", linewidths=0.6,
                          c=[_slot_color(r["inst"])], label="2D head (oracle slot)")
                e = np.linalg.norm(r["uv3"][r["inst"]][:, c] - g2, axis=-1)[vis]
                ax.set_title(f"{cam} {e.mean():.1f}px" if e.size else cam, fontsize=7)
            legacy = "" if r["n_instances"] == N_SLOTS else f" legacy I={r['n_instances']}"
            pol = "miss" if r["slot_policy"] is None else f"s{r['slot_policy']}"
            axes[r_i, 0].set_ylabel(
                f"#{r['i']} {'F' if r['female'] else 'M'} grp{r['group']}{legacy}\n"
                f"oracle s{r['slot']} policy {pol} pF={r['sex_prob']:.2f}\n"
                f"exist {' '.join(f'{p:.2f}' for p in r['exist'])}\n"
                f"{r['mpjpe_units']*MM_PER_UNIT:.2f}mm contain={r['mask_containment']:.2f}",
                fontsize=6)
        # One deduplicated legend for the whole figure (each slot appears in many axes).
        handles, labels = [], []
        for ax in axes.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h); labels.append(l)
        fig.legend(handles, labels, fontsize=6, loc="upper left",
                  bbox_to_anchor=(0.0, 1.0), bbox_transform=fig.transFigure)
        fig.suptitle(f"mvq {os.path.basename(os.path.dirname(a.run.rstrip('/')))} — {case} — "
                    f"white x = human 2D; one colour per existing slot (dots = reprojected 3D, "
                    f"+ = oracle slot's 2D head)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.955)); fig.savefig(os.path.join(a.out, f"gate1_{case}.png"), dpi=130); plt.close(fig)
        print("wrote", os.path.join(a.out, f"gate1_{case}.png"))


if __name__ == "__main__":
    main()
