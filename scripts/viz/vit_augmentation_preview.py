"""Show what the ViTPose 2D train-time augmentations actually do to real crops.

Renders the augmentation stack in ``jarvis_jax.data.augment`` (the one
``jarvis_jax/train/train.py::make_train_step`` calls inside the jitted step)
against real ``V5Dataset`` crops from the promoted checkpoint's data root,
using the parameter values read out of ``configs/aug/*.yaml`` -- never the
``AugParams`` dataclass defaults, which are a separate copy of the numbers and
could drift.

EXPECTATION, stated before anything is generated (per CLAUDE.md)
----------------------------------------------------------------
Figure 1 (each stage alone, default vs heavy):
  * ``affine``  -- the fly rotates/scales/shifts and the overlaid keypoints
    rotate/scale/shift WITH it, staying pinned to the same anatomy. Corners
    that rotate in from outside the crop are black. Keypoints pushed out of
    the 224-unit grid flip to vis=False (drawn as a red x). If a dot stays put
    while the fly turns, the kp transform and the image warp disagree -- a bug
    no loss curve would show.
  * ``flip``    -- a clean mirror; keypoints mirror too, and the LEFT/RIGHT
    colour assignment EXCHANGES (see Figure 4, which is the decisive test).
  * ``cutout``  -- 2 hard black squares of 112 px side (0.25 x 448) land
    anywhere in the crop, often clipped by the border. Keypoints under them
    stay drawn and stay visible: that is the point of the augmentation.
  * ``photometric``/``blur``/``noise``/``per-channel colour`` -- appearance
    only: not one keypoint may move, and the fly's outline must not move.
  * ``heavy`` columns must be visibly stronger than ``default`` on every row
    (larger rotations, 3 boxes not 2, more colour swing).

Figure 2 (full stack, 8 draws): 8 visibly DIFFERENT images. If several draws
  look the same, the per-step key is not reaching the stages. Every draw keeps
  its keypoints on the fly.

Figure 3 (the 4-channel truth): the SAM-mask channel (ch 3) must show NO
  cutout holes while the RGB right beside it does. ``cutout_batch`` writes
  only channels 0-2, so for a mask-ON checkpoint cutout never hid the mask
  evidence -- if a hole appears in the mask panel, that claim is false.
  The mask DOES follow the affine (warped + re-binarised) -- so a mask that
  stays upright while the RGB rotates is the bug to look for here.

Figure 4 (L/R swap): a blue disc is painted into the RGB at ``WingL_base``
  and an orange disc at ``WingR_base`` BEFORE the flip, so the tags travel
  with the pixels. After ``flip_batch`` with the real ``build_lr_swap`` table,
  the BLUE disc (source left wing) must sit under an ORANGE (right-labelled)
  keypoint and vice versa -- that is what makes a mirrored fly a valid
  training example instead of a chirality contradiction. The third panel is
  the deliberate control: the same flip with an IDENTITY swap table, where
  the blue disc keeps a blue dot. If the real table looks like the control,
  the swap is broken -- and this repo has twice shipped confident, completely
  wrong numbers from exactly this class of index-mapping error.

Figure 6 (the affine bounds fix, 2026-09-02): see ``fig6_boundsfix``'s own
  docstring -- legacy vs fixed on the same keys, plus the realised parameter
  histograms that decide whether the fix kept its diversity.

Also measured and written to ``augmentation_stats.json``:
  * mean fraction of crop pixels actually erased by cutout over many draws
    (< 2 x 6.25% = 12.5%, because centre-sampled boxes get clipped by the
    border; and far below the 2 x 25% a careless read of ``cutout_frac``
    suggests -- 0.25 is a SIDE fraction, not an area fraction),
  * how many keypoints land under a cutout box and how many of those are
    still vis=True (expected: all of them),
  * an image/keypoint agreement check: the SAM mask's centroid pushed through
    ``affine_transform_kp`` vs the centroid of the actually-warped mask. These
    must agree to well under a heatmap pixel or "keypoints follow the image"
    is false.

Usage
-----
    unset LD_LIBRARY_PATH
    JAX_PLATFORMS=cpu python scripts/viz/vit_augmentation_preview.py \\
        --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v8_gm_only \\
        --out-dir figures/2026-09-02-vit-augmentations
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from dataclasses import replace as dataclasses_replace

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "jarvis_jax"))

AUG_CFG_DIR = os.path.join(REPO, "third_party", "jarvis_jax", "configs", "aug")

CROP = 448          # V5Dataset img4 side, px
HM = 224            # heatmap_size; kp_xy live in these units. crop_px = 2 * hm
KP2PX = CROP / HM   # 2.0

# Colour = body SIDE here (not fly identity). Cyan/orange is the repo's
# established high-contrast pair (viz/core/colors.py PALETTE, BGR there).
C_LEFT = "#00ffff"    # PALETTE["fly0"] BGR (255,255,0) -> RGB cyan
C_RIGHT = "#ffa500"   # PALETTE["fly1"] BGR (0,165,255) -> RGB orange
C_MID = "#ffffff"
C_INVIS = "#ff2020"


# ----------------------------------------------------------------- config io
def load_aug_yaml(path, **overrides):
    """Read a configs/aug/*.yaml into an AugParams, by field name.

    Deliberately does NOT fall back to the dataclass defaults for a missing
    key: the YAML is the record of what a run actually trained with, and a
    silent default would misreport it. A field the YAML does not carry must
    therefore be supplied explicitly as a keyword -- which is how the two
    arms of the ``keep_kp_in_bounds`` A/B are built here (that knob is not in
    either preset file, so the training runs take the AugParams default).
    """
    import yaml
    from jarvis_jax.data.augment import AugParams

    with open(path) as f:
        raw = yaml.safe_load(f)
    fields = {f.name for f in __import__("dataclasses").fields(AugParams)}
    unknown = (set(raw) | set(overrides)) - fields
    if unknown:
        raise ValueError(f"{path}: keys not on AugParams: {sorted(unknown)}")
    missing = fields - set(raw) - set(overrides)
    if missing:
        raise ValueError(f"{path}: missing keys (refusing dataclass defaults): "
                         f"{sorted(missing)}")
    merged = {**raw, **overrides}
    return AugParams(**merged), merged


# ------------------------------------------------------------- sample choice
def side_of(name):
    """'L', 'R' or None (midline) for a keypoint name, by NAME not index."""
    head = name.split("_", 1)[0]
    if head.endswith("L"):
        return "L"
    if head.endswith("R"):
        return "R"
    return None


def pick_crops(ds, explicit=None):
    """Two hard female crops: a two-fly contact/occlusion frame and a fly
    pressed against the chamber wall. Chosen by measurable properties (nearest
    other annotated fly; how far crop_origin had to clamp off the bbox centre),
    restricted to females with a real SAM mask -- not by eye, and not the
    flattering clean male."""
    from jarvis_jax.data.transforms import crop_origin

    if explicit:
        return [(int(i), f"idx{i}", ds.file_names[int(i)]) for i in explicit]

    by_file = {}
    for i, f in enumerate(ds.file_names):
        by_file.setdefault(f, []).append(i)

    best_close, best_wall, best_centred = None, None, None
    for i, sex in enumerate(ds.sex):
        if sex != "female":
            continue
        bb = ds.bboxes[i]
        cx, cy = bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0
        w, h = ds.img_wh[i]
        x0, y0 = crop_origin(bb, w, h, CROP)
        clamp = max(abs((cx - CROP / 2) - x0), abs((cy - CROP / 2) - y0))
        others = [j for j in by_file[ds.file_names[i]] if j != i]
        d = min((float(np.hypot(cx - (ds.bboxes[j][0] + ds.bboxes[j][2] / 2),
                                cy - (ds.bboxes[j][1] + ds.bboxes[j][3] / 2)))
                 for j in others), default=np.inf)
        if np.isfinite(d) and (best_close is None or d < best_close[0]):
            best_close = (d, i)
        if best_wall is None or clamp > best_wall[0]:
            best_wall = (clamp, i)
        if best_centred is None and clamp < 1.0:      # rounding only, no clamp
            best_centred = i

    out = []
    if best_close:
        d, i = best_close
        out.append((i, "female-occlusion",
                    f"female, 2-fly contact ({d:.0f} px between fly centres)"))
    if best_wall:
        c, i = best_wall
        out.append((i, "female-wall",
                    f"female, against chamber wall (crop clamped {c:.0f} px)"))
    if best_centred is not None:
        out.append((best_centred, "female-centred",
                    "female, crop_origin unclamped (control: the easy case the "
                    "bounds fix must leave alone)"))
    # keep only crops that carry a real SAM mask -- fig 3 is about that channel
    keep = []
    for i, tag, cap in out:
        img4, _, _ = ds[i]
        if int(img4[..., 3].sum()) > 500:
            keep.append((i, tag, cap))
        else:
            print(f"[warn] dropping {tag} (idx {i}): empty SAM mask channel")
    return keep


# ------------------------------------------------------------------ plotting
def draw_kp(ax, kp_hm, vis, names, *, label_names=(), s=14):
    """Overlay keypoints on a 448-px crop axis. Colour = body side, resolved
    from the NAME. vis=False drawn as a red x so affine's out-of-bounds
    invalidation is visible."""
    x = np.asarray(kp_hm)[:, 0] * KP2PX
    y = np.asarray(kp_hm)[:, 1] * KP2PX
    vis = np.asarray(vis).astype(bool)
    for sd, col in (("L", C_LEFT), ("R", C_RIGHT), (None, C_MID)):
        m = np.array([side_of(n) == sd for n in names]) & vis
        if m.any():
            ax.scatter(x[m], y[m], s=s, c=col, edgecolors="k",
                       linewidths=0.3, zorder=3)
    m = ~vis
    if m.any():
        ax.scatter(x[m], y[m], s=s + 6, c=C_INVIS, marker="x",
                   linewidths=1.0, zorder=4)
    idx = {n: k for k, n in enumerate(names)}
    for n in label_names:
        k = idx[n]
        ax.annotate(n, (x[k], y[k]), fontsize=5.0, color="w", zorder=5,
                    xytext=(3, 3), textcoords="offset points",
                    path_effects=_stroke())


def _suptitle(fig, headline, caption, note, fontsize=8.0, width=None):
    """Three-deck title that wraps instead of running off the canvas. Returns
    the tight_layout `rect` top that leaves exactly enough room for it."""
    import textwrap
    if width is None:      # chars that actually fit at this figure width
        width = max(60, int(fig.get_figwidth() * 72.0 / (fontsize * 0.62)))
    lines = [headline]
    lines += textwrap.wrap(caption, width)
    lines += sum([textwrap.wrap(l, width) for l in note.split("\n")], [])
    fig.suptitle("\n".join(lines), fontsize=fontsize, y=0.995, va="top",
                 linespacing=1.35)
    inches = 0.02 + len(lines) * fontsize * 1.35 / 72.0
    return 1.0 - inches / fig.get_figheight()


def _stroke():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=1.6, foreground="black")]


def show_crop(ax, img4, title=None, *, channel="rgb", margin=26):
    if channel == "rgb":
        ax.imshow(np.asarray(img4)[..., :3])
    else:
        ax.imshow(np.asarray(img4)[..., 3] * 255, cmap="gray", vmin=0, vmax=255)
    # fixed limits (small margin) so keypoints the affine pushed OUT of the
    # crop still show just outside the border without rescaling the panel
    ax.set_xlim(-margin, CROP + margin)
    ax.set_ylim(CROP + margin, -margin)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    if title:
        ax.set_title(title, fontsize=6.0, pad=2)


# ------------------------------------------------------------------- stages
def stage_apply(name, key, img4_b, kp_b, vis_b, p, lr_swap):
    """Run ONE augmentation stage alone on a (1,H,W,4) batch. Returns
    (img, kp, vis, label) -- label reports the sampled magnitude where it can
    be recovered without duplicating augment.py's RNG (flip: by comparing the
    image to its mirror; affine: by re-splitting the key the same way, which
    is cross-checked numerically by measure_affine_alignment)."""
    import jax
    from jarvis_jax.data import augment as A

    if name == "affine":
        k1, k2, k3, k4 = jax.random.split(key, 4)
        th = float(jax.random.uniform(k1, (1,), minval=-p.rot_deg, maxval=p.rot_deg)[0])
        sc = float(jax.random.uniform(k2, (1,), minval=p.scale_min, maxval=p.scale_max)[0])
        tx = float(jax.random.uniform(k3, (1,), minval=-p.translate_frac,
                                      maxval=p.translate_frac)[0])
        ty = float(jax.random.uniform(k4, (1,), minval=-p.translate_frac,
                                      maxval=p.translate_frac)[0])
        i, k, v = A.affine_batch(key, img4_b, kp_b, vis_b, rot_deg=p.rot_deg,
                                 scale_min=p.scale_min, scale_max=p.scale_max,
                                 translate_frac=p.translate_frac, heatmap_size=HM)
        lab = f"rot {th:+.1f}° scale {sc:.2f}\nshift ({tx:+.2f},{ty:+.2f})·W"
        return i, k, v, lab
    if name == "flip":
        # flip_p forced to 1.0 so the row demonstrates the stage; the real
        # config value is reported in the row label.
        i, k, v = A.flip_batch(key, img4_b, kp_b, vis_b, lr_swap, 1.0, HM)
        flipped = bool(np.array_equal(np.asarray(i)[0], np.asarray(img4_b)[0, :, ::-1, :]))
        return i, k, v, f"mirrored={flipped} (+L/R label swap)"
    if name == "cutout":
        i = A.cutout_batch(key, img4_b, p.cutout_n, p.cutout_frac)
        side = max(1, int(round(p.cutout_frac * CROP)))
        er = cutout_box_mask(key, p)
        return i, kp_b, vis_b, f"{p.cutout_n} boxes, {side}px side, {100*er.mean():.1f}% erased"
    if name == "photometric":
        i = A.photometric_batch(key, img4_b, p.brightness, p.contrast, p.gamma)
        return i, kp_b, vis_b, f"b±{p.brightness} c±{p.contrast} γ±{p.gamma}"
    if name == "blur":
        k1 = jax.random.uniform(key, (1,), minval=0.0, maxval=p.blur_max)
        i = A.gaussian_blur_batch(key, img4_b, p.blur_max)
        return i, kp_b, vis_b, f"lerp α={float(k1[0]):.2f} of max {p.blur_max}"
    if name == "noise":
        k1, _ = jax.random.split(key)
        s = float(jax.random.uniform(k1, (1, 1, 1, 1), minval=0.0,
                                     maxval=p.noise_scale)[0, 0, 0, 0])
        i = A.gaussian_noise_batch(key, img4_b, p.noise_scale)
        return i, kp_b, vis_b, f"σ={s:.3f} of max {p.noise_scale} (0-1 units)"
    if name == "pc_color":
        f = jax.random.uniform(key, (1, 1, 1, 3), minval=1.0 - p.pc_color,
                               maxval=1.0 + p.pc_color)[0, 0, 0]
        i = A.per_channel_multiply_batch(key, img4_b, p.pc_color)
        return (i, kp_b, vis_b,
                f"RGB x ({f[0]:.2f},{f[1]:.2f},{f[2]:.2f})")
    raise ValueError(name)


STAGES = [
    ("affine", "affine\n(rot/scale/shift)"),
    ("flip", "flip\n(+ L/R swap)"),
    ("cutout", "cutout\n(RGB only)"),
    ("photometric", "photometric\n(bright/contr/gamma)"),
    ("blur", "gaussian blur"),
    ("noise", "gaussian noise"),
    ("pc_color", "per-channel colour"),
]


def cutout_box_mask(key, p):
    """Exact (H,W) bool of which pixels cutout erases for this key, obtained by
    running cutout_batch on an all-255 probe (so already-black image pixels
    cannot be mistaken for erased ones)."""
    import jax.numpy as jnp
    from jarvis_jax.data.augment import cutout_batch
    probe = jnp.full((1, CROP, CROP, 4), 255, dtype=jnp.uint8)
    out = np.asarray(cutout_batch(key, probe, p.cutout_n, p.cutout_frac))[0]
    return (out[..., :3] == 0).all(axis=-1)


# ------------------------------------------------------------------ figures
def fig1_stages(ds_item, names, lr_swap, presets, out_png, caption, seed=0):
    """One row per stage, applied alone to the same crop, next to the original."""
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img4, kp, vis = ds_item
    b = (jnp.asarray(img4)[None], jnp.asarray(kp)[None], jnp.asarray(vis)[None])
    n_def, n_hvy = 3, 2
    ncol = 1 + n_def + n_hvy
    fig, axes = plt.subplots(len(STAGES), ncol,
                             figsize=(2.05 * ncol, 2.32 * len(STAGES)))
    labelled = ("Antenna_Base", "Abd_tip", "WingL_base", "WingR_base")
    for r, (stage, pretty) in enumerate(STAGES):
        show_crop(axes[r, 0], img4, "ORIGINAL (no aug)")
        draw_kp(axes[r, 0], kp, vis, names, label_names=labelled)
        axes[r, 0].set_ylabel(pretty, fontsize=7.5, rotation=0, ha="right",
                              va="center", labelpad=44)
        axes[r, 0].set_yticks([])
        c = 1
        for pname, npanel in (("default", n_def), ("heavy", n_hvy)):
            p = presets[pname]
            for d in range(npanel):
                key = jax.random.PRNGKey(seed + 1000 * r + 17 * d
                                         + (0 if pname == "default" else 500))
                i2, k2, v2, lab = stage_apply(stage, key, *b, p, lr_swap)
                show_crop(axes[r, c], np.asarray(i2)[0], f"{pname}\n{lab}")
                draw_kp(axes[r, c], np.asarray(k2)[0], np.asarray(v2)[0], names,
                        label_names=labelled)
                axes[r, c].title.set_color("#1a6fd4" if pname == "default" else "#b33000")
                c += 1
    top = _suptitle(fig, "ViTPose 2D train augmentations — EACH STAGE ALONE",
                    caption,
                    "cyan = LEFT-side keypoint, orange = RIGHT-side, white = "
                    "midline, red x = vis=False (colour is body SIDE, not fly "
                    "identity)\nparams read from configs/aug/{default,heavy}.yaml; "
                    "the flip row forces flip_p=1.0 to show the stage "
                    "(config flip_p=0.5)")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    fig.savefig(out_png, dpi=135)
    plt.close(fig)


def fig2_fullstack(ds_item, names, lr_swap, presets, out_png, caption, seed=0,
                   ndraw=8):
    """The whole pipeline, several independent draws -- the distribution the
    model actually saw, not one lucky sample."""
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jarvis_jax.data.augment import augment_batch

    img4, kp, vis = ds_item
    b = (jnp.asarray(img4)[None], jnp.asarray(kp)[None], jnp.asarray(vis)[None])
    ncol = 1 + ndraw
    fig, axes = plt.subplots(2, ncol, figsize=(1.85 * ncol, 6.0))
    for r, pname in enumerate(("default", "heavy")):
        p = presets[pname]
        show_crop(axes[r, 0], img4, "ORIGINAL")
        draw_kp(axes[r, 0], kp, vis, names, s=9)
        axes[r, 0].set_ylabel(f"aug={pname}", fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=26)
        for d in range(ndraw):
            key = jax.random.PRNGKey(seed + 7919 * (d + 1) + 31 * r)
            i2, k2, v2 = augment_batch(key, *b, p, lr_swap, HM)
            nlost = int((~np.asarray(v2)[0]).sum() - (~np.asarray(vis)).sum())
            show_crop(axes[r, d + 1], np.asarray(i2)[0],
                      f"draw {d+1}   vis {int(np.asarray(v2)[0].sum())}/{len(names)}"
                      + (f"  ({nlost:+d})" if nlost else ""))
            draw_kp(axes[r, d + 1], np.asarray(k2)[0], np.asarray(v2)[0], names, s=9)
    top = _suptitle(fig, "FULL augmentation stack, independent draws  (affine → "
                    "flip → cutout → photometric → blur → noise → per-channel "
                    "colour); size sweep at fixed targeting", caption,
                    "cyan = LEFT-side keypoint, orange = RIGHT-side, white = "
                    "midline, red x = vis=False — the affine pushed it out of the "
                    "224-unit grid and it no longer supervises anything")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    fig.savefig(out_png, dpi=135)
    plt.close(fig)


def fig3_channels(ds_item, names, lr_swap, p, out_png, caption, seed=0, ndraw=4):
    """RGB and the 4th (SAM mask) channel side by side: cutout writes channels
    0-2 only, so the mask must stay hole-free."""
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jarvis_jax.data.augment import augment_batch, cutout_batch

    img4, kp, vis = ds_item
    b = (jnp.asarray(img4)[None], jnp.asarray(kp)[None], jnp.asarray(vis)[None])
    cols = [("ORIGINAL", np.asarray(img4), kp, vis, None)]
    for d in range(ndraw):
        key = jax.random.PRNGKey(seed + 104729 * (d + 1))
        kc = jax.random.split(key, 7)[2]           # augment_batch's cutout key
        i2, k2, v2 = augment_batch(key, *b, p, lr_swap, HM)
        cols.append((f"full stack, draw {d+1}", np.asarray(i2)[0],
                     np.asarray(k2)[0], np.asarray(v2)[0], kc))
    # cutout alone, so the boxes are unmistakable against a clean crop
    for d in range(2):
        key = jax.random.PRNGKey(seed + 555 + d)
        i2 = np.asarray(cutout_batch(key, b[0], p.cutout_n, p.cutout_frac))[0]
        cols.append((f"cutout ONLY, draw {d+1}", i2, kp, vis, key))

    fig, axes = plt.subplots(2, len(cols), figsize=(2.1 * len(cols), 6.4))
    stats = []
    for c, (title, im, k, v, ckey) in enumerate(cols):
        show_crop(axes[0, c], im, title)
        draw_kp(axes[0, c], k, v, names, s=9)
        show_crop(axes[1, c], im, channel="mask")
        nin = nvis_in = 0
        if ckey is not None:
            box = cutout_box_mask(ckey, p)
            # outline the boxes on BOTH rows so a reader can check the mask
            # panel at exactly the places the RGB was erased
            for ax in (axes[0, c], axes[1, c]):
                ax.contour(box.astype(float), levels=[0.5], colors="#ff2bd6",
                           linewidths=0.9)
            xy = np.clip((np.asarray(k) * KP2PX).astype(int), 0, CROP - 1)
            under = box[xy[:, 1], xy[:, 0]]
            nin = int(under.sum())
            nvis_in = int((under & np.asarray(v).astype(bool)).sum())
            mask_holes = int((box & (np.asarray(im)[..., 3] > 0)).sum())
            stats.append(dict(panel=title, kp_under_box=nin,
                              kp_under_box_visible=nvis_in,
                              mask_px_inside_box=mask_holes,
                              erased_frac=float(box.mean())))
            axes[1, c].set_xlabel(
                f"{mask_holes} mask px still SET\ninside the boxes",
                fontsize=6.0, labelpad=2, color="#127a12")
            axes[0, c].set_xlabel(f"{nvis_in}/{nin} kp under a box\nare vis=True",
                                  fontsize=6.0, labelpad=2, color="#1a6fd4")
        else:
            axes[1, c].set_title("mask ch3 (4th channel)", fontsize=6.0, pad=2)
    axes[0, 0].set_ylabel("RGB (ch 0-2)", fontsize=8, rotation=0, ha="right",
                          va="center", labelpad=26)
    axes[1, 0].set_ylabel("SAM mask (ch 3)", fontsize=8, rotation=0, ha="right",
                          va="center", labelpad=26)
    top = _suptitle(fig, "THE 4-CHANNEL TRUTH — cutout zeroes RGB channels 0-2 "
                    "only", caption,
                    "magenta outline = the cutout boxes, drawn identically on both "
                    "rows. The SAM-mask row must show NO holes inside them.\n"
                    "(the mask DOES follow the affine — warped then re-binarised — "
                    "so it rotates with the RGB in the full-stack columns)")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    fig.savefig(out_png, dpi=135)
    plt.close(fig)
    return stats


def fig4_lrswap(ds_item, names, lr_swap, out_png, caption, seed=0):
    """The decisive left/right test: paint chirality tags into the RGB before
    the flip, then see which LABEL lands on which TAG."""
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jarvis_jax.data.augment import flip_batch

    img4, kp, vis = ds_item
    idx = {n: i for i, n in enumerate(names)}
    tags = [("WingL_base", (0, 200, 255)), ("WingR_base", (255, 165, 0))]
    painted = np.asarray(img4).copy()
    yy, xx = np.mgrid[0:CROP, 0:CROP]
    for n, col in tags:
        cx, cy = np.asarray(kp)[idx[n]] * KP2PX
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= 13.0 ** 2
        for ch in range(3):
            painted[..., ch][disc] = col[ch]

    b = (jnp.asarray(painted)[None], jnp.asarray(kp)[None], jnp.asarray(vis)[None])
    ident = np.arange(len(names), dtype=np.int32)
    i_ok, k_ok, v_ok = flip_batch(jax.random.PRNGKey(seed), *b, jnp.asarray(lr_swap), 1.0, HM)
    i_no, k_no, v_no = flip_batch(jax.random.PRNGKey(seed), *b, jnp.asarray(ident), 1.0, HM)

    # exact algebraic check: kp_out[i] must equal mirror_x(kp_in[lr_swap[i]])
    kin = np.asarray(kp)
    expect = np.stack([(HM - 1) - kin[lr_swap, 0], kin[lr_swap, 1]], -1)
    max_err = float(np.abs(np.asarray(k_ok)[0] - expect).max())
    # how far the label lands from the pixel-exact mirror of the source point
    # (image flip in crop px is x -> CROP-1-x, i.e. HM-0.5-x in heatmap units)
    subpx = float(np.abs(np.asarray(k_ok)[0][:, 0]
                         - ((HM - 0.5) - kin[lr_swap, 0])).max()) * KP2PX

    panels = [
        ("SOURCE (tags painted in)", painted, kp, vis, None),
        (f"flip + REAL build_lr_swap\nblue tag → ORANGE dot = correct",
         np.asarray(i_ok)[0], np.asarray(k_ok)[0], np.asarray(v_ok)[0], "#127a12"),
        ("CONTROL: flip + IDENTITY swap\nblue tag keeps a BLUE dot = the bug",
         np.asarray(i_no)[0], np.asarray(k_no)[0], np.asarray(v_no)[0], "#b30000"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.4))
    for ax, (t, im, k, v, col) in zip(axes, panels):
        show_crop(ax, im, t)
        if col:
            ax.title.set_color(col)
        draw_kp(ax, k, v, names, s=26,
                label_names=("WingL_base", "WingR_base", "EyeL", "EyeR"))
    top = _suptitle(fig, "LEFT / RIGHT SWAP VERIFICATION", caption,
                    "tags painted into the RGB BEFORE the flip: blue disc = "
                    "source WingL_base, orange disc = source WingR_base. Dot "
                    "colour = the LABEL's side (cyan L, orange R).\n"
              f"algebraic check  max|kp_out − mirror(kp_in[lr_swap])| = "
              f"{max_err:.3g} heatmap px     •     label-vs-pixel mirror offset "
              f"= {subpx:.2f} crop px", fontsize=8.5)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    fig.savefig(out_png, dpi=145)
    plt.close(fig)

    # which label actually lands on which painted tag
    kout = np.asarray(k_ok)[0]
    kno = np.asarray(k_no)[0]
    res = {"max_algebraic_err_hm_px": max_err,
           "label_vs_pixel_mirror_offset_crop_px": subpx}
    for n, _ in tags:
        src = np.asarray(kp)[idx[n]] * KP2PX
        tag_after = np.array([(CROP - 1) - src[0], src[1]])     # tag pixel after mirror
        d_ok = np.linalg.norm(kout * KP2PX - tag_after, axis=1)
        d_no = np.linalg.norm(kno * KP2PX - tag_after, axis=1)
        res[f"{n}_tag_gets_label__real_swap"] = names[int(d_ok.argmin())]
        res[f"{n}_tag_gets_label__identity_swap"] = names[int(d_no.argmin())]
    return res


def fig5_flip_sidecolour(ds_item, names, lr_swap, out_png, caption, seed=0):
    """Original vs flipped at full panel size, keypoints coloured by BODY SIDE.

    Expectation: the flipped panel is a bitwise mirror (checked, printed in the
    title), and the side colours EXCHANGE -- the tarsal cluster that is cyan
    (LEFT) at the bottom-left of the original must be ORANGE (RIGHT) at the
    mirrored bottom-right of the flipped panel. Same colour at the mirrored
    position would mean the flip mirrored the pixels but not the labels."""
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jarvis_jax.data.augment import flip_batch

    img4, kp, vis = ds_item
    b = (jnp.asarray(img4)[None], jnp.asarray(kp)[None], jnp.asarray(vis)[None])
    i2, k2, v2 = flip_batch(jax.random.PRNGKey(seed), *b, jnp.asarray(lr_swap), 1.0, HM)
    mirrored = bool(np.array_equal(np.asarray(i2)[0], np.asarray(img4)[:, ::-1, :]))
    lab = ("WingL_base", "WingR_base", "Abd_tip", "Antenna_Base")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.9))
    show_crop(axes[0], img4, "ORIGINAL")
    draw_kp(axes[0], kp, vis, names, s=30, label_names=lab)
    show_crop(axes[1], np.asarray(i2)[0],
              f"FLIPPED  (bitwise mirror of the original: {mirrored})")
    draw_kp(axes[1], np.asarray(k2)[0], np.asarray(v2)[0], names, s=30,
            label_names=lab)
    top = _suptitle(fig, "FLIP — do the SIDE LABELS exchange with the pixels?",
                    caption,
                    "cyan = LEFT-side keypoint, orange = RIGHT-side, white = "
                    "midline. Correct behaviour: the cluster at a given place in "
                    "the original appears at the MIRRORED place with the OPPOSITE "
                    "colour.\nSame colour at the mirrored position = the pixels "
                    "flipped but the labels did not, and every flipped sample "
                    "would then contradict every un-flipped one.", fontsize=8.5)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return mirrored


# --------------------------------------------------------------- measurements
def measure_cutout_coverage(p, ndraw=400, seed=0):
    import jax
    means = []
    for d in range(ndraw):
        means.append(float(cutout_box_mask(jax.random.PRNGKey(seed + d), p).mean()))
    m = np.asarray(means)
    side = max(1, int(round(p.cutout_frac * CROP)))
    per_box_if_inside = (side / CROP) ** 2
    return dict(n_draws=ndraw, cutout_n=p.cutout_n, cutout_frac=p.cutout_frac,
                box_side_px=side,
                mean_erased_frac=float(m.mean()), std_erased_frac=float(m.std()),
                min_erased_frac=float(m.min()), max_erased_frac=float(m.max()),
                naive_n_times_frac=float(p.cutout_n * p.cutout_frac),
                n_times_box_area_if_fully_inside=float(p.cutout_n * per_box_if_inside))


def measure_affine_alignment(ds_item, p, ndraw=12, seed=0):
    """Do keypoints follow the image? Push the SAM mask's centroid through
    affine_transform_kp with the SAME sampled params the image warp used, and
    compare against the centroid of the actually-warped mask. A rigid, purely
    geometric invariant -- not a smoothness check."""
    import jax
    import jax.numpy as jnp
    from jarvis_jax.data import augment as A

    img4, _, _ = ds_item
    m0 = np.asarray(img4)[..., 3] > 0
    c0_px = np.array([np.argwhere(m0)[:, 1].mean(), np.argwhere(m0)[:, 0].mean()])
    c0_hm = c0_px / KP2PX
    errs, clipped = [], []
    for d in range(ndraw):
        key = jax.random.PRNGKey(seed + 4001 * (d + 1))
        img_b = jnp.asarray(img4)[None]
        kp_b = jnp.asarray(c0_hm, dtype=jnp.float32)[None, None]
        vis_b = jnp.ones((1, 1), dtype=bool)
        iw, kw, _ = A.affine_batch(key, img_b, kp_b, vis_b, rot_deg=p.rot_deg,
                                   scale_min=p.scale_min, scale_max=p.scale_max,
                                   translate_frac=p.translate_frac, heatmap_size=HM)
        mw = np.asarray(iw)[0, ..., 3] > 0
        if mw.sum() < 200:
            continue          # mask warped mostly out of frame; centroid meaningless
        w = np.argwhere(mw)
        cw_hm = np.array([w[:, 1].mean(), w[:, 0].mean()]) / KP2PX
        err = float(np.linalg.norm(np.asarray(kw)[0, 0] - cw_hm))
        # a centroid is only affine-equivariant if the WHOLE shape survived the
        # warp; when the crop border clips part of the mask (the wall crop does
        # this constantly) the centroid moves for reasons that have nothing to
        # do with the kp transform, so those draws are reported separately.
        s_used = float(jax.random.uniform(jax.random.split(key, 4)[1], (1,),
                                          minval=p.scale_min, maxval=p.scale_max)[0])
        area_ratio = float(mw.sum()) / (m0.sum() * s_used ** 2)
        (errs if area_ratio > 0.97 else clipped).append(err)
    e = np.asarray(errs) if errs else np.array([np.nan])
    c = np.asarray(clipped) if clipped else np.array([np.nan])
    return dict(n_uncliped=len(errs), n_border_clipped=len(clipped),
                mean_err_hm_px=float(np.nanmean(e)),
                max_err_hm_px=float(np.nanmax(e)),
                mean_err_crop_px=float(np.nanmean(e) * KP2PX),
                border_clipped_mean_err_hm_px=float(np.nanmean(c)),
                note="border-clipped draws excluded from mean_err: a clipped "
                     "mask's centroid is not affine-equivariant, so their error "
                     "measures the clipping, not the keypoint transform")


def measure_supervision_loss(ds_item, names, lr_swap, p, ndraw=200, seed=0):
    """How much annotated supervision the AFFINE throws away on this crop.

    ``affine_batch`` ANDs an in-bounds test into ``vis``, so a keypoint pushed
    outside the 224-unit grid silently stops supervising. A fly sitting near a
    crop edge (the wall case) loses far more than a centred one -- exactly the
    hard case this pipeline already fails on, so it is worth a number and not
    just a picture."""
    import jax
    import jax.numpy as jnp
    from jarvis_jax.data.augment import augment_batch

    img4, kp, vis = ds_item
    b = (jnp.asarray(img4)[None], jnp.asarray(kp)[None], jnp.asarray(vis)[None])
    n0 = int(np.asarray(vis).sum())
    kept, worst = [], None
    per_kp = np.zeros(len(names), dtype=np.int64)
    for d in range(ndraw):
        _, _, v2 = augment_batch(jax.random.PRNGKey(seed + 7919 * (d + 1)),
                                 *b, p, lr_swap, HM)
        v2 = np.asarray(v2)[0].astype(bool)
        kept.append(int(v2.sum()))
        per_kp += (~v2).astype(np.int64)
        if worst is None or v2.sum() < worst:
            worst = int(v2.sum())
    k = np.asarray(kept)
    order = np.argsort(-per_kp)[:6]
    return dict(n_draws=ndraw, n_visible_unaugmented=n0,
                mean_visible_after_aug=float(k.mean()),
                mean_frac_kept=float(k.mean() / max(n0, 1)),
                min_visible_after_aug=int(worst),
                most_often_dropped={names[int(i)]: f"{100*per_kp[int(i)]/ndraw:.0f}%"
                                    for i in order})


def measure_stage_strength(ds_item, names, lr_swap, p, ndraw=40, seed=0):
    """Mean |ΔRGB| per stage in 0-255 units, so "you can barely see the blur /
    noise rows" is a measured claim and not an impression. Geometric stages
    (affine/flip) move pixels wholesale and their |Δ| is not comparable to the
    appearance-only stages -- reported, but flagged."""
    import jax
    import jax.numpy as jnp

    img4, kp, vis = ds_item
    b = (jnp.asarray(img4)[None], jnp.asarray(kp)[None], jnp.asarray(vis)[None])
    ref = np.asarray(img4)[..., :3].astype(np.float32)
    out = {}
    for si, (stage, _) in enumerate(STAGES):
        ds_ = []
        for d in range(ndraw):
            # deterministic across runs -- str hashing is salted per process
            key = jax.random.PRNGKey(seed + 601 * (d + 1) + 97 * si)
            i2, _, _, _ = stage_apply(stage, key, *b, p, lr_swap)
            ds_.append(float(np.abs(np.asarray(i2)[0, ..., :3].astype(np.float32)
                                    - ref).mean()))
        out[stage] = dict(mean_abs_delta_rgb_0_255=float(np.mean(ds_)),
                          max_abs_delta_rgb_0_255=float(np.max(ds_)),
                          geometric=stage in ("affine", "flip"))
    return out



# --------------------------------------------- affine bounds fix (Task 1)
def _sampled_affine(key, p, B):
    """Re-derive the affine ``affine_batch`` drew from `key`, before any
    bounds fitting. Mirrors its key split exactly."""
    import jax
    import jax.numpy as jnp
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return (jnp.deg2rad(jax.random.uniform(k1, (B,), minval=-p.rot_deg, maxval=p.rot_deg)),
            jax.random.uniform(k2, (B,), minval=p.scale_min, maxval=p.scale_max),
            jax.random.uniform(k3, (B,), minval=-p.translate_frac, maxval=p.translate_frac),
            jax.random.uniform(k4, (B,), minval=-p.translate_frac, maxval=p.translate_frac))


BOUNDS_ARMS = ("legacy", "fixed", "cropcentre")


def measure_bounds_fix(ds_item, names, lr_swap, p, ndraw=200, seed=0):
    """Three-arm A/B of the affine bounds fix, on ONE crop and ONE preset.

    arms
      ``legacy``     -- ``keep_kp_in_bounds=False``: the pre-2026-09-02 path.
      ``fixed``      -- the shipped fix: transform centred on the visible-kp
                        centroid, then translation clipped (then scale shrunk)
                        to fit.
      ``cropcentre`` -- the alternative that was considered and rejected: keep
                        the crop centre and rely on the clip/shrink alone. It
                        reaches the same 100% kept, so kept-fraction cannot
                        choose between them; ``n_tx_clipped`` can, and does.

    Measured on the AFFINE ALONE, driven with the exact sub-key
    ``augment_batch`` hands it (``jax.random.split(key, 7)[0]``). The affine is
    the only stage that touches ``vis``, and flip/cutout/photometric leave the
    count alone, so ``frac_kept`` here is bit-identical to the full-stack
    number and directly comparable to the pre-fix values in this file's git
    history (female-wall 82.7% default / 74.4% heavy, female-occlusion 100% /
    99.8%). Driving the affine directly is also the only way to measure the
    ``cropcentre`` arm on the same footing -- the centre override is not a
    training option and ``augment_batch`` has no route to it.

    ``drop_rate`` counts ONLY keypoints that were visible in the unaugmented
    crop, and is measured pre-flip for a second reason: the wall crop ships
    with T2R_TaTip already vis=False (it lies outside the clamped 448-px
    window), and the L/R flip shuffles that dead label into the T2L_TaTip slot
    on half the draws. Reading a full-stack per-keypoint table therefore
    reports "T2R_TaTip 72% / T2L_TaTip 68% dropped" for ONE annotation that
    was never there -- which is exactly what the pre-fix table did.
    """
    import jax
    import jax.numpy as jnp
    from jarvis_jax.data.augment import (augment_batch, affine_batch,
                                         fit_affine_to_bounds)

    img4, kp, vis = ds_item
    v0 = np.asarray(vis).astype(bool)
    n0 = int(v0.sum())
    hm_centre = jnp.array([HM / 2.0, HM / 2.0])
    out = {}
    for arm in BOUNDS_ARMS:
        pa = dataclasses_replace(p, keep_kp_in_bounds=(arm != "legacy"))
        kept, th, sc, tx, ty, shift, black = [], [], [], [], [], [], []
        drops = np.zeros(len(names), np.int64)
        n_sh = n_tc = 0
        for d in range(ndraw):
            key = jax.random.PRNGKey(seed + 7919 * (d + 1))
            b = (jnp.asarray(img4)[None], jnp.asarray(kp)[None], jnp.asarray(vis)[None])
            kkey = jax.random.split(key, 7)[0]      # augment_batch's affine key
            i2, k2, v2 = affine_batch(
                kkey, *b, rot_deg=p.rot_deg, scale_min=p.scale_min,
                scale_max=p.scale_max, translate_frac=p.translate_frac,
                heatmap_size=HM, keep_kp_in_bounds=pa.keep_kp_in_bounds,
                fit_center=(hm_centre if arm == "cropcentre" else None))
            if arm != "cropcentre":                 # cross-check vs the real stack
                assert int(np.asarray(augment_batch(key, *b, pa, lr_swap, HM)[2]).sum()) \
                    == int(np.asarray(v2).sum()), "affine-only kept count diverged"
            v2 = np.asarray(v2)[0].astype(bool)
            kept.append(int(v2.sum()))
            drops += (v0 & ~v2).astype(np.int64)
            # realised (post-fit) affine for this draw
            t0, s0, x0, y0 = _sampled_affine(kkey, p, 1)
            if arm == "legacy":
                t1, s1, x1, y1 = t0, s0, x0, y0
            else:
                ctr = None if arm == "fixed" else hm_centre
                t1, s1, x1, y1, _ = fit_affine_to_bounds(
                    jnp.asarray(kp)[None], jnp.asarray(vis)[None],
                    t0, s0, x0, y0, HM, 0.01, ctr)
            th.append(float(jnp.rad2deg(t1[0]))); sc.append(float(s1[0]))
            tx.append(float(x1[0])); ty.append(float(y1[0]))
            n_sh += int(float(s1[0]) < float(s0[0]) - 1e-6)
            n_tc += int(abs(float(x1[0]) - float(x0[0])) > 1e-6
                        or abs(float(y1[0]) - float(y0[0])) > 1e-6)
            shift.append(float(np.linalg.norm(
                np.asarray(k2)[0][v0].mean(0) - np.asarray(kp)[v0].mean(0))))
            black.append(float((np.asarray(i2)[0, ..., :3].sum(-1) == 0).mean()))
        k = np.asarray(kept)
        f = lambda a: dict(mean=round(float(np.mean(a)), 4), std=round(float(np.std(a)), 4),
                           p5=round(float(np.percentile(a, 5)), 4),
                           p95=round(float(np.percentile(a, 95)), 4))
        order = np.argsort(-drops)[:6]
        out[arm] = dict(
            n_draws=ndraw, n_visible_unaugmented=n0,
            mean_visible_after_aug=float(k.mean()),
            frac_kept=round(float(k.mean() / max(n0, 1)), 4),
            min_visible_after_aug=int(k.min()),
            drop_rate={names[int(i)]: f"{100 * drops[int(i)] / ndraw:.0f}%"
                       for i in order if drops[int(i)] > 0},
            rot_deg=f(th), scale=f(sc), tx=f(tx), ty=f(ty),
            n_scale_shrunk=n_sh, n_translation_clipped=n_tc,
            fly_centroid_shift_hm=f(shift), black_pad_frac=f(black),
            draws=dict(rot_deg=[round(v, 3) for v in th],
                       scale=[round(v, 4) for v in sc],
                       tx=[round(v, 4) for v in tx], ty=[round(v, 4) for v in ty]))
    return out


def fig6_boundsfix(ds_item, names, lr_swap, presets, stats, out_png, caption,
                   seed=0, ndraw=6):
    """Legacy vs fixed affine on the SAME random keys, plus the realised
    parameter distributions.

    EXPECTATION, written before the figure exists
    ---------------------------------------------
    Row 1 (``keep_kp_in_bounds=False``, what shipped until 2026-09-02): on a
      wall crop several draws swing the fly partly outside the 448-px window
      and its distal leg keypoints sit OUTSIDE the panel border as red x --
      annotated tarsal tips that supervise nothing. Panel titles show vis N/49.
    Row 2 (the fix, SAME keys, so flip/cutout/colour are identical and the
      only difference is the affine): zero red x, the fly whole inside the
      crop -- AND the six panels must still show six visibly DIFFERENT
      rotations. If row 2 is six near-upright flies, the "fix" bought its
      100% by collapsing the transform toward identity, which is a failure,
      not a success, and this figure is how you would see it.
    Row 3: realised rotation / scale / translation histograms, legacy vs
      fixed. Rotation and scale must lie exactly on top of each other (the
      fit never touches them); translation may be narrower for the fixed arm
      on the wall crop only -- that is the price, and it is the cheapest of
      the three to pay.
    """
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jarvis_jax.data.augment import augment_batch

    p = presets["default"]
    img4, kp, vis = ds_item
    b = (jnp.asarray(img4)[None], jnp.asarray(kp)[None], jnp.asarray(vis)[None])
    n0 = int(np.asarray(vis).sum())
    fig = plt.figure(figsize=(1.85 * (ndraw + 1), 8.4))
    gs = fig.add_gridspec(3, ndraw + 1, height_ratios=[1.0, 1.0, 0.72], hspace=0.16)
    for r, arm in enumerate(("legacy", "fixed")):
        pa = dataclasses_replace(p, keep_kp_in_bounds=(arm != "legacy"))
        ax0 = fig.add_subplot(gs[r, 0])
        show_crop(ax0, img4, "ORIGINAL")
        draw_kp(ax0, kp, vis, names, s=9)
        ax0.set_ylabel("keep_kp_in_bounds\n" + ("False (old)" if r == 0 else "True (fix)"),
                       fontsize=7.5, rotation=0, ha="right", va="center", labelpad=34)
        for d in range(ndraw):
            key = jax.random.PRNGKey(seed + 7919 * (d + 1))
            i2, k2, v2 = augment_batch(key, *b, pa, lr_swap, HM)
            v2 = np.asarray(v2)[0]
            # COUNT-based, not slot-based: flip_batch permutes vis through
            # lr_swap, so comparing slot-for-slot against the unflipped vis
            # reports a phantom loss of 1 on every flipped draw of a crop that
            # carries a dead label (the wall crop's T2R_TaTip). The count is
            # permutation-invariant and is what "supervision lost" means.
            lost = n0 - int(v2.sum())
            ax = fig.add_subplot(gs[r, d + 1])
            show_crop(ax, np.asarray(i2)[0],
                      f"draw {d+1}   vis {int(v2.sum())}/{n0}"
                      + (f"   LOST {lost}" if lost else ""))
            draw_kp(ax, np.asarray(k2)[0], v2, names, s=9)
            if lost:
                for sp in ax.spines.values():
                    sp.set_edgecolor(C_INVIS); sp.set_linewidth(1.6)
    hist = [("rot_deg", "rotation (deg)"), ("scale", "scale"), ("tx", "translate x (frac)")]
    for c, (kk, lab) in enumerate(hist):
        ax = fig.add_subplot(gs[2, c * 2:(c * 2 + 2)])
        ax.hist(stats["fixed"]["draws"][kk], bins=22, histtype="stepfilled",
                alpha=0.65, color="#33dd77",
                label=f"fixed (std {stats['fixed'][kk]['std']:.3g})")
        # legacy drawn as an OUTLINE on top: for rotation and scale the two are
        # identical by construction, and a filled grey patch hidden underneath
        # would read as "only one arm was plotted"
        ax.hist(stats["legacy"]["draws"][kk], bins=22, histtype="step",
                color="#222222", linewidth=1.1,
                label=f"legacy (std {stats['legacy'][kk]['std']:.3g})")
        ax.set_xlabel(lab, fontsize=7); ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False)
    ax = fig.add_subplot(gs[2, 6:])
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(
        [f"{a:<11}kept {stats[a]['frac_kept']*100:5.1f}%   worst draw "
         f"{stats[a]['min_visible_after_aug']}/{n0}" for a in BOUNDS_ARMS]
        + ["", f"fixed: scale shrunk on {stats['fixed']['n_scale_shrunk']}"
              f"/{stats['fixed']['n_draws']} draws,",
           f"       translation clipped on {stats['fixed']['n_translation_clipped']}",
           f"cropcentre: translation",
           f"       clipped on "
           f"{stats['cropcentre']['n_translation_clipped']}  <-- why it lost"]),
            fontsize=6.0, family="monospace", va="top")
    top = _suptitle(fig, "AFFINE BOUNDS FIX — legacy vs fixed on the SAME keys "
                    "(aug=default; flip/cutout/colour identical, only the "
                    "affine differs)", caption,
                    "cyan = LEFT-side keypoint, orange = RIGHT-side, white = "
                    "midline, red x = vis=False. A red panel border means the "
                    "draw destroyed supervision that the unaugmented crop had. "
                    "Row 2 must keep the labels AND stay visibly diverse — a "
                    "row of near-identical upright flies would mean the fix "
                    "collapsed the augmentation.")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    fig.savefig(out_png, dpi=135)
    plt.close(fig)



# ------------------------------------------ targeted cutout (Task 2 probe)
# A SIZE SWEEP, not a targeting sweep. With uniform centres the box size sets
# how OFTEN the fly is hit; once the centres are on the fly the hit rate is 1
# and the size alone sets how MUCH is hidden. Moving the boxes onto the animal
# without shrinking them therefore turns a mostly-wasted augmentation into an
# aggressive one -- the fly's mask bbox is 278 +- 21 px long and the shipping
# box is 112 px, so a mask-centred 112-px box swallows the thorax and the leg
# bases (visible in fig 7's mask_p1.0 examples).
CUTOUT_ARMS = (("uniform 112px (shipping)", dict()),
               ("mask_p1.0 112px", dict(mask_target_p=1.0)),
               ("mask_p1.0 rel-fly 0.30", dict(mask_target_p=1.0, size_rel_fly=0.30)),
               ("mask_p1.0 rel-fly 0.20", dict(mask_target_p=1.0, size_rel_fly=0.20)),
               ("mask_p1.0 rel-fly 0.12", dict(mask_target_p=1.0, size_rel_fly=0.12)),
               ("mask_p0.5 rel-fly 0.20", dict(mask_target_p=0.5, size_rel_fly=0.20)))


def measure_cutout_targeting(ds, p, indices, names, ndraw=200, seed=0):
    """Where does cutout's erasure budget actually go?

    ``cutout_batch`` samples box centres uniformly over the 448-px crop, but
    the fly is a small object in that crop, so most of the budget lands on
    arena floor. Reported per arm, averaged over `indices` real crops:

      ``frac_erased_on_fly``  of the pixels cutout blanked, what fraction sat
        on the SAM silhouette. Compare it against ``fly_frac_of_crop``: equal
        means the sampler is hitting the animal no better than chance.
      ``frac_of_fly_erased``  how much of the animal is hidden per draw.
      ``p_any_tatip_hidden``  fraction of draws hiding at least one visible
        tarsal tip, resolved BY NAME (``*_TaTip``). The distal tips are the
        keypoints the detector A/B measured worst, so this is the number the
        whole proposal is aimed at.

    and, because a targeted box is far more destructive than a uniform one of
    the same size, three budget numbers that decide whether the sample is
    still LEARNABLE:

      ``mean_kp_hidden`` / ``p95_kp_hidden``  visible keypoints under a box,
        out of 50.
      ``worst_leg_chain_hidden``  the largest fraction of any ONE leg chain
        (``viz.core.colors.leg_chains``, so the limb definition is the repo's
        and not a local invention) that a single draw hides. A box that takes
        the tip AND the whole chain leading to it has removed the context that
        would let the network infer the tip -- that sample is unlearnable, not
        hard. Near 1.0 is the failure signature.
      ``p_whole_chain_hidden``  fraction of draws that do exactly that to at
        least one leg.

    Visibility here is ``transform_keypoints``'s in-crop flag, NOT an
    annotator occlusion judgement -- 1864/1871 val annotations are marked
    fully visible, so the occlusion flags carry no information and nothing in
    this proposal keys off them.

    The silhouette is read at DATA-PREP time; it does not require the model to
    see channel 3, so this applies unchanged to the promoted mask-off
    checkpoint (v5vf_maskoff, trained with train.mask_ablation=true, whose
    first layer provably discards the mask channel).
    """
    import jax
    import jax.numpy as jnp
    from jarvis_jax.data.augment import cutout_batch

    from viz.core.colors import leg_chains

    tip = [i for i, n in enumerate(names) if n.endswith("_TaTip")]
    chains = leg_chains(names)
    out = {}
    fly_fracs = []
    for arm, kw in CUTOUT_ARMS:
        on_fly, of_fly, p_tip, cov = [], [], [], []
        nkp, nkp95, worst_chain, p_chain, sides = [], [], [], [], []
        for idx in indices:
            img4, kp, vis = ds[int(idx)]
            m = np.asarray(img4)[..., 3] > 0
            if m.sum() < 500:
                continue
            fly_fracs.append(float(m.mean()))
            b = jnp.broadcast_to(jnp.asarray(img4)[None], (ndraw,) + img4.shape)
            o = np.asarray(jax.jit(lambda x: cutout_batch(
                jax.random.PRNGKey(seed + int(idx)), x, p.cutout_n,
                p.cutout_frac, **kw))(b))
            er = (o[..., :3].sum(-1) == 0) & (np.asarray(img4)[..., :3].sum(-1) != 0)
            hit = (er & m[None]).sum(axis=(1, 2))
            on_fly.append(float(np.mean(hit / np.maximum(er.sum(axis=(1, 2)), 1))))
            of_fly.append(float(np.mean(hit / m.sum())))
            cov.append(float(er.mean()))
            px = np.clip(np.round(np.asarray(kp) * KP2PX).astype(int), 0, CROP - 1)
            v = np.asarray(vis).astype(bool)
            hid = er[:, px[:, 1], px[:, 0]] & v[None, :]       # (ndraw, K)
            nkp.append(float(hid.sum(1).mean()))
            nkp95.append(float(np.percentile(hid.sum(1), 95)))
            t = [j for j in tip if v[j]]
            if t:
                p_tip.append(float(hid[:, t].any(axis=1).mean()))
            fr = []
            for leg, ch in chains.items():
                c = [j for j in ch if v[j]]
                if c:
                    fr.append(hid[:, c].mean(axis=1))          # (ndraw,)
            if fr:
                mx = np.max(np.stack(fr, 1), axis=1)           # worst leg per draw
                worst_chain.append(float(mx.mean()))
                p_chain.append(float((mx > 0.85).mean()))
            sides.append(float(np.mean(np.abs(
                hid[:, [i for i, n in enumerate(names) if side_of(n) == "L"]].sum(1)
                - hid[:, [i for i, n in enumerate(names) if side_of(n) == "R"]].sum(1)))))
        box = ("%d px" % round(p.cutout_frac * CROP)) if "size_rel_fly" not in kw \
            else ("%.2f x fly (~%d px)" % (kw["size_rel_fly"],
                                           round(kw["size_rel_fly"] * 278)))
        out[arm] = dict(box=box, n_boxes=int(p.cutout_n),
                        frac_erased_on_fly=round(float(np.mean(on_fly)), 4),
                        frac_of_fly_erased=round(float(np.mean(of_fly)), 4),
                        frac_of_crop_erased=round(float(np.mean(cov)), 4),
                        p_any_tatip_hidden=round(float(np.mean(p_tip)), 4),
                        mean_kp_hidden=round(float(np.mean(nkp)), 2),
                        p95_kp_hidden=round(float(np.mean(nkp95)), 2),
                        worst_leg_chain_hidden=round(float(np.mean(worst_chain)), 4),
                        p_whole_chain_hidden=round(float(np.mean(p_chain)), 4),
                        mean_LR_imbalance_kp=round(float(np.mean(sides)), 2),
                        n_crops=len(on_fly))
    out["fly_frac_of_crop"] = round(float(np.mean(fly_fracs)), 4)
    out["note"] = ("frac_erased_on_fly == fly_frac_of_crop means the sampler "
                   "hits the animal no better than chance; "
                   "worst_leg_chain_hidden near 1.0 means the box took a whole "
                   "limb AND the context needed to infer it back")
    return out


def fig7_cutout_targeting(ds_item, names, p, out_png, caption, seed=0, ndraw=300):
    """Where the cutout boxes land: uniform (shipping) vs mask-centred.

    EXPECTATION, written before the figure exists
    ---------------------------------------------
    This is a SIZE sweep at fixed targeting, because targeting alone is not
    the interesting axis: once the centres sit on the animal every box hits
    it, and the size decides whether the sample is hard or unlearnable.
    Top row, the accumulated erase-frequency map over 300 draws with the
      silhouette contour drawn on top:
      * ``uniform 112px`` (what ships) -- a nearly FLAT field across the whole
        448-px crop, dimmer only near the border where boxes get clipped, with
        no visible relationship to the fly contour, and an on-fly percentage
        equal to the fly's share of the crop. That flatness IS the defect.
      * ``mask_p1.0 112px`` -- a blob on the silhouette, but WIDER than it: a
        112-px box on a 278-px fly reaches from the thorax across the leg
        bases. Its example panels should look alarming, and its
        ``mean_kp_hidden`` should be large -- that is the point of showing it.
      * shrinking to ``rel-fly 0.20`` (~56 px) and ``0.12`` (~33 px) must
        tighten the blob onto the animal and drop the keypoints-hidden count,
        WITHOUT the on-fly fraction collapsing back toward chance.
      * ``mask_p0.5 rel-fly 0.20`` -- visibly a blob on a dim floor: the
        mixture, so the model still sees plenty of unoccluded flies.
    Bottom rows: four example crops per arm, with the count of visible
      keypoints hidden printed per panel. In the uniform row several draws
      leave the fly untouched; in the small targeted rows every draw covers a
      limb-sized patch and the keypoints under it stay drawn and stay vis=True
      (cutout must not invalidate labels -- that is its whole point).
    The failure to look for: a box that hides a whole leg chain. The tip is
      then unguessable because its own context went with it, and the sample
      teaches noise. If the 112-px targeted panels show that and the small
      ones do not, the sweep has found its answer.
    """
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jarvis_jax.data.augment import cutout_batch

    img4, kp, vis = ds_item
    m = np.asarray(img4)[..., 3] > 0
    v = np.asarray(vis).astype(bool)
    px = np.clip(np.round(np.asarray(kp) * KP2PX).astype(int), 0, CROP - 1)
    arms = list(CUTOUT_ARMS)
    nex = 3
    fig, axes = plt.subplots(1 + nex, len(arms), figsize=(2.6 * len(arms),
                                                          2.45 * (1 + nex)))
    for c, (label, kw) in enumerate(arms):
        b = jnp.broadcast_to(jnp.asarray(img4)[None], (ndraw,) + img4.shape)
        o = np.asarray(jax.jit(lambda x: cutout_batch(
            jax.random.PRNGKey(seed), x, p.cutout_n, p.cutout_frac, **kw))(b))
        er = (o[..., :3].sum(-1) == 0) & (np.asarray(img4)[..., :3].sum(-1) != 0)
        freq = er.mean(0)
        ax = axes[0, c]
        im = ax.imshow(freq, cmap="magma", vmin=0, vmax=max(0.02, freq.max()))
        ax.contour(m.astype(float), levels=[0.5], colors="#00ffff", linewidths=1.0)
        on = float((er & m[None]).sum()) / max(int(er.sum()), 1)
        hid = er[:, px[:, 1], px[:, 0]] & v[None, :]
        ax.set_title(f"{label}\n{100*on:.0f}% of erased px on the fly "
                     f"(fly = {100*m.mean():.0f}% of crop)\n"
                     f"hides {hid.sum(1).mean():.1f} of {int(v.sum())} "
                     f"keypoints per draw", fontsize=6.2)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(labelsize=5)
        for r in range(nex):
            a = axes[1 + r, c]
            show_crop(a, o[r * 11], None)
            draw_kp(a, kp, vis, names, s=8)
            hit = float((er[r * 11] & m).sum()) / max(int(er[r * 11].sum()), 1)
            a.set_title(f"draw {r+1}: {100*hit:.0f}% on fly, "
                        f"{int(hid[r*11].sum())} kp hidden", fontsize=6.0, pad=2)
    top = _suptitle(fig, "CUTOUT TARGETING — box centres uniform over the crop "
                    "(what ships) vs drawn from the SAM silhouette (proposed, "
                    "default-off)", caption,
                    "cyan contour = SAM silhouette; cyan/orange/white dots = "
                    "LEFT/RIGHT/midline keypoints, which cutout must leave "
                    "vis=True even under a box. The uniform map being flat, "
                    "and its on-fly percentage matching the fly's share of "
                    "the crop, is the measurement that says the erasure "
                    "budget is spent on arena floor. But moving the boxes onto "
                    "the animal at the SHIPPING size hides most of it: read "
                    "the keypoints-hidden counts, not just the on-fly "
                    "percentage. The proposal is targeting AND shrinking.")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    fig.savefig(out_png, dpi=125)
    plt.close(fig)


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # red_data_3d_v5_valfix was deleted 2026-09-02; this is the current root.
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/"
                                      "red_data/red_data_3d_v8_gm_only")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", default="figures/2026-09-02-vit-augmentations")
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="override the automatic hard-crop selection")
    ap.add_argument("--ndraw", type=int, default=8)
    ap.add_argument("--coverage-draws", type=int, default=400)
    ap.add_argument("--targeting-crops", type=int, default=40,
                    help="random train crops for the cutout-targeting measurement")
    ap.add_argument("--bounds-draws", type=int, default=200,
                    help="draws per arm for the affine bounds-fix A/B")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from jarvis_jax.data.v5_2d import V5Dataset
    from jarvis_jax.data.augment import build_lr_swap

    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)

    presets = {}
    raws = {}
    for nm in ("default", "heavy"):
        # Neither preset file writes keep_kp_in_bounds or the two targeted-
        # cutout knobs, so training takes the AugParams defaults (bounds fix
        # ON since 2026-09-02, targeted cutout OFF -- proposed, not adopted).
        # Name them here rather than letting load_aug_yaml pick a default.
        presets[nm], raws[nm] = load_aug_yaml(
            os.path.join(AUG_CFG_DIR, f"{nm}.yaml"), keep_kp_in_bounds=True,
            cutout_mask_target_p=0.0, cutout_size_rel_fly=0.0)
        print(f"[aug/{nm}.yaml] {raws[nm]}")

    ds = V5Dataset(args.root, args.split)
    names = json.load(open(os.path.join(
        args.root, "annotations", f"instances_{args.split}.json")))["keypoint_names"]
    lr_swap = build_lr_swap(names)          # asserts involution internally
    swap_pairs = {n: names[lr_swap[i]] for i, n in enumerate(names)
                  if lr_swap[i] != i}
    print(f"[lr_swap] {len(swap_pairs)}/{len(names)} keypoints swap; "
          f"{len(names) - len(swap_pairs)} midline")

    crops = pick_crops(ds, args.indices)
    stats = {"aug_yaml": raws, "keypoint_names": names,
             "lr_swap_pairs": swap_pairs,
             "lr_swap_midline": [n for i, n in enumerate(names) if lr_swap[i] == i],
             "crops": {}}
    for idx, tag, cap in crops:
        item = ds[idx]
        rec, cam, frame = ds.file_names[idx].split("/")
        caption = (f"{tag} — {cap}   [{rec} / {cam} / {frame}, sex={ds.sex[idx]}, "
                   f"behavior={ds.behavior[idx]}, ann idx {idx}]")
        print(f"\n=== {tag}  idx={idx}  {ds.file_names[idx]}")
        fig1_stages(item, names, lr_swap, presets,
                    os.path.join(out, f"fig1_stages_{tag}.png"), caption, args.seed)
        fig2_fullstack(item, names, lr_swap, presets,
                       os.path.join(out, f"fig2_fullstack_{tag}.png"), caption,
                       args.seed, args.ndraw)
        ch = fig3_channels(item, names, lr_swap, presets["default"],
                           os.path.join(out, f"fig3_channels_{tag}.png"), caption,
                           args.seed)
        lr = fig4_lrswap(item, names, lr_swap,
                         os.path.join(out, f"fig4_lrswap_{tag}.png"), caption, args.seed)
        lr["image_is_bitwise_mirror"] = fig5_flip_sidecolour(
            item, names, lr_swap,
            os.path.join(out, f"fig5_flip_sidecolour_{tag}.png"), caption, args.seed)
        bf = {nm: measure_bounds_fix(item, names, lr_swap, presets[nm],
                                     ndraw=args.bounds_draws, seed=args.seed)
              for nm in ("default", "heavy")}
        fig6_boundsfix(item, names, lr_swap, presets, bf["default"],
                       os.path.join(out, f"fig6_boundsfix_{tag}.png"), caption,
                       args.seed)
        fig7_cutout_targeting(item, names, presets["default"],
                              os.path.join(out, f"fig7_cutout_targeting_{tag}.png"),
                              caption, args.seed)
        al = measure_affine_alignment(item, presets["default"], seed=args.seed)
        sl = {nm: measure_supervision_loss(item, names, lr_swap, presets[nm],
                                           seed=args.seed)
              for nm in ("default", "heavy")}
        st = {nm: measure_stage_strength(item, names, lr_swap, presets[nm],
                                         seed=args.seed)
              for nm in ("default", "heavy")}
        stats["crops"][tag] = dict(index=int(idx), file=ds.file_names[idx],
                                   sex=ds.sex[idx], behavior=ds.behavior[idx],
                                   note=cap, channel_panels=ch, lr_swap_check=lr,
                                   affine_kp_image_alignment=al,
                                   supervision_kept=sl, bounds_fix=bf,
                                   stage_strength=st)
        print(f"  lr_swap check: {lr}")
        print(f"  affine kp/image alignment: {al}")
        for nm, v in sl.items():
            print(f"  supervision kept ({nm}): {v}")
        for nm, v in bf.items():
            print(f"  bounds fix ({nm}): " + "  ".join(
                f"{a} kept={v[a]['frac_kept']:.3f} worst={v[a]['min_visible_after_aug']}"
                f" rot_std={v[a]['rot_deg']['std']:.2f} tclip={v[a]['n_translation_clipped']}"
                for a in BOUNDS_ARMS))
        print("  stage |dRGB| (0-255, default): "
              + ", ".join(f"{k}={v['mean_abs_delta_rgb_0_255']:.2f}"
                          for k, v in st["default"].items()))

    stats["cutout_coverage"] = {
        nm: measure_cutout_coverage(presets[nm], args.coverage_draws, args.seed)
        for nm in ("default", "heavy")}
    tgt_idx = np.random.RandomState(0).choice(len(ds), args.targeting_crops,
                                              replace=False)
    stats["cutout_targeting"] = {
        nm: measure_cutout_targeting(ds, presets[nm], tgt_idx, names,
                                     ndraw=args.bounds_draws, seed=args.seed)
        for nm in ("default", "heavy")}
    print("\n[cutout targeting]", json.dumps(stats["cutout_targeting"], indent=2))
    print("\n[cutout coverage]", json.dumps(stats["cutout_coverage"], indent=2))

    with open(os.path.join(out, "augmentation_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
