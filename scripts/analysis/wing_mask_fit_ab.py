#!/usr/bin/env python3
"""A/B the post-STAC wing-mask PITCH refinement (Stage D2) against the STAC pose.

Runs `scripts.run_bout.wing_mask_fit_bout` -- the PRODUCTION code path, not a
re-implementation -- at one or more `coverage_weight` values on a bout-fly, and
scores every acceptance criterion in
`docs/specs/2026-09-01-wing-orientation-from-masks-design.md` §5.

READ-ONLY on the source run. Nothing is written under `--bout-dir`; the refined
poses, the scorecard and the per-arm npz all go to `--out`.

THE SIX CRITERIA (§5; criterion 6 is a figure, made by scripts/viz/wing_fit_7cam.py):

  1. PER-DOF SONG GUARD, and it comes first. `hp_rms` of `wing_pitch_*` on the
     SINGING wing must stay within 20% of control. This is per-DOF on purpose:
     the rejected rest prior kept `|yawL-yawR|` at 38->39 deg while removing 75%
     of the extended wing's PITCH dynamics, so a yaw-only guard passed it.
  2. SONG UNCHANGED. `|yawL-yawR|` mean and its pulse stats within noise.
     Yaw is not in `opt_mask`, so a change here means the wrong DOFs moved.
  3. PENETRATION REDUCED on BOTH wings: `mj_geomDistance(wing, abdomen)` median
     from ~-0.043 toward the model's own -0.0013 grazing value.
  4. MARKER FIT NOT DEGRADED: wing-keypoint residual rise <= 20%.
  5. PITCH LANDS NEAR THE MASK OPTIMUM (-20..-40 deg on the folded wing), NOT at
     the springref rest value of -57.3 deg. Rest is the WRONG target -- that is
     what the rejected prior overshot to.
  7. BLADE NORMAL IS REPORTED, NEVER GATED ON. It is a hypersensitive function
     of the marker null direction (a 12% residual change once swung it 59 deg),
     so it cannot adjudicate a change to that direction.

INVARIANTS CHECKED BEFORE ANY NUMBER IS BELIEVED (CLAUDE.md: "check a rigid
invariant, not just smoothness"):

  * `WingX_V12`-`WingX_V13` both sit on the `wing_left`/`wing_right` BODY, so
    their FK separation is a RIGID length and must be bit-identical between the
    arms -- wing pitch rotates the body they live on. If it moves, the fit did
    something other than rotate the blade. (`WingX_base` maps to the THORAX, so
    base->V12 is NOT invariant and is not checked.)
  * every qpos address except `wing_pitch_left`/`wing_pitch_right`, resolved BY
    JOINT NAME, must be bit-identical. That is `refine_wing_pitch`'s `opt_mask`
    contract; an all-False mask would silently return `q_init` and every metric
    would read "no change" rather than "broken".

MASK-SIDE METRICS (the coverage-weight question, Task 5's two objective numbers,
recomputed here on BOTH flies over the requested window):
  inside%    -- projected wing vertices landing INSIDE the SAM mask. Should stay
                high; a drop means the wing was driven out of the silhouette.
  explained% -- mask pixels the projected BODY does not explain that a wing
                vertex is within ~6 px of. The coverage term's own goal.
  The body basis is held FIXED (stride 8) regardless of the optimiser's
  `body_vertex_stride`, so rows stay comparable across arms.

Run (GPU node; `unset LD_LIBRARY_PATH` first):
  python scripts/analysis/wing_mask_fit_ab.py \
      --bout-dir <run_root>/bouts/bout_00028 --flies 0,1 \
      --coverage-weights 0.0,0.1,0.3,1.0 --t0 0 --nt 1500 \
      --out figures/2026-09-01-wing-mask-fit

--------------------------------------------------------------------------------
`--plot`: THE FIGURES, drawn from saved arms, running no fits
--------------------------------------------------------------------------------

`--plot` reads the `arms_flyN.npz` a scoring run already wrote and draws the
three figures below. It runs no optimiser and writes nothing outside `--out`;
the only thing it reads under the run root is `stac_ik.h5`, for the model these
poses were solved against. Every DOF is resolved with `mj_name2id` +
`jnt_qposadr` (`joint_qposadr`), never by a hardcoded address -- the wing joints
happen to sit at 7..12 in the v1 model and at DIFFERENT addresses in v2_3, and
this repo has twice produced confident, self-consistent, completely wrong
numbers by indexing an axis positionally.

  --plot pitch   FIGURE A. Wing PITCH, control vs one arm, 4 rows
                 (fly1 singing / fly1 folded / fly0 left / fly0 right) x
                 [full window | zoom]. One PNG PER ARM, all with the same axes,
                 so two configurations can be laid side by side. Every panel
                 carries `hp_rms` (criterion 1's own number) AND `lp_std`.
  --plot alldof  FIGURE B. All six wing DOFs -- yaw/roll/pitch, left and right --
                 one PNG per fly, every arm overlaid on control. This is the
                 picture of the "wing PITCH only" guarantee.
  --plot delta   FIGURE C. `max |dqpos|` over all `nq` addresses, per arm.
  --plot crit1   FIGURE D. Criterion 1 close up: a short window INSIDE the fuzzy
                 stretch with every arm overlaid, beside the distribution of the
                 frame-to-frame step that `hp_rms` is summarising.
  --plot bilateral FIGURE E, THE DECISIVE ONE. LEFT vs RIGHT wing pitch on one
                 axis per arm, their coherence, and their cross-correlation.
                 Both wings are driven during song, so song-driven pitch is
                 phase-locked ACROSS wings and independent mask noise is not.
  --plot song    FIGURE F. Song-band peak vs broadband floor in each arm's pitch
                 spectrum, and coherence with the SAME wing's yaw.
  --plot pulses  FIGURE G. Every detected pulse, one common threshold, full
                 window: preserved, delayed, or smoothed over.

  All of these score PER EXTENDED-WING EPOCH (`singer_epochs`), because fly1
  swaps which wing it extends at ~frame 869 on this bout.

EXPECTATIONS, WRITTEN BEFORE THE FIGURES WERE GENERATED (CLAUDE.md: "a figure
you can't be wrong about proves nothing"):

  A. The shipped arm (`smooth_weight` 0.005) and the `smooth_weight = 30` arm
     should sit in the SAME place -- both pulled down out of the control's
     ~-6..-8 deg into the -20..-40 deg measured mask optimum, neither anywhere
     near the springref rest line at -57.3 deg -- and differ only in TEXTURE:
     the shipped green trace visibly fuzzy frame-to-frame, the smooth-30 trace
     following the same slow structure cleanly.
     THE WAY THIS FAILS, and it is the failure a previous attempt at this
     problem actually hit (a rest prior that removed 75% of the extended wing's
     pitch dynamics): smooth-30 could buy criterion 1 by FLATTENING the wing,
     not by denoising it. Then the singing wing's real excursions -- the slow
     10-30 deg swings the control shows -- would be gone, the trace nearly a
     line, and the low-frequency std `lp_std` would COLLAPSE relative to
     control while `hp_rms` fell. `hp_rms` alone cannot tell those apart, which
     is exactly why `lp_std` is printed next to it in every panel. A pass on
     criterion 1 with a collapsed `lp_std` is not a fix, it is the old bug.
  B. `wing_yaw_*` and `wing_roll_*` are NOT in `refine_wing_pitch`'s `opt_mask`,
     so their traces should lie EXACTLY on control -- not close, identical, with
     `max |d|` printing as 0.000000 deg. Anything else means the wrong DOFs
     moved and criterion 2's "bit-identical" verdict was luck. Only the two
     `wing_pitch_*` rows should separate.
  C. The same claim as a spectrum: exactly two of the `nq` bars nonzero, both
     named `wing_pitch_*`, every other address flat at zero. If a third bar
     appears the `opt_mask` contract is broken.
  D. The shipped arm's frame-to-frame step should have a long tail -- per-frame
     excursions of tens of degrees that neither control nor smooth-30 has --
     while all three arms sit at the SAME median pitch. Texture, not placement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.analysis.wing_pitch_rest_prior_ab import pulse_stats  # noqa: E402

# The two DOFs Stage D2 is allowed to move, and the four it must not.
PITCH = ("wing_pitch_left", "wing_pitch_right")
WING_DOFS = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
             "wing_yaw_right", "wing_roll_right", "wing_pitch_right")


def compose_cfg(overrides):
    """The real shipped pipeline config, with `overrides` applied."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver(
        "basename", lambda p: os.path.basename(os.path.normpath(str(p))),
        replace=True)
    os.environ.setdefault("USER", "eabe")
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.join(_REPO, "configs")):
        return compose(config_name="pipeline", overrides=list(overrides))


def joint_qposadr(mj, names):
    """qpos address of each joint BY NAME. Never positional: which address is
    left and which is right is a property of the XML."""
    import mujoco
    out = {}
    for n in names:
        j = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j < 0:
            raise KeyError(f"joint {n!r} is not in this model")
        out[n] = int(mj.jnt_qposadr[j])
    return out


def singing_wing(q, adr):
    """Which wing is EXTENDED (the singer). Measured, not assumed.

    `wing_yaw_*`'s springref sits AT the joint's upper stop (+85.94 deg, the
    known "wing_yaw rest == joint stop" gotcha), so a FOLDED wing reads near the
    stop and the extended wing is the one furthest from it. Returns
    ('left'|'right', mean |yawL - yawR| in deg); a small asymmetry means neither
    wing is extended (the female), and the caller should treat criterion 1 as
    not applicable rather than gating on noise.
    """
    fin = np.isfinite(q).all(axis=1)
    yl = np.degrees(q[fin, adr["wing_yaw_left"]])
    yr = np.degrees(q[fin, adr["wing_yaw_right"]])
    return ("left" if yl.mean() < yr.mean() else "right",
            float(np.mean(np.abs(yl - yr))))


def pose_metrics(q, adr, rest, fs):
    """Per-DOF pulse statistics + the song asymmetry, in degrees."""
    out = {}
    for n in WING_DOFS:
        v = np.degrees(q[:, adr[n]])
        out[n] = dict(median_deg=float(np.nanmedian(v)),
                      off_rest_deg=float(np.nanmedian(
                          np.abs(v - np.degrees(rest[adr[n]])))),
                      pulse=pulse_stats(v, fs))
    song = np.degrees(q[:, adr["wing_yaw_left"]] - q[:, adr["wing_yaw_right"]])
    out["song"] = dict(absmean_deg=float(np.nanmean(np.abs(song))),
                       pulse=pulse_stats(song, fs))
    return out


def _wing_geoms(mj):
    import mujoco
    wg = {}
    for sd in ("left", "right"):
        bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, f"wing_{sd}")
        wg[sd] = [g for g in range(mj.ngeom) if mj.geom_bodyid[g] == bid]
    abd = [g for g in range(mj.ngeom)
           if "abdomen" in (mujoco.mj_id2name(
               mj, mujoco.mjtObj.mjOBJ_BODY, int(mj.geom_bodyid[g])) or "")]
    return wg, abd


def penetration(mj, dat, q, stride=10):
    """Median `mj_geomDistance(wing, abdomen)` per wing. NEGATIVE = the blade is
    inside the abdomen. MuJoCo's own distance, not a proxy; the model at its own
    rest pose merely grazes at -0.0013."""
    import mujoco
    wg, abd = _wing_geoms(mj)
    if not abd:
        return {}
    out = {}
    for sd in ("left", "right"):
        vals = []
        for t in range(0, len(q), stride):
            if not np.isfinite(q[t]).all():
                continue
            dat.qpos[:] = q[t]
            mujoco.mj_forward(mj, dat)
            vals.append(min(mujoco.mj_geomDistance(mj, dat, g1, g2, 1.0, None)
                            for g1 in wg[sd] for g2 in abd))
        out[sd] = float(np.median(vals)) if vals else float("nan")
    return out


def site_traj(mj, dat, q, site_idx):
    """FK site positions (T, len(site_idx), 3) in MODEL units."""
    import mujoco
    out = np.full((len(q), len(site_idx), 3), np.nan)
    for t in range(len(q)):
        if not np.isfinite(q[t]).all():
            continue
        dat.qpos[:] = q[t]
        mujoco.mj_forward(mj, dat)
        out[t] = dat.site_xpos[site_idx]
    return out


def blade_normal_deg(fit_xyz, meas_xyz):
    """Angle between the MEASURED and FITTED base-V12-V13 plane normals, deg.

    REPORTED, NEVER GATED ON (spec §5.7): the normal is a hypersensitive
    function of the very null direction this stage acts in.
    """
    def nrm(X):
        v = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
        L = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.where(L > 1e-12, v / np.where(L > 1e-12, L, 1.0), np.nan)
    a, b = nrm(fit_xyz), nrm(meas_xyz)
    ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    if not ok.any():
        return float("nan")
    c = np.abs(np.sum(a[ok] * b[ok], axis=-1))
    return float(np.median(np.degrees(np.arccos(np.clip(c, -1, 1)))))


def mask_metrics(cfg, q_by_arm, bridges, masks, present, frames, cameras,
                 body_stride=8):
    """inside% / explained% per arm, over `frames` x the present cameras.

    The body raster is built from the CONTROL pose and reused for every arm --
    the body does not move, and holding it fixed is what makes the arms
    comparable. `body_stride` is deliberately independent of the optimiser's
    `body_vertex_stride` for the same reason.
    """
    import jax.numpy as jnp
    from scipy import ndimage
    from jarvis_jax.tracking.appendage_dof import appendage_vertex_indices
    from jarvis_jax.tracking.fk import load_anatomy, make_fk_repose
    from jarvis_jax.tracking.wing_mask_refine import (affine_cameras_by_name,
                                                      body_vertex_indices)
    anat = load_anatomy(str(cfg.ik.xml), str(cfg.ik.mesh_npz))
    fk = make_fk_repose(anat)
    widx = appendage_vertex_indices(str(cfg.ik.mesh_npz),
                                    subset=str(cfg.ik.mesh_subset),
                                    include=("wing",))
    bidx = body_vertex_indices(str(cfg.ik.mesh_npz), stride=body_stride)
    cam_Ms, cam_ts = affine_cameras_by_name(str(cfg.recording.calib_dir), cameras)
    bs, bR, bt = bridges

    def uv_all(qt, t, idx):
        v = np.asarray(fk(jnp.asarray(qt), 1.0, jnp.asarray(idx)))
        w = bs[t] * (v @ bR[t].T) + bt[t]                       # model -> mm
        return np.stack([w @ cam_Ms[c].T + cam_ts[c] for c in range(len(cameras))])

    acc = {a: dict(inside=[], expl=[]) for a in q_by_arm}
    ctrl = q_by_arm["control"]
    for t in frames:
        buv = uv_all(ctrl[t], t, bidx)
        wuv = {a: uv_all(q[t], t, widx) for a, q in q_by_arm.items()}
        for c in range(len(cameras)):
            if not present[t, c]:
                continue
            mk = np.asarray(masks[t][c], bool)
            H, W = mk.shape
            xy = np.rint(buv[c]).astype(int)
            ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
            body = np.zeros_like(mk)
            body[xy[ok, 1], xy[ok, 0]] = True
            body = ndimage.binary_dilation(body, iterations=3)
            unexpl = mk & ~body
            for a in q_by_arm:
                xy = np.rint(wuv[a][c]).astype(int)
                ok = ((xy[:, 0] >= 0) & (xy[:, 0] < W)
                      & (xy[:, 1] >= 0) & (xy[:, 1] < H))
                ins = np.zeros(len(xy), bool)
                ins[ok] = mk[xy[ok, 1], xy[ok, 0]]
                acc[a]["inside"].append(ins.mean())
                wim = np.zeros_like(mk)
                wim[xy[ok, 1], xy[ok, 0]] = True
                wim = ndimage.binary_dilation(wim, iterations=6)
                acc[a]["expl"].append((unexpl & wim).sum() / max(1, unexpl.sum()))
    return {a: dict(inside_pct=100 * float(np.mean(v["inside"])),
                    explained_pct=100 * float(np.mean(v["expl"])),
                    n_samples=len(v["inside"]))
            for a, v in acc.items()}


def score_fly(cfg_base, bout_dir, fly, weights, t0, nt, fs, out_dir,
              mask_frames=40, from_arms=None):
    import h5py
    import mujoco
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from omegaconf import OmegaConf
    from jarvis_jax.tracking.bout_masks import load_bout_masks
    from scripts.run_bout import wing_mask_fit_bout
    from viz.config import resolve_body_model_xml

    fdir = os.path.join(bout_dir, f"fly{fly}")
    stac_h5 = os.path.join(fdir, "stac_ik.h5")
    ref = np.load(os.path.join(fdir, "qpos_refined.npz"))
    q_ctrl_full = np.asarray(ref["qpos"], np.float32)
    bridges = (ref["bridge_s"], ref["bridge_R"], ref["bridge_t"])
    bok = ref["bridge_ok"]

    d = ioh5.load(stac_h5)
    kpn = [x.decode() if isinstance(x, bytes) else str(x)
           for x in np.asarray(d["kp_names"])]
    with h5py.File(stac_h5, "r") as f:
        icfg = OmegaConf.create(f["config"][()].decode())
    xml = resolve_body_model_xml(icfg.model.MJCF_PATH)
    mj = mujoco.MjModel.from_xml_path(xml)
    # The run's OWN fitted marker offsets onto the model sites -- without them
    # the "wing-keypoint residual" would be measured against nominal landmarks
    # the solve never targeted.
    sites = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i)
             for i in range(mj.nsite)]
    off = np.asarray(d["offsets"], float)
    for i, n in enumerate(kpn):
        if f"tracking[{n}]" in sites:
            mj.site_pos[sites.index(f"tracking[{n}]")] = off[i]
    dat = mujoco.MjData(mj)
    rest = np.asarray(mj.qpos_spring, float)
    adr = joint_qposadr(mj, WING_DOFS)
    kp_model = np.asarray(d["kp_data"]).reshape(len(q_ctrl_full), len(kpn), 3)

    wing_kp = [n for n in kpn if "Wing" in n and f"tracking[{n}]" in sites]
    widx = [sites.index(f"tracking[{n}]") for n in wing_kp]
    wrow = [kpn.index(n) for n in wing_kp]

    cameras = [str(c) for c in cfg_base.recording.cameras]
    bout_idx = int(os.path.basename(bout_dir).split("_")[-1])
    bout_npz = os.path.join(str(cfg_base.recording.predictions_dir),
                            f"bout_{bout_idx:05d}", "sam3_masks.npz")
    masks_dict = load_bout_masks(bout_npz, fly, expected_cameras=cameras)

    # --- run the PRODUCTION stage worker, once per coverage weight -----------
    # `--from-arms` re-scores poses a previous run already produced, so a new
    # metric can be added without paying for the fits again. It CANNOT invent an
    # arm: what is in the npz is what is scored.
    arms = {"control": q_ctrl_full}
    stats = {}
    if from_arms is not None:
        with np.load(from_arms) as z:
            for k in z.files:
                if k.startswith("q_"):
                    arms[k[2:]] = np.asarray(z[k], np.float32)
        if not np.array_equal(np.nan_to_num(arms["control"]),
                              np.nan_to_num(q_ctrl_full)):
            raise SystemExit(f"{from_arms}'s q_control is not this fly's "
                             f"qpos_refined.npz -- wrong file or wrong fly")
        print(f"[ab] fly{fly} re-scoring {sorted(arms)} from {from_arms}")
        weights = []
    for name, over in weights:
        cfg_w = compose_cfg(["paths=hyak", "wing_mask_fit.enabled=true", *over])
        t_start = time.time()
        q_w, st = wing_mask_fit_bout(cfg_w, q_ctrl_full, *bridges, bok,
                                     masks_dict, cameras)
        st["wall_s"] = round(time.time() - t_start, 1)
        st["overrides"] = list(over)
        arms[name] = np.asarray(q_w, np.float32)
        stats[name] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                       for k, v in st.items()}
        print(f"[ab] fly{fly} {name}: {st['wall_s']}s  "
              f"refined {st['n_refined']}/{st['n_frames']}  "
              f"dpitch L {st['dpitch_left_deg']:+.2f} / "
              f"R {st['dpitch_right_deg']:+.2f} deg", flush=True)
    if from_arms is None:
        np.savez_compressed(os.path.join(out_dir, f"arms_fly{fly}.npz"),
                            **{f"q_{k}": v for k, v in arms.items()})

    sl = slice(t0, t0 + nt)
    singer, song_asym = singing_wing(q_ctrl_full[sl], adr)
    res = {"fly": fly, "window": [t0, t0 + nt], "fs_hz": fs,
           "singing_wing": singer, "control_song_absmean_deg": song_asym,
           "folded_wing": "right" if singer == "left" else "left",
           "stage_stats": stats, "arms": {}}

    # --- rigid invariant + frozen-DOF contract, BEFORE any conclusion --------
    v12v13 = {}
    for sd, tag in (("left", "L"), ("right", "R")):
        pair = [f"Wing{tag}_V12", f"Wing{tag}_V13"]
        if all(f"tracking[{p}]" in sites for p in pair):
            v12v13[sd] = [sites.index(f"tracking[{p}]") for p in pair]
    free = np.array([adr[n] for n in PITCH])
    frozen = np.setdiff1d(np.arange(q_ctrl_full.shape[1]), free)

    for name, q in arms.items():
        qw = q[sl]
        m = dict(pose=pose_metrics(qw, adr, rest, fs),
                 penetration_median=penetration(mj, dat, qw))
        S = site_traj(mj, dat, qw, widx)
        R = np.linalg.norm(S - kp_model[sl][:, wrow], axis=-1)
        m["wing_resid_median"] = float(np.nanmedian(R))
        m["wing_resid_per_kp"] = {n: float(np.nanmedian(R[:, i]))
                                  for i, n in enumerate(wing_kp)}
        m["blade_normal_deg"] = {}
        for sd, tag in (("left", "L"), ("right", "R")):
            trip = [f"Wing{tag}_{p}" for p in ("base", "V12", "V13")]
            if all(t_ in kpn and f"tracking[{t_}]" in sites for t_ in trip):
                si = [sites.index(f"tracking[{t_}]") for t_ in trip]
                ki = [kpn.index(t_) for t_ in trip]
                m["blade_normal_deg"][sd] = blade_normal_deg(
                    site_traj(mj, dat, qw, si), kp_model[sl][:, ki])
        # RIGID INVARIANT: V12-V13 both live on the wing body, so wing PITCH
        # cannot change their separation. A drift here means something other
        # than the blade rotated.
        m["v12v13_len"] = {}
        for sd, si in v12v13.items():
            P = site_traj(mj, dat, qw, si)
            L = np.linalg.norm(P[:, 0] - P[:, 1], axis=-1)
            m["v12v13_len"][sd] = dict(
                median=float(np.nanmedian(L)),
                cv=float(np.nanstd(L) / max(1e-12, np.nanmean(L))))
        if name != "control":
            fin = np.isfinite(arms["control"]).all(1) & np.isfinite(q).all(1)
            m["max_abs_change_frozen_dofs"] = float(
                np.abs(q[fin][:, frozen] - arms["control"][fin][:, frozen]).max())
            m["max_abs_change_pitch_deg"] = float(np.degrees(
                np.abs(q[fin][:, free] - arms["control"][fin][:, free]).max()))
            yaw = np.array([adr["wing_yaw_left"], adr["wing_yaw_right"]])
            m["max_abs_change_yaw"] = float(
                np.abs(q[fin][:, yaw] - arms["control"][fin][:, yaw]).max())
            # WHAT KIND of hp_rms change is it? A criterion-1 miss can be the
            # fit REMOVING real dynamics (the rest prior's failure: the change
            # is smooth and the treatment's hp_rms falls) or ADDING frame-to-
            # frame jitter of its own (the treatment's hp_rms rises). The
            # high-passed RMS of the CHANGE itself separates them: a smooth
            # correction has a small one, per-frame mask noise a large one.
            m["delta_pulse"] = {
                n: pulse_stats(np.degrees(qw[:, adr[n]]
                                          - arms["control"][sl][:, adr[n]]), fs)
                for n in PITCH}
        res["arms"][name] = m

    # --- mask-side metrics (the coverage-weight question) --------------------
    present = np.asarray(masks_dict["valid"], bool) & np.asarray(bok, bool)[:, None]
    fin = np.isfinite(q_ctrl_full).all(1)
    cand = [t for t in range(t0, t0 + nt) if fin[t] and present[t].sum() >= 4]
    step = max(1, len(cand) // mask_frames)
    frames = cand[::step][:mask_frames]
    res["mask_metrics"] = mask_metrics(
        cfg_base, {k: v for k, v in arms.items()}, bridges,
        masks_dict["masks"], present, frames, cameras)
    res["mask_metric_frames"] = len(frames)
    return res


def _fmt(v, spec=".4f"):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else format(v, spec)


def report(res, springref_deg=-57.29578):
    """Print the acceptance table. Gates are RATIOS TO THIS RUN'S OWN CONTROL."""
    fly, singer, folded = res["fly"], res["singing_wing"], res["folded_wing"]
    ctl = res["arms"]["control"]
    print(f"\n===== fly{fly}  frames {res['window'][0]}..{res['window'][1]}  "
          f"fs={res['fs_hz']} Hz  singer={singer} (|yawL-yawR| "
          f"{res['control_song_absmean_deg']:.2f} deg) =====")
    hdr = (f"{'arm':>10}{'pitchL hp_rms':>15}{'pitchR hp_rms':>15}"
           f"{'d(pitchL) hp':>14}{'d(pitchR) hp':>14}"
           f"{'|yawL-yawR|':>13}{'song hp_rms':>13}{'song kurt':>11}"
           f"{'song peaks':>12}")
    print(hdr)
    for a, m in res["arms"].items():
        p = m["pose"]
        dp = m.get("delta_pulse") or {}
        print(f"{a:>10}{p['wing_pitch_left']['pulse']['hp_rms']:>15.4f}"
              f"{p['wing_pitch_right']['pulse']['hp_rms']:>15.4f}"
              f"{_fmt(dp.get('wing_pitch_left', {}).get('hp_rms'), '.4f'):>14}"
              f"{_fmt(dp.get('wing_pitch_right', {}).get('hp_rms'), '.4f'):>14}"
              f"{p['song']['absmean_deg']:>13.2f}"
              f"{p['song']['pulse']['hp_rms']:>13.4f}"
              f"{p['song']['pulse']['kurtosis']:>11.2f}"
              f"{p['song']['pulse']['n_peaks']:>12d}")
    print(f"\n{'arm':>10}{'pen L':>10}{'pen R':>10}{'wing resid':>12}"
          f"{'pitchL med':>12}{'pitchR med':>12}{'bladeN L':>10}{'bladeN R':>10}"
          f"{'inside%':>9}{'expl%':>8}")
    for a, m in res["arms"].items():
        mm = res["mask_metrics"].get(a, {})
        print(f"{a:>10}{_fmt(m['penetration_median'].get('left'), '.5f'):>10}"
              f"{_fmt(m['penetration_median'].get('right'), '.5f'):>10}"
              f"{m['wing_resid_median']:>12.4f}"
              f"{m['pose']['wing_pitch_left']['median_deg']:>12.1f}"
              f"{m['pose']['wing_pitch_right']['median_deg']:>12.1f}"
              f"{_fmt(m['blade_normal_deg'].get('left'), '.1f'):>10}"
              f"{_fmt(m['blade_normal_deg'].get('right'), '.1f'):>10}"
              f"{_fmt(mm.get('inside_pct'), '.1f'):>9}"
              f"{_fmt(mm.get('explained_pct'), '.1f'):>8}")
    print(f"  (penetration: mj_geomDistance, NEGATIVE = blade inside the "
          f"abdomen; the model's own rest pose grazes at -0.0013)")
    print(f"  (blade normal is REPORTED, never a gate -- spec 5.7)")

    verdicts = {}
    for a, m in res["arms"].items():
        if a == "control":
            continue
        v, p = {}, m["pose"]
        # 1. per-DOF song guard on the SINGING wing
        if res["control_song_absmean_deg"] < 10.0:
            v["1_song_pitch_dynamics"] = dict(
                verdict="N/A", reason="no wing extension: this fly is not singing")
        else:
            n = f"wing_pitch_{singer}"
            c0 = ctl["pose"][n]["pulse"]["hp_rms"]
            c1 = p[n]["pulse"]["hp_rms"]
            v["1_song_pitch_dynamics"] = dict(
                dof=n, control=c0, treatment=c1, ratio=c1 / c0,
                # LOST vs ADDED: ratio<1 means the fit removed real dynamics
                # (the rest prior's failure); ratio>1 with a large delta_hp_rms
                # means it injected per-frame mask noise instead.
                delta_hp_rms=m["delta_pulse"][n]["hp_rms"],
                direction="dynamics LOST" if c1 < c0 else "jitter ADDED",
                verdict="PASS" if abs(c1 / c0 - 1) <= 0.20 else "FAIL")
        # 2. |yawL-yawR| + its pulse stats. Yaw is NOT in `opt_mask`, so a
        #    correct implementation makes this bit-identical, not merely close;
        #    the tolerant thresholds are kept so a near-miss is still visible as
        #    a number rather than only as a boolean.
        y0, y1 = ctl["pose"]["song"], p["song"]
        dm = abs(y1["absmean_deg"] - y0["absmean_deg"])
        dr = abs(y1["pulse"]["hp_rms"] / max(1e-12, y0["pulse"]["hp_rms"]) - 1)
        yaw_exact = m.get("max_abs_change_yaw", None)
        v["2_song_unchanged"] = dict(
            control_absmean_deg=y0["absmean_deg"],
            treatment_absmean_deg=y1["absmean_deg"],
            hp_rms_ratio=y1["pulse"]["hp_rms"] / max(1e-12, y0["pulse"]["hp_rms"]),
            n_peaks=[y0["pulse"]["n_peaks"], y1["pulse"]["n_peaks"]],
            max_abs_yaw_change_rad=yaw_exact,
            verdict="PASS" if (dm < 0.5 and dr < 0.05
                               and (yaw_exact is None or yaw_exact == 0.0))
            else "FAIL")
        # 3. penetration toward -0.0013 on BOTH wings
        ok = True
        pen = {}
        for sd in ("left", "right"):
            a0 = ctl["penetration_median"].get(sd)
            a1 = m["penetration_median"].get(sd)
            pen[sd] = [a0, a1]
            if a0 is None or a1 is None or not (a1 > a0):
                ok = False
        v["3_penetration"] = dict(control_vs_treatment=pen,
                                  target=-0.0013,
                                  verdict="PASS" if ok else "FAIL")
        # 4. wing-keypoint residual rise <= 20%
        rr = m["wing_resid_median"] / ctl["wing_resid_median"]
        v["4_marker_residual"] = dict(control=ctl["wing_resid_median"],
                                      treatment=m["wing_resid_median"],
                                      ratio=rr,
                                      verdict="PASS" if rr <= 1.20 else "FAIL")
        # 5. folded-wing pitch near the mask optimum, NOT at springref rest
        fp = m["pose"][f"wing_pitch_{folded}"]["median_deg"]
        v["5_folded_pitch"] = dict(
            wing=folded, treatment_deg=fp,
            control_deg=ctl["pose"][f"wing_pitch_{folded}"]["median_deg"],
            mask_optimum_deg=[-40.0, -20.0], springref_deg=springref_deg,
            verdict="PASS" if (-50.0 <= fp <= -10.0) else "FAIL")
        # invariants
        v["invariants"] = dict(
            max_abs_change_frozen_dofs=m["max_abs_change_frozen_dofs"],
            v12v13_len_drift={
                sd: abs(m["v12v13_len"][sd]["median"]
                        - ctl["v12v13_len"][sd]["median"])
                for sd in m["v12v13_len"]},
            verdict="PASS" if (m["max_abs_change_frozen_dofs"] == 0.0 and all(
                abs(m["v12v13_len"][sd]["median"]
                    - ctl["v12v13_len"][sd]["median"]) < 1e-9
                for sd in m["v12v13_len"])) else "FAIL")
        v["7_blade_normal_reported_not_gated"] = m["blade_normal_deg"]
        verdicts[a] = v
    res["acceptance"] = verdicts

    print(f"\n  ACCEPTANCE (gates are ratios to THIS run's own control)")
    print(f"{'arm':>10}{'1 song pitch':>16}{'2 song yaw':>13}{'3 penetration':>15}"
          f"{'4 residual':>13}{'5 folded pitch':>17}{'invariants':>13}")
    for a, v in verdicts.items():
        print(f"{a:>10}{v['1_song_pitch_dynamics']['verdict']:>16}"
              f"{v['2_song_unchanged']['verdict']:>13}"
              f"{v['3_penetration']['verdict']:>15}"
              f"{v['4_marker_residual']['verdict']:>13}"
              f"{v['5_folded_pitch']['verdict']:>17}"
              f"{v['invariants']['verdict']:>13}")
    return res


# ---------------------------------------------------------------------------
# FIGURES.  `--plot` only reads arms npz + the bout's stac_ik.h5; no fits run.
# ---------------------------------------------------------------------------

CONTROL_COLOR = "0.35"
ARM_COLORS = ("#3aa757", "#1f6fb4", "#b8459b", "#d1892b")   # green = fit
ARM_STYLES = ("-", "--", "-.", ":")
# Grouped so the two DOFs Stage D2 is allowed to move are the LAST two rows.
DOF_ROWS = ("wing_yaw_left", "wing_yaw_right", "wing_roll_left",
            "wing_roll_right", "wing_pitch_left", "wing_pitch_right")


def lp_std(x, fs, hp_hz=40.0):
    """Std of the content BELOW `hp_hz`, in the units of `x` -- the complement
    of `pulse_stats`' `hp_rms`, and the number that separates the two ways
    criterion 1 can move.

    Added per-frame mask noise raises `hp_rms` and leaves `lp_std` alone. A
    prior that smooths REAL dynamics away lowers `hp_rms` AND collapses
    `lp_std`. `hp_rms` on its own cannot tell those apart, so a criterion-1
    PASS is only believable next to an `lp_std` that survived.
    """
    from scipy.signal import butter, filtfilt
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 32:
        return float("nan")
    b, a = butter(2, hp_hz / (fs / 2.0), btype="low")
    return float(np.std(filtfilt(b, a, x - x.mean())))


def plot_model(bout_dir, fly):
    """The model these poses were solved against + wing qpos addresses BY NAME.

    Read-only on the bout: opens `stac_ik.h5` for its embedded config and
    nothing else. Marker offsets are NOT applied -- no site position is used
    here, only `qpos_spring` and the joint->qpos map.
    """
    import h5py
    import mujoco
    from omegaconf import OmegaConf
    from viz.config import resolve_body_model_xml
    with h5py.File(os.path.join(bout_dir, f"fly{fly}", "stac_ik.h5"), "r") as f:
        icfg = OmegaConf.create(f["config"][()].decode())
    mj = mujoco.MjModel.from_xml_path(resolve_body_model_xml(icfg.model.MJCF_PATH))
    return mj, joint_qposadr(mj, WING_DOFS)


def qpos_address_names(mj):
    """Name of the joint owning every qpos address (free/ball get a suffix).

    Used to LABEL the delta figure. A bar chart of 93 anonymous integers is
    exactly the kind of figure this repo's index bugs hid inside.
    """
    import mujoco
    names = [f"qpos[{i}]" for i in range(mj.nq)]
    for j in range(mj.njnt):
        a = int(mj.jnt_qposadr[j])
        n = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        w = {int(mujoco.mjtJoint.mjJNT_FREE): 7,
             int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(int(mj.jnt_type[j]), 1)
        for k in range(w):
            if a + k < mj.nq:
                names[a + k] = n if w == 1 else f"{n}[{k}]"
    return names


def load_plot_arms(specs, flies):
    """`--plot-arm LABEL=PATH:KEY` -> ({fly: control}, {label: {fly: qpos}}).

    `PATH` may contain `{fly}`. Every file's own `q_control` must agree, or the
    arms are not against the same control pose and laying the figures side by
    side would compare two different baselines -- that is a hard error, not a
    warning.
    """
    control, arms = {}, {}
    for spec in specs:
        lab, _, rest = spec.partition("=")
        path, _, key = rest.rpartition(":")
        if not (lab and path and key):
            raise SystemExit(f"--plot-arm wants LABEL=PATH:KEY, got {spec!r}")
        for fly in flies:
            p = path.format(fly=fly)
            with np.load(p) as z:
                if key not in z.files:
                    raise SystemExit(f"{p} has no {key!r}; it has {sorted(z.files)}")
                if "q_control" not in z.files:
                    raise SystemExit(f"{p} has no q_control")
                q = np.asarray(z[key], np.float32)
                c = np.asarray(z["q_control"], np.float32)
            prev = control.get(fly)
            if prev is not None and not np.array_equal(np.nan_to_num(prev),
                                                       np.nan_to_num(c)):
                raise SystemExit(
                    f"fly{fly}: {p}'s q_control differs from an earlier arm "
                    f"file's. These arms are NOT against the same control pose.")
            control[fly] = c
            arms.setdefault(lab, {})[fly] = q
            print(f"[plot] fly{fly} {lab:>10}: {key} <- {p} {q.shape}")
    return control, arms


def trace_stats(control, arms, mjadr, flies, t0, t1, fs):
    """Per fly / per pitch DOF: hp_rms + lp_std for control and every arm, the
    singer measured from the control pose, and max |delta| per DOF."""
    out = {}
    for fly in flies:
        mj, adr = mjadr[fly]
        singer, asym = singing_wing(control[fly][t0:t1], adr)
        f = {"singing_wing": singer, "song_absmean_deg": asym,
             "criterion_1_applies": bool(asym >= 10.0), "dofs": {}}
        for dof in WING_DOFS:
            a = adr[dof]
            c = np.degrees(control[fly][t0:t1, a])
            row = {"control": {"hp_rms": pulse_stats(c, fs)["hp_rms"],
                               "lp_std": lp_std(c, fs),
                               "median_deg": float(np.nanmedian(c))},
                   "springref_deg": float(np.degrees(mj.qpos_spring[a]))}
            for lab, per in arms.items():
                v = np.degrees(per[fly][t0:t1, a])
                fin = np.isfinite(v) & np.isfinite(c)
                row[lab] = {
                    "hp_rms": pulse_stats(v, fs)["hp_rms"],
                    "lp_std": lp_std(v, fs),
                    "median_deg": float(np.nanmedian(v)),
                    "max_abs_change_deg": float(np.abs(v[fin] - c[fin]).max()),
                    "hp_ratio": pulse_stats(v, fs)["hp_rms"]
                    / max(1e-12, row["control"]["hp_rms"]),
                    "lp_ratio": lp_std(v, fs) / max(1e-12, row["control"]["lp_std"]),
                }
            f["dofs"][dof] = row
        out[fly] = f
    return out


def _crit1_subtitle(stats, flies):
    """One line naming criterion 1's number for EVERY arm, with its verdict."""
    bits = []
    for fly in flies:
        f = stats[fly]
        if not f["criterion_1_applies"]:
            continue
        dof = f"wing_pitch_{f['singing_wing']}"
        r = f["dofs"][dof]
        parts = [f"criterion 1, fly{fly} {dof} (SINGING wing): "
                 f"control hp_rms {r['control']['hp_rms']:.3f}"]
        for lab in [k for k in r if k not in ("control", "springref_deg")]:
            v = r[lab]
            parts.append(f"{lab} {v['hp_rms']:.3f} ({v['hp_ratio']:.2f}x, "
                         f"{'PASS' if abs(v['hp_ratio'] - 1) <= 0.20 else 'FAIL'})")
        bits.append("  ->  ".join(parts))
    return "\n".join(bits) if bits else "criterion 1 N/A (no wing extension)"


def _wing_role(side, eps, st):
    """What this wing is DOING, per epoch -- never a single whole-window label.

    On this bout fly1 extends the LEFT wing to ~frame 869 and the RIGHT wing
    after it, so "the singing wing" is a different DOF in each half and any
    figure or score that carries one label for the whole window is wrong for a
    large part of it.
    """
    if eps:
        ext = [e for e in eps if e["extended"]]
        if ext:
            return "  ".join(
                (f"EXTENDED {e['t0']}-{e['t1']}" if e["wing"] == side
                 else f"folded {e['t0']}-{e['t1']}") for e in ext)
        return "folded throughout (no wing extension: not singing)"
    if st["criterion_1_applies"]:
        return "EXTENDED" if side == st["singing_wing"] else "folded"
    return "folded (no song)"


def _panel(ax, t, ctl, trt_list, rest, shade, lo, hi, t0, ctl_lw=0.9):
    s = slice(lo - t0, hi - t0)
    ax.plot(t[s], ctl[s], color=CONTROL_COLOR, lw=ctl_lw,
            label="control (qpos_refined.npz)", zorder=3)
    for i, (lab, v) in enumerate(trt_list):
        ax.plot(t[s], v[s], color=ARM_COLORS[i % len(ARM_COLORS)], lw=0.8,
                ls=ARM_STYLES[i % len(ARM_STYLES)] if len(trt_list) > 1 else "-",
                label=f"wing-mask fit ({lab})", zorder=4)
    ax.axhline(rest, color="crimson", ls=":", lw=1,
               label=f"springref rest {rest:.1f} deg")
    if shade:
        ax.axhspan(-40, -20, color="#3aa757", alpha=0.10,
                   label="measured mask optimum -20..-40 deg")
    # Headroom at the bottom so the per-row note never sits on the springref
    # line -- an unreadable number is the same as no number.
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo - 0.15 * (yhi - ylo), yhi)


_NOTE_BOX = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5)


def plot_pitch_traces(control, arms, mjadr, flies, t0, t1, zoom, fs,
                      out_dir, prefix, stats, epochs=None):
    """FIGURE A -- one PNG per arm, identical axes, so they lay side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(f, d) for f in sorted(flies, reverse=True) for d in PITCH]
    sub = _crit1_subtitle(stats, flies)
    paths = []
    for lab, per in arms.items():
        fig, axes = plt.subplots(len(rows), 2, figsize=(15, 2.8 * len(rows)),
                                 sharex="col", squeeze=False)
        for r, (fly, dof) in enumerate(rows):
            mj, adr = mjadr[fly]
            a = adr[dof]
            st = stats[fly]
            ctl = np.degrees(control[fly][t0:t1, a])
            trt = np.degrees(per[fly][t0:t1, a])
            rest = float(np.degrees(mj.qpos_spring[a]))
            t = np.arange(t0, t1)
            side = dof.rsplit("_", 1)[1]
            role = _wing_role(side, epochs[fly] if epochs else None, st)
            d = st["dofs"][dof]
            note = (f"hp_rms {d['control']['hp_rms']:.3f} -> {d[lab]['hp_rms']:.3f}"
                    f" ({d[lab]['hp_ratio']:.2f}x)   "
                    f"lp_std {d['control']['lp_std']:.2f} -> "
                    f"{d[lab]['lp_std']:.2f} deg ({d[lab]['lp_ratio']:.2f}x)")
            gated = st["criterion_1_applies"] and side == st["singing_wing"]
            if gated:
                ok = abs(d[lab]["hp_ratio"] - 1) <= 0.20
                note += f"   crit 1: {'PASS' if ok else 'FAIL'}"
                ncol = "#127a2e" if ok else "#c2261a"
            else:
                ncol = "0.2"
            for c, (lo, hi) in enumerate([(t0, t1), zoom]):
                ax = axes[r, c]
                _panel(ax, t, ctl, [(lab, trt)], rest, True, lo, hi, t0)
                ax.set_ylabel(f"fly{fly} {dof}\n[deg]", fontsize=8)
                ax.tick_params(labelsize=7)
                if r == 0:
                    ax.set_title(f"frames {t0}-{t1 - 1}" if c == 0
                                 else f"zoom {lo}-{hi}", fontsize=9)
                if r == 0 and c == 1:
                    ax.legend(fontsize=6, loc="lower right")
                if epochs and c == 0:
                    _draw_epochs(ax, epochs[fly], lo, hi)
                ax.text(0.01, 0.87, role, transform=ax.transAxes, fontsize=7,
                        va="top", color="0.2")
                if c == 0:      # left column only: the right one holds the legend
                    ax.text(0.99, 0.03, note, transform=ax.transAxes,
                            fontsize=7, va="bottom", ha="right", color=ncol,
                            fontweight="bold" if gated else "normal",
                            bbox=_NOTE_BOX)
        for ax in axes[-1]:
            ax.set_xlabel("frame", fontsize=8)
        fig.suptitle(f"{prefix} -- wing PITCH, control vs wing-mask fit "
                     f"[{lab}]\n{sub}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p = os.path.join(out_dir, f"pitch_traces_{lab}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print("wrote", p)
        paths.append(p)
    return paths


def plot_all_dofs(control, arms, mjadr, flies, t0, t1, zoom, out_dir, prefix,
                  stats, epochs=None):
    """FIGURE B -- all six wing DOFs, one PNG per fly, every arm on control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for fly in flies:
        mj, adr = mjadr[fly]
        st = stats[fly]
        fig, axes = plt.subplots(len(DOF_ROWS), 2, figsize=(15, 2.5 * len(DOF_ROWS)),
                                 sharex="col", squeeze=False)
        frozen_ok = True
        for r, dof in enumerate(DOF_ROWS):
            a = adr[dof]
            ctl = np.degrees(control[fly][t0:t1, a])
            trt = [(lab, np.degrees(per[fly][t0:t1, a])) for lab, per in arms.items()]
            rest = float(np.degrees(mj.qpos_spring[a]))
            t = np.arange(t0, t1)
            d = st["dofs"][dof]
            mx = {lab: d[lab]["max_abs_change_deg"] for lab, _ in trt}
            if "pitch" not in dof and any(v != 0.0 for v in mx.values()):
                frozen_ok = False
            note = "max |d| vs control:  " + "   ".join(
                f"{lab} {v:.6f} deg" for lab, v in mx.items())
            if "pitch" not in dof:
                # A frozen DOF's arm traces land exactly ON control, so ONE
                # apparent line is the CORRECT picture -- say so in the SAME
                # note, or the reader cannot tell it from a plotting bug.
                # (Control is drawn thick underneath, the thin arms on top.)
                note = ("all three traces coincide exactly -- the thin arm "
                        "traces are drawn ON TOP of the thick grey control\n"
                        + note)
            side = dof.rsplit("_", 1)[1]
            role = _wing_role(side, epochs[fly] if epochs else None, st)
            in_mask = "pitch" in dof
            tag = ("IN opt_mask -- Stage D2 may move this"
                   if in_mask else "FROZEN (not in opt_mask)")
            for c, (lo, hi) in enumerate([(t0, t1), zoom]):
                ax = axes[r, c]
                _panel(ax, t, ctl, trt, rest, in_mask, lo, hi, t0, ctl_lw=2.2)
                if epochs and c == 0:
                    _draw_epochs(ax, epochs[fly], lo, hi)
                ax.set_ylabel(f"{dof}\n[deg]", fontsize=8)
                ax.tick_params(labelsize=7)
                if r == 0:
                    ax.set_title(f"frames {t0}-{t1 - 1}" if c == 0
                                 else f"zoom {lo}-{hi}", fontsize=9)
                if r == 0 and c == 1:
                    ax.legend(fontsize=6, loc="lower right")
                ax.text(0.01, 0.87, f"{role}   [{tag}]", transform=ax.transAxes,
                        fontsize=7, va="top",
                        color="#c2261a" if in_mask else "0.2")
                if c == 0:
                    ax.text(0.99, 0.03, note, transform=ax.transAxes,
                            fontsize=7, va="bottom", ha="right", bbox=_NOTE_BOX,
                            color="0.2" if in_mask else "#127a2e")
        for ax in axes[-1]:
            ax.set_xlabel("frame", fontsize=8)
        # NEVER a whole-window "singer=" label: fly1 swaps extended wing at
        # ~869 on this bout, and the whole-window label is wrong after that.
        ext = [e for e in (epochs[fly] if epochs else []) if e["extended"]]
        who = ("male" if fly == 1 else "female") + (
            "  extends " + " then ".join(
                f"{e['wing'].upper()} {e['t0']}-{e['t1']}" for e in ext)
            if ext else "  no wing extension (not singing)")
        verdict = ("yaw and roll are EXACTLY unchanged (max |d| = 0.000000 deg, "
                   "bit-identical) -- only wing_pitch_* moves"
                   if frozen_ok else
                   "WARNING: a NON-PITCH wing DOF moved -- the opt_mask contract "
                   "is broken, see the per-row max |d|")
        fig.suptitle(f"{prefix} -- fly{fly} ({who}) all six wing DOFs, control "
                     f"vs wing-mask fit\n{verdict}", fontsize=10,
                     color="black" if frozen_ok else "#c2261a")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        p = os.path.join(out_dir, f"alldof_traces_fly{fly}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"wrote {p}   (non-pitch wing DOFs unchanged: {frozen_ok})")
        paths.append(p)
    return paths


def plot_delta_qpos(control, arms, mjadr, flies, t0, t1, out_dir, prefix):
    """FIGURE C -- `max |dqpos|` over every address, per arm. The "wing pitch
    only" claim as a spectrum: exactly two bars should be nonzero."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(flies), 1, figsize=(14, 3.4 * len(flies)),
                             squeeze=False)
    summary = {}
    for r, fly in enumerate(flies):
        ax = axes[r, 0]
        mj, adr = mjadr[fly]
        names = qpos_address_names(mj)
        nq = int(mj.nq)
        w = 0.8 / max(1, len(arms))
        nz_all, top = {}, 0.0
        # Every address gets a visible marker at its own value, so the 91 zeros
        # read as "measured and exactly zero" instead of as absent data.
        ax.plot(np.arange(nq), np.zeros(nq), ".", ms=3, color="0.55",
                label=f"all {nq} addresses (a dot at 0.0 = no change at all)")
        for i, (lab, per) in enumerate(arms.items()):
            c, q = control[fly][t0:t1], per[fly][t0:t1]
            fin = np.isfinite(c).all(1) & np.isfinite(q).all(1)
            dq = np.abs(q[fin] - c[fin]).max(axis=0)
            x = np.arange(nq) + (i - (len(arms) - 1) / 2) * w
            ax.bar(x, dq, width=w, color=ARM_COLORS[i % len(ARM_COLORS)],
                   label=f"{lab}  (nonzero at {int((dq > 0).sum())}/{nq} addresses)")
            top = max(top, float(dq.max()))
            nz = {names[k]: float(dq[k]) for k in np.nonzero(dq)[0]}
            nz_all[lab] = nz
            for k in np.nonzero(dq)[0]:
                # hinge addresses carry radians, so also print degrees; the
                # free-joint block (0-6) is mm + quaternion and is left raw.
                txt = (f"{names[k]}\n{dq[k]:.4f} rad = {np.degrees(dq[k]):.1f} deg"
                       if k >= 7 else f"{names[k]}\n{dq[k]:.6f}")
                ax.annotate(txt, (x[k], dq[k]),
                            xytext=(-70 + 140 * (k > adr["wing_pitch_left"]),
                                    14 + 26 * i),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=ARM_COLORS[i % len(ARM_COLORS)])
        ax.set_xlim(-1, nq)
        ax.set_ylim(0, top * 1.75 if top > 0 else 1.0)
        ax.set_xlabel("qpos address (0-2 root translation, 3-6 root quaternion, "
                      "7+ hinge joints; addresses resolved by joint NAME)",
                      fontsize=8)
        ax.set_ylabel(f"fly{fly}\nmax |d qpos| [native units]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        labels = sorted({n for nz in nz_all.values() for n in nz})
        ax.text(0.01, 0.95,
                f"moved: {', '.join(labels) if labels else 'nothing'}   |   "
                f"exactly 0.0 at the other {nq - len(labels)} addresses",
                transform=ax.transAxes, fontsize=8, va="top", color="#c2261a",
                bbox=_NOTE_BOX)
        summary[f"fly{fly}"] = {lab: nz for lab, nz in nz_all.items()}
    fig.suptitle(f"{prefix} -- max |d qpos| across all {int(mjadr[flies[0]][0].nq)}"
                 f" addresses, wing-mask fit vs control\n"
                 f"the 'wing PITCH only' guarantee: every other address is "
                 f"EXACTLY 0.0", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(out_dir, "delta_qpos_spectrum.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p, json.dumps(summary))
    return [p], summary


def plot_crit1_jitter(control, arms, mjadr, flies, t0, t1, jzoom, fs,
                      out_dir, prefix, stats):
    """FIGURE D -- criterion 1 close up: is the hp_rms rise TEXTURE or PLACEMENT?

    Criterion 1's `hp_rms` is a single scalar over 1500 frames, and the trace
    figure's 100-frame zoom happens to fall in a quiet stretch, so neither shows
    what the number is made of. This one does: a short window chosen INSIDE the
    fuzzy stretch, all arms overlaid, next to the distribution of the
    frame-to-frame step |dpitch| that `hp_rms` is summarising.

    EXPECTATION: the shipped arm's step distribution should have a long tail
    (per-frame excursions of tens of degrees) that neither control nor the
    smooth_weight=30 arm has, while all three arms sit at the SAME median pitch
    -- i.e. the failure is texture, not placement. If instead the smooth-30
    trace is displaced from the shipped one, the temporal term moved the wing
    rather than steadying it.

    The DOF shown is the criterion-1 DOF where criterion 1 applies (the singing
    wing), else the pitch DOF with the worst shipped hp_rms ratio -- the hard
    case, never the flattering one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = jzoom
    rows = []
    for fly in sorted(flies, reverse=True):
        st = stats[fly]
        if st["criterion_1_applies"]:
            dof, why = f"wing_pitch_{st['singing_wing']}", "criterion-1 DOF (singing wing)"
        else:
            first = next(iter(arms))
            dof = max(PITCH, key=lambda d: st["dofs"][d][first]["hp_ratio"])
            why = "worst-hp_rms pitch DOF (criterion 1 N/A: not singing)"
        rows.append((fly, dof, why))
    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 3.4 * len(rows)),
                             squeeze=False,
                             gridspec_kw=dict(width_ratios=[2.2, 1.0]))
    for r, (fly, dof, why) in enumerate(rows):
        mj, adr = mjadr[fly]
        a = adr[dof]
        st = stats[fly]
        series = [("control", np.degrees(control[fly][t0:t1, a]), CONTROL_COLOR)]
        for i, (lab, per) in enumerate(arms.items()):
            series.append((lab, np.degrees(per[fly][t0:t1, a]),
                           ARM_COLORS[i % len(ARM_COLORS)]))
        t = np.arange(t0, t1)
        ax = axes[r, 0]
        sl = slice(lo - t0, hi - t0)
        for lab, v, col in series:
            ax.plot(t[sl], v[sl], color=col,
                    lw=2.0 if lab == "control" else 1.0,
                    label=f"{lab}  (median {np.nanmedian(v):.1f} deg)")
        ax.axhspan(-40, -20, color="#3aa757", alpha=0.10,
                   label="measured mask optimum -20..-40 deg")
        ax.set_xlabel("frame", fontsize=8)
        ax.set_ylabel(f"fly{fly} {dof}\n[deg]", fontsize=8)
        ax.set_title(f"fly{fly} {dof} -- {why};  frames {lo}-{hi}", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.tick_params(labelsize=7)

        ax = axes[r, 1]
        bins = np.logspace(-3, 2, 60)
        txt = []
        for lab, v, col in series:
            d = np.abs(np.diff(v))
            d = d[np.isfinite(d)]
            ax.hist(d, bins=bins, histtype="step", color=col,
                    lw=2.0 if lab == "control" else 1.2, label=lab)
            txt.append(f"{lab:>16}  median {np.median(d):6.3f}  "
                       f"p99 {np.percentile(d, 99):7.2f}  max {d.max():6.1f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("frame-to-frame |d pitch| [deg]  (what hp_rms summarises)",
                      fontsize=8)
        ax.set_ylabel("frames", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)
        # headroom above the histograms so the numbers never sit on a curve
        ax.set_ylim(top=ax.get_ylim()[1] * 40)
        ax.text(0.98, 0.97, "\n".join(txt), transform=ax.transAxes, fontsize=6.5,
                family="monospace", va="top", ha="right", bbox=_NOTE_BOX)
    fig.suptitle(f"{prefix} -- criterion 1 close up: the hp_rms rise is TEXTURE, "
                 f"not placement\n{_crit1_subtitle(stats, flies)}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(out_dir, "crit1_jitter.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)
    return [p]


# ---------------------------------------------------------------------------
# SONG vs JITTER.  Criterion 1's `hp_rms` is a scalar power measure and cannot
# tell "the fit added broadband noise" from "song energy moved into this DOF".
# Everything below separates them, and scores PER EXTENDED-WING EPOCH, because
# fly1 swaps which wing it extends part-way through this bout and a whole-window
# "singing wing = left" label is wrong for the rest of it.
# ---------------------------------------------------------------------------


def hp_filt(x, fs, hp_hz=40.0):
    """The same high-pass `pulse_stats` uses, returned as a signal."""
    from scipy.signal import butter, filtfilt
    b, a = butter(2, hp_hz / (fs / 2.0), btype="high")
    return filtfilt(b, a, np.asarray(x, float) - np.nanmean(x))


def singer_epochs(q, adr, t0, t1, win=101, min_len=200, min_asym_deg=10.0):
    """Which wing is EXTENDED over time -> epochs, MEASURED not assumed.

    `wing_yaw_*`'s springref sits AT the joint's upper stop (+85.94 deg), so the
    FOLDED wing reads near the stop and the extended wing is the smaller yaw.
    A `win`-frame majority filter turns the per-frame comparison into epochs.

    This exists because `singing_wing()` returns ONE label for the whole window,
    and on this bout that label is wrong for roughly 40% of it: fly1 sings with
    the left wing, pauses, and resumes on the RIGHT. Every criterion-1 number
    computed against the whole-window label mixes a singing epoch with a folded
    one, which is how a per-DOF guard can pass on a DOF whose song was destroyed.
    """
    from scipy.ndimage import uniform_filter1d
    yl = np.degrees(q[t0:t1, adr["wing_yaw_left"]])
    yr = np.degrees(q[t0:t1, adr["wing_yaw_right"]])
    left_ext = (yl < yr).astype(float)
    maj = uniform_filter1d(left_ext, win, mode="nearest") > 0.5
    edges = [0, *(np.nonzero(np.diff(maj.astype(int)))[0] + 1).tolist(), len(maj)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < min_len:
            continue
        asym = float(np.mean(np.abs(yl[a:b] - yr[a:b])))
        out.append(dict(t0=t0 + a, t1=t0 + b,
                        wing="left" if maj[a] else "right",
                        song_absmean_deg=asym,
                        extended=bool(asym >= min_asym_deg)))
    return out


def song_stats(control, arms, mjadr, cases, fs, hp_hz=40.0):
    """Per case (fly, epoch, wing), per arm: is the high-frequency content on
    `wing_pitch_<wing>` SONG or NOISE?

    Five numbers, each of which fails differently:

      P(f0)/floor  song-band power over the broadband floor of the SAME spectrum.
                   `hp_rms` is P(f0) + floor summed; this splits them, and it is
                   the number that shows a fit keeping `hp_rms` constant while
                   replacing song with noise.
      MSC@f0       magnitude-squared coherence with the SAME wing's `wing_yaw`,
                   at the yaw's own dominant frequency. Real song-driven pitch is
                   phase-locked to the wing's motion; mask noise is not. Yaw is
                   bit-identical across arms, so this is the same reference
                   signal for every arm. Reported against its own 95%
                   significance level, which depends on the segment count.
      xcorr        normalised cross-correlation of hp(pitch) with hp(yaw) and its
                   lag, so a phase flip or a delay is visible as well as a level.
      corr_ctl     correlation of hp(pitch) with the CONTROL's hp(pitch): how
                   much of the pose the fit started from survived, independent of
                   how much was added.
      pulses       peak count at a COMMON threshold (2x the control's own hp std,
                   for every arm), plus how many control pulses have a treatment
                   pulse within +-2 frames, and the median offset -- the literal
                   "are the pulses preserved, later, or gone" question. The
                   chance match rate for that window is reported next to it,
                   because an arm with many pulses matches everything by luck.
    """
    from scipy.signal import welch, coherence, find_peaks
    out = {}
    for key, fly, a0, a1, wing, label in cases:
        mj, adr = mjadr[fly]
        s = slice(a0, a1)
        n = a1 - a0
        # Segment length: coherence is a RATIO of averaged spectra, so it is
        # biased hard upward when there are few segments. nperseg=256 on a
        # 631-frame epoch gives K~3 and a 95% significance floor of 0.78, which
        # is useless. 128 gives K~8. Both were checked; the verdict is the same.
        nper = int(min(128, max(64, (n // 6) * 2)))
        y = np.degrees(control[fly][s, adr[f"wing_yaw_{wing}"]])
        hy = hp_filt(y, fs, hp_hz)
        fy, Py = welch(y - y.mean(), fs=fs, nperseg=nper)
        band = fy > hp_hz
        f0 = float(fy[band][np.argmax(Py[band])])
        K = max(2, int(2 * n / nper) - 1)
        c = {"fly": fly, "t0": a0, "t1": a1, "wing": wing, "label": label,
             "f0_hz": f0, "f0_period_frames": fs / f0, "n_frames": n,
             "nperseg": nper,
             "msc_sig95": 1 - 0.05 ** (1 / (K - 1)), "n_segments": K,
             "song_absmean_deg": float(np.mean(np.abs(
                 np.degrees(control[fly][s, adr["wing_yaw_left"]])
                 - np.degrees(control[fly][s, adr["wing_yaw_right"]])))),
             "arms": {}}
        hc = hp_filt(np.degrees(control[fly][s, adr[f"wing_pitch_{wing}"]]), fs, hp_hz)
        thr = 2.0 * float(np.std(hc))
        pk_c, _ = find_peaks(np.abs(hc), height=thr)
        for lab, q in [("control", control[fly])] + [(k, v[fly]) for k, v in arms.items()]:
            v = hp_filt(np.degrees(q[s, adr[f"wing_pitch_{wing}"]]), fs, hp_hz)
            f_, P = welch(v, fs=fs, nperseg=nper)
            i0 = int(np.argmin(np.abs(f_ - f0)))
            side = (np.abs(f_ - f0) > 2 * (f_[1] - f_[0])) & (f_ > hp_hz)
            floor = float(np.median(P[side]))
            fc, C = coherence(v, hy, fs=fs, nperseg=nper)
            j0 = int(np.argmin(np.abs(fc - f0)))
            a_ = (v - v.mean()) / (v.std() + 1e-12)
            b_ = (hy - hy.mean()) / (hy.std() + 1e-12)
            cc = np.correlate(a_, b_, "full") / len(a_)
            lags = np.arange(-len(a_) + 1, len(a_))
            m = np.abs(lags) <= 20
            k = int(np.argmax(np.abs(cc[m])))
            pk, _ = find_peaks(np.abs(v), height=thr)
            d = np.array([(pk - ci)[np.argmin(np.abs(pk - ci))] for ci in pk_c]) \
                if (len(pk) and len(pk_c)) else np.array([])
            matched = int((np.abs(d) <= 2).sum()) if d.size else 0
            c["arms"][lab] = dict(
                hp_rms=float(np.std(v)),
                P_f0=float(P[i0]), floor=floor,
                peak_over_floor=float(P[i0] / max(floor, 1e-30)),
                msc_at_f0=float(C[j0]),
                xcorr_max=float(cc[m][k]), xcorr_lag=int(lags[m][k]),
                corr_with_control_hp=float(np.corrcoef(hc, v)[0, 1]),
                n_pulses_common_thr=int(len(pk)),
                n_control_pulses=int(len(pk_c)),
                matched_within_2fr=matched,
                chance_match_rate=float(min(1.0, 5.0 * len(pk) / max(1, n))),
                median_offset_fr=float(np.median(d)) if d.size else float("nan"),
                pulse_times=(pk + a0).tolist(),
                median_abs_amp_deg=(float(np.median(np.abs(v[pk])))
                                    if len(pk) else float("nan")))
        out[key] = c
    return out


def build_cases(control, mjadr, flies, t0, t1):
    """(key, fly, a0, a1, wing, label) per extended-wing epoch, both flies."""
    cases = []
    for fly in sorted(flies, reverse=True):
        mj, adr = mjadr[fly]
        eps = singer_epochs(control[fly], adr, t0, t1)
        ext = [e for e in eps if e["extended"]]
        if not ext:
            yl = np.nanmean(control[fly][t0:t1, adr["wing_yaw_left"]])
            yr = np.nanmean(control[fly][t0:t1, adr["wing_yaw_right"]])
            w = "left" if yl < yr else "right"
            cases.append((f"fly{fly}_nosong", fly, t0, t1, w,
                          f"fly{fly} NOT singing (negative control)"))
            continue
        for i, e in enumerate(ext):
            cases.append((f"fly{fly}_E{i + 1}", fly, e["t0"], e["t1"], e["wing"],
                          f"fly{fly} epoch {i + 1}: {e['wing']} wing extended "
                          f"(|yawL-yawR| {e['song_absmean_deg']:.1f} deg)"))
    return cases


def epoch_marks(control, mjadr, fly, t0, t1):
    """Frames where the extended wing swaps, for drawing on a time axis."""
    mj, adr = mjadr[fly]
    eps = singer_epochs(control[fly], adr, t0, t1)
    return [(e["t0"], e["wing"], e["song_absmean_deg"]) for e in eps if e["t0"] > t0], eps


def _draw_epochs(ax, eps, t0, t1, fontsize=6.5):
    """Mark the extended-wing epochs on a time axis, without ever widening it:
    a label placed outside [t0, t1] would silently rescale the panel and break
    the alignment between the zoom panels of different figures."""
    for e in eps:
        if t0 < e["t0"] < t1:
            ax.axvline(e["t0"], color="#7a3fbf", ls="--", lw=1.2, zorder=6)
        lo, hi = max(e["t0"], t0), min(e["t1"], t1)
        if hi - lo < 0.08 * (t1 - t0):
            continue
        ax.text((lo + hi) / 2, 0.985,
                f"{e['wing'][0].upper()} extended\n|L-R| {e['song_absmean_deg']:.0f} deg"
                + ("" if e["extended"] else "\n(no extension)"),
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=fontsize, color="#7a3fbf")


def plot_song(control, arms, mjadr, cases, st, fs, out_dir, prefix, hp_hz=40.0):
    """FIGURE E -- is the high-frequency pitch content SONG or NOISE?

    THE TWO HYPOTHESES, and what each one looks like here. Written before the
    figure was drawn; the numbers behind it are in `song_stats.json`.

      "SONG MOVED INTO PITCH": the refined wing sits at a new orientation, so
      motion that used to project onto yaw now projects onto pitch. Signature:
      the treatment's spectrum keeps (or grows) the SHARP peak at the yaw's own
      frequency f0, `peak/floor` holds up, coherence with that wing's yaw at f0
      stays high or RISES, and the control's pulses survive at their original
      times. `hp_rms` rising is then a correct measurement of real signal and
      criterion 1 is the thing that is wrong.

      "BROADBAND NOISE ADDED": the per-frame mask minimum is broad, so the fit
      wanders. Signature: the floor rises much more than the peak, `peak/floor`
      falls, coherence with yaw FALLS, and extra threshold crossings appear
      between the control's pulses.

    The female is the negative control: she does not sing, so whatever the fit
    does to HER pitch spectrum is what the fit does in the ABSENCE of song.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch, coherence, find_peaks
    rows = len(cases)
    fig, axes = plt.subplots(rows, 3, figsize=(17, 3.3 * rows), squeeze=False,
                             gridspec_kw=dict(width_ratios=[2.0, 1.0, 1.0]))
    for r, (key, fly, a0, a1, wing, label) in enumerate(cases):
        mj, adr = mjadr[fly]
        c = st[key]
        s = slice(a0, a1)
        f0 = c["f0_hz"]
        nper = int(c["nperseg"])
        hy = hp_filt(np.degrees(control[fly][s, adr[f"wing_yaw_{wing}"]]), fs, hp_hz)
        series = [("control", control[fly], CONTROL_COLOR)]
        for i, (lab, per) in enumerate(arms.items()):
            series.append((lab, per[fly], ARM_COLORS[i % len(ARM_COLORS)]))
        # --- col 0: the pulse train itself, hp(pitch), arms stacked ---------
        ax = axes[r, 0]
        w0 = a0 + (a1 - a0) // 2
        w1 = min(a1, w0 + 120)
        off = 0.0
        hc = hp_filt(np.degrees(control[fly][s, adr[f"wing_pitch_{wing}"]]), fs, hp_hz)
        thr = 2.0 * float(np.std(hc))
        step = 6.0 * float(np.std(hc))
        for lab, q, col in series:
            v = hp_filt(np.degrees(q[s, adr[f"wing_pitch_{wing}"]]), fs, hp_hz)
            t = np.arange(a0, a1)
            m = (t >= w0) & (t < w1)
            ax.plot(t[m], v[m] + off, color=col, lw=1.1, label=f"hp({lab})")
            ax.axhline(off, color=col, lw=0.4, alpha=0.4)
            pk, _ = find_peaks(np.abs(v), height=thr)
            pk = pk[(pk + a0 >= w0) & (pk + a0 < w1)]
            ax.plot(pk + a0, np.full(len(pk), off + 2.6 * float(np.std(hc))),
                    "v", ms=4, color=col)
            off -= step
        # the SAME wing's yaw, scaled, as the phase reference
        t = np.arange(a0, a1)
        m = (t >= w0) & (t < w1)
        ax.plot(t[m], hy[m] / (np.std(hy) + 1e-12) * float(np.std(hc)) + off,
                color="#7a3fbf", lw=1.0, ls="--",
                label=f"hp(wing_yaw_{wing}) [scaled] -- the phase reference")
        ax.set_title(f"{label}\nhp(wing_pitch_{wing}) pulse train, frames "
                     f"{w0}-{w1} (markers = |hp| > 2x control sigma)", fontsize=8)
        ax.set_xlabel("frame", fontsize=8)
        ax.set_yticks([])
        ax.legend(fontsize=6, loc="lower left", ncol=2)
        ax.tick_params(labelsize=7)
        # --- col 1: the spectrum -- peak vs floor ---------------------------
        ax = axes[r, 1]
        for lab, q, col in series:
            v = hp_filt(np.degrees(q[s, adr[f"wing_pitch_{wing}"]]), fs, hp_hz)
            f_, P = welch(v, fs=fs, nperseg=nper)
            a = c["arms"][lab]
            ax.loglog(f_, P, color=col, lw=1.2,
                      label=f"{lab}: peak/floor {a['peak_over_floor']:.0f}")
        ax.axvline(f0, color="#7a3fbf", ls="--", lw=1.2)
        ax.text(f0, 0.02, f" f0 = {f0:.0f} Hz\n (yaw's own, {fs / f0:.1f} frames)",
                transform=ax.get_xaxis_transform(), fontsize=6.5, color="#7a3fbf")
        ax.set_title(f"PSD of hp(wing_pitch_{wing})", fontsize=8)
        ax.set_xlabel(f"Hz at the assumed fs={fs:g} (frame rate UNVERIFIED)",
                      fontsize=7)
        ax.set_ylabel("power [deg^2/Hz]", fontsize=7)
        ax.legend(fontsize=6, loc="lower left")
        ax.tick_params(labelsize=7)
        # --- col 2: coherence with the same wing's yaw ----------------------
        ax = axes[r, 2]
        for lab, q, col in series:
            v = hp_filt(np.degrees(q[s, adr[f"wing_pitch_{wing}"]]), fs, hp_hz)
            fc, C = coherence(v, hy, fs=fs, nperseg=nper)
            ax.semilogx(fc, C, color=col, lw=1.2,
                        label=f"{lab}: MSC@f0 {c['arms'][lab]['msc_at_f0']:.2f}")
        ax.axvline(f0, color="#7a3fbf", ls="--", lw=1.2)
        ax.axhline(c["msc_sig95"], color="crimson", ls=":", lw=1.2,
                   label=f"95% significance {c['msc_sig95']:.2f}")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"coherence  hp(pitch_{wing})  vs  hp(yaw_{wing})\n"
                     f"(yaw is bit-identical across arms)", fontsize=8)
        ax.set_xlabel("Hz", fontsize=7)
        ax.set_ylabel("magnitude-squared coherence", fontsize=7)
        ax.legend(fontsize=6, loc="lower left")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{prefix} -- SONG or NOISE? the high-frequency content of "
                 f"wing PITCH, per extended-wing epoch\n"
                 f"song = a sharp peak at the yaw's own f0 + coherence with that "
                 f"yaw;  noise = a raised broadband floor + coherence falling",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out_dir, "song_or_noise.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)
    return [p]


def plot_pulse_trains(control, arms, mjadr, cases, st, fs, t0, t1, out_dir,
                      prefix):
    """FIGURE F -- WHERE are the pulses? Every detected pulse, full window.

    EXPECTATION: if the fit preserves the song, the treatment's tick rows line
    up with the control's, one tick per control tick, no systematic shift. Ticks
    appearing BETWEEN the control's are added crossings (noise); control ticks
    with no treatment tick under them are pulses the fit smoothed away; a
    constant horizontal offset would be the delay the user asked about.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = len(cases)
    fig, axes = plt.subplots(rows, 1, figsize=(16, 2.5 * rows), squeeze=False)
    for r, (key, fly, a0, a1, wing, label) in enumerate(cases):
        ax = axes[r, 0]
        c = st[key]
        names = ["control"] + list(arms)
        for i, lab in enumerate(names):
            a = c["arms"][lab]
            col = CONTROL_COLOR if lab == "control" else ARM_COLORS[(i - 1) % len(ARM_COLORS)]
            pt = np.asarray(a["pulse_times"], float)
            ax.vlines(pt, -i - 0.4, -i + 0.4, color=col, lw=1.0)
            note = (f"{lab}: {a['n_pulses_common_thr']} pulses")
            if lab != "control":
                note += (f";  {a['matched_within_2fr']}/{a['n_control_pulses']} "
                         f"control pulses matched within +-2 fr "
                         f"(chance {a['chance_match_rate'] * 100:.0f}%);  "
                         f"median offset {a['median_offset_fr']:+.1f} fr")
            ax.text(a0 + 3, -i + 0.44, note, fontsize=7, color=col,
                    va="bottom", bbox=_NOTE_BOX)
        ax.set_ylim(-len(names) + 0.3, 1.1)
        ax.set_xlim(t0, t1)
        ax.set_yticks([])
        ax.set_xlabel("frame", fontsize=8)
        ax.set_title(f"{label}  --  hp(wing_pitch_{wing}) pulses at ONE common "
                     f"threshold (2x the control's own sigma), frames {a0}-{a1}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axvspan(t0, a0, color="0.9")
        ax.axvspan(a1, t1, color="0.9")
    fig.suptitle(f"{prefix} -- are the song pulses preserved, delayed, or "
                 f"smoothed over?\ncommon threshold for every arm, so the counts "
                 f"are comparable; grey = outside this epoch", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, "pulse_trains.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)
    return [p]


def report_epochs(st, cases, arms):
    """Criterion 1, recomputed PER EPOCH, next to the song/noise numbers."""
    print("\n===== criterion 1 PER EXTENDED-WING EPOCH, with what hp_rms is "
          "made of =====")
    print(f"{'case':>13}{'wing':>7}{'frames':>12}{'|L-R|':>7}{'arm':>10}"
          f"{'hp_rms':>9}{'ratio':>7}{'P(f0)':>11}{'floor':>11}"
          f"{'pk/floor':>10}{'MSC@f0':>8}{'corr_ctl':>10}{'pulses':>8}")
    for key, fly, a0, a1, wing, label in cases:
        c = st[key]
        base = c["arms"]["control"]["hp_rms"]
        for lab in ["control"] + list(arms):
            a = c["arms"][lab]
            print(f"{key:>13}{wing:>7}{f'{a0}-{a1}':>12}"
                  f"{c['song_absmean_deg']:>7.1f}{lab:>10}"
                  f"{a['hp_rms']:>9.4f}{a['hp_rms'] / max(base, 1e-12):>7.2f}"
                  f"{a['P_f0']:>11.3e}{a['floor']:>11.3e}"
                  f"{a['peak_over_floor']:>10.1f}{a['msc_at_f0']:>8.3f}"
                  f"{a['corr_with_control_hp']:>10.3f}"
                  f"{a['n_pulses_common_thr']:>8d}")
        print(f"{'':>13}(95% MSC significance {c['msc_sig95']:.2f}; f0 "
              f"{c['f0_hz']:.0f} Hz = {c['f0_period_frames']:.1f} frames; "
              f"pulse threshold = 2x control sigma, common to every arm)")


def bilateral_stats(control, arms, mjadr, cases, fs, hp_hz=40.0):
    """THE DECISIVE TEST: are the LEFT and RIGHT wing pitch oscillations
    phase-locked to each other?

    Bilateral motor drive means song-driven wing motion is phase-locked ACROSS
    wings -- both wings are driven during song, the non-extended one merely
    pinned against extension by its yaw stop. Independent per-frame mask noise
    on two separately-optimised DOFs cannot become phase-locked by accident.
    So this separates the two readings of criterion 1's `hp_rms` rise in a way
    `hp_rms` itself never can:

      the treatment's high-frequency pitch is RECOVERED SONG  -> its L/R
        coherence at the song frequency is high, and at least as high as the
        control's, with a stable phase.
      the treatment's high-frequency pitch is ADDED NOISE     -> its L/R
        coherence collapses toward the significance floor and its phase
        scatters, even if `hp_rms` is unchanged or larger.

    PHASE CONVENTION, and it matters for reading the number: `wing_pitch_left`
    and `wing_pitch_right` are both hinges about the local +y of their own wing
    body, and the two wing bodies are mirror-placed (body quats
    [0,-0.4031,0,-0.9152] and [0,0.9152,0,-0.4031], both 180 deg rotations about
    perpendicular axes in the x-z plane, each mapping local +y onto world -y).
    A bilaterally SYMMETRIC motion therefore appears at ~180 deg in these joint
    coordinates, not at 0. Report the measured phase and read +-180 as "in phase
    in the animal".
    """
    from scipy.signal import coherence, csd, welch
    out = {}
    for key, fly, a0, a1, wing, label in cases:
        mj, adr = mjadr[fly]
        s = slice(a0, a1)
        n = a1 - a0
        nper = int(min(128, max(64, (n // 6) * 2)))
        y = np.degrees(control[fly][s, adr[f"wing_yaw_{wing}"]])
        fy, Py = welch(y - y.mean(), fs=fs, nperseg=nper)
        band = fy > hp_hz
        f0 = float(fy[band][np.argmax(Py[band])])
        K = max(2, int(2 * n / nper) - 1)
        c = {"f0_hz": f0, "f0_period_frames": fs / f0,
             "nperseg": nper, "n_segments": K,
             "msc_sig95": 1 - 0.05 ** (1 / (K - 1)), "arms": {}}
        for lab, q in [("control", control[fly])] + [(k, v[fly]) for k, v in arms.items()]:
            L = hp_filt(np.degrees(q[s, adr["wing_pitch_left"]]), fs, hp_hz)
            R = hp_filt(np.degrees(q[s, adr["wing_pitch_right"]]), fs, hp_hz)
            fc, C = coherence(L, R, fs=fs, nperseg=nper)
            j = int(np.argmin(np.abs(fc - f0)))
            fx, Pxy = csd(L, R, fs=fs, nperseg=nper)
            ph = float(np.degrees(np.angle(Pxy[int(np.argmin(np.abs(fx - f0)))])))
            a_ = (L - L.mean()) / (L.std() + 1e-12)
            b_ = (R - R.mean()) / (R.std() + 1e-12)
            cc = np.correlate(a_, b_, "full") / len(a_)
            lags = np.arange(-len(a_) + 1, len(a_))
            m = np.abs(lags) <= 20
            k = int(np.argmax(np.abs(cc[m])))
            c["arms"][lab] = dict(msc_LR_at_f0=float(C[j]),
                                  msc_LR_max=float(C[fc > hp_hz].max()),
                                  phase_LR_deg=ph,
                                  xcorr_LR=float(cc[m][k]),
                                  xcorr_lag_fr=int(lags[m][k]),
                                  xcorr_curve=cc[m].tolist(),
                                  xcorr_lags=lags[m].tolist(),
                                  hp_rms_left=float(L.std()),
                                  hp_rms_right=float(R.std()))
        out[key] = c
    return out


def plot_bilateral(control, arms, mjadr, cases, bst, st, fs, out_dir, prefix,
                   hp_hz=40.0):
    """FIGURE E -- the decisive test, drawn: are the two wings phase-locked?

    EXPECTATION, written before drawing. In the left column the two wings'
    high-passed pitch are overlaid on ONE axis per arm, over a song window. If
    the fly is singing bilaterally, the two traces should rise and fall together
    on a fixed phase relation (near mirror-image, i.e. ~180 deg in these joint
    coordinates) for whichever arm carries the song. If an arm's high-frequency
    pitch is independent per-frame mask noise, its two traces will wander with
    no fixed relation, its coherence curve (middle) will sit at the significance
    floor, and its cross-correlation (right) will be a flat, low, non-periodic
    curve instead of an oscillation at the song period.

    Whichever arm shows the lock is the arm that carries the song. That is the
    whole question, and it is decided by looking at three panels, not by the
    sign of an hp_rms ratio.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import coherence
    rows = len(cases)
    fig, axes = plt.subplots(rows, 3, figsize=(17, 3.6 * rows), squeeze=False,
                             gridspec_kw=dict(width_ratios=[2.1, 1.0, 1.0]))
    for r, (key, fly, a0, a1, wing, label) in enumerate(cases):
        mj, adr = mjadr[fly]
        b = bst[key]
        f0 = b["f0_hz"]
        s = slice(a0, a1)
        series = [("control", control[fly], CONTROL_COLOR)]
        for i, (lab, per) in enumerate(arms.items()):
            series.append((lab, per[fly], ARM_COLORS[i % len(ARM_COLORS)]))
        # --- col 0: L and R pitch overlaid, one block per arm ---------------
        ax = axes[r, 0]
        w0 = a0 + (a1 - a0) // 2
        w1 = min(a1, w0 + 90)
        hcL = hp_filt(np.degrees(control[fly][s, adr["wing_pitch_left"]]), fs, hp_hz)
        step = 11.0 * float(np.std(hcL))
        t = np.arange(a0, a1)
        m = (t >= w0) & (t < w1)
        off = 0.0
        for lab, q, col in series:
            L = hp_filt(np.degrees(q[s, adr["wing_pitch_left"]]), fs, hp_hz)
            R = hp_filt(np.degrees(q[s, adr["wing_pitch_right"]]), fs, hp_hz)
            ax.plot(t[m], L[m] + off, color=col, lw=1.3)
            ax.plot(t[m], R[m] + off, color=col, lw=1.1, ls="--")
            a = b["arms"][lab]
            ax.text(w0 + 1, off + 3.4 * float(np.std(hcL)),
                    f"{lab}:  MSC(L,R)@f0 {a['msc_LR_at_f0']:.3f}"
                    f"  (95% sig {b['msc_sig95']:.2f})"
                    f"   phase {a['phase_LR_deg']:+.0f} deg"
                    f"   xcorr {a['xcorr_LR']:+.2f} @ lag {a['xcorr_lag_fr']:+d} fr",
                    fontsize=7, color=col, va="bottom", bbox=_NOTE_BOX)
            ax.axhline(off, color=col, lw=0.4, alpha=0.4)
            off -= step
        ax.set_title(f"{label}\nhp(wing_pitch_LEFT) solid vs hp(wing_pitch_RIGHT) "
                     f"dashed, SAME axis per arm; frames {w0}-{w1}", fontsize=8)
        ax.set_xlabel("frame", fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + 0.10 * (yhi - ylo))
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color="0.2", lw=1.6,
                                  label="wing_pitch_LEFT"),
                           Line2D([], [], color="0.2", lw=1.3, ls="--",
                                  label="wing_pitch_RIGHT")],
                  fontsize=7, loc="lower right")
        # --- col 1: L/R coherence spectrum -----------------------------------
        ax = axes[r, 1]
        for lab, q, col in series:
            L = hp_filt(np.degrees(q[s, adr["wing_pitch_left"]]), fs, hp_hz)
            R = hp_filt(np.degrees(q[s, adr["wing_pitch_right"]]), fs, hp_hz)
            fc, C = coherence(L, R, fs=fs, nperseg=int(b["nperseg"]))
            ax.semilogx(fc, C, color=col, lw=1.3,
                        label=f"{lab}: {b['arms'][lab]['msc_LR_at_f0']:.3f}")
        ax.axvline(f0, color="#7a3fbf", ls="--", lw=1.2)
        ax.axhline(b["msc_sig95"], color="crimson", ls=":", lw=1.3,
                   label=f"95% significance {b['msc_sig95']:.2f}")
        ax.text(f0, 0.02, f" f0 {f0:.0f} Hz ({fs / f0:.1f} fr)",
                transform=ax.get_xaxis_transform(), fontsize=6.5, color="#7a3fbf")
        ax.set_ylim(0, 1.05)
        ax.set_title("coherence  hp(pitch_LEFT) vs hp(pitch_RIGHT)\n"
                     "bilateral phase lock -- noise cannot fake this", fontsize=8)
        ax.set_xlabel("Hz", fontsize=7)
        ax.set_ylabel("magnitude-squared coherence", fontsize=7)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.tick_params(labelsize=7)
        # --- col 2: cross-correlation vs lag ---------------------------------
        ax = axes[r, 2]
        for lab, q, col in series:
            a = b["arms"][lab]
            ax.plot(a["xcorr_lags"], a["xcorr_curve"], color=col, lw=1.3,
                    label=f"{lab}: {a['xcorr_LR']:+.2f} @ {a['xcorr_lag_fr']:+d} fr")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvline(0, color="0.7", lw=0.8)
        for kk in (-2, -1, 1, 2):
            ax.axvline(kk * fs / f0, color="#7a3fbf", ls=":", lw=0.8)
        ax.set_title(f"cross-correlation L vs R pitch\n(dotted = the song period, "
                     f"{fs / f0:.1f} frames)", fontsize=8)
        ax.set_xlabel("lag [frames]", fontsize=7)
        ax.set_ylabel("normalised cross-correlation", fontsize=7)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{prefix} -- BILATERAL PHASE LOCK: are BOTH wings oscillating "
                 f"together?\nboth wings are driven during song (the folded one "
                 f"is only pinned against extension by its yaw stop), so "
                 f"song-driven pitch must be phase-locked ACROSS wings; "
                 f"independent per-frame mask noise cannot be",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(out_dir, "bilateral_phase_lock.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)
    return [p]


def report_bilateral(bst, st, cases, arms):
    print("\n===== BILATERAL PHASE LOCK (the decisive test) + folded-wing "
          "oscillation =====")
    print(f"{'case':>13}{'arm':>10}{'MSC(L,R)@f0':>13}{'sig95':>8}{'phase':>9}"
          f"{'xcorr':>8}{'lag':>6}{'hp_rms L':>10}{'hp_rms R':>10}")
    for key, fly, a0, a1, wing, label in cases:
        b = bst[key]
        for lab in ["control"] + list(arms):
            a = b["arms"][lab]
            print(f"{key:>13}{lab:>10}{a['msc_LR_at_f0']:>13.3f}"
                  f"{b['msc_sig95']:>8.2f}{a['phase_LR_deg']:>9.0f}"
                  f"{a['xcorr_LR']:>8.2f}{a['xcorr_lag_fr']:>6d}"
                  f"{a['hp_rms_left']:>10.4f}{a['hp_rms_right']:>10.4f}")
        print(f"{'':>13}(f0 {b['f0_hz']:.0f} Hz = "
              f"{b['f0_period_frames']:.1f} frames; nperseg {b['nperseg']}, "
              f"K~{b['n_segments']}; "
              f"phase ~+-180 deg = bilaterally SYMMETRIC in these mirrored "
              f"joint coordinates)")


def run_plots(args):
    """`--plot` entry point: figures from saved arms, no fits."""
    flies = [int(x) for x in args.flies.split(",")]
    if not args.plot_arm:
        raise SystemExit("--plot needs at least one --plot-arm LABEL=PATH:KEY")
    os.makedirs(args.out, exist_ok=True)
    control, arms = load_plot_arms(args.plot_arm, flies)
    mjadr = {f: plot_model(args.bout_dir, f) for f in flies}
    t0, t1 = args.t0, args.t0 + args.nt
    zoom = tuple(int(x) for x in args.zoom.split(","))
    for f in flies:
        n = min(len(control[f]), *(len(a[f]) for a in arms.values()))
        if t1 > n:
            raise SystemExit(f"fly{f}: window {t0}..{t1} exceeds {n} frames")
    stats = trace_stats(control, arms, mjadr, flies, t0, t1, args.fs)
    for f in flies:
        s = stats[f]
        print(f"\n== fly{f}  singer={s['singing_wing']} "
              f"(|yawL-yawR| {s['song_absmean_deg']:.2f} deg, criterion 1 "
              f"{'applies' if s['criterion_1_applies'] else 'N/A'})")
        head = f"{'dof':>18}{'arm':>12}{'hp_rms':>10}{'ratio':>8}" \
               f"{'lp_std':>10}{'ratio':>8}{'median':>9}{'max|d| deg':>12}"
        print(head)
        for dof in DOF_ROWS:
            d = stats[f]["dofs"][dof]
            print(f"{dof:>18}{'control':>12}{d['control']['hp_rms']:>10.4f}"
                  f"{'-':>8}{d['control']['lp_std']:>10.3f}{'-':>8}"
                  f"{d['control']['median_deg']:>9.1f}{'-':>12}")
            for lab in [k for k in d if k not in ("control", "springref_deg")]:
                v = d[lab]
                print(f"{'':>18}{lab:>12}{v['hp_rms']:>10.4f}"
                      f"{v['hp_ratio']:>8.2f}{v['lp_std']:>10.3f}"
                      f"{v['lp_ratio']:>8.2f}{v['median_deg']:>9.1f}"
                      f"{v['max_abs_change_deg']:>12.6f}")
    # Extended-wing EPOCHS, measured from yaw. Everything song-related is scored
    # per epoch: fly1 swaps which wing it extends part-way through this bout, so
    # a whole-window "singing wing" label is wrong for a large part of it.
    epochs = {f: singer_epochs(control[f], mjadr[f][1], t0, t1) for f in flies}
    for f in flies:
        print(f"\n[epochs] fly{f}: " + " | ".join(
            f"{e['t0']}-{e['t1']} {e['wing']} extended "
            f"(|yawL-yawR| {e['song_absmean_deg']:.1f} deg"
            f"{'' if e['extended'] else ', NO extension'})" for e in epochs[f]))
    cases = build_cases(control, mjadr, flies, t0, t1)
    st_song = song_stats(control, arms, mjadr, cases, args.fs)
    bst = bilateral_stats(control, arms, mjadr, cases, args.fs)
    report_epochs(st_song, cases, arms)
    report_bilateral(bst, st_song, cases, arms)

    wanted = (("pitch", "alldof", "delta", "crit1", "bilateral", "song",
               "pulses") if args.plot == "all" else (args.plot,))
    paths, delta = [], None
    if "pitch" in wanted:
        paths += plot_pitch_traces(control, arms, mjadr, flies, t0, t1, zoom,
                                   args.fs, args.out, args.plot_title, stats,
                                   epochs=epochs)
    if "alldof" in wanted:
        paths += plot_all_dofs(control, arms, mjadr, flies, t0, t1, zoom,
                               args.out, args.plot_title, stats, epochs=epochs)
    if "bilateral" in wanted:
        paths += plot_bilateral(control, arms, mjadr, cases, bst, st_song,
                                args.fs, args.out, args.plot_title)
    if "song" in wanted:
        paths += plot_song(control, arms, mjadr, cases, st_song, args.fs,
                           args.out, args.plot_title)
    if "pulses" in wanted:
        paths += plot_pulse_trains(control, arms, mjadr, cases, st_song,
                                   args.fs, t0, t1, args.out, args.plot_title)
    if "crit1" in wanted:
        paths += plot_crit1_jitter(control, arms, mjadr, flies, t0, t1,
                                   tuple(int(x) for x in
                                         args.jitter_zoom.split(",")),
                                   args.fs, args.out, args.plot_title, stats)
    if "delta" in wanted:
        pp, delta = plot_delta_qpos(control, arms, mjadr, flies, t0, t1,
                                    args.out, args.plot_title)
        paths += pp
    # The numbers behind the pictures, beside the pictures (CLAUDE.md).
    j = os.path.join(args.out, "trace_stats.json")
    with open(j, "w") as fh:
        json.dump({"window": [t0, t1], "fs_hz": args.fs, "zoom": list(zoom),
                   "arms": list(arms), "plot_arm_specs": list(args.plot_arm),
                   "delta_qpos_nonzero": delta,
                   "epochs": {f"fly{f}": epochs[f] for f in flies},
                   "song_stats": st_song, "bilateral_stats": bst,
                   "stats": {f"fly{f}": stats[f] for f in flies}},
                  fh, indent=2, default=float)
    print(f"\nwrote {j}\n" + "\n".join(paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout-dir", required=True, help=".../bouts/bout_000NN")
    ap.add_argument("--flies", default="0,1")
    ap.add_argument("--coverage-weights", default="0.0,0.3",
                    help="sugar for --arm cov<W>=wing_mask_fit.coverage_weight=<W>")
    ap.add_argument("--arm", action="append", default=[],
                    help="LABEL=override[;override...] -- any wing_mask_fit "
                         "hydra override, repeatable. Lets Task 8 sweep a knob "
                         "other than coverage_weight without editing this file.")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--nt", type=int, default=1500)
    ap.add_argument("--fs", type=float, default=800.0)
    ap.add_argument("--mask-frames", type=int, default=40)
    ap.add_argument("--from-arms", default=None,
                    help="re-score the poses in this arms_flyN.npz instead of "
                         "re-running the fits ('{fly}' is substituted)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", default=None,
                    choices=("all", "pitch", "alldof", "delta", "crit1",
                             "bilateral", "song", "pulses"),
                    help="draw figures from saved arms npz and exit; runs no "
                         "fits. See the FIGURES section of the module docstring "
                         "for what each one is and what it should look like.")
    ap.add_argument("--plot-arm", action="append", default=[],
                    help="LABEL=PATH:KEY -- an arm to draw, repeatable. PATH may "
                         "contain '{fly}'. KEY is the npz key, e.g. 'q_sm30'.")
    ap.add_argument("--jitter-zoom", default="900,1000",
                    help="short window for the criterion-1 close up; pick it "
                         "INSIDE a fuzzy stretch, not a quiet one")
    ap.add_argument("--zoom", default="600,700",
                    help="zoom-panel frame range; keep it fixed across figures "
                         "so they line up")
    ap.add_argument("--plot-title", default="wing-mask fit",
                    help="recording/bout prefix drawn in the suptitle")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.plot:
        return run_plots(args)

    cfg = compose_cfg(["paths=hyak"])
    if args.arm:
        weights = []
        for spec in args.arm:
            lab, _, ov = spec.partition("=")
            weights.append((lab, [s for s in ov.split(";") if s]))
    else:
        weights = [(f"cov{float(x):g}",
                    [f"wing_mask_fit.coverage_weight={float(x)}"])
                   for x in args.coverage_weights.split(",")]
    all_res = {}
    for fly in [int(x) for x in args.flies.split(",")]:
        fa = (args.from_arms.format(fly=fly) if args.from_arms else None)
        r = score_fly(cfg, args.bout_dir, fly, weights, args.t0, args.nt,
                      args.fs, args.out, mask_frames=args.mask_frames,
                      from_arms=fa)
        all_res[f"fly{fly}"] = report(r)
    p = os.path.join(args.out, "wing_mask_fit_ab.json")
    with open(p, "w") as f:
        json.dump(all_res, f, indent=2, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
