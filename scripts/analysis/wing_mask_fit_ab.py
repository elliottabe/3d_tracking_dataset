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
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

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
