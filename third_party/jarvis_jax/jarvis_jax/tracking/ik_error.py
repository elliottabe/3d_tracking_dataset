"""IK error quantification: marker Jacobian + qpos sensitivity + fit residual +
implied bias, plus the 3D-noise model and a Monte-Carlo re-solve.

All quantities are PROXIES (flies have no ground-truth qpos). The linking object
is the marker Jacobian J = d(marker site_xpos)/dqpos at the solved pose, obtained
by jax autodiff through mjx.kinematics (marker sites are the STAC `tracking[{kp}]`
sites; FK mirrors jarvis_jax.tracking.segment_fit). See
docs/superpowers/plans/2026-07-16-ik-error-quantification.md.
"""
from __future__ import annotations
import numpy as np


def marker_site_ids(mj_model, kp_names):
    """Site id per keypoint (the STAC `tracking[{kp}]` marker site)."""
    import mujoco
    return np.array([mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f"tracking[{k}]")
                     for k in kp_names], dtype=int)


def marker_jacobian(mjx_model, d0, qpos_frame, site_ids):
    """d(marker site_xpos, flattened (3K,)) / dqpos at one frame -> (3K, nq)."""
    import jax
    import jax.numpy as jnp
    import mujoco.mjx as mjx
    sid = jnp.asarray(np.asarray(site_ids))

    def fk(q):
        d = mjx.kinematics(mjx_model, d0.replace(qpos=q))
        return d.site_xpos[sid].reshape(-1)                 # (3K,)
    return jax.jacobian(fk)(jnp.asarray(qpos_frame))        # (3K, nq)


def qpos_sensitivity(J, W, sigma2, eps=1e-8):
    """Local IK sensitivity of qpos to marker (3D) error.

    S = (JᵀWJ + εI)⁻¹ JᵀW  (nq, 3K). transfer[j] = ‖S[j, :]‖ (qpos change per unit
    isotropic 3D noise, per DOF). cov = S sigma2 Sᵀ; std = sqrt(diag(cov)).
    `sigma2` is the (3K, 3K) 3D-noise covariance (same length units as J's marker
    space)."""
    import jax.numpy as jnp
    J = jnp.asarray(J); W = jnp.asarray(W); sigma2 = jnp.asarray(sigma2)
    H = J.T @ W @ J
    H = H + eps * jnp.eye(H.shape[0])
    S = jnp.linalg.solve(H, J.T @ W)                        # (nq, 3K)
    cov = S @ sigma2 @ S.T                                  # (nq, nq)
    return dict(S=S, transfer=jnp.linalg.norm(S, axis=1),
                cov=cov, std=jnp.sqrt(jnp.clip(jnp.diag(cov), 0.0, None)))


def implied_bias(J, W, residual_vec, eps=1e-8):
    """qpos shift implied by a marker residual (3K,) via the same pseudo-inverse:
    S @ r (nq,). W is the marker weight matrix (3K,3K)."""
    import jax.numpy as jnp
    S = qpos_sensitivity(J, W, jnp.eye(J.shape[0]), eps)["S"]
    return S @ jnp.asarray(residual_vec)


def dof_units(names_qpos):
    """Classify each qpos DOF and give its natural report unit:
    root_trans -> mm, root_quat -> deg, hinge -> deg. Heuristic on names_qpos."""
    out = []
    for n in names_qpos:
        ln = str(n).lower()
        is_quat = any(t in ln for t in ("qw", "qx", "qy", "qz", "quat"))
        is_root = any(t in ln for t in ("root", "free", "world"))
        is_trans = any(ln.endswith(a) or f"_{a}" in ln for a in ("x", "y", "z"))
        if is_quat:
            out.append(dict(name=n, kind="root_quat", unit="deg"))
        elif is_root and is_trans:
            out.append(dict(name=n, kind="root_trans", unit="mm"))
        else:
            out.append(dict(name=n, kind="hinge", unit="deg"))
    return out


# --- noise model + Monte-Carlo (Task 2) -------------------------------------

def sigma_kp_from_conf(conf3d, kp3d, *, floor_mm=1.0, scale_mm_per_lowconf=10.0):
    """Per-keypoint isotropic 3D sigma (mm) from mean confidence:
    sigma = floor_mm + scale*(1 - mean_conf). conf3d (T,K); kp3d (T,K,3) only used
    to keep the signature aligned with callers (mask handled via conf)."""
    import warnings
    conf3d = np.asarray(conf3d, float)
    with warnings.catch_warnings():
        # an all-zero-confidence keypoint -> empty slice -> nan (intended: nan_to_num
        # below maps it to mean_conf=0 => max sigma). Suppress the noisy RuntimeWarning.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_conf = np.nanmean(np.where(conf3d > 0, conf3d, np.nan), axis=0)   # (K,)
    mean_conf = np.nan_to_num(mean_conf, nan=0.0)
    return floor_mm + scale_mm_per_lowconf * np.clip(1.0 - mean_conf, 0.0, 1.0)


def sigma2_diag(sigma_kp):
    """(K,) per-keypoint sigma -> (3K,3K) diagonal covariance (sigma^2 per xyz)."""
    s = np.repeat(np.asarray(sigma_kp, float), 3)          # (3K,)
    return np.diag(s ** 2)


def monte_carlo_qpos(cfg, kp3d, kp_names, sigma_kp, *, offsets_path, save_path,
                     scale=1.0, n=20, seed=0):
    """Re-solve ik_only_bout on n noise-perturbed copies of kp3d; return (n,T,nq).
    Per-DOF Monte-Carlo std = returned.std(axis=0)."""
    import numpy as _np
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.tracking.stac import ik_only_bout
    rng = _np.random.default_rng(seed)
    kp3d = _np.asarray(kp3d, float)
    T, K, _ = kp3d.shape
    s3 = _np.repeat(_np.asarray(sigma_kp, float), 3).reshape(K, 3)
    out = []
    for i in range(n):
        noisy = kp3d + rng.normal(0.0, 1.0, kp3d.shape) * s3[None, :, :]
        h5 = ik_only_bout(cfg, noisy, kp_names, offsets_path=offsets_path,
                          out_h5=f"mc_{i:03d}.h5", save_path=save_path, scale=scale)
        out.append(_np.asarray(ioh5.load(h5)["qpos"]))
    return _np.stack(out)                                  # (n, T, nq)
