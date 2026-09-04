import numpy as np
import jax, jax.numpy as jnp
import pytest


def test_enumerate_assignments_is_injective_and_complete():
    from jarvis_jax.train.matching import enumerate_assignments
    a = enumerate_assignments(3, 2)
    assert a.shape == (6, 2) and all(len(set(r)) == 2 for r in a)
    assert enumerate_assignments(2, 2).shape == (2, 2)


def test_match_agrees_with_scipy_hungarian():
    from scipy.optimize import linear_sum_assignment
    from jarvis_jax.train.matching import match
    rng = np.random.default_rng(0)
    cost = rng.uniform(size=(16, 3, 2)).astype(np.float32)
    fv = np.ones((16, 2), bool)
    assign, inst_m = match(jnp.asarray(cost), jnp.asarray(fv), jnp.zeros(16, bool))
    for b in range(16):
        rows, cols = linear_sum_assignment(cost[b].T)          # flies x instances
        ref = np.full(2, -1); ref[rows] = cols
        assert list(np.asarray(assign[b])) == list(ref)
    assert np.asarray(inst_m).sum(1).tolist() == [2] * 16


def test_match_pins_fly0_to_instance0_and_ignores_invalid_flies():
    from jarvis_jax.train.matching import match
    cost = jnp.asarray([[[0.0, 5.0], [1.0, 0.1], [9.0, 9.0]]])        # instance 1 is best for fly 0
    assign, inst_m = match(cost, jnp.asarray([[True, False]]), jnp.asarray([True]))
    assert np.asarray(assign).tolist() == [[0, -1]]
    assert np.asarray(inst_m).tolist() == [[True, False, False]]


def _perfect_batch(B=2, I=3, T=1, C=3, K=5, seed=0):
    """Labels + an output that reproduces them exactly (instance 0 <- fly 0, 2 <- fly 1)."""
    from jarvis_jax.models.mvq.geometry import project_local
    rng = np.random.default_rng(seed)
    M = np.stack([np.array([[8.0, 0.1 * c, 0.0], [0.0, -8.0, 0.2 * c]]) for c in range(C)]).astype(np.float32)
    M = np.broadcast_to(M, (B, C, 2, 3)).copy()
    tl = np.broadcast_to(np.array([[224.0, 224.0]] * C, np.float32), (B, T, C, 2)).copy()
    X = rng.normal(size=(B, 2, T, K, 3)).astype(np.float32) * 5                    # flies' 3D
    kp2d = np.stack([np.stack([np.stack([np.asarray(project_local(jnp.asarray(X[b, f, t]), jnp.asarray(M[b]), jnp.asarray(tl[b, t])))
                                          for t in range(T)])
                                for f in range(2)])
                      for b in range(B)])  # (B,F,T,K,C,2)
    kp2d = np.moveaxis(kp2d, 4, 3)                                                  # (B,F,T,C,K,2)
    batch = {"M": M, "t_local": tl, "kp3d_local": X, "has3d": np.ones((B, 2, T, K), bool),
             "kp2d": kp2d.astype(np.float32), "vis2d": np.ones((B, 2, T, C, K), bool),
             "fly_valid": np.ones((B, 2), bool), "px_scale": np.full((B,), 8.0, np.float32),
             "cam_valid": np.ones((B, T, C), bool), "prompt_on": np.zeros((B,), bool)}
    xyz = np.zeros((B, I, T, K, 3), np.float32); xyz[:, 0] = X[:, 0]; xyz[:, 2] = X[:, 1]; xyz[:, 1] = 50.0
    uv = np.zeros((B, I, T, C, K, 2), np.float32); uv[:, 0] = kp2d[:, 0]; uv[:, 2] = kp2d[:, 1]
    big = np.full((B, I, T, K), 6.0, np.float32)
    out = {"xyz": xyz, "conf_logit": big, "exist_logit": np.array([[6.0, -6.0, 6.0]] * B, np.float32),
           "uv": uv, "vis_logit": np.full((B, I, T, C, K), 6.0, np.float32), "aux_pass1": None, "aux_layers": []}
    out["aux_pass1"] = {k: v for k, v in out.items() if k not in ("aux_pass1", "aux_layers")}
    j = lambda d: {k: (jnp.asarray(v) if not isinstance(v, (dict, list)) and v is not None else v) for k, v in d.items()}
    out = j(out); out["aux_pass1"] = j(out["aux_pass1"])
    return out, j(batch)


def test_loss_near_zero_at_ground_truth_and_metrics():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    part_of_k = np.arange(5, dtype=np.int32)
    total, m = mvq_loss(out, batch, LossWeights(), part_of_k)
    assert float(m["reproj"]) < 1e-3 and float(m["l3d"]) < 1e-3 and float(m["uv2d"]) < 1e-3
    assert float(m["rep"]) == 0.0 and float(m["exist_acc"]) == 1.0
    assert float(m["match_reproj_px"]) < 1e-2 and float(m["mpjpe3d_units"]) < 1e-3
    assert {"uv2d_px", "head_vs_reproj_px"} <= set(m)
    assert float(m["uv2d_px"]) < 1e-2 and float(m["head_vs_reproj_px"]) < 1e-2
    # vis must equal the per-entry BCE of a logit of 6.0 against target 1, independent of K
    # (a broadcast-mask bug previously made this K-times too large)
    expected_vis = float(jnp.log1p(jnp.exp(-6.0)))
    assert abs(float(m["vis"]) - expected_vis) < 1e-5
    # only the -log c and BCE floors remain
    assert float(total) < 0.05


def test_masked_entries_do_not_contribute():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    out["xyz"] = out["xyz"].at[:, 0, :, 0].add(30.0)                    # corrupt keypoint 0 of fly 0
    _, m_bad = mvq_loss(out, batch, LossWeights(), np.arange(5, dtype=np.int32))
    batch["vis2d"] = batch["vis2d"].at[:, 0, :, :, 0].set(False)
    batch["has3d"] = batch["has3d"].at[:, 0, :, 0].set(False)
    _, m_masked = mvq_loss(out, batch, LossWeights(), np.arange(5, dtype=np.int32))
    assert float(m_bad["reproj"]) > 1.0 and float(m_masked["reproj"]) < 1e-3
    assert float(m_masked["l3d"]) < 1e-3


def test_repulsion_fires_only_near_other_fly_same_part():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    part_of_k = np.arange(5, dtype=np.int32)
    # positive: move fly-0's prediction for keypoint 1 onto fly-1's keypoint 1 (same part)
    out, batch = _perfect_batch()
    out["xyz"] = out["xyz"].at[:, 0, :, 1].set(batch["kp3d_local"][:, 1, :, 1])
    out["uv"] = out["uv"].at[:, 0, :, :, 1].set(batch["kp2d"][:, 1, :, :, 1])
    _, m_same = mvq_loss(out, batch, LossWeights(), part_of_k)
    # zero: move fly-0's prediction for keypoint 1 onto fly-1's keypoint 2 (a DIFFERENT part)
    out2, batch2 = _perfect_batch()
    out2["xyz"] = out2["xyz"].at[:, 0, :, 1].set(batch2["kp3d_local"][:, 1, :, 2])
    out2["uv"] = out2["uv"].at[:, 0, :, :, 1].set(batch2["kp2d"][:, 1, :, :, 2])
    _, m_diff = mvq_loss(out2, batch2, LossWeights(), part_of_k)
    assert float(m_same["rep"]) > 0.0 and float(m_diff["rep"]) == 0.0


def test_confidence_term_prefers_low_c_on_bad_points():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    out["xyz"] = out["xyz"].at[:, 0].add(3.0)                             # ~24 px reprojection error
    hi = dict(out); hi["conf_logit"] = jnp.full_like(out["conf_logit"], 4.0)
    lo = dict(out); lo["conf_logit"] = jnp.full_like(out["conf_logit"], -4.0)
    _, m_hi = mvq_loss(hi, batch, LossWeights(), np.arange(5, dtype=np.int32))
    _, m_lo = mvq_loss(lo, batch, LossWeights(), np.arange(5, dtype=np.int32))
    assert float(m_lo["conf"]) < float(m_hi["conf"])


def test_loss_is_differentiable_and_jittable():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    pk = np.arange(5, dtype=np.int32)
    f = jax.jit(lambda xyz: mvq_loss({**out, "xyz": xyz}, batch, LossWeights(), pk)[0])
    g = jax.grad(f)(out["xyz"])
    assert g.shape == out["xyz"].shape and bool(jnp.isfinite(g).all())


def test_px_scale_broadcasts_by_batch_not_fly_axis():
    # B=3 != F=2 (the fixture always has 2 flies): the matching cost's 3D term must scale
    # each batch element by ITS OWN px_scale, not by whichever fly index the batch axis
    # happened to alias onto. Per-sample-varying scale (6, 8, 10) with a batch size that
    # differs from the fly count used to raise (or silently mis-scale at B==F).
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch(B=3)
    batch["px_scale"] = jnp.array([6.0, 8.0, 10.0], jnp.float32)
    total, m = mvq_loss(out, batch, LossWeights(), np.arange(5, dtype=np.int32))
    assert bool(jnp.isfinite(total))
    assert float(m["reproj"]) < 1e-3
    assert float(m["l3d"]) < 1e-3


def test_single_valid_fly_zero_repulsion_and_correct_exist_acc():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    B, I = out["xyz"].shape[0], out["xyz"].shape[1]
    batch["fly_valid"] = jnp.asarray([[True, False]] * B)
    total, m = mvq_loss(out, batch, LossWeights(), np.arange(5, dtype=np.int32))
    assert float(m["rep"]) == 0.0
    assert bool(jnp.isfinite(total))
    # with fly 1 invalid, only instance 0 (matched to fly 0) should be "matched" ground truth
    expected_matched = jnp.asarray([[True, False, False]] * B)
    expected_acc = ((out["exist_logit"] > 0) == expected_matched).astype(jnp.float32).mean()
    assert float(m["exist_acc"]) == float(expected_acc)


def test_aux_layers_deep_supervision_raises_total_and_stays_jittable():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    pk = np.arange(5, dtype=np.int32)
    total0, _ = mvq_loss(out, batch, LossWeights(), pk)
    out2 = dict(out)
    out2["aux_layers"] = [{"xyz": out["xyz"] + 2.0, "conf_logit": out["conf_logit"], "exist_logit": out["exist_logit"]}]
    total1, _ = mvq_loss(out2, batch, LossWeights(), pk)
    assert float(total1) > float(total0)

    def f(xyz):
        o = {**out2, "aux_layers": [{**out2["aux_layers"][0], "xyz": xyz}]}
        return mvq_loss(o, batch, LossWeights(), pk)[0]

    g = jax.grad(jax.jit(f))(out2["aux_layers"][0]["xyz"])
    assert bool(jnp.isfinite(g).all())
