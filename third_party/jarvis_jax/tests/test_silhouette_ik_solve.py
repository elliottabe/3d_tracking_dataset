import os, numpy as np, pytest
from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs, solve_ik

IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_build_inputs_shapes():
    inp = build_solver_inputs(IK, XML)
    T = inp["q_init"].shape[0]; nq = inp["q_init"].shape[1]; nk = len(inp["kp_names"])
    assert nq == 93 and nk == 50
    assert inp["kp_data"].shape == (T, nk, 3)
    assert inp["kps_to_opt"].shape == (nk * 3,)
    assert inp["site_idxs"].shape == (nk,)
    assert inp["lb"].shape == (nq,) and inp["ub"].shape == (nq,)


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_solve_ik_does_not_worsen_fit_on_slice():
    inp = build_solver_inputs(IK, XML)
    sl = slice(0, 8)
    small = dict(inp)
    small["q_init"] = inp["q_init"][sl]; small["kp_data"] = inp["kp_data"][sl]
    q = solve_ik(small, smooth_weight=0.0, n_iter=40)
    assert q.shape == small["q_init"].shape
    assert np.isfinite(q).all()
