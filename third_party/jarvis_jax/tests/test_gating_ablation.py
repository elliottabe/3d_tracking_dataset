import os
import json
import jax.numpy as jnp
import numpy as np
import pytest

RUNS = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs"
GATED = f"{RUNS}/cse_vit350_gated/final"
UNGATED = f"{RUNS}/cse_vit350_wings/final"
SUMMARY = f"{RUNS}/cse_vit350_gated/ablation.json"


def test_off_fly_mass_metric_matches_mask_containment():
    """The headline metric IS mean mask_containment over the batch (dilated)."""
    from jarvis_jax.cse.eval_gating_ablation import off_fly_mass_frac
    from jarvis_jax.train.losses import mask_containment
    pred = jnp.zeros((1, 8, 8, 1)).at[0, 6, 6, 0].set(1.0)   # peak outside
    mask = jnp.zeros((1, 8, 8)).at[0, 1, 1].set(1.0)
    assert abs(off_fly_mass_frac(pred, mask, dilate=0)
               - float(mask_containment(pred, mask, dilate=0))) < 1e-6
    # dilating to reach the peak drops the metric toward 0.
    assert off_fly_mass_frac(pred, mask, dilate=0) > off_fly_mass_frac(
        jnp.zeros((1, 8, 8, 1)).at[0, 2, 2, 0].set(1.0), mask, dilate=3)


@pytest.mark.skipif(not (os.path.exists(GATED) and os.path.exists(UNGATED)),
                    reason="run T6 gated fine-tune first (need both checkpoints)")
def test_ablation_summary_exists_and_is_well_formed():
    if not os.path.exists(SUMMARY):
        pytest.skip("run eval_gating_ablation.py first to produce ablation.json")
    d = json.load(open(SUMMARY))
    for arm in ("ungated_ckpt", "gated_ckpt"):
        assert arm in d
        for mode in ("raw", "gated_decode"):
            m = d[arm][mode]
            assert np.isfinite(m["mpjpe_px"]) and m["mpjpe_px"] > 0
            assert 0.0 <= m["off_fly_mass_frac"] <= 1.0
    # HEADLINE assertion (honest): the GATED checkpoint (raw decode) puts LESS mass
    # off-fly than the UNGATED checkpoint (raw decode). This is the payoff of the
    # containment loss. If it does NOT hold, the test FAILS loudly -> report it.
    assert (d["gated_ckpt"]["raw"]["off_fly_mass_frac"]
            <= d["ungated_ckpt"]["raw"]["off_fly_mass_frac"] + 1e-4), (
        "gated fine-tune did NOT reduce off-fly mass -- report this honestly")
