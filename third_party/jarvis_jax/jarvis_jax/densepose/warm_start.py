"""Warm-start a 50+M-output ViTPose from the trained 50-keypoint v3 checkpoint.

All params are shape-identical between the 50- and (50+M)-output models EXCEPT the
decoder head (1x1 conv): kernel (1,1,256,K) and bias (K,).  We copy every matching
param from v3, and for the head copy v3's first 50 output channels, leaving the M
new vertex channels at their fresh init.  This gives the keypoints v3-level
accuracy immediately and a fly-fine-tuned backbone for the vertex channels to
build on.
"""
from __future__ import annotations

import numpy as np


def warm_start_from_v3(new_model, v3_ckpt_path, cfg50):
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import nnx
    from jarvis_jax.models.vitpose import ViTPose

    v3 = ViTPose(cfg50, rngs=nnx.Rngs(0))
    gdef50, st50 = nnx.split(v3)
    st50 = ocp.StandardCheckpointer().restore(v3_ckpt_path, st50)

    gdefN, stN = nnx.split(new_model)
    f50 = dict(st50.flat_state())
    fN = dict(stN.flat_state())

    n_copy, n_splice, n_fresh = 0, 0, 0
    for path, varN in fN.items():
        if path not in f50:
            n_fresh += 1
            continue
        v50 = f50[path].value
        if varN.value.shape == v50.shape:
            varN.value = v50
            n_copy += 1
        else:
            arr = np.array(varN.value)
            k = v50.shape[-1]               # last axis = output channels
            arr[..., :k] = np.array(v50)
            varN.value = jnp.asarray(arr)
            n_splice += 1
    print(f"[warm_start] copied {n_copy}, spliced {n_splice}, fresh {n_fresh} param tensors")
    return nnx.merge(gdefN, stN)
