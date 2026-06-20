"""Generate the timm reference features for the backbone parity test (Task 9).

Run in the *torch/timm* env (e.g. the ``jarvis`` conda env), NOT the jax env::

    /gscratch/portia/eabe/miniconda3/envs/jarvis/bin/python scripts/make_ref_features.py

Writes two /tmp artifacts shared with the jax-side parity test:
  * /tmp/parity_rgb_nhwc.npy  -- the exact input pixels (1,448,448,3), NHWC
  * /tmp/ref_feats.npy        -- timm forward_features at img_size=448, cls dropped

The jax test LOADS /tmp/parity_rgb_nhwc.npy (does not regenerate it) so both sides
see identical pixels; otherwise parity would fail for the wrong reason.

Optional 224 diagnostic (proves the porting mapping is correct, isolating the
448 pos-embed-interpolation drift)::

    IMG=224 /gscratch/portia/eabe/miniconda3/envs/jarvis/bin/python scripts/make_ref_features.py
"""
import os
import numpy as np
import timm
import torch

IMG = int(os.environ.get("IMG", "448"))
RGB_NPY = os.environ.get("RGB_NPY", "/tmp/parity_rgb_nhwc.npy")
REF_NPY = os.environ.get("REF_NPY", "/tmp/ref_feats.npy")

rng = np.random.default_rng(0)
rgb_nhwc = rng.standard_normal((1, IMG, IMG, 3)).astype("float32")
np.save(RGB_NPY, rgb_nhwc)

m = timm.create_model(
    "vit_base_patch16_224.mae", pretrained=True, num_classes=0, img_size=IMG
).eval()

rgb_nchw = np.ascontiguousarray(rgb_nhwc.transpose(0, 3, 1, 2))  # (1,3,IMG,IMG)
with torch.no_grad():
    feats = m.forward_features(torch.from_numpy(rgb_nchw))       # (1, N+1, 768)
feats_no_cls = feats[:, 1:].numpy()                              # drop cls token
np.save(REF_NPY, feats_no_cls)

print(f"IMG={IMG}  rgb={rgb_nhwc.shape} -> {RGB_NPY}")
print(f"ref_feats={feats_no_cls.shape} -> {REF_NPY}")
