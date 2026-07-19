"""Export a golden torchvision EfficientNet-b3 (ImageNet) fixture to npz.

Oracle for the faithful JAX port of the STANDARD (BatchNorm) EfficientNet-b3 that
we warm-start from ImageNet for the fair ViT-vs-EfficientNet comparison. Dumps
per-feature-block activations (to identify the /8,/16,/32 FPN taps), the final
feature map, and the full state_dict (conv + BatchNorm params + running stats).

Run on CPU (fine — 2-sample inference) with the libstdc++ preload::

    LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 \
    TORCH_HOME=/gscratch/portia/eabe/data/Johnson_lab/torch_cache \
    python jarvis_jax/convert/export_effnet_b3_imagenet_fixture.py --out <npz>
"""
import argparse
import os

import numpy as np
import torch
import torchvision.models as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    weights = M.EfficientNet_B3_Weights.IMAGENET1K_V1
    net = M.efficientnet_b3(weights=weights).eval()

    # Capture each features[i] output (Sequential of 9 blocks: stem, 7 MBConv
    # stages, head conv) to identify the FPN taps by spatial stride.
    feats = {}
    handles = []
    for i, block in enumerate(net.features):
        def mk(idx):
            def hook(_m, _inp, out):
                feats[idx] = out.detach().cpu().numpy()
            return hook
        handles.append(block.register_forward_hook(mk(i)))

    torch.manual_seed(a.seed)
    x = torch.rand(2, 3, a.image_size, a.image_size)
    with torch.no_grad():
        _ = net.features(x)
    for h in handles:
        h.remove()

    out = {
        "input_nchw": x.numpy(),
        "image_size": np.int64(a.image_size),
    }
    for i, arr in feats.items():
        out[f"feat_{i}"] = arr
    for k, v in net.state_dict().items():
        out[f"w::{k}"] = v.detach().cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}")
    print("feature-block output shapes (NCHW) — identify /8,/16,/32 FPN taps:")
    for i in sorted(feats):
        s = feats[i].shape
        stride = a.image_size // s[2]
        print(f"  features[{i}]: {s}  (stride /{stride}, C={s[1]})")
    print(f"{sum(1 for k in out if k.startswith('w::'))} weight tensors")


if __name__ == "__main__":
    main()
