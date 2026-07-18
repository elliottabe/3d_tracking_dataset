"""Export golden EfficientTrack-large activations + weights to npz (PyTorch side).

This is the PyTorch-side oracle for the faithful JAX EfficientTrack port. It runs
the trained ``EfficientTrackBackbone`` in eval mode on a fixed-seed input and dumps
the three backbone feature maps (P3/P4/P5), the two head outputs (res1/res2), and
every state_dict tensor to a single npz. The JAX parity tests (Tasks 2-4) compare
against these arrays.

Run inside the JARVIS-HybridNet PyTorch env (``jarvis``), on a COMPUTE NODE::

    python third_party/jarvis_jax/jarvis_jax/convert/export_efficienttrack_fixture.py \
        --weights <EfficientTrack-large_final.pth> --num-joints 50 \
        --out third_party/jarvis_jax/jarvis_jax/convert/fixtures/efficienttrack_large.npz

The script imports only ``torch``, ``numpy`` and the JARVIS ``jarvis`` package (via
``--jarvis-root`` on sys.path). It does NOT import ``jarvis_jax`` (JAX), so it runs
in the PyTorch-only env.
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="EfficientTrack-large_final.pth")
    ap.add_argument("--jarvis-root", default="third_party/JARVIS-HybridNet")
    ap.add_argument("--num-joints", type=int, default=50)
    ap.add_argument("--in-channels", type=int, default=4,
                    help="input channels; unified_V3_masked weights are 4 (RGB+mask)")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, a.jarvis_root)
    from jarvis.efficienttrack.model import EfficientTrackBackbone

    class Cfg:
        """Minimal KEYPOINTDETECT cfg stand-in (backbone reads none of its fields
        in __init__/forward; model_size + output_channels drive the architecture)."""
        MODEL_SIZE = "large"
        NUM_JOINTS = a.num_joints

    net = EfficientTrackBackbone(Cfg(), model_size="large", output_channels=a.num_joints,
                                 in_channels=a.in_channels)
    state_dict = torch.load(a.weights, map_location="cpu")
    net.load_state_dict(state_dict, strict=True)
    net.eval()

    # Capture the three backbone feature maps (list output of EfficientNet wrapper)
    # via a forward hook — robust vs. monkeypatching .forward.
    captured = {}

    def _hook(_module, _inp, output):
        # output is the list [P3, P4, P5] returned by model.py::EfficientNet.forward
        for i, f in enumerate(output):
            captured[f"backbone_feat_{i}"] = f.detach().cpu().numpy()

    handle = net.backbone_net.register_forward_hook(_hook)

    torch.manual_seed(a.seed)
    x = torch.rand(2, a.in_channels, a.image_size, a.image_size)  # fixed seed, eval mode
    with torch.no_grad():
        res1, res2 = net(x)
    handle.remove()

    assert {"backbone_feat_0", "backbone_feat_1", "backbone_feat_2"} <= set(captured), (
        f"expected 3 backbone feature maps, got {sorted(captured)}")

    out = {
        "input_nchw": x.numpy(),
        "feat_p3": captured["backbone_feat_0"],
        "feat_p4": captured["backbone_feat_1"],
        "feat_p5": captured["backbone_feat_2"],
        "res1": res1.detach().cpu().numpy(),
        "res2": res2.detach().cpu().numpy(),
        "num_joints": np.int64(a.num_joints),
        "image_size": np.int64(a.image_size),
        "in_channels": np.int64(a.in_channels),
    }
    for k, v in net.state_dict().items():
        out[f"w::{k}"] = v.detach().cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    n_w = sum(1 for k in out if k.startswith("w::"))
    print(f"wrote {a.out}")
    print(f"  input {out['input_nchw'].shape}  res1 {out['res1'].shape}  res2 {out['res2'].shape}")
    print(f"  feat_p3 {out['feat_p3'].shape}  feat_p4 {out['feat_p4'].shape}  feat_p5 {out['feat_p5'].shape}")
    print(f"  {n_w} weight tensors")


if __name__ == "__main__":
    main()
