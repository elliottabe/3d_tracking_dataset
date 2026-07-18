"""Export a golden PyTorch HybridNetBackbone full-forward fixture to npz.

This is the end-to-end oracle for the faithful JAX HybridNet3D port (Task 5):
crops -> EfficientTrack -> reproject -> v2vNet -> soft-argmax -> points3D.

It instantiates the real ``HybridNetBackbone`` with the trained
``HybridNet-large_final.pth`` (whose ``effTrack`` sub-model is 3-channel — the
end-to-end HybridNet was trained without the mask channel, unlike the standalone
4-channel masked KeypointDetect model), reuses the 7-camera geometry from
``reproject_fixture.npz`` (center3D / centerHM / cameraMatrices), feeds a
fixed-seed random image batch, and dumps points3D + confidences.

MUST run on a GPU node — ``ReprojectionLayer`` and ``HybridNetBackbone`` are
CUDA-hardcoded (``.cuda()`` / ``torch.device('cuda')`` / ``torch.cuda.IntTensor``).
Run in the ``jarvis`` PyTorch env::

    python third_party/jarvis_jax/jarvis_jax/convert/export_hybridnet_fixture.py \
        --weights <HybridNet-large_final.pth> --num-joints 50 \
        --geometry third_party/jarvis_jax/jarvis_jax/convert/reproject_fixture.npz \
        --out third_party/jarvis_jax/jarvis_jax/convert/fixtures/hybridnet_large.npz
"""
import argparse
import os
import sys

import numpy as np
import torch


class _Node(dict):
    """Attribute-accessible config node (cfg.HYBRIDNET.GRID_SPACING style)."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _build_cfg(num_joints, num_cameras, image_size):
    return _Node(
        DATASET=_Node(DATASET_ROOT_DIR="", IMAGE_SIZE=[image_size, image_size]),
        HYBRIDNET=_Node(GRID_SPACING=1, ROI_CUBE_SIZE=48, NUM_CAMERAS=num_cameras),
        KEYPOINTDETECT=_Node(MODEL_SIZE="large", NUM_JOINTS=num_joints,
                             BOUNDING_BOX_SIZE=image_size),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="HybridNet-large_final.pth")
    ap.add_argument("--jarvis-root", default="third_party/JARVIS-HybridNet")
    ap.add_argument("--geometry", required=True, help="reproject_fixture.npz for cam geometry")
    ap.add_argument("--num-joints", type=int, default=50)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "HybridNetBackbone is CUDA-hardcoded; run on a GPU node"
    sys.path.insert(0, a.jarvis_root)
    from jarvis.hybridnet.model import HybridNetBackbone

    g = np.load(a.geometry, allow_pickle=True)
    center3D = torch.as_tensor(np.asarray(g["center3D"]), dtype=torch.float32).cuda()      # (B,3)
    centerHM = torch.as_tensor(np.asarray(g["centerHM"]), dtype=torch.float32).cuda()       # (B,cam,2)
    cameraMatrices = torch.as_tensor(np.asarray(g["cameraMatrices"]), dtype=torch.float32).cuda()  # (B,cam,4,3)
    B, num_cam = centerHM.shape[0], centerHM.shape[1]

    cfg = _build_cfg(a.num_joints, num_cam, a.image_size)
    net = HybridNetBackbone(cfg)
    state_dict = torch.load(a.weights, map_location="cpu")
    net.load_state_dict(state_dict, strict=True)
    net = net.cuda().eval()

    # Match effTrack's trained input-channel count (HybridNet effTrack is 3ch).
    in_ch = net.effTrack.backbone_net.model._conv_stem.weight.shape[1]
    torch.manual_seed(a.seed)
    imgs = torch.rand(B, num_cam, in_ch, a.image_size, a.image_size).cuda()
    img_size = torch.tensor([a.image_size, a.image_size]).cuda()

    with torch.no_grad():
        heatmap_final, heatmaps_padded, points3D, confidences = net(
            imgs, img_size, centerHM, center3D, cameraMatrices)

    out = {
        "imgs_nchw": imgs.detach().cpu().numpy(),          # (B,cam,in_ch,H,W)
        "centerHM": centerHM.detach().cpu().numpy(),
        "center3D": center3D.detach().cpu().numpy(),
        "cameraMatrices": cameraMatrices.detach().cpu().numpy(),
        "points3D": points3D.detach().cpu().numpy(),        # (B,J,3)
        "confidences": confidences.detach().cpu().numpy(),  # (B,J)
        "num_joints": np.int64(a.num_joints),
        "in_channels": np.int64(in_ch),
        "image_size": np.int64(a.image_size),
        "num_cam": np.int64(num_cam),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}")
    print(f"  imgs {out['imgs_nchw'].shape} (in_ch={in_ch})  points3D {out['points3D'].shape}"
          f"  conf {out['confidences'].shape}  num_cam={num_cam}")
    print(f"  points3D[0,:3]=\n{out['points3D'][0, :3]}")


if __name__ == "__main__":
    main()
