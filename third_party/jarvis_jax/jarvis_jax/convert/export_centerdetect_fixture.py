"""Export golden CenterDetect (EfficientTrack-medium) activations + weights
to npz (PyTorch side), on REAL camera frames -- not synthetic noise.

CenterDetect is the same ``EfficientTrackBackbone`` as KeypointDetect (see
``jarvis/efficienttrack/efficienttrack.py``), just with ``model_size='medium'``
(EfficientNet-b1 backbone), ``in_channels=3`` (RGB only, no SAM3 mask), and a
320x320 input -- see ``export_efficienttrack_fixture.py`` for the sibling
``large``/KeypointDetect exporter this mirrors.

Real frames: by default this pulls a handful of frames straight out of the
fly50_V6 project's own reprojection videos
(``projects/fly50_V6/visualization/Videos_3D_*/Cam*.mp4``) -- the only real
imagery from this exact project checked into the repo (the raw training
dataset lives at the training host's ``/home/user/red_data/merge_fly50_V6``,
not present here). Those videos have a thin keypoint-skeleton overlay drawn
near the fly (a handful of small colored dots/lines covering a small corner
of the frame); the rest of the 1936x448 frame is untouched camera pixels.
Preprocessing follows ``jarvis/dataset/datasetBase.py::_load_image`` (BGR->RGB,
float32/255) followed by the project's ``DATASET.MEAN``/``STD`` normalization
(fly50_V6 does not override these, so they are the config.py ImageNet
defaults ``[0.485,0.456,0.406]``/``[0.229,0.224,0.225]``), then resized to
``IMAGE_SIZE`` (320) -- matching ``JarvisMultiAnimalPredictor3D``'s CenterDetect
preprocessing path (``jarvis3D_multi.py``: resize-then-normalize, same op
order used here).

Run inside the JARVIS-HybridNet PyTorch env (``jarvis``), on a COMPUTE NODE::

    python third_party/jarvis_jax/jarvis_jax/convert/export_centerdetect_fixture.py \
        --weights third_party/JARVIS-HybridNet/projects/fly50_V6/models/CenterDetect/Run_20260810-094955/EfficientTrack-medium_final.pth \
        --out third_party/jarvis_jax/jarvis_jax/convert/fixtures/centerdetect_medium.npz

The script imports only ``torch``, ``numpy``, ``cv2`` and the JARVIS ``jarvis``
package (via ``--jarvis-root`` on sys.path). It does NOT import ``jarvis_jax``
(JAX), so it runs in the PyTorch-only env.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

# (video_path_glob, frame_index) pairs, resolved relative to --jarvis-root.
# Spans multiple cameras + times within the one available real-frame source
# (fly50_V6's own reprojection videos -- see module docstring).
_DEFAULT_FRAME_SOURCES = [
    ("projects/fly50_V6/visualization/Videos_3D_20260813-152325/Cam2012630.mp4", 50),
    ("projects/fly50_V6/visualization/Videos_3D_20260813-152325/Cam2012631.mp4", 300),
    ("projects/fly50_V6/visualization/Videos_3D_20260813-152325/Cam2012857.mp4", 150),
    ("projects/fly50_V6/visualization/Videos_3D_20260813-152325/Cam2012862.mp4", 600),
]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_real_frame(path: str, frame_idx: int, image_size: int) -> np.ndarray:
    """Read one real frame, RGB float32/255, resize to (image_size,image_size),
    normalize by (ImageNet) DATASET.MEAN/STD. Returns CHW float32."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {path}")
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(img, (2, 0, 1))  # HWC -> CHW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="EfficientTrack-medium_final.pth (CenterDetect)")
    ap.add_argument("--jarvis-root", default="third_party/JARVIS-HybridNet")
    ap.add_argument("--num-joints", type=int, default=1,
                    help="CENTERDETECT.NUM_JOINTS default (config.py: 1)")
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.jarvis_root)
    from jarvis.efficienttrack.model import EfficientTrackBackbone

    class Cfg:
        """Minimal CENTERDETECT cfg stand-in (backbone reads none of its
        fields in __init__/forward; model_size + output_channels drive the
        architecture, same as export_efficienttrack_fixture.py's Cfg)."""
        MODEL_SIZE = "medium"
        NUM_JOINTS = a.num_joints

    net = EfficientTrackBackbone(Cfg(), model_size="medium",
                                 output_channels=a.num_joints,
                                 in_channels=a.in_channels)
    state_dict = torch.load(a.weights, map_location="cpu")
    net.load_state_dict(state_dict, strict=True)
    net.eval()

    captured = {}

    def _hook(_module, _inp, output):
        for i, f in enumerate(output):
            captured[f"backbone_feat_{i}"] = f.detach().cpu().numpy()

    handle = net.backbone_net.register_forward_hook(_hook)

    frames = []
    resolved_sources = []
    for rel_path, frame_idx in _DEFAULT_FRAME_SOURCES:
        full_path = os.path.join(a.jarvis_root, rel_path)
        matches = glob.glob(full_path)
        if not matches:
            raise FileNotFoundError(full_path)
        frames.append(_load_real_frame(matches[0], frame_idx, a.image_size))
        resolved_sources.append(f"{matches[0]}#{frame_idx}")

    x = torch.from_numpy(np.stack(frames, axis=0))  # (N, in_channels, H, W)
    assert x.shape == (len(frames), a.in_channels, a.image_size, a.image_size), x.shape

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
        "model_size": np.array("medium"),
        "frame_sources": np.array(resolved_sources),
    }
    for k, v in net.state_dict().items():
        out[f"w::{k}"] = v.detach().cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    n_w = sum(1 for k in out if k.startswith("w::"))
    print(f"wrote {a.out}")
    print(f"  frame sources: {resolved_sources}")
    print(f"  input {out['input_nchw'].shape}  res1 {out['res1'].shape}  res2 {out['res2'].shape}")
    print(f"  feat_p3 {out['feat_p3'].shape}  feat_p4 {out['feat_p4'].shape}  feat_p5 {out['feat_p5'].shape}")
    print(f"  {n_w} weight tensors")


if __name__ == "__main__":
    main()
