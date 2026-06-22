"""Dump a ReprojectionLayer parity fixture from the PyTorch JARVIS HybridNet.

Run ONCE in the torch `jarvis` env:
    conda activate jarvis
    LD_LIBRARY_PATH=/gscratch/portia/eabe/miniconda3/envs/jarvis/lib \\
      python -m jarvis_jax.convert.export_reproject_fixture \\
        --project unified_V3_masked --out jarvis_jax/convert/reproject_fixture.npz

The output npz is committed and consumed by tests/test_reproject_parity.py in the
JAX env, which never imports torch.

Dataset3D.__getitem__ returns:
  [img_l, keypoints3D, centerHM, center3D, heatmap3D, cameraMatrices, datasetName]
  idx:  0          1          2         3          4             5             6
After Normalizer transform the list is the same order.
The DataLoader cannot collate the string at index 6, so we index the dataset
directly and add a batch dimension manually.
"""
import argparse
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="unified_V3_masked")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from jarvis.config.project_manager import ProjectManager
    from jarvis.hybridnet.repro_layer import ReprojectionLayer
    from jarvis.dataset.dataset3D import Dataset3D

    pm = ProjectManager()
    assert pm.load(args.project), f"could not load project {args.project}"
    cfg = pm.get_cfg()
    layer = ReprojectionLayer(cfg).cuda()

    # Pull one real frameset's geometry from Dataset3D.
    # dataset3D.__getitem__ returns:
    #   [img_l, keypoints3D, centerHM, center3D, heatmap3D, cameraMatrices, datasetName]
    # The DataLoader cannot auto-collate the string at index 6, so we index
    # directly and add a batch=1 dimension manually.
    ds = Dataset3D(cfg, set="val")
    sample = ds[0]

    # sample[2] = centerHM  shape: (num_cameras, 2)
    # sample[3] = center3D  shape: (3,)
    # sample[5] = cameraMatrices  shape: (num_cameras, 4, 3)
    centerHM = torch.from_numpy(
        np.array(sample[2]).astype("float32")).unsqueeze(0).cuda()   # (1, num_cam, 2)
    center3D = torch.from_numpy(
        np.array(sample[3]).astype("float32")).unsqueeze(0).cuda()   # (1, 3)
    camM = torch.from_numpy(
        np.array(sample[5]).astype("float32")).unsqueeze(0).cuda()   # (1, num_cam, 4, 3)

    rng = np.random.RandomState(0)
    num_cam = cfg.HYBRIDNET.NUM_CAMERAS
    J = cfg.KEYPOINTDETECT.NUM_JOINTS
    hm_size = int(cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE / 2 + 2)
    # Synthetic but well-formed per-camera heatmaps (batch 1).
    heatmaps = torch.from_numpy(
        rng.rand(1, num_cam, J, hm_size, hm_size).astype("float32")).cuda()

    grid_size = int(cfg.HYBRIDNET.ROI_CUBE_SIZE / cfg.HYBRIDNET.GRID_SPACING)
    print(f"num_cam={num_cam}, J={J}, hm_size={hm_size}, grid_size={grid_size}")
    print(f"heatmaps shape:      {tuple(heatmaps.shape)}")
    print(f"center3D shape:      {tuple(center3D.shape)}")
    print(f"centerHM shape:      {tuple(centerHM.shape)}")
    print(f"cameraMatrices shape:{tuple(camM.shape)}")

    with torch.no_grad():
        out = layer(heatmaps, center3D, centerHM, camM)   # (1, J, 48, 48, 48)

    print(f"heatmaps3D shape:    {tuple(out.shape)}")

    np.savez(
        args.out,
        heatmaps=heatmaps.cpu().numpy(),
        center3D=center3D.cpu().numpy(),
        centerHM=centerHM.cpu().numpy(),
        cameraMatrices=camM.cpu().numpy(),
        heatmaps3D=out.cpu().numpy(),
        grid_size=np.int64(grid_size),
        grid_spacing=np.int64(cfg.HYBRIDNET.GRID_SPACING),
        roi_cube=np.int64(cfg.HYBRIDNET.ROI_CUBE_SIZE),
        heatmap_size=np.int64(hm_size),
    )
    print(f"wrote {args.out}  heatmaps3D {out.shape}")


if __name__ == "__main__":
    main()
