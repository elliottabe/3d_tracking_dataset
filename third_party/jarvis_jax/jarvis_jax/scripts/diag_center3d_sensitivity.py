"""Diagnostic: how sensitive is HybridNet3D's 3-D reconstruction to center3D error?

HYPOTHESIS under test
----------------------
The reproject cube is centred at ``center3D``. Training/the reprojected-volume
cache used ``center3D`` = the truncated midrange of triangulated GT keypoints
(``V3FramesetDataset``'s ``center3D`` field -- see ``jarvis_jax/data/v3_3d.py``).
At inference (``predict_batch``, see ``jarvis_jax/predict/infer_3d.py``),
``center3D`` instead comes from ``estimate_center3d_from_masks`` -- triangulated
SAM3 mask centroids, which can differ from the GT-keypoint centre. A wrong
``center3D`` shifts where the 2-D heatmaps reproject into the 48-unit cube
(possibly clamping some of them at the cube edge), which could degrade the
V2VNet volume and make the 3-D soft-argmax diffuse.

This script tests that hypothesis directly: on red_data val (which has BOTH a
GT ``center3D`` and GT ``kp3d``), it runs the SAME trained model forward
several times per frameset, each time with ``center3D`` offset from the GT
value by a controlled amount (``delta``), and measures:

  * ``mpjpe_raw``   -- 3-D MPJPE vs GT kp3d in WORLD coords. This necessarily
                       grows with ``delta`` even if the model does nothing
                       wrong, because a shifted ``center3D`` shifts the whole
                       predicted skeleton by the same rigid offset (world
                       coords = local + center3D). It's reported for context
                       only -- it does NOT isolate a volume/localization
                       problem from the trivial shift.
  * ``mpjpe_local`` -- 3-D MPJPE in CUBE-LOCAL coords: ``(pred - used_center3D)``
                       vs ``(gt_kp3d - GT_center3D)``. This subtracts the
                       rigid shift, so it measures localization quality
                       independent of ``delta``.
  * ``spread``      -- bbox diagonal of the 50 predicted keypoints for one
                       frameset. Translation-invariant by construction (a
                       rigid shift of every point does not change the bbox
                       diagonal), so this is a DIRECT measure of how diffuse/
                       spread-out the predicted skeleton is, independent of
                       any world-coordinate shift.

INTERPRETATION
---------------
If ``spread`` grows sharply with ``delta`` (e.g. spread at delta=10 much
bigger than at delta=0), a wrong ``center3D`` degrades the V2VNet volume /
soft-argmax localization itself -- CONFIRMS the hypothesis. If ``spread`` is
roughly flat across ``delta``, center3D error is NOT the cause of diffuse 3-D
predictions -- look elsewhere (OOD input / under-trained V2VNet). The script
prints this verdict automatically (see ``print_report``).

ALSO (cheap, included): quantifies the ACTUAL center3D discrepancy on val --
V3FramesetDataset's GT center3D already carries the SAME per-image SAM3 mask
data (crops4's channel-3) that ``estimate_center3d_from_masks`` consumes, so
no courtship packed-mask format is needed here; this comparison is computed
for every frameset with >=2 valid mask cameras.

Model loading reuses ``jarvis_jax.predict.infer_3d.load_inference_model``
(the exact production load path -- see ``predict_3d.py``/
``test_infer_3d_efficienttrack_bn.py``). Unlike ``predict_batch`` (which
always computes ``center3D`` internally from masks), this script calls
``HybridNet3D.__call__`` directly on host arrays with an EXPLICIT
``center3D`` -- see ``run_forward`` -- bypassing the mask-centroid estimate
entirely so each delta is an exact, controlled perturbation of the true GT
``center3D``. No batch sharding is used (one frameset, B=1, at a time): the
model was already replicated across the mesh by ``load_inference_model``, but
plain host-array forward calls work fine on any device topology and keep this
script simple (per module docstring in infer_3d.py: "If sharding is awkward,
call model.__call__ directly on host arrays").

CLI (Hydra; mirrors eval_keypoints_2d.py / mine_hard_frames.py). All knobs are
ad-hoc ``+diag.*`` keys (not a registered config group, same convention as
``+eval.*``/``+mine.*`` in those two scripts)::

    python -m jarvis_jax.scripts.diag_center3d_sensitivity \\
        paths=hyak \\
        +diag.front_end=efficienttrack_bn \\
        +diag.efficienttrack_ckpt=/gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs/et2d_bn_imagenet/final \\
        +diag.v2v_final_dir=/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/v2v_etbn/final \\
        +diag.n=40

Optional overrides (each a NEW key under the ad-hoc ``diag.*`` node, so also
``+``-prefixed unless noted): ``diag.num_keypoints`` (default 50),
``diag.sharpen`` (default 3.0), ``diag.data_root`` (default
``paths.data_root``), ``diag.n`` (default 40 framesets), ``diag.deltas``
(default ``[0.0, 3.0, 6.0, 10.0, 15.0]``), ``diag.direction`` (default
``[1.0, 0.0, 0.0]`` -- unit-normalized internally), ``diag.out`` (optional
``.npz`` path to save the summary). For the ViTPose arm instead:
``+diag.front_end=vitpose`` with ``diag.vitpose_ckpt`` (default
``paths.vitpose_ckpt``) and ``diag.v2v_vitpose`` (default
``.../jax_cached3d_runs/run4/final``, the ViTPose+run4 arm from
``tests/test_infer_3d.py``).
"""
import hydra
import jax.numpy as jnp
import numpy as np

from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers

register_resolvers()

from jarvis_jax.data.v3_3d import V3FramesetDataset
from jarvis_jax.geometry.center3d import estimate_center3d_from_masks
from jarvis_jax.predict.infer_3d import load_inference_model

DEFAULT_ET_BN_CKPT = ("/gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs/"
                      "et2d_bn_imagenet/final")
DEFAULT_V2V_ETBN = "/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/v2v_etbn/final"
DEFAULT_V2V_VITPOSE = "/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/run4/final"
DEFAULT_DELTAS = (0.0, 3.0, 6.0, 10.0, 15.0)
DEFAULT_DIRECTION = (1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Pure-python metrics (unit-tested against a stub model in
# tests/test_diag_center3d.py, no real checkpoint needed).
# ---------------------------------------------------------------------------

def keypoint_spread(points) -> float:
    """Bbox diagonal of a set of ``(J, 3)`` points -- a translation-invariant
    diffuseness metric: shifting every point by the same rigid offset leaves
    ``max - min`` per axis unchanged, so this is independent of any
    ``center3D`` world-space shift."""
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def mpjpe(pred, gt, vis) -> float:
    """Mean per-joint 3-D error, masked by ``vis`` (both ``(J, 3)``,
    ``vis`` ``(J,)`` bool/0-1). NaN if no joint is visible."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    d = np.linalg.norm(pred - gt, axis=-1)          # (J,)
    w = np.asarray(vis, dtype=np.float64)           # (J,)
    denom = w.sum()
    if denom <= 0:
        return float("nan")
    return float((d * w).sum() / denom)


def perturb_center3d(center3D, delta, direction=DEFAULT_DIRECTION):
    """Return ``center3D + delta * unit(direction)`` (float64, shape (3,)).

    ``direction`` is normalized internally, so any non-zero vector may be
    passed (e.g. an un-normalized axis). A zero vector (or delta=0) is a
    no-op -- returns ``center3D`` unchanged (up to dtype)."""
    center3D = np.asarray(center3D, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm > 0:
        d = d / norm
    return center3D + float(delta) * d


def run_forward(model, crops4, center3D, centerHM, cameraMatrices):
    """Call ``HybridNet3D.__call__`` (or a stub with the same contract)
    directly on host arrays with an EXPLICIT ``center3D`` -- bypasses
    ``predict_batch``'s internal ``estimate_center3d_from_masks`` so the
    caller controls ``center3D`` exactly. No sharding: single-frameset (B=1)
    calls are cheap enough for a diagnostic sweep.

    Args:
        model: a HybridNet3D (or test stub) with the same ``__call__``
            contract: ``model(crops, center3D, centerHM, cameraMatrices,
            use_running_average=True) -> (vol, points3D, conf)``. If the
            model exposes ``_frontend_channels`` (set by
            ``load_inference_model``), the crop's channel axis is sliced to
            match -- mirrors ``predict_batch``'s slicing.
        crops4:         ``(B, num_cam, 448, 448, 4)`` uint8/float array.
        center3D:       ``(B, 3)`` array -- the EXPLICIT center to reproject
                        around (already perturbed by the caller, if desired).
        centerHM:       ``(B, num_cam, 2)`` array.
        cameraMatrices: ``(B, num_cam, 4, 3)`` array.

    Returns:
        points3D: ``(B, J, 3)`` float32 world-space keypoints.
        conf:     ``(B, J)`` float32 confidence.
    """
    frontend_channels = getattr(model, "_frontend_channels", None)
    crops = np.asarray(crops4)
    if frontend_channels is not None:
        crops = crops[..., :frontend_channels]
    _vol, points3D, conf = model(
        jnp.asarray(crops),
        jnp.asarray(np.asarray(center3D), dtype=jnp.float32),
        jnp.asarray(np.asarray(centerHM), dtype=jnp.float32),
        jnp.asarray(np.asarray(cameraMatrices), dtype=jnp.float32),
        use_running_average=True,
    )
    return np.asarray(points3D), np.asarray(conf)


def sweep_center3d_sensitivity(model, dataset, *, n=40, deltas=DEFAULT_DELTAS,
                               direction=DEFAULT_DIRECTION, quiet=False):
    """Run the delta sweep over the first ``n`` framesets of ``dataset``.

    For each frameset: compute the ACTUAL center3D discrepancy (GT vs
    ``estimate_center3d_from_masks`` on the frameset's own SAM3 mask
    channel), then for each ``delta`` in ``deltas``, perturb the GT
    ``center3D`` along ``direction`` by ``delta`` and run the model forward
    with that explicit center3D (see ``run_forward``), recording
    ``mpjpe_raw``, ``mpjpe_local``, and ``spread`` (see module docstring for
    definitions).

    Args:
        model:   HybridNet3D (or stub) -- see ``run_forward``.
        dataset: anything supporting ``len()`` and ``__getitem__(i)`` that
            returns a dict with keys ``crops4``, ``centerHM``, ``center3D``,
            ``cameraMatrices``, ``kp3d``, ``vis`` (see
            ``jarvis_jax.data.v3_3d.V3FramesetDataset``).
        n:       number of framesets to process (clamped to ``len(dataset)``).
        deltas:  offset magnitudes to sweep (world units); include 0.0 as
            the reference (GT center3D -- what training/caching used).
        direction: fixed offset direction (normalized internally).
        quiet:   suppress the printed table/verdict.

    Returns:
        dict with keys ``summary`` (per-delta dict of mean
        mpjpe_raw/mpjpe_local/spread), ``n`` (framesets actually processed),
        and ``center3d_discrepancy`` (list of ||GT - estimated|| center3D
        errors, one per frameset with >=2 valid SAM3 mask cameras -- may be
        empty if none qualify).
    """
    n = min(n, len(dataset))
    per_delta = {d: {"mpjpe_raw": [], "mpjpe_local": [], "spread": []} for d in deltas}
    center3d_discrepancy = []

    for i in range(n):
        sample = dataset[i]
        crops4 = np.asarray(sample["crops4"])[None]             # (1, nc, 448, 448, 4)
        centerHM = np.asarray(sample["centerHM"])[None]         # (1, nc, 2)
        cameraMatrices = np.asarray(sample["cameraMatrices"])[None]  # (1, nc, 4, 3)
        gt_center3D = np.asarray(sample["center3D"], dtype=np.float64)  # (3,)
        gt_kp3d = np.asarray(sample["kp3d"], dtype=np.float64)   # (J, 3)
        vis = np.asarray(sample["vis"])                          # (J,)
        gt_local = gt_kp3d - gt_center3D

        # Cheap ACTUAL center3D discrepancy check: crops4's channel-3 mask IS
        # the per-image SAM3 mask V3FramesetDataset already loaded (see
        # jarvis_jax/data/v3_3d.py::_load_mask) -- the same format
        # estimate_center3d_from_masks expects. No courtship packed-mask
        # format needed.
        est_center3D, n_valid = estimate_center3d_from_masks(
            crops4, centerHM, cameraMatrices)
        if n_valid[0] >= 2:
            center3d_discrepancy.append(
                float(np.linalg.norm(est_center3D[0] - gt_center3D)))

        for delta in deltas:
            used_center3D = perturb_center3d(gt_center3D, delta, direction)
            points3D, _conf = run_forward(
                model, crops4, used_center3D[None], centerHM, cameraMatrices)
            pred = points3D[0]                        # (J, 3) world coords
            pred_local = pred - used_center3D          # shift-corrected (cube-local)

            per_delta[delta]["mpjpe_raw"].append(mpjpe(pred, gt_kp3d, vis))
            per_delta[delta]["mpjpe_local"].append(mpjpe(pred_local, gt_local, vis))
            per_delta[delta]["spread"].append(keypoint_spread(pred))

    summary = {}
    for delta in deltas:
        rec = per_delta[delta]
        summary[delta] = {
            "mpjpe_raw_mean": float(np.nanmean(rec["mpjpe_raw"])) if rec["mpjpe_raw"] else float("nan"),
            "mpjpe_local_mean": float(np.nanmean(rec["mpjpe_local"])) if rec["mpjpe_local"] else float("nan"),
            "spread_mean": float(np.mean(rec["spread"])) if rec["spread"] else float("nan"),
        }

    result = {"summary": summary, "n": n, "center3d_discrepancy": center3d_discrepancy}
    if not quiet:
        print_report(result)
    return result


def print_report(result) -> None:
    summary = result["summary"]
    n = result["n"]
    center3d_discrepancy = result["center3d_discrepancy"]

    print(f"center3D-sensitivity sweep (n={n} framesets)")
    print(f"{'delta':>8} {'mpjpe_raw':>12} {'mpjpe_local':>13} {'spread':>10}   (world units)")
    for delta in sorted(summary):
        rec = summary[delta]
        print(f"{delta:8.1f} {rec['mpjpe_raw_mean']:12.3f} {rec['mpjpe_local_mean']:13.3f} "
              f"{rec['spread_mean']:10.3f}")

    deltas_sorted = sorted(summary)
    base_spread = summary[deltas_sorted[0]]["spread_mean"]
    max_spread = max(rec["spread_mean"] for rec in summary.values())
    growth = ((max_spread - base_spread) / base_spread) if base_spread > 0 else float("inf")

    print()
    if growth > 0.5:
        print(f"INTERPRETATION: spread grows sharply with delta "
              f"({base_spread:.2f} -> {max_spread:.2f}, +{growth * 100:.0f}%). "
              "center3D error DOES cause diffuse 3-D reconstruction -- "
              "CONFIRMS the hypothesis.")
    else:
        print(f"INTERPRETATION: spread is roughly flat vs delta "
              f"({base_spread:.2f} -> {max_spread:.2f}, +{growth * 100:.0f}%). "
              "center3D error does NOT explain the diffuseness -- look "
              "elsewhere (OOD input / under-trained V2VNet).")

    print()
    if center3d_discrepancy:
        arr = np.asarray(center3d_discrepancy)
        print(f"actual center3D discrepancy (GT vs estimate_center3d_from_masks), "
              f"n={arr.size}: mean={arr.mean():.3f} median={np.median(arr):.3f} "
              f"p90={np.percentile(arr, 90):.3f} max={arr.max():.3f}  (world units)")
    else:
        print("actual center3D discrepancy: SKIPPED (no frameset had >=2 valid "
              "SAM3 mask cameras for triangulation)")


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------

def build_model(cfg):
    """Load the HybridNet3D model for the arm selected by ``cfg.diag.*``.

    Reuses ``load_inference_model`` exactly (the production load path) --
    see module docstring for the two supported arms.
    """
    diag = cfg.get("diag", {})
    front_end = diag.get("front_end", "efficienttrack_bn")
    num_keypoints = int(diag.get("num_keypoints", 50))
    sharpen = float(diag.get("sharpen", 3.0))

    if front_end == "vitpose":
        vitpose_ckpt = diag.get("vitpose_ckpt", cfg.paths.vitpose_ckpt)
        v2v_final_dir = diag.get("v2v_vitpose", DEFAULT_V2V_VITPOSE)
        return load_inference_model(
            vitpose_ckpt, v2v_final_dir, sharpen=sharpen,
            num_keypoints=num_keypoints, front_end="vitpose")

    efficienttrack_ckpt = diag.get("efficienttrack_ckpt", DEFAULT_ET_BN_CKPT)
    v2v_final_dir = diag.get("v2v_final_dir", DEFAULT_V2V_ETBN)
    return load_inference_model(
        None, v2v_final_dir, sharpen=sharpen, num_keypoints=num_keypoints,
        front_end=front_end, efficienttrack_ckpt=efficienttrack_ckpt)


def main_from_cfg(cfg):
    diag = cfg.get("diag", {})
    root = diag.get("data_root", cfg.paths.data_root)
    n = int(diag.get("n", 40))
    deltas = tuple(float(x) for x in diag.get("deltas", DEFAULT_DELTAS))
    direction = tuple(float(x) for x in diag.get("direction", DEFAULT_DIRECTION))

    model = build_model(cfg)
    ds = V3FramesetDataset(root, "val")
    print(f"root: {root}")
    print(f"val framesets available: {len(ds)}  (using first {min(n, len(ds))})")

    result = sweep_center3d_sensitivity(model, ds, n=n, deltas=deltas, direction=direction)

    out = diag.get("out", None)
    if out:
        np.savez(
            str(out),
            deltas=np.asarray(sorted(result["summary"])),
            mpjpe_raw=np.asarray([result["summary"][d]["mpjpe_raw_mean"] for d in sorted(result["summary"])]),
            mpjpe_local=np.asarray([result["summary"][d]["mpjpe_local_mean"] for d in sorted(result["summary"])]),
            spread=np.asarray([result["summary"][d]["spread_mean"] for d in sorted(result["summary"])]),
            n=result["n"],
            center3d_discrepancy=np.asarray(result["center3d_discrepancy"]),
        )
        print(f"saved: {out}")

    return result


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
