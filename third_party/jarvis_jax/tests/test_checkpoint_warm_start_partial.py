import numpy as np, jax, orbax.checkpoint as ocp
from flax import nnx


def _tiny(n_inst):
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    cfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=n_inst,
                    n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                    refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                    backbone_heads=4, remat=False)
    return MVQModel(cfg, rngs=nnx.Rngs(0))


def test_warm_start_partial_copies_matching_leaves_and_reports_the_rest(tmp_path):
    from jarvis_jax.train.checkpoint import warm_start_partial
    src = _tiny(3)
    # make the source distinguishable from a fresh init
    src.decoder.e_inst.value = src.decoder.e_inst.value + 1.0
    src.decoder.heads.exist.kernel.value = src.decoder.heads.exist.kernel.value + 2.0
    ck = ocp.StandardCheckpointer(); ck.save(str(tmp_path / "final"), nnx.split(src)[1]); ck.wait_until_finished()
    dst = _tiny(4)
    fresh_row3 = np.asarray(dst.decoder.e_inst.value[3]).copy()
    fresh_sex = np.asarray(dst.decoder.heads.sex.kernel.value).copy()
    dst, skipped = warm_start_partial(dst, str(tmp_path / "final"))
    np.testing.assert_allclose(np.asarray(dst.decoder.heads.exist.kernel.value), np.asarray(src.decoder.heads.exist.kernel.value))
    np.testing.assert_allclose(np.asarray(dst.decoder.e_inst.value[:3]), np.asarray(src.decoder.e_inst.value))
    np.testing.assert_allclose(np.asarray(dst.decoder.e_inst.value[3]), fresh_row3)
    np.testing.assert_allclose(np.asarray(dst.decoder.heads.sex.kernel.value), fresh_sex)
    assert sorted(skipped) == sorted(["decoder/e_inst (partial rows 0:3)", "decoder/heads/sex/bias", "decoder/heads/sex/kernel"])
