import os
import numpy as np
import pytest

MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"

BASE_50 = [
    "Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "EyeL", "EyeR",
    "WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13", "Abd_A4", "Abd_tip",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
]

_HAVE_MESH = os.path.exists(MESH)


@pytest.mark.skipif(not _HAVE_MESH, reason="canonical mesh not present")
def test_dense_swap_is_involution_full_length():
    from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap
    swap = build_dense_lr_swap(MESH, "fps_300", BASE_50)
    assert swap.shape == (50 + 300,)
    assert swap.dtype == np.int32
    assert np.array_equal(swap[swap], np.arange(50 + 300)), "dense swap is not an involution"
    # the first 50 entries are exactly the named-keypoint swap.
    from jarvis_jax.data.augment import build_lr_swap
    assert np.array_equal(swap[:50], build_lr_swap(BASE_50))
    # dense block indexes only into the dense block (50..350), never into the kp block.
    assert swap[50:].min() >= 50 and swap[50:].max() < 350


@pytest.mark.skipif(not _HAVE_MESH, reason="canonical mesh not present")
def test_named_kp_left_maps_to_right():
    from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap
    swap = build_dense_lr_swap(MESH, "fps_300", BASE_50)
    i_L = BASE_50.index("WingL_V12"); i_R = BASE_50.index("WingR_V12")
    assert swap[i_L] == i_R and swap[i_R] == i_L
    i_eL = BASE_50.index("EyeL"); i_eR = BASE_50.index("EyeR")
    assert swap[i_eL] == i_eR
    assert swap[BASE_50.index("Scutellum")] == BASE_50.index("Scutellum")  # midline self


@pytest.mark.skipif(not _HAVE_MESH, reason="canonical mesh not present")
def test_left_wing_vertex_maps_to_a_right_wing_vertex():
    from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap
    z = np.load(MESH, allow_pickle=True)
    fps = z["fps_300"]; seg = z["vertex_segment"][fps]   # per-fps segment id
    swap = build_dense_lr_swap(MESH, "fps_300", BASE_50)
    wl = np.where(seg == 9)[0]                             # wing_left fps verts
    assert len(wl) > 0
    partners_dense = swap[50 + wl] - 50                    # dense-local partner idx
    assert np.all(seg[partners_dense] == 10), "a left-wing vertex must map into wing_right (seg 10)"
    # and it is a genuine swap (not self) for the wing verts.
    assert np.all(partners_dense != wl)


@pytest.mark.skipif(not _HAVE_MESH, reason="canonical mesh not present")
def test_no_lateral_dense_vertex_self_maps():
    """Final-review regression test (Critical finding).

    _dense_vertex_swap used to leave ~26/300 dense verts self-mapped even
    though they were LATERAL (off-midline, |Y| >= midline_tol) vertices. Since
    flip_batch unconditionally mirrors x for every channel, a self-mapped
    lateral vertex's flip-augmented target lands on the WRONG side of the
    midline every time flip fires (~50% of batches) -- this is the root cause
    of the 3x keypoint MPJPE regression. Only TRUE MIDLINE verts (|Y| <
    midline_tol) may self-map; every lateral vertex must have a genuine mirror
    partner.

    This test FAILS on the pre-fix code: the old `_dense_vertex_swap` gated
    pairing on mutual-NN-within-tol AND same-segment-consistency, which left
    26 lateral fps_300 vertices (in abdomen/leg/thorax segments, |Y| up to
    ~0.087) unpaired and therefore self-mapped, well above midline_tol=0.02.
    Deliberately calls `build_dense_lr_swap` WITHOUT the new `midline_tol`
    kwarg (using the module's `_MIRROR_AXIS` and a locally-hardcoded
    midline_tol=0.02 for the assertion only) so this test exercises the actual
    pairing logic rather than merely the new keyword argument's presence.
    """
    from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap, _MIRROR_AXIS

    midline_tol = 0.02
    swap = build_dense_lr_swap(MESH, "fps_300", BASE_50)

    z = np.load(MESH, allow_pickle=True)
    fps = z["fps_300"]
    V = z["vertices"][fps]

    dense = swap[50:] - 50                       # dense-local swap targets
    self_mapped = np.where(dense == np.arange(len(dense)))[0]
    assert len(self_mapped) > 0, "expected some true midline verts to self-map"

    lateral_self_mapped = self_mapped[np.abs(V[self_mapped, _MIRROR_AXIS]) >= midline_tol]
    assert len(lateral_self_mapped) == 0, (
        f"{len(lateral_self_mapped)} LATERAL dense vertex/vertices self-map "
        f"(|Y| >= {midline_tol}): {lateral_self_mapped.tolist()} -- these would "
        "land on the wrong side of the midline under flip augmentation"
    )

    # every self-mapped vertex is a genuine midline vertex.
    assert np.all(np.abs(V[self_mapped, _MIRROR_AXIS]) < midline_tol)

    # sanity: still an involution, and first-50 unchanged (don't let the fix
    # regress the existing contract).
    assert np.array_equal(swap[swap], np.arange(len(swap)))
    from jarvis_jax.data.augment import build_lr_swap
    assert np.array_equal(swap[:50], build_lr_swap(BASE_50))


@pytest.mark.skipif(not _HAVE_MESH, reason="canonical mesh not present")
def test_flipping_a_synthetic_example_swaps_L_and_R():
    """End-to-end: apply the dense swap the way flip_batch does (kp[:, swap]) and
    confirm a synthetic labeled example with distinct L/R coords swaps correctly."""
    from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap
    swap = build_dense_lr_swap(MESH, "fps_300", BASE_50)
    J = 50 + 300
    kp = np.arange(J * 2).reshape(J, 2).astype(np.float32)  # unique per-channel coords
    kp_sw = kp[swap]
    # applying the involution twice restores the original (what flip-then-flip does).
    assert np.array_equal(kp_sw[swap], kp)
    # a specific left-wing vertex's coords land on its right-wing partner's slot.
    z = np.load(MESH, allow_pickle=True); seg = z["vertex_segment"][z["fps_300"]]
    wl = int(np.where(seg == 9)[0][0]); partner = int(swap[50 + wl] - 50)
    assert np.array_equal(kp_sw[50 + partner], kp[50 + wl])
