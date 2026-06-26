import numpy as np
from jarvis_jax.data.augment import build_lr_swap

V3_NAMES = ['Antenna_Base', 'EyeL', 'EyeR', 'Scutellum', 'Abd_A4', 'Abd_tip',
            'WingL_base', 'WingL_V12', 'WingL_V13', 'T1L_ThxCx', 'T1L_Tro',
            'T1L_FeTi', 'T1L_TiTa', 'T1L_TaT1', 'T1L_TaT3', 'T1L_TaTip',
            'T2L_Tro', 'T2L_FeTi', 'T2L_TiTa', 'T2L_TaT1', 'T2L_TaT3', 'T2L_TaTip',
            'T3L_Tro', 'T3L_FeTi', 'T3L_TiTa', 'T3L_TaT1', 'T3L_TaT3', 'T3L_TaTip',
            'WingR_base', 'WingR_V12', 'WingR_V13', 'T1R_ThxCx', 'T1R_Tro',
            'T1R_FeTi', 'T1R_TiTa', 'T1R_TaT1', 'T1R_TaT3', 'T1R_TaTip',
            'T2R_Tro', 'T2R_FeTi', 'T2R_TiTa', 'T2R_TaT1', 'T2R_TaT3', 'T2R_TaTip',
            'T3R_Tro', 'T3R_FeTi', 'T3R_TiTa', 'T3R_TaT1', 'T3R_TaT3', 'T3R_TaTip']


def test_build_lr_swap_pairs_and_involution():
    sw = build_lr_swap(V3_NAMES)
    idx = {n: i for i, n in enumerate(V3_NAMES)}
    # L<->R pairs
    assert sw[idx['EyeL']] == idx['EyeR'] and sw[idx['EyeR']] == idx['EyeL']
    assert sw[idx['T1L_TaTip']] == idx['T1R_TaTip']
    assert sw[idx['WingL_base']] == idx['WingR_base']
    # midline keypoints map to themselves
    for m in ('Antenna_Base', 'Scutellum', 'Abd_A4', 'Abd_tip'):
        assert sw[idx[m]] == idx[m]
    # involution covering all K
    assert np.array_equal(sw[sw], np.arange(len(V3_NAMES)))
    assert sw.dtype == np.int32
