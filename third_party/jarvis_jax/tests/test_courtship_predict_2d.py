import numpy as np
import jax.numpy as jnp
from jarvis_jax.cse.courtship_predict_2d import peaks_and_conf
from jarvis_jax.geometry.center3d import centroids_to_fullpx


def test_peaks_and_conf_locates_peak_and_reports_value():
    # one 224x224 heatmap, K=2, sharp peaks at known locations
    hm = np.full((1, 224, 224, 2), -5.0, np.float32)
    hm[0, 50, 30, 0] = 9.0     # (row=50,col=30) -> in_size=448 => (x=60,y=100)
    hm[0, 100, 200, 1] = 4.0
    kp, conf = peaks_and_conf(jnp.asarray(hm))
    kp = np.asarray(kp); conf = np.asarray(conf)
    assert abs(kp[0, 0, 0] - 60) < 2 and abs(kp[0, 0, 1] - 100) < 2   # x=col*2, y=row*2
    assert conf[0, 0] > conf[0, 1] > 0                                # peak values, relu'd


def test_crop_to_fullframe_mapping():
    # centerHM = crop center in full px; kp at crop center -> full == centerHM
    kp_crop = np.array([[[224.0, 224.0]]])          # (nc=1,K=1,2)
    centerHM = np.array([[900.0, 300.0]])           # (nc=1,2)
    full = np.asarray(centroids_to_fullpx(kp_crop, centerHM, 448))
    assert np.allclose(full[0, 0], [900.0, 300.0])
