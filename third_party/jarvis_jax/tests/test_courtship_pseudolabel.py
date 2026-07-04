import numpy as np
from jarvis_jax.cse.courtship_pseudolabel import gate_pseudolabels, GateCfg

def test_gates_frame_and_consensus():
    T,C,K = 2,2,3
    mesh2d = np.zeros((T,C,K,2), float)            # all at origin
    det = np.zeros((T,C,K,2), float)
    det[0,0,2] = [100,100]                          # kp2 in (t0,c0) far from mesh -> consensus fail
    conf = np.ones((T,C,K), float)
    conf[0,1,1] = 0.1                               # low conf -> Gate B drop
    qc_pf = {"soft_iou": np.array([0.5, 0.001]),    # frame1 fails Gate A (iou<tau)
             "reproj_px": np.array([5.0, 5.0]),
             "cont_px": np.array([1.0, 1.0])}
    mv = np.ones((T,C), bool)
    cfg = GateCfg(tau_iou=0.05, consensus_px=25.0, tau_conf=0.5, tau_reproj=30.0, tau_cont=8.0)
    labels, keep = gate_pseudolabels(mesh2d, det, conf, qc_pf, mv, cfg=cfg)
    assert keep.tolist() == [True, False]           # frame1 dropped by Gate A
    assert labels[0,0,2,2] == 0                      # consensus fail -> vis 0
    assert labels[0,1,1,2] == 0                      # low conf -> vis 0
    assert labels[0,0,0,2] == 1                      # agree + conf + frame ok -> vis 1
    assert (labels[1,:,:,2] == 0).all()             # dropped frame -> all vis 0
