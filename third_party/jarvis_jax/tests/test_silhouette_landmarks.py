import numpy as np
from jarvis_jax.cse.affine_camera import factor_affine, reconstruct_affine, project_affine
from jarvis_jax.cse.silhouette_landmarks import wing_side_vertices, mask_wing_tip_2d, triangulate_wing_tips

MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
P_REAL = np.array([[8.1001,0.0074869,-0.031773,-2.828],[0.0093308,-8.0788,-0.17912,462.78],[0,0,0,1.0]])


def test_wing_side_vertices_are_distinct_and_in_range():
    import numpy as np
    z = np.load(MESH, allow_pickle=True); nv = len(z["fps_300"])
    sides = wing_side_vertices(MESH)
    assert set(sides) == {"left", "right"}
    for s in ("left", "right"):
        assert 0 <= sides[s]["tip"] < nv and 0 <= sides[s]["prox"] < nv
        assert sides[s]["tip"] != sides[s]["prox"]


def test_mask_wing_tip_2d_finds_far_edge():
    # a horizontal bar mask from x=100..180 at y=50; axis points +x from prox (90,50)
    mask = np.zeros((100, 300), bool); mask[48:53, 100:181] = True
    tip = mask_wing_tip_2d(mask, np.array([90.0, 50.0]), np.array([140.0, 50.0]), corridor=6.0)
    assert tip is not None and abs(tip[0] - 180) <= 1 and abs(tip[1] - 50) <= 2


def test_triangulate_wing_tips_recovers_known_tip():
    # 2-camera affine rig; place a known left-wing tip; render bar masks; triangulate.
    # NOTE: world points are offset by [120, 30, 0] mm so they land inside P_REAL's
    # actual (448, 1936) field of view -- P_REAL's x-row has a -2.828 translation and
    # ~8x magnification, so points near the world origin (as in a first draft of this
    # test) project to negative/off-frame pixel coordinates in both cameras and paint
    # an empty mask. [120, 30, 0] is the mid-frame world point at z=11 for P_REAL
    # (solved from project_affine(P_REAL, [x,y,11]) == [968, 224], the image center).
    K2, R, t = factor_affine(P_REAL)
    th = np.deg2rad(20.0); Ry = np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
    cams = [P_REAL, reconstruct_affine(K2, Ry @ R, t)]
    shift = np.array([120.0, 30.0, 0.0])
    prox3d = {"left": np.array([0.0, 0.5, 11.0]) + shift, "right": np.array([0.0, -0.5, 11.0]) + shift}
    tip3d_true = {"left": np.array([0.0, 2.5, 11.0]) + shift, "right": np.array([0.0, -2.5, 11.0]) + shift}
    masks = []
    for P in cams:
        m = np.zeros((448, 1936), bool)
        for side in ("left", "right"):
            a = project_affine(P, prox3d[side]); b = project_affine(P, tip3d_true[side])
            # Dense sampling + round() (not truncation) + a thin +-1px box: truncating
            # (int()) and thick (+-3px) boxes at 60 samples bias the painted "far edge"
            # up to 3*sqrt(2)~4.2 px beyond the true tip (the box corner, not its
            # center, maximizes distance along the axis), which is ~0.5mm at this
            # magnification -- enough to blow the < 0.3mm tolerance below.
            for f in np.linspace(0, 1, 400):
                p = np.round(a + f * (b - a)).astype(int)
                m[max(0,p[1]-1):p[1]+1, max(0,p[0]-1):p[0]+1] = True
        masks.append(m)
    out = triangulate_wing_tips(masks, cams, prox3d, tip3d_true, corridor=8.0)
    assert out["left"] is not None
    X, ncam = out["left"]
    assert ncam == 2 and np.linalg.norm(X - tip3d_true["left"]) < 0.3
