"""Unified viz CLI:  python -m viz <subcommand> [options]."""
import argparse

def _add_shared(sp):
    sp.add_argument("--run", help="run root (…/Session0_bouts_<date>)")
    sp.add_argument("--bout", type=int)
    sp.add_argument("--fly", type=int, default=0)
    sp.add_argument("--frame", type=int, default=0)
    sp.add_argument("--cams", nargs="*", default=None)
    sp.add_argument("--out", default=None)

def _overlay(args):     from viz.views import overlay;      return overlay.run(args)
def _legskel(args):     from viz.views import legskel;      return legskel.run(args)
def _kp_qc(args):       from viz.views import kp_qc;        return kp_qc.run(args)
def _fit_check(args):   from viz.views import fit_check;    return fit_check.run(args)
def _reproj_video(args):from viz.views import reproj_video; return reproj_video.run(args)
def _clip(args):        from viz.views import clip;         return clip.run(args)

def build_parser():
    p = argparse.ArgumentParser(prog="viz", description="Centralized 3d_tracking visualizations")
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("overlay", help="mesh+kp+mask reproj overlay (per-camera montage)")
    _add_shared(o); o.add_argument("--show", default="mesh,kp,mask,axis")
    o.add_argument("--compare", default=None); o.add_argument("--bodyalign", action="store_true")
    o.set_defaults(func="_overlay")

    l = sub.add_parser("legskel", help="leg-joint chains: detector vs fitted vs triangulated")
    _add_shared(l); l.add_argument("--compare", default=None); l.set_defaults(func="_legskel")

    k = sub.add_parser("kp-qc", help="detector pred-vs-GT keypoints + per-kp error")
    k.add_argument("--run-dir"); k.add_argument("--ckpt"); k.add_argument("--recording")
    k.add_argument("--n", type=int, default=None); k.add_argument("--female-vs-male", action="store_true")
    k.add_argument("--female-rec", default="2026_05_27_11_56_05")
    k.add_argument("--male-rec", default="2026_05_27_11_57_05")
    k.add_argument("--data-root",
                    default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3")
    k.add_argument("--out", default=None); k.set_defaults(func="_kp_qc")

    f = sub.add_parser("fit-check", help="STAC-fit verification frames")
    f.add_argument("ik_h5"); f.add_argument("--start", type=int, default=0)
    f.add_argument("--n", type=int, default=10); f.add_argument("--camera", default=None)
    f.add_argument("--out", default=None)
    f.add_argument("--body-model-dir", default="/home/eabe/Research/MyRepos/fruitfly_body_models")
    f.add_argument("--no-error", action="store_true")
    f.add_argument("--n-stills", type=int, default=3)
    f.set_defaults(func="_fit_check")

    r = sub.add_parser("reproj-video", help="predicted 3D reprojected onto a bout video")
    r.add_argument("--session-dir", required=True); r.add_argument("--pred-dir", required=True)
    r.add_argument("--bout", type=int, required=True); r.add_argument("--cameras", nargs="*", default=None)
    r.add_argument("--with-masks", action="store_true"); r.add_argument("--out", default=None)
    r.add_argument("--max-frames", type=int, default=0,
                   help="cap the number of frames rendered per camera (0 = all)")
    r.set_defaults(func="_reproj_video")

    c = sub.add_parser("clip", help="multi-camera clip cut|stack|render")
    c.add_argument("mode", choices=["cut", "stack", "render"])
    c.add_argument("--session-dir"); c.add_argument("--start", type=int); c.add_argument("--end", type=int)
    c.add_argument("--cameras", nargs="*", default=None); c.add_argument("--bout-dir")
    c.add_argument("--out", default=None); c.add_argument("--fps", type=int, default=30)
    c.set_defaults(func="_clip")
    return p

def main(argv=None):
    # func is the NAME of a module-level handler; resolve at call time via globals()
    # so tests (and future edits) can monkeypatch a handler by attribute.
    args = build_parser().parse_args(argv)
    return int(globals()[args.func](args) or 0)
