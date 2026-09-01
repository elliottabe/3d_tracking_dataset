"""Unified viz CLI:  python -m viz <subcommand> [options]."""
import argparse

def _add_shared(sp):
    sp.add_argument("--run", help="run root (…/Session0_bouts_<date>)")
    sp.add_argument("--bout", type=int, required=True)
    sp.add_argument("--fly", type=int, default=0)
    sp.add_argument("--frame", type=int, default=0)
    sp.add_argument("--cams", nargs="*", default=None)
    sp.add_argument("--out", default=None)
    # Recording overrides: without these the view resolves the DEFAULT recording
    # (Session0) via viz.config.courtship_recording -> wrong videos/frames for any
    # other recording (free-running, Session1, ...). Pipeline hooks pass these.
    sp.add_argument("--session-dir", dest="session_dir", default=None,
                    help="recording dir holding the Cam*.mp4 (default: Session0)")
    sp.add_argument("--start-frame", dest="start_frame", type=int, default=None,
                    help="absolute first video frame of the bout (default: bout_start_frame)")

def _overlay(args):     from viz.views import overlay;      return overlay.run(args)
def _legskel(args):     from viz.views import legskel;      return legskel.run(args)
def _kp_qc(args):       from viz.views import kp_qc;        return kp_qc.run(args)
def _fit_check(args):   from viz.views import fit_check;    return fit_check.run(args)
def _reproj_video(args):from viz.views import reproj_video; return reproj_video.run(args)
def _clip(args):        from viz.views import clip;         return clip.run(args)
def _sidebyside(args):  from viz.views import sidebyside;   return sidebyside.run(args)
def _maskvid(args):     from viz.views import maskvid;      return maskvid.run(args)
def _rigcam(args):      from viz.views import rigcam;       return rigcam.run(args)

def build_parser():
    p = argparse.ArgumentParser(prog="viz", description="Centralized 3d_tracking visualizations")
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("overlay", help="mesh+kp+mask reproj overlay (per-camera montage)")
    _add_shared(o); o.add_argument("--show", default="mesh,kp,mask,axis")
    o.add_argument("--compare", default=None); o.add_argument("--bodyalign", action="store_true")
    o.set_defaults(func="_overlay")

    l = sub.add_parser("legskel", help="leg-joint chains: detector-2D vs fitted-3D (+ --compare)")
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
    c.add_argument("--out", default=None); c.add_argument("--fps", type=int, default=None)
    c.set_defaults(func="_clip")

    rc = sub.add_parser("rigcam",
                        help="render the fit through MuJoCo cameras built from the REAL "
                             "rig calibration, beside the video from the same camera")
    rc.add_argument("--run", help="run root (…/pose)")
    rc.add_argument("--bout", type=int, required=True)
    rc.add_argument("--fly", type=int, default=0)
    rc.add_argument("--frame", type=int, default=0, help="frame offset within the bout")
    rc.add_argument("--cams", nargs="*", default=None, help="cameras (default: all)")
    rc.add_argument("--pad", type=int, default=170, help="crop half-size in px")
    rc.add_argument("--model-xml", dest="model_xml", required=True)
    rc.add_argument("--session-dir", dest="session_dir", default=None)
    rc.add_argument("--predictions-dir", dest="predictions_dir", default=None)
    rc.add_argument("--calib-dir", dest="calib_dir", default=None)
    rc.add_argument("--out", default=None)
    rc.add_argument("--pose", choices=("auto", "wingfit", "refined"), default="auto",
                    help="which fitted pose to draw: 'auto' (default) prefers a stamped "
                        "qpos_wingfit.npz over qpos_refined.npz; 'wingfit'/'refined' "
                        "force one, so both arms of an A/B can be rendered deliberately")
    rc.set_defaults(func="_rigcam")

    s = sub.add_parser("sidebyside", help="side-by-side (bout,fly) QC: raw video+SAM+2D | MuJoCo IK render")
    s.add_argument("--run", help="run root (…/Session0_bouts_<date>)")
    s.add_argument("--bout", type=int, required=True); s.add_argument("--fly", type=int, default=0)
    s.add_argument("--start", type=int, default=0, help="segment start offset within the bout")
    s.add_argument("--n", type=int, default=None, help="segment length in frames (default: to end)")
    s.add_argument("--cams", nargs="*", default=None,
                   help="restrict LEFT-camera auto-pick to these cameras")
    s.add_argument("--camera", default=None, help="MuJoCo render camera (default: track1); "
                   "only used when --right mujoco")
    s.add_argument("--right", choices=("rigcam", "reproj", "mujoco"), default="rigcam",
                   help="RIGHT panel: 'rigcam' (default) renders MuJoCo through a camera "
                        "built from the LEFT camera's own calibration -- the rig is "
                        "telecentric, so an orthographic camera matches it exactly; "
                        "'reproj' draws the fitted mesh/sites reprojected as points (no "
                        "MuJoCo); 'mujoco' renders the model-space track1 camera, which is "
                        "NOT view-matched and cannot be used to judge orientation")
    s.add_argument("--views", default=None,
                   help="opt-in MULTI-VIEW mode: comma-separated roles and/or explicit "
                        "camera names, e.g. 'left,top,right' -- one output ROW per entry, "
                        "each its own [video+SAM+2D | view-matched MuJoCo rigcam] built from "
                        "THAT camera's own calibration. Roles are derived from the rig "
                        "geometry (viz.core.rigviews), never hardcoded per session. Default "
                        "(omitted) is the single-view behaviour above, unchanged. Requires "
                        "--right rigcam (the only view-matched multi-camera mode).")
    s.add_argument("--calib-dir", dest="calib_dir", default=None,
                   help="camera calibration dir (default: <session-dir>/calibration). "
                        "MUST match the session being rendered: the right panel is built "
                        "from these camera poses.")
    s.add_argument("--pose", choices=("auto", "wingfit", "refined"), default="auto",
                   help="which fitted pose to draw: 'auto' (default) prefers a stamped "
                        "qpos_wingfit.npz over qpos_refined.npz; 'wingfit'/'refined' "
                        "force one, so both arms of an A/B can be rendered deliberately")
    s.add_argument("--conf", type=float, default=0.3, help="2D keypoint confidence threshold")
    s.add_argument("--verify", action="store_true",
                   help="LEFT panel: three-level verification overlay -- detections coloured "
                        "BLUE=left / ORANGE=right, magenta crosses for reprojected measured kp3d, "
                        "and sparse anatomy labels. Catches keypoint-order scrambles and left/right "
                        "swaps, which no qc.json metric can see. Multi-view (--views) only.")
    s.add_argument("--panel-h", dest="panel_h", type=int, default=480)
    s.add_argument("--fps", type=int, default=30); s.add_argument("--out", default=None)
    s.add_argument("--session-dir", dest="session_dir", default=None,
                   help="recording dir with Cam*.mp4 (default: Session0 via config)")
    s.add_argument("--predictions-dir", dest="predictions_dir", default=None,
                   help="SAM3 masks dir (default: recording.predictions_dir)")
    s.add_argument("--start-frame", dest="start_frame", type=int, default=None,
                   help="absolute first video frame of the bout (default: bout_start_frame)")
    s.set_defaults(func="_sidebyside")

    m = sub.add_parser("maskvid",
                       help="stacked SAM-mask overlay per bout (fly0/fly1 colored) across a few cameras")
    m.add_argument("--run", help="predictions dir (…/Predictions_3D_*), holds bout_NNNNN/sam3_masks.npz")
    m.add_argument("--bout", type=int, required=True)
    m.add_argument("--n", type=int, default=300, help="max frames rendered (default 300)")
    m.add_argument("--start-t", dest="start_t", type=int, default=0,
                   help="offset INTO the bout to start at (default 0). Shifts "
                        "the video seek and the mask index together, so they "
                        "stay aligned.")
    m.add_argument("--n-cams", dest="n_cams", type=int, default=3,
                   help="number of top-mask-pixel cameras to stack (default 3)")
    m.add_argument("--cams", nargs="*", default=None,
                   help="explicit camera-name list (overrides auto-pick)")
    m.add_argument("--session-dir", dest="session_dir", default=None,
                   help="recording dir with Cam*.mp4 (default: Session0 via config)")
    m.add_argument("--start-frame", dest="start_frame", type=int, default=None,
                   help="absolute first video frame of the bout (default: bout_start_frame)")
    m.add_argument("--panel-h", dest="panel_h", type=int, default=320,
                   help="per-camera panel height in px (default 320)")
    m.add_argument("--fps", type=int, default=30); m.add_argument("--out", default=None)
    m.set_defaults(func="_maskvid")
    return p

def main(argv=None):
    # func is the NAME of a module-level handler; resolve at call time via globals()
    # so tests (and future edits) can monkeypatch a handler by attribute.
    args = build_parser().parse_args(argv)
    return int(globals()[args.func](args) or 0)
