"""Window dataset over the v12 root for the multi-view query model (mvq).

A sample is (recording, host fly, start frame, T): T consecutive labelled
frames of the host, all 7 cameras, cropped at 448 around the projection of
ONE window-level center3D -- the inference convention of
predict/session_frameset.build_frameset, not the per-camera bbox crop of
data/v5_3d.py. Every other labelled fly inside the crops is an extra
instance (fly_valid), so the model can be trained as a set predictor.

Camera order is rt.cameras' sorted-glob order and slots are placed BY NAME
(see data/v5_3d.py docstring for why). Keypoint order is asserted against
annotations/keypoint_names.json.

Sex is resolved PER WINDOW and ANNOTATION-FIRST (`_resolve_fs_sex`, which is
`data/v5_3d._resolve_sex`'s chain applied to the window's OWN frameset):

  1. that frameset's own annotation `sex`, when not "unknown"
  2. else `manifest[rec]["fly_sex"]["fly<id>"]`
  3. else `manifest[rec]["sex"]`
  4. else "unknown"

Why per window rather than one value per (recording, fly): a (recording, fly)
pair can carry framesets from more than one annotation SUBSET, and those
subsets can label DIFFERENT ANIMALS under the same fly id. Measured on
`red_data_3d_v12_export0902`, `2025_10_20_13_20_04` fly0 has 677 framesets
from subset `courtship_20_04_male` (annotation `sex` = male, frames
84143-439478) and 15 from `20_04_female_climbing` (female, frames
446642-447638). The user judged a full-frame render against known-sex
reference recordings on 2026-09-04: the labelled fly really is the MALE in
the courtship block and the FEMALE in the climbing block, so the annotators'
own `sex` field is right for both and the manifest's single per-fly value
(`{fly0: female, fly1: male}`, `sex_source: "dirname"`) simply cannot
represent this recording. Collapsing to one value per (recording, fly) in an
unordered loop -- the old behaviour -- was last-one-wins, a silent coin flip
that made all 692 windows female.

Any (recording, fly) whose framesets disagree on annotation sex gets ONE
warning at init naming the counts and the manifest value they disagree with:
the annotation wins per window, and the warning is there so a recording like
this one is noticed rather than silently averaged.
"""
from __future__ import annotations

import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from jarvis_jax.data.build_v5 import iter_resolved_slots
from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.data.v5_3d import _load_mask, _resolve_sex, _frameset_own_sex
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.train.matching import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, SEX_PRESENT_UNKNOWN

CROP = 448
WINDOW_KEYS = ("crops", "cam_valid", "M", "t_local", "center3D", "kp3d_local", "has3d",
               "kp2d", "vis2d", "fly_valid", "px_scale", "is_female", "prompt_mask", "crop_origin",
               "fly_sex", "unlabelled_sex")

_SEX_CODE = {"female": SEX_FEMALE, "male": SEX_MALE}


def _parse_key(key):
    rec, frame, fly = key.split("/")
    return rec, int(frame.split("_")[1]), int(fly[3:])


def _affine_np(cam_mats):
    P = np.swapaxes(np.asarray(cam_mats, np.float64), 1, 2)          # (C,3,4)
    if not np.allclose(P[:, 2, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("non-affine calibration")
    return P[:, :2, :3], P[:, :2, 3]


class V12WindowDataset:
    def __init__(self, root, split, T=1, *, max_flies=2, jitter_units=3.0, seed=0,
                 train=True, recordings=None, copy_paste=None, center_shift_units=0.0):
        self.root, self.split, self.T = root, split, int(T)
        self.max_flies, self.jitter, self.train = int(max_flies), float(jitter_units), bool(train)
        self.seed = int(seed)
        self.center_shift = float(center_shift_units)
        self.epoch = 0
        self.copy_paste = copy_paste
        coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
        self.manifest = json.load(open(os.path.join(root, "manifest.json")))["recordings"]
        self.keypoint_names = list(coco["keypoint_names"])
        canon = json.load(open(os.path.join(root, "annotations", "keypoint_names.json")))
        if self.keypoint_names != canon:
            raise ValueError("instances keypoint_names != annotations/keypoint_names.json")
        self.K = len(self.keypoint_names)
        self._img = {i["id"]: i for i in coco["images"]}
        self._ann = {a["id"]: a for a in coco["annotations"]}
        self._fs = {}                                   # (rec, frame, fly) -> frameset
        self._tools = {}
        for key, fsv in coco["framesets"].items():
            rec, frame, fly = _parse_key(key)
            if recordings is not None and rec not in recordings:
                continue
            grp = self.manifest[rec]["calib_group"]
            if grp not in self._tools:
                self._tools[grp] = ReprojectionTool(os.path.join(root, "calibrations", str(grp)))
            self._fs[(rec, frame, fly)] = fsv
        self.windows = []
        for (rec, frame, fly) in sorted(self._fs):
            if all((rec, frame + k, fly) in self._fs for k in range(self.T)):
                self.windows.append((rec, fly, frame))
        # Per-WINDOW host sex (the host's own frame-0 frameset annotation first),
        # not one collapsed value per (rec, fly): see the module docstring.
        self._win_sex = [self._resolve_fs_sex(rec, f0, fly) for (rec, fly, f0) in self.windows]
        self._warn_sex_disagreements()
        self._donors = {}
        if self.copy_paste is not None and self.T == 1:
            for i, (rec, fly, _) in enumerate(self.windows):
                self._donors.setdefault((self.manifest[rec]["calib_group"],
                                         _SEX_CODE.get(self._win_sex[i], SEX_UNKNOWN)), []).append(i)

    def _warn_sex_disagreements(self):
        """Print ONE warning per (recording, fly) whose framesets carry more
        than one KNOWN annotation-level sex -- the state that made a collapsed
        per-(rec, fly) sex wrong in the first place (module docstring). The
        annotation still WINS per window; the warning exists so a recording
        whose annotation subsets label different animals under one fly id is
        NOTICED, and it names the manifest value that disagrees. "unknown" is
        not counted as a disagreeing value: an R15-unresolved frameset (every
        camera slot None, see data/v5_3d.py) simply has no annotation sex of
        its own and falls back to the manifest, which is the documented chain,
        not a conflict. Today `2025_10_20_13_20_04` fly0 is the only pair in
        `red_data_3d_v12_export0902` that triggers this."""
        census = collections.defaultdict(collections.Counter)
        for (rec, frame, fly), fsv in self._fs.items():
            s = _frameset_own_sex(fsv, self._ann)
            if s and s != "unknown":
                census[(rec, fly)][s] += 1
        for (rec, fly), c in sorted(census.items()):
            if len(c) > 1:
                man_sex = (self.manifest.get(rec, {}).get("fly_sex") or {}).get(f"fly{fly}")
                print(f"[v12_windows] {rec} fly{fly}: framesets disagree on annotation sex "
                      f"{dict(sorted(c.items()))} -- the ANNOTATION sex WINS per window "
                      f"(annotation-first resolution, see the module docstring); the manifest's "
                      f"single fly_sex={man_sex!r} disagrees and is NOT used for these framesets",
                      flush=True)

    def __len__(self):
        return len(self.windows)

    def calib_group(self, i):
        return self.manifest[self.windows[i][0]]["calib_group"]

    def is_female(self, i):
        """Host sex of THIS window (`_resolve_fs_sex`: this frameset's own
        annotation first, then the manifest), not one collapsed value per
        (recording, fly) -- read by the balanced sampler's `female_weight`
        and by the `female` val cohort."""
        return self._win_sex[i] == "female"

    def n_flies(self, i):
        rec, fly, f0 = self.windows[i]
        others = {k[2] for k in self._fs if k[0] == rec and f0 <= k[1] < f0 + self.T and k[2] != fly}
        return 1 + min(len(others), self.max_flies - 1)

    def _resolve_fs_sex(self, rec, frame, fly):
        """Resolved sex STRING of ONE (recording, frame, fly), ANNOTATION-FIRST:
        `v5_3d._resolve_sex`'s chain (that frameset's own annotation `sex`,
        else the manifest's per-fly `fly_sex`, else the recording's `sex`,
        else "unknown") applied to THIS window's frameset -- see the module
        docstring for why the per-frameset annotation is authoritative here.
        A frame this fly has no frameset for (e.g. the other fly of a T=2
        window labelled only in the second frame) has no annotation of its
        own and falls through to the manifest by fly id."""
        fsv = self._fs.get((rec, frame, fly))
        own = _frameset_own_sex(fsv, self._ann) if fsv is not None else "unknown"
        return _resolve_sex(own, fly, self.manifest.get(rec, {}))

    def fly_sex_code(self, rec, fly, frame):
        """Sex code of ONE fly at ONE frame (0 female, 1 male, -1 unknown).
        `frame` is REQUIRED (it was not, before 2026-09-04): the annotation
        step of the chain is per frameset, and the same (recording, fly) can
        be a male in one annotation subset and a female in another -- see the
        module docstring."""
        return _SEX_CODE.get(self._resolve_fs_sex(rec, frame, fly), SEX_UNKNOWN)

    def window_fly_sex(self, i):
        """Sex code per LABELLED fly of window i (host first, `_window_flies`
        order), each resolved from that fly's own frame-0 frameset."""
        rec, flies = self._window_flies(i)
        f0 = self.windows[i][2]
        return [self.fly_sex_code(rec, fly, f0) for fly in flies]

    def _window_flies(self, i):
        """Labelled fly ids in window i, host first, capped at max_flies (same rule as __getitem__)."""
        rec, host, f0 = self.windows[i]
        frames = [f0 + k for k in range(self.T)]
        others = sorted({k[2] for k in self._fs if k[0] == rec and k[1] in frames and k[2] != host})
        return rec, [host] + others[: self.max_flies - 1]

    def unlabelled_sex(self, i):
        """-1 if every animal the manifest says is in this recording is labelled in the
        window; else the sex code of the one unlabelled animal (SEX_PRESENT_UNKNOWN if
        the manifest does not name its sex).

        A MIXED-sex pair with exactly one of its two animals labelled is
        resolved from the HOST's own per-window sex (`1 - host`), not from the
        manifest's fly ids: `2025_10_20_13_20_04`'s two annotation subsets
        label DIFFERENT animals under fly id 0 (male in
        `courtship_20_04_male`, female in `20_04_female_climbing` -- module
        docstring), so "the missing fly id is fly1, and the manifest calls
        fly1 male" would claim a male unlabelled animal in the 677 windows
        whose LABELLED animal already is that male. Otherwise the unlabelled
        animal is whichever `fly_sex` entry this window has no labels for."""
        rec, flies = self._window_flies(i)
        meta = self.manifest.get(rec, {})
        n_present = int(meta.get("n_flies", len(flies)))
        if n_present <= len(flies):
            return SEX_UNKNOWN
        if meta.get("sex") == "mixed" and n_present == 2 and len(flies) == 1:
            host = self.fly_sex_code(rec, flies[0], self.windows[i][2])
            return (1 - host) if host in (SEX_FEMALE, SEX_MALE) else SEX_PRESENT_UNKNOWN
        named = {int(k[3:]): v for k, v in (meta.get("fly_sex") or {}).items()}
        missing = [fid for fid in sorted(named) if fid not in flies]
        if not missing:
            return SEX_PRESENT_UNKNOWN
        return _SEX_CODE.get(named[missing[0]], SEX_PRESENT_UNKNOWN)

    def fly_centroids(self, i):
        """(n_labelled_flies, 3) world centroid of each fly's DLT-able labels at frame 0
        (host first). Labels only -- no JPEG decode -- so it is cheap enough for cohorts."""
        rec, flies = self._window_flies(i)
        rt = self._rt(rec); f0 = self.windows[i][2]
        out = np.zeros((len(flies), 3), np.float32)
        for fi, fly in enumerate(flies):
            fsv = self._fs.get((rec, f0, fly))
            if fsv is None:
                out[fi] = np.nan; continue
            kp, _ = self._labels_full(fsv, rt)
            X, has = self._dlt(kp, rt)
            out[fi] = X[has].mean(0) if has.any() else np.nan
        return out

    def camera_names(self, i):
        """The window's camera-axis names, in the SAME order as `crops`/`kp2d`/
        `M` (`rt.cameras` order -- the calibration serials, e.g. 'Cam2012630'),
        for labelling figures by name instead of a bare `cam{c+1}` index (see
        CLAUDE.md's keypoint/camera-order-trap history)."""
        rec = self.windows[i][0]
        return list(self._rt(rec).cameras.keys())

    def _frame_infos(self, rec, frame, fly=0):
        """`[(img_id, ann_id) | None]` per CAMERA ROW for one frameset.

        The row order is `camera_names`' (== `_rt(rec).cameras`, the order of
        the `crops`/`M`/`t_local` camera axis), and a camera with no resolved
        slot in this frameset is None. Tests and figure scripts use it to
        decode the SAME full frames `_build` cropped, so a window built by
        another code path (`tracking/lift_mvq.py::MVQRunner.windows`) can be
        compared against this loader's pixel for pixel -- resolving the
        images BY CAMERA NAME here rather than trusting the frameset's own
        slot order.
        """
        rt = self._rt(rec)
        cam_to_row = {n: i for i, n in enumerate(rt.cameras.keys())}
        out = [None] * rt.num_cameras
        for img_id, ann_id in iter_resolved_slots(self._fs[(rec, int(frame), int(fly))]):
            c = cam_to_row.get(self._img[img_id]["file_name"].split("/")[1])
            if c is not None:
                out[c] = (img_id, ann_id)
        return out

    # ------------------------------------------------------------------ helpers
    def _rt(self, rec):
        return self._tools[self.manifest[rec]["calib_group"]]

    def _labels_full(self, fsv, rt):
        """Per camera (BY NAME) full-frame (K,3) labels, image infos, ann infos."""
        C = rt.num_cameras
        cam_to_row = {n: i for i, n in enumerate(rt.cameras.keys())}
        kp = np.zeros((C, self.K, 3), np.float32)
        infos = [None] * C
        for img_id, ann_id in iter_resolved_slots(fsv):
            info, ann = self._img[img_id], self._ann[ann_id]
            c = cam_to_row.get(info["file_name"].split("/")[1])
            if c is None:
                continue
            k = np.asarray(ann["keypoints"], np.float32)
            if k.size == self.K * 3:
                kp[c] = k.reshape(-1, 3)
            infos[c] = (info, ann)
        return kp, infos

    def _dlt(self, kp, rt):
        C = rt.num_cameras
        X = np.zeros((self.K, 3), np.float32); has = np.zeros(self.K, bool)
        for j in range(self.K):
            use = [c for c in range(C) if kp[c, j, 2] > 0]
            if len(use) >= 2:
                pts = np.zeros((C, 2)); pts[use] = kp[use, j, :2]
                X[j] = rt.reconstruct_point(pts, cams_to_use=use); has[j] = True
        return X, has

    def _decode(self, info):
        with Image.open(os.path.join(self.root, "images", info["file_name"])) as im:
            return np.asarray(im.convert("RGB"), np.uint8)

    # ------------------------------------------------------------------ sample
    def _build(self, i):
        rec, host, f0 = self.windows[i]
        rt = self._rt(rec); C = rt.num_cameras; T, K, F = self.T, self.K, self.max_flies
        cams = list(rt.cameras.keys())
        M, t = _affine_np(rt.camera_matrices)                         # (C,2,3),(C,2) float64

        # --- labels per frame per fly (full-frame), 3D via DLT, host first
        frames = [f0 + k for k in range(T)]
        rec_, flies = self._window_flies(i)
        kp_full = np.zeros((F, T, C, K, 3), np.float32)
        X3 = np.zeros((F, T, K, 3), np.float32); has3d = np.zeros((F, T, K), bool)
        infos = {}
        for fi, fly in enumerate(flies):
            for ti, f in enumerate(frames):
                fsv = self._fs.get((rec, f, fly))
                if fsv is None:
                    continue
                kp, inf = self._labels_full(fsv, rt)
                kp_full[fi, ti] = kp
                X3[fi, ti], has3d[fi, ti] = self._dlt(kp, rt)
                for c in range(C):
                    if inf[c] is not None:
                        infos.setdefault((ti, c), inf[c])
        cam_valid = np.zeros((T, C), bool)
        for (ti, c) in infos:
            cam_valid[ti, c] = True
        # host frameset None-slots: camera absent for this window frame
        for ti, f in enumerate(frames):
            fsv = self._fs[(rec, f, host)]
            present = {self._img[img]["file_name"].split("/")[1] for img, _ in iter_resolved_slots(fsv)}
            for c, name in enumerate(cams):
                if name not in present:
                    cam_valid[ti, c] = False

        # --- window centre from the host's 3D (frame 0), jittered in train mode
        vis0 = has3d[0, 0]
        pts = X3[0, 0][vis0] if vis0.any() else np.zeros((1, 3), np.float32)
        center = 0.5 * (pts.max(0) + pts.min(0))
        if self.train and self.jitter > 0:
            # Per-sample generator, not a shared self.rng: window_batches draws
            # samples concurrently from a ThreadPoolExecutor, and a numpy
            # Generator is not thread-safe for concurrent draws (corrupted or
            # non-reproducible jitter). Seeding on (seed, index, epoch) keeps
            # this deterministic and lock-free -- reproducible for a given
            # (seed, i, epoch) regardless of thread interleaving, and varying
            # across epochs since window_batches sets ds.epoch = seed.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(i), int(self.epoch)]))
            center = center + rng.uniform(-self.jitter, self.jitter, size=3)
        if not self.train and self.center_shift > 0:
            # Eval-only, deterministic per (seed, i) regardless of epoch: a
            # fixed random in-plane direction and an EXACT magnitude (unlike
            # the train jitter's uniform box) so a centre-shift sweep moves
            # every window by precisely `center_shift` and nothing else
            # (spec doc 2026-09-04-mvq-maskfree-p4a-p4b task 1, §6 spike).
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(i), 99]))
            ang = rng.uniform(0, 2 * np.pi)
            center = center + self.center_shift * np.array([np.cos(ang), np.sin(ang), 0.0])
        center = center.astype(np.float32)

        # --- crops around the projection of center (same origin for all frames of the window)
        origin = np.zeros((C, 2), np.int32)
        for c in range(C):
            info = next((infos[(ti, c)][0] for ti in range(T) if (ti, c) in infos), None)
            w, h = (info["width"], info["height"]) if info else (1936, 448)
            u, v = M[c] @ center + t[c]
            origin[c] = crop_origin([u, v, 0, 0], w, h, CROP)
        crops = np.zeros((T, C, CROP, CROP, 3), np.uint8)
        prompt = np.zeros((T, C, CROP, CROP), bool)
        for (ti, c), (info, ann) in infos.items():
            if not cam_valid[ti, c]:
                continue
            img = self._decode(info)
            x0, y0 = origin[c]
            crops[ti, c] = img[y0:y0 + CROP, x0:x0 + CROP]
            # host mask (may be absent -> zeros)
            fsv = self._fs[(rec, frames[ti], host)]
            for img_id, ann_id in iter_resolved_slots(fsv):
                if self._img[img_id]["file_name"] == info["file_name"]:
                    a = self._ann[ann_id]
                    m = _load_mask(self.root, info["file_name"], a.get("src_ann_id", ann_id),
                                   ann_id, info["width"], info["height"])
                    prompt[ti, c] = m[y0:y0 + CROP, x0:x0 + CROP].astype(bool)

        # --- to crop/local coordinates
        t_local = np.zeros((T, C, 2), np.float32)
        for ti in range(T):
            t_local[ti] = (M @ center + t - origin).astype(np.float32)
        kp2d = kp_full[..., :2] - origin[None, None, :, None, :]
        inside = ((kp2d >= 0) & (kp2d <= CROP - 1)).all(-1)
        vis2d = (kp_full[..., 2] > 0) & inside & cam_valid[None, :, :, None]
        fly_valid = np.array([fi < len(flies) and vis2d[fi].any() for fi in range(F)])
        fly_valid[0] = True
        px_scale = float(np.mean(np.sqrt((M ** 2).sum((1, 2)) / 2.0)))
        return {
            "crops": crops, "cam_valid": cam_valid,
            "M": M.astype(np.float32), "t_local": t_local, "center3D": center,
            "kp3d_local": (X3 - center).astype(np.float32) * has3d[..., None],
            "has3d": has3d, "kp2d": kp2d.astype(np.float32), "vis2d": vis2d,
            "fly_valid": fly_valid, "px_scale": np.float32(px_scale),
            "is_female": np.bool_(self.is_female(i)), "prompt_mask": prompt,
            "crop_origin": origin,
            "fly_sex": np.array([self.fly_sex_code(rec, flies[fi], f0) if fi < len(flies) else SEX_UNKNOWN
                                 for fi in range(F)], np.int8),
            "unlabelled_sex": np.int8(self.unlabelled_sex(i)),
        }

    def paste_window(self, i, rng):
        """Copy-paste per spec §6; None when no donor fits after max_tries."""
        from jarvis_jax.data.mv_copy_paste import body_plane_axes, composite, sample_offset
        p = self.copy_paste
        rec, host, _ = self.windows[i]
        grp = self.manifest[rec]["calib_group"]
        host_sex = _SEX_CODE.get(self._win_sex[i], SEX_UNKNOWN)   # this window's host, not the (rec, fly) pair's
        tgt = self._build(i)
        axes = body_plane_axes(tgt["kp3d_local"][0, 0], tgt["has3d"][0, 0])
        for _ in range(p.max_tries):
            want = (1 - host_sex) if (host_sex in (0, 1) and rng.uniform() < p.opposite_sex_p) else host_sex
            other = (1 - want) if want in (0, 1) else host_sex
            pool = [j for j in self._donors.get((grp, want), []) if j != i] \
                or [j for j in self._donors.get((grp, other), []) if j != i]
            if not pool:
                return None          # no donor of either sex in this calibration group
            j = int(pool[rng.integers(len(pool))])
            D = sample_offset(rng, axes, p)
            out = composite(tgt, self._build(j), D, p)
            if out is not None:
                sep = float(np.linalg.norm(D))
                return out, {"donor": j, "D": D, "sep": sep, "contact": sep <= p.contact_sep[1]}
        return None

    def __getitem__(self, i):
        p = self.copy_paste
        if (p is not None and self.train and self.T == 1 and self.n_flies(i) == 1
                and self.unlabelled_sex(i) == SEX_UNKNOWN):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(i), int(self.epoch), 7]))
            if rng.uniform() < p.p:
                r = self.paste_window(i, rng)
                if r is not None:
                    return r[0]
        return self._build(i)


def window_batches(ds, batch_size, *, shuffle=True, seed=0, weights=None, num_workers=8,
                   drop_last=True):
    ds.epoch = int(seed)                 # trainer passes a different seed per epoch
    rng = np.random.default_rng(seed)
    n = len(ds)
    if weights is not None:
        w = np.asarray(weights, np.float64); w = w / w.sum()
        idx = rng.choice(n, size=n, replace=True, p=w)
    else:
        idx = rng.permutation(n) if shuffle else np.arange(n)
    stop = (n // batch_size) * batch_size if drop_last else n
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for s in range(0, stop, batch_size):
            samples = list(pool.map(ds.__getitem__, [int(i) for i in idx[s:s + batch_size]]))
            yield {k: np.stack([smp[k] for smp in samples]) for k in WINDOW_KEYS}
