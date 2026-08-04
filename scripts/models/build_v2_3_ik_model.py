"""Build a derived, adhesion-free v2.3 MJCF for IK/reference-kinematics use.

Why this exists
----------------
The shared body-model repo (fruitfly_body_models) keeps its 8 ``<adhesion>``
actuators enabled -- ``mjx.put_model(m, impl="warp")`` accepts them (warp is
what training/rollout uses), and removing them breaks checkpoints trained
with adhesion. See fruitfly_body_models commit 072a293.

The plain (non-warp) MJX path used by STAC/postprocess in *this* repo cannot
load ``mjTRN_BODY`` (adhesion) actuators. Rather than re-editing the shared
repo (which this project must never write to), this script derives a
kinematically-identical copy with the adhesion actuators stripped, living
entirely inside this repo at ``models/fruitfly_v2_3_ik/``.

The two models are kinematically identical except for ``nu`` (272 vs 264):
same nq/nv/nbody/njnt/nsite/ngeom, identical joint/body/site name order,
identical jnt_type/jnt_range/body_pos/body_quat/site_pos/body_parentid, and
FK from a random qpos agrees exactly. The dataset stores qpos/qvel/xpos/xquat
-- none of which depend on nu -- so output produced with the derived model
stays valid against checkpoints trained on the 272-actuator model.

This script also folds in what ``add_v2_3_tracking_sites.py`` used to do:
STAC creates its own marker sites at runtime (stac.py _create_body_sites), so
IK does not need the XML sites. Two other stages read them out of the XML:

  * preprocess_keypoints_for_ik.py:256 selects sites via ``'tracking[' in
    name`` to fix the keypoint column order and the rest-pose Procrustes
    reference. With zero tracking sites it does not raise -- it reports every
    node unmatched and produces a degenerate reorder.
  * postprocess_stac_data.py:826 selects via ``'tracking' in name`` for the
    egocentric outputs, silently yielding empty (T, 0, 3) arrays.

Positions come from the anatomy config's KEYPOINT_INITIAL_OFFSETS rather than
from the v2.1 XML, so the XML sites and STAC's runtime sites cannot drift
apart.

Idempotent and safe to re-run whenever the upstream shared XML changes: reads
the shared XML fresh each time, strips adhesion actuators, strips any
pre-existing tracking[...] sites, reinserts them, and recompiles to verify
the expected shape. Never writes to the source tree.

Usage:
    python scripts/models/build_v2_3_ik_model.py \
        --source models/fruitfly_v2.3/fruitfly_muscles_warp.xml \
        --out-dir models/fruitfly_v2_3_ik \
        --anatomy configs/anatomy/v2_3.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import mujoco as mj
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE = REPO_ROOT / 'models' / 'fruitfly_v2.3' / 'fruitfly_muscles_warp.xml'
DEFAULT_OUT_DIR = REPO_ROOT / 'models' / 'fruitfly_v2_3_ik'
DEFAULT_ANATOMY = REPO_ROOT / 'configs' / 'anatomy' / 'v2_3.yaml'
OUT_XML_NAME = 'fruitfly_v2_3_ik.xml'

ACTUATOR_SECTION_RE = re.compile(r'<actuator>.*?</actuator>', flags=re.S)
ADHESION_RE = re.compile(r'[ \t]*<adhesion\b[^>]*/>[ \t]*\n?')
SITE_RE = re.compile(r'[ \t]*<site name="tracking\[[^\]]*\]"[^>]*/>[ \t]*\n')

EXPECTED_NQ = 101
EXPECTED_NV = 100
EXPECTED_NBODY = 74
EXPECTED_NJNT = 95
EXPECTED_NU = 264
EXPECTED_N_TRACKING_SITES = 50


def refresh_symlink(link_path: Path, target_path: Path) -> None:
    """Create/replace a relative symlink at ``link_path`` pointing at ``target_path``.

    Uses os.path.relpath so the derived tree stays movable together. Replaces
    an existing symlink rather than failing; refuses to clobber a real file
    or directory that isn't already a symlink.
    """
    if link_path.is_symlink() or link_path.exists():
        if not link_path.is_symlink():
            raise SystemExit(
                f'ERROR: {link_path} exists and is not a symlink -- refusing to overwrite')
        link_path.unlink()
    rel_target = os.path.relpath(target_path, start=link_path.parent)
    link_path.symlink_to(rel_target)


def strip_adhesion_actuators(xml: str) -> tuple[str, int]:
    """Remove every ``<adhesion .../>`` actuator inside the ``<actuator>`` section.

    Scoped to the actuator section only, so ``<default class="adhesion*">``
    blocks (which also contain bare ``<adhesion .../>`` elements setting
    defaults, not actuators) and the ``adhesion-collision`` geom class are
    left untouched.
    """
    m = ACTUATOR_SECTION_RE.search(xml)
    if not m:
        raise SystemExit('ERROR: <actuator> section not found in source XML')
    section = m.group(0)
    new_section, n_removed = ADHESION_RE.subn('', section)
    xml = xml[:m.start()] + new_section + xml[m.end():]
    return xml, n_removed


def strip_existing_tracking_sites(xml: str) -> tuple[str, int]:
    """Remove any tracking[...] sites so re-running cannot duplicate them."""
    n = len(SITE_RE.findall(xml))
    return SITE_RE.sub('', xml), n


def build_site_xml(kp_name: str, pos: str) -> str:
    return f'      <site name="tracking[{kp_name}]" pos="{pos}" size="0.01" group="3"/>\n'


def insert_tracking_sites(xml: str, pairs: dict, offsets: dict) -> tuple[str, int]:
    """Insert one <site> per keypoint as the first child of its parent body.

    Matches ``<body name="X" ...>`` and inserts immediately after that tag.
    """
    by_body: dict[str, list[str]] = {}
    for kp, body in pairs.items():
        by_body.setdefault(body, []).append(
            build_site_xml(kp, str(offsets[kp]).strip()))

    inserted = 0
    for body, site_lines in by_body.items():
        pat = re.compile(r'(<body name="' + re.escape(body) + r'"[^>]*>[ \t]*\n)')
        m = pat.search(xml)
        if not m:
            raise SystemExit(f'ERROR: body "{body}" not found in XML')
        xml = xml[:m.end()] + ''.join(site_lines) + xml[m.end():]
        inserted += len(site_lines)
    return xml, inserted


def verify_compiled_model(xml_path: Path, n_tracking_expected: int) -> mj.MjModel:
    m = mj.MjModel.from_xml_path(str(xml_path))

    shape = (m.nq, m.nv, m.nbody, m.njnt, m.nu)
    expected_shape = (EXPECTED_NQ, EXPECTED_NV, EXPECTED_NBODY, EXPECTED_NJNT, EXPECTED_NU)
    if shape != expected_shape:
        raise SystemExit(
            f'ERROR: compiled model shape {shape} != expected {expected_shape} '
            '(nq, nv, nbody, njnt, nu)')

    n_body_trn = int((m.actuator_trntype == mj.mjtTrn.mjTRN_BODY).sum())
    if n_body_trn != 0:
        raise SystemExit(f'ERROR: {n_body_trn} mjTRN_BODY actuators remain (expected 0)')

    site_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    n_tracking = sum(1 for n in site_names if n and n.startswith('tracking['))
    if n_tracking != n_tracking_expected:
        raise SystemExit(
            f'ERROR: {n_tracking} tracking[...] sites found, expected {n_tracking_expected}')

    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                     help='Shared, adhesion-enabled source MJCF (never modified).')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                     help='Directory for the derived, adhesion-free MJCF.')
    ap.add_argument('--anatomy', type=Path, default=DEFAULT_ANATOMY,
                     help='Anatomy config providing KEYPOINT_MODEL_PAIRS / '
                          'KEYPOINT_INITIAL_OFFSETS.')
    args = ap.parse_args()

    source = args.source.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source_dir = source.parent
    refresh_symlink(out_dir / 'assets', source_dir / 'assets')
    refresh_symlink(out_dir / 'floor.xml', source_dir / 'floor.xml')

    cfg = OmegaConf.load(args.anatomy)
    pairs = OmegaConf.to_container(cfg.model.KEYPOINT_MODEL_PAIRS)
    offsets = OmegaConf.to_container(cfg.model.KEYPOINT_INITIAL_OFFSETS)
    missing = [k for k in pairs if k not in offsets]
    if missing:
        raise SystemExit(f'ERROR: no KEYPOINT_INITIAL_OFFSETS for {missing}')

    xml = source.read_text()
    xml, n_removed_adhesion = strip_adhesion_actuators(xml)
    xml, n_removed_sites = strip_existing_tracking_sites(xml)
    xml, n_inserted_sites = insert_tracking_sites(xml, pairs, offsets)

    out_xml_path = out_dir / OUT_XML_NAME
    out_xml_path.write_text(xml)

    print(f'removed {n_removed_adhesion} adhesion actuators')
    print(f'removed {n_removed_sites} pre-existing tracking sites; '
          f'inserted {n_inserted_sites}')

    m = verify_compiled_model(out_xml_path, n_tracking_expected=len(pairs))
    print(f'compiled OK: nq={m.nq} nv={m.nv} nbody={m.nbody} njnt={m.njnt} nu={m.nu}, '
          f'0 mjTRN_BODY, {len(pairs)} tracking sites')
    print(f'wrote {out_xml_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
