"""Insert the 50 STAC ``tracking[...]`` sites into the v2.3 MJCF.

STAC creates its own marker sites at runtime (stac.py _create_body_sites), so IK
does not need these. Two other stages read them out of the XML and DO:

  * preprocess_keypoints_for_ik.py:256 selects sites via ``'tracking[' in name``
    to fix the keypoint column order and the rest-pose Procrustes reference.
    With zero tracking sites it does not raise -- it reports every node
    unmatched and produces a degenerate reorder.
  * postprocess_stac_data.py:826 selects via ``'tracking' in name`` for the
    egocentric outputs, silently yielding empty (T, 0, 3) arrays.

Positions come from the anatomy config's KEYPOINT_INITIAL_OFFSETS rather than
from the v2.1 XML, so the XML sites and STAC's runtime sites cannot drift apart.

Idempotent: re-running replaces any existing tracking[...] sites.

Usage:
    python scripts/models/add_v2_3_tracking_sites.py \
        --xml models/fruitfly_v2.3/fruitfly_muscles_warp.xml \
        --anatomy configs/anatomy/v2_3.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

SITE_RE = re.compile(r'[ \t]*<site name="tracking\[[^\]]*\]"[^>]*/>[ \t]*\n')


def build_site_xml(kp_name: str, pos: str) -> str:
    return f'      <site name="tracking[{kp_name}]" pos="{pos}" size="0.01" group="3"/>\n'


def strip_existing(xml: str) -> tuple[str, int]:
    """Remove any tracking[...] sites so the script is idempotent."""
    n = len(SITE_RE.findall(xml))
    return SITE_RE.sub('', xml), n


def insert_sites(xml: str, pairs: dict, offsets: dict) -> tuple[str, int]:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml', required=True, type=Path)
    ap.add_argument('--anatomy', required=True, type=Path)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.anatomy)
    pairs = OmegaConf.to_container(cfg.model.KEYPOINT_MODEL_PAIRS)
    offsets = OmegaConf.to_container(cfg.model.KEYPOINT_INITIAL_OFFSETS)

    missing = [k for k in pairs if k not in offsets]
    if missing:
        raise SystemExit(f'ERROR: no KEYPOINT_INITIAL_OFFSETS for {missing}')

    xml = args.xml.read_text()
    xml, removed = strip_existing(xml)
    xml, inserted = insert_sites(xml, pairs, offsets)
    args.xml.write_text(xml)

    print(f'removed {removed} pre-existing tracking sites; inserted {inserted}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
