#!/usr/bin/env python3
"""
Add missing wing tracking sites to the MuJoCo fruitfly model.

This script uses the MuJoCo spec API to programmatically add tracking sites
for wing keypoints (base, V12, V13) to both left and right wings.
"""

import mujoco
from pathlib import Path
import shutil
from datetime import datetime

# Paths
XML_PATH = Path('/home/eabe/Research/MyRepos/Fly_tracking/assets/fruitfly_v1/fruitfly_v1_free.xml')
BACKUP_SUFFIX = f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

# Wing tracking site definitions
# Positions are in local body coordinates (meters)
WING_SITES = {
    'wing_left': [
        {'name': 'tracking[WingL_base]', 'pos': [0, 0, 0]},
        {'name': 'tracking[WingL_V12]', 'pos': [0, -0.08, -0.01]},
        {'name': 'tracking[WingL_V13]', 'pos': [0, -0.14, -0.025]},
    ],
    'wing_right': [
        {'name': 'tracking[WingR_base]', 'pos': [0, 0, 0]},
        {'name': 'tracking[WingR_V12]', 'pos': [0, 0.08, 0.01]},
        {'name': 'tracking[WingR_V13]', 'pos': [0, 0.14, 0.025]},
    ]
}

# Site properties (matching other tracking sites)
SITE_SIZE = [0.002, 0.002, 0.002]  # 3-element array for site size
SITE_GROUP = 3
SITE_RGBA = [1, 0, 0, 1]


def find_body_by_name(spec, body_name):
    """Find a body in the spec by name."""
    for body in spec.bodies:
        if body.name == body_name:
            return body
    return None


def check_existing_sites(spec, site_names):
    """Check if any of the sites already exist."""
    existing_sites = [site.name for site in spec.sites]
    conflicts = [name for name in site_names if name in existing_sites]
    return conflicts


def add_wing_tracking_sites(xml_path, backup=True, overwrite=True):
    """
    Add wing tracking sites to the MuJoCo model.

    Args:
        xml_path: Path to the XML file
        backup: If True, create a backup before modifying
        overwrite: If True, overwrite original file. If False, save to new file.

    Returns:
        Path to the modified XML file
    """
    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"Loading MuJoCo model from: {xml_path}")

    # Load the spec
    try:
        spec = mujoco.MjSpec.from_file(str(xml_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load XML: {e}")

    # Check for existing tracking sites
    all_site_names = []
    for sites in WING_SITES.values():
        all_site_names.extend([s['name'] for s in sites])

    conflicts = check_existing_sites(spec, all_site_names)
    if conflicts:
        print(f"WARNING: The following sites already exist and will be skipped:")
        for name in conflicts:
            print(f"  - {name}")

    # Find wing bodies
    print("\nFinding wing bodies...")
    wing_bodies = {}
    for body_name in WING_SITES.keys():
        body = find_body_by_name(spec, body_name)
        if body is None:
            raise ValueError(f"Body '{body_name}' not found in model")
        wing_bodies[body_name] = body
        print(f"  ✓ Found body: {body_name}")

    # Add tracking sites
    print("\nAdding tracking sites...")
    sites_added = 0

    for body_name, sites_config in WING_SITES.items():
        body = wing_bodies[body_name]

        for site_config in sites_config:
            site_name = site_config['name']

            # Skip if already exists
            if site_name in conflicts:
                continue

            # Add site to body
            site = body.add_site()
            site.name = site_name
            site.pos = site_config['pos']
            site.size = SITE_SIZE
            site.group = SITE_GROUP
            site.rgba = SITE_RGBA

            print(f"  ✓ Added {site_name} to {body_name} at pos={site_config['pos']}")
            sites_added += 1

    print(f"\n Total sites added: {sites_added}")

    # Compile to validate
    print("\nValidating modified model...")
    try:
        model = spec.compile()
        print("  ✓ Model compiled successfully")

        # Count tracking sites
        tracking_sites = [s for s in spec.sites if 'tracking' in s.name]
        print(f"  ✓ Total tracking sites in model: {len(tracking_sites)}")

        wing_sites = [s for s in spec.sites if 'Wing' in s.name]
        print(f"  ✓ Wing tracking sites: {len(wing_sites)}")
        for site in wing_sites:
            print(f"     - {site.name}")
    except Exception as e:
        raise RuntimeError(f"Model validation failed: {e}")

    # Save the modified XML
    if backup and overwrite:
        backup_path = xml_path.with_suffix(xml_path.suffix + BACKUP_SUFFIX)
        print(f"\nCreating backup: {backup_path}")
        shutil.copy2(xml_path, backup_path)

    if overwrite:
        output_path = xml_path
        print(f"\nOverwriting original file: {output_path}")
    else:
        output_path = xml_path.with_name(xml_path.stem + '_with_wings.xml')
        print(f"\nSaving to new file: {output_path}")

    # to_xml() returns string, need to write to file
    xml_string = spec.to_xml()
    with open(output_path, 'w') as f:
        f.write(xml_string)
    print(f"  ✓ XML saved successfully")

    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Add wing tracking sites to MuJoCo fruitfly model'
    )
    parser.add_argument(
        '--xml-path',
        type=str,
        default=str(XML_PATH),
        help='Path to the XML file'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup of original file'
    )
    parser.add_argument(
        '--no-overwrite',
        action='store_true',
        help='Save to new file instead of overwriting'
    )

    args = parser.parse_args()

    try:
        output_path = add_wing_tracking_sites(
            args.xml_path,
            backup=not args.no_backup,
            overwrite=not args.no_overwrite
        )
        print(f"\n{'='*60}")
        print("SUCCESS: Wing tracking sites added successfully!")
        print(f"Modified model saved to: {output_path}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        exit(1)
