#!/usr/bin/env python3
"""
Add aligned keypoint sites to MuJoCo fly model for dynamic visualization.

This utility adds sites to the worldbody that can be updated frame-by-frame
to visualize aligned keypoint trajectories overlaid on the MuJoCo model.
"""

import mujoco
from pathlib import Path
import shutil
from datetime import datetime
from typing import List, Optional


def add_aligned_keypoint_sites_to_model(xml_path: str,
                                        output_path: Optional[str] = None,
                                        node_names: Optional[List[str]] = None,
                                        backup: bool = True,
                                        color_coded: bool = False) -> Path:
    """
    Add sites to worldbody for visualizing aligned keypoints.

    Sites are attached to worldbody (world coordinates) and can be updated
    dynamically during rendering by setting mj_data.site_xpos[site_id].

    Args:
        xml_path: Path to source MuJoCo XML file
        output_path: Path to save modified XML (default: same as xml_path)
        node_names: List of node names for sites (default: courtship 13 nodes)
        backup: Whether to create backup of original file
        color_coded: Whether to use different colors for different body parts

    Returns:
        Path to modified XML file
    """
    xml_path = Path(xml_path)

    # Default node names (courtship dataset with 13 keypoints)
    if node_names is None:
        node_names = [
            'Antenna_Base',
            'WingL_Base', 'WingL_V12', 'WingL_V13',
            'WingR_Base', 'WingR_V12', 'WingR_V13',
            'T1L_TaTip', 'T1R_TaTip',
            'T2L_TaTip', 'T2R_TaTip',
            'T3L_TaTip', 'T3R_TaTip'
        ]

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"Loading MuJoCo model from: {xml_path}")

    # Load spec
    try:
        spec = mujoco.MjSpec.from_file(str(xml_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load XML: {e}")

    # Get worldbody
    worldbody = spec.worldbody

    # Check for existing aligned sites
    existing_names = [site.name for site in spec.sites]
    sites_to_add = []

    for node_name in node_names:
        site_name = f'aligned[{node_name}]'
        if site_name not in existing_names:
            sites_to_add.append((node_name, site_name))
        else:
            print(f"  Site already exists: {site_name}")

    if not sites_to_add:
        print("All aligned sites already exist in model!")
        return xml_path

    print(f"\nAdding {len(sites_to_add)} aligned keypoint sites to worldbody...")

    # Color scheme for different body parts
    if color_coded:
        colors = {
            'antenna': [1, 0, 0, 1],      # Red
            'wing': [0, 0, 1, 1],         # Blue
            'leg': [0, 1, 0, 1],          # Green
        }
    else:
        # All green
        colors = {'default': [0, 1, 0, 1]}

    # Add sites to worldbody
    for node_name, site_name in sites_to_add:
        # Determine color based on node name
        if color_coded:
            if 'Antenna' in node_name:
                color = colors['antenna']
            elif 'Wing' in node_name:
                color = colors['wing']
            else:  # Legs
                color = colors['leg']
        else:
            color = colors['default']

        # Add site
        site = worldbody.add_site()
        site.name = site_name
        site.pos = [0, 0, 0]  # Initial position (will be updated dynamically)
        site.size = [0.005, 0.005, 0.005]  # Slightly larger than tracking sites
        site.group = 3  # Same group as tracking sites
        site.rgba = color

        print(f"  ✓ Added {site_name}")

    print(f"\nTotal sites added: {len(sites_to_add)}")

    # Compile to validate
    print("\nValidating modified model...")
    try:
        model = spec.compile()
        print("  ✓ Model compiled successfully")

        # Count sites
        aligned_sites = [s for s in spec.sites if 'aligned[' in s.name]
        tracking_sites = [s for s in spec.sites if 'tracking[' in s.name]
        print(f"  ✓ Total aligned sites in model: {len(aligned_sites)}")
        print(f"  ✓ Total tracking sites in model: {len(tracking_sites)}")
    except Exception as e:
        raise RuntimeError(f"Model validation failed: {e}")

    # Determine output path
    if output_path is None:
        output_path = xml_path

    # Create backup if overwriting
    if backup and Path(output_path) == xml_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = xml_path.with_suffix(f'.xml.backup_{timestamp}')
        print(f"\nCreating backup: {backup_path}")
        shutil.copy2(xml_path, backup_path)

    # Save modified XML
    print(f"\nSaving modified XML to: {output_path}")
    xml_string = spec.to_xml()
    with open(output_path, 'w') as f:
        f.write(xml_string)
    print("  ✓ XML saved successfully")

    return Path(output_path)


def get_aligned_site_indices(mj_model: mujoco.MjModel,
                             node_names: List[str]) -> dict:
    """
    Get mapping from node index to site index for aligned keypoint sites.

    Args:
        mj_model: Compiled MuJoCo model
        node_names: List of node names in order

    Returns:
        Dict mapping node index (0-12) to site index in mj_data.site_xpos
    """
    aligned_site_ids = {}

    for i, node_name in enumerate(node_names):
        site_name = f'aligned[{node_name}]'
        try:
            site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            aligned_site_ids[i] = site_id
        except Exception as e:
            print(f"Warning: Could not find site {site_name}: {e}")

    return aligned_site_ids


def remove_aligned_sites(xml_path: str,
                        output_path: Optional[str] = None,
                        backup: bool = True) -> Path:
    """
    Remove all aligned keypoint sites from the model.

    Useful for cleaning up or starting fresh.

    Args:
        xml_path: Path to MuJoCo XML file
        output_path: Path to save cleaned XML (default: same as xml_path)
        backup: Whether to create backup

    Returns:
        Path to cleaned XML file
    """
    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"Loading MuJoCo model from: {xml_path}")
    spec = mujoco.MjSpec.from_file(str(xml_path))

    # Find and remove aligned sites
    aligned_sites = [s for s in spec.sites if 'aligned[' in s.name]
    print(f"\nFound {len(aligned_sites)} aligned sites to remove")

    # Note: MuJoCo spec API doesn't have a direct remove_site() method
    # We need to rebuild the worldbody without aligned sites
    # For now, just report what would be removed
    for site in aligned_sites:
        print(f"  - {site.name}")

    print("\nNote: Automatic removal not implemented yet.")
    print("To remove sites, manually edit the XML file or reload from backup.")

    return xml_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Add aligned keypoint sites to MuJoCo fruitfly model'
    )
    parser.add_argument(
        '--xml-path',
        type=str,
        required=True,
        help='Path to the XML file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path (default: overwrite input)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup of original file'
    )
    parser.add_argument(
        '--color-coded',
        action='store_true',
        help='Use different colors for different body parts'
    )
    parser.add_argument(
        '--node-names',
        type=str,
        nargs='+',
        default=None,
        help='Custom list of node names'
    )

    args = parser.parse_args()

    try:
        output_path = add_aligned_keypoint_sites_to_model(
            args.xml_path,
            output_path=args.output_path,
            node_names=args.node_names,
            backup=not args.no_backup,
            color_coded=args.color_coded
        )
        print(f"\n{'='*60}")
        print("SUCCESS: Aligned keypoint sites added successfully!")
        print(f"Modified model saved to: {output_path}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        exit(1)
