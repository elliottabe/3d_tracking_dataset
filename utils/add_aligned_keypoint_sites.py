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
            'antenna': [1.0, 0.0, 0.0, 1],      # Red
            'eye': [1.0, 0.5, 0.0, 1],          # Orange (if eye nodes exist)
            'wing_left': [0.0, 0.0, 1.0, 1],    # Blue
            'wing_right': [0.0, 0.8, 1.0, 1],   # Cyan
            'T1L': [0.0, 1.0, 0.0, 1],          # Green (front left)
            'T1R': [0.5, 1.0, 0.0, 1],          # Lime (front right)
            'T2L': [1.0, 1.0, 0.0, 1],          # Yellow (mid left)
            'T2R': [1.0, 0.65, 0.0, 1],         # Orange (mid right)
            'T3L': [0.8, 0.0, 0.8, 1],          # Purple (back left)
            'T3R': [1.0, 0.0, 0.5, 1],          # Pink (back right)
        }
    else:
        # All green
        colors = {'default': [0, 1, 0, 1]}

    # Add sites to worldbody
    for node_name, site_name in sites_to_add:
        # Determine color based on node name
        if color_coded:
            # Check for specific body parts with detailed color coding
            if 'Antenna' in node_name or 'antenna' in node_name.lower():
                color = colors['antenna']
            elif 'Eye' in node_name or 'eye' in node_name.lower():
                color = colors['eye']
            elif 'WingL' in node_name:
                color = colors['wing_left']
            elif 'WingR' in node_name:
                color = colors['wing_right']
            elif 'T1L' in node_name:
                color = colors['T1L']
            elif 'T1R' in node_name:
                color = colors['T1R']
            elif 'T2L' in node_name:
                color = colors['T2L']
            elif 'T2R' in node_name:
                color = colors['T2R']
            elif 'T3L' in node_name:
                color = colors['T3L']
            elif 'T3R' in node_name:
                color = colors['T3R']
            else:  # Fallback for any other parts
                color = [0.5, 0.5, 0.5, 1]  # Gray
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
                             node_names: List[str], 
                             suffix: str='') -> dict:
    """
    Get mapping from node index to site index for aligned keypoint sites.

    Args:
        mj_model: Compiled MuJoCo model
        node_names: List of node names in order
        suffix: Suffix to append to site names
    Returns:
        Dict mapping node index (0-12) to site index in mj_data.site_xpos
    """
    aligned_site_ids = {}

    for i, node_name in enumerate(node_names):
        site_name = f'aligned[{node_name}]{suffix}'
        try:
            site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            aligned_site_ids[i] = site_id
        except Exception as e:
            print(f"Warning: Could not find site {site_name}: {e}")

    return aligned_site_ids


def set_aligned_site_colors(spec: mujoco.MjSpec, color_coded: bool = True, suffix: str = '') -> mujoco.MjSpec:
    """
    Set colors for aligned keypoint sites in a MuJoCo spec.
    
    This function modifies the RGBA colors of all 'aligned[*]' sites in the spec
    based on body part (antenna, eyes, wings, legs).
    
    Args:
        spec: MuJoCo spec object
        color_coded: Whether to use different colors for different body parts
                    If False, all sites will be green
    
    Returns:
        Modified spec with updated site colors
    """
    # Define color scheme
    if color_coded:
        colors = {
            'antenna': [1.0, 0.0, 0.0, 1],      # Red
            'eye': [1.0, 0.5, 0.0, 1],          # Orange
            'wing_left': [0.0, 0.0, 1.0, 1],    # Blue
            'wing_right': [0.0, 0.8, 1.0, 1],   # Cyan
            'T1L': [0.0, 1.0, 0.0, 1],          # Green (front left)
            'T1R': [0.5, 1.0, 0.0, 1],          # Lime (front right)
            'T2L': [1.0, 1.0, 0.0, 1],          # Yellow (mid left)
            'T2R': [1.0, 0.65, 0.0, 1],         # Orange (mid right)
            'T3L': [0.8, 0.0, 0.8, 1],          # Purple (back left)
            'T3R': [1.0, 0.0, 0.5, 1],          # Pink (back right)
            'default': [0.5, 0.5, 0.5, 1]       # Gray (fallback)
        }
    else:
        colors = {'default': [0, 1, 0, 1]}
    
    # Find and update aligned sites
    updated_count = 0
    for site in spec.sites:
        if f'aligned[' in site.name and site.name.endswith(suffix):
            # Extract node name from 'aligned[NodeName]'
            node_name = site.name.replace('aligned[', '').replace(']', '').replace(suffix, '')
            
            # Determine color based on node name
            if color_coded:
                if 'Antenna' in node_name or 'antenna' in node_name.lower():
                    color = colors['antenna']
                elif 'Eye' in node_name or 'eye' in node_name.lower():
                    color = colors['eye']
                elif 'WingL' in node_name:
                    color = colors['wing_left']
                elif 'WingR' in node_name:
                    color = colors['wing_right']
                elif 'T1L' in node_name:
                    color = colors['T1L']
                elif 'T1R' in node_name:
                    color = colors['T1R']
                elif 'T2L' in node_name:
                    color = colors['T2L']
                elif 'T2R' in node_name:
                    color = colors['T2R']
                elif 'T3L' in node_name:
                    color = colors['T3L']
                elif 'T3R' in node_name:
                    color = colors['T3R']
                else:
                    color = colors['default']
            else:
                color = colors['default']
            
            # Update site color
            site.rgba = color
            updated_count += 1
    
    print(f"✓ Updated colors for {updated_count} aligned sites")
    return spec
# New approach: Create mocap bodies for aligned keypoints
# These will automatically work when attaching fly model to floor

from typing import List, Dict

def add_aligned_mocap_bodies(spec: mujoco.MjSpec, 
                              node_names: List[str],
                              color_coded: bool = True,
                              prefix: str = 'aligned_') -> mujoco.MjSpec:
    """
    Add mocap bodies with colored sites for aligned keypoint visualization.
    
    Mocap bodies are free-floating and can be positioned anywhere in the scene
    by updating mj_data.mocap_pos[mocap_id].
    
    Args:
        spec: MuJoCo spec object
        node_names: List of keypoint node names
        color_coded: Whether to use different colors for body parts
        prefix: Prefix for mocap body names
        
    Returns:
        Modified spec with mocap bodies added
    """
    # Color scheme
    if color_coded:
        colors = {
            'antenna': [1.0, 0.0, 0.0, 1],      # Red
            'eye': [1.0, 0.5, 0.0, 1],          # Orange
            'wing_left': [0.0, 0.0, 1.0, 1],    # Blue
            'wing_right': [0.0, 0.8, 1.0, 1],   # Cyan
            'T1L': [0.0, 1.0, 0.0, 1],          # Green
            'T1R': [0.5, 1.0, 0.0, 1],          # Lime
            'T2L': [1.0, 1.0, 0.0, 1],          # Yellow
            'T2R': [1.0, 0.65, 0.0, 1],         # Orange
            'T3L': [0.8, 0.0, 0.8, 1],          # Purple
            'T3R': [1.0, 0.0, 0.5, 1],          # Pink
            'default': [0.5, 0.5, 0.5, 1]       # Gray
        }
    else:
        colors = {'default': [0, 1, 0, 1]}
    
    # Add mocap body for each keypoint
    for node_name in node_names:
        # Determine color
        if color_coded:
            if 'Antenna' in node_name or 'antenna' in node_name.lower():
                color = colors['antenna']
            elif 'Eye' in node_name or 'eye' in node_name.lower():
                color = colors['eye']
            elif 'WingL' in node_name:
                color = colors['wing_left']
            elif 'WingR' in node_name:
                color = colors['wing_right']
            elif 'T1L' in node_name:
                color = colors['T1L']
            elif 'T1R' in node_name:
                color = colors['T1R']
            elif 'T2L' in node_name:
                color = colors['T2L']
            elif 'T2R' in node_name:
                color = colors['T2R']
            elif 'T3L' in node_name:
                color = colors['T3L']
            elif 'T3R' in node_name:
                color = colors['T3R']
            else:
                color = colors['default']
        else:
            color = colors['default']
        
        # Create mocap body
        mocap_body = spec.worldbody.add_body()
        mocap_body.name = f'{prefix}{node_name}'
        mocap_body.mocap = True
        mocap_body.pos = [0, 0, 0]
        
        # Add visualization site
        site = mocap_body.add_site()
        site.name = f'{prefix}site_{node_name}'
        site.size = [0.005, 0.005, 0.005]
        site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        site.group = 3
        site.rgba = color
    
    print(f"✓ Added {len(node_names)} mocap bodies with colored sites")
    return spec


def get_aligned_mocap_indices(mj_model: mujoco.MjModel,
                               node_names: List[str],
                               prefix: str = 'aligned_') -> Dict[int, int]:
    """
    Get mapping from keypoint index to mocap index.
    
    Args:
        mj_model: Compiled MuJoCo model
        node_names: List of node names in order
        prefix: Prefix used for mocap body names
        
    Returns:
        Dict mapping keypoint index to mocap index for mj_data.mocap_pos
    """
    mocap_indices = {}
    
    for i, node_name in enumerate(node_names):
        body_name = f'{prefix}{node_name}'
        try:
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            mocap_id = mj_model.body_mocapid[body_id]
            mocap_indices[i] = mocap_id
        except Exception as e:
            print(f"Warning: Could not find mocap body {body_name}: {e}")
    
    return mocap_indices


def prune_spec_to_keypoints(spec: mujoco.MjSpec,
                            present_kp_names: List[str],
                            site_prefixes: tuple = ('tracking', 'aligned'),
                            verbose: bool = True) -> List[str]:
    """Remove body subtrees whose keypoint sites are all absent from the data.

    For an amputation (or any recording with missing markers), the body model
    XML still contains the full limb (e.g. the front-left T1L leg) with dangling
    ``tracking[...]`` sites that have no corresponding keypoint. This walks the
    spec and deletes the *top-most* body whose entire subtree of keypoint sites
    (``tracking[KP]`` / ``aligned[KP]``) is absent from ``present_kp_names`` —
    so a distal leg truncation drops the missing chain while keeping the most
    proximal body that still carries a present keypoint (e.g. coxa for T1L_ThxCx).

    Operates in place on ``spec`` BEFORE ``compile()`` / attaching. Because it
    removes joints, the resulting model has fewer DOFs — use it for keypoint /
    reference-pose visualization, not for replaying full-model STAC ``qpos``
    (which still contains the unconstrained amputated-leg joints).

    Args:
        spec: A fly-model ``mujoco.MjSpec`` (raw, before suffix-attach).
        present_kp_names: Keypoint names present in the data (e.g. ``kp_names``).
        site_prefixes: Site-name prefixes that encode a keypoint as ``prefix[KP]``.
        verbose: Print which bodies were removed.

    Returns:
        List of removed (top-most) body names.
    """
    import re

    present = set(present_kp_names)
    pattern = re.compile(r'(?:' + '|'.join(site_prefixes) + r')\[(.+)\]$')

    def site_kp(name: str):
        m = pattern.match(name or '')
        return m.group(1) if m else None

    # child-name -> parent-name map via a worldbody walk
    parent_of: Dict[str, Optional[str]] = {}

    def _walk(body, parent_name):
        parent_of[body.name] = parent_name
        child = body.first_body()
        while child is not None:
            _walk(child, body.name)
            child = body.next_body(child)

    _walk(spec.worldbody, None)

    def subtree_kps(body) -> set:
        kps = set()
        for site in body.find_all(mujoco.mjtObj.mjOBJ_SITE):
            kp = site_kp(site.name)
            if kp is not None:
                kps.add(kp)
        return kps

    # A body is "fully absent" if its subtree has >=1 keypoint site and none of
    # those keypoints are present in the data.
    fully_absent: Dict[str, object] = {}
    for body in spec.bodies:
        if not body.name:
            continue
        kps = subtree_kps(body)
        if kps and kps.isdisjoint(present):
            fully_absent[body.name] = body

    # Delete only the top-most fully-absent bodies (parent not itself absent);
    # deleting a body cascades to its subtree.
    topmost = [b for name, b in fully_absent.items()
               if parent_of.get(name) not in fully_absent]
    removed = sorted(b.name for b in topmost)
    for body in topmost:
        spec.delete(body)

    if verbose:
        if removed:
            print(f"✓ Pruned {len(removed)} body subtree(s) for missing keypoints: {removed}")
        else:
            print("✓ No body parts to prune — all model keypoints present in data")
    return removed


def render_keypoint_overlay(keypoints,
                            kp_names: List[str],
                            flybody_path,
                            floor_path,
                            out_path,
                            n_frames: int = 3,
                            cameras=('track1', 'track2'),
                            height: int = 512,
                            width: int = 512,
                            floor_offset: float = 0.0,
                            trunk_keypoints=('Scutellum', 'WingL_base', 'WingR_base',
                                             'Abd_A4', 'Abd_tip'),
                            anchor_keypoint: str = 'Scutellum',
                            segment_scales=None) -> Optional[Path]:
    """Per-bout QC render: body model + aligned keypoint markers overlaid.

    The (already Procrustes-scaled) keypoint cloud is rigidly fit to the model's
    rest pose using the trunk markers (Kabsch rotation+translation, NO scale) and
    drawn as colored mocap markers over the model, so you can visually confirm
    the body scale matches the model. The body model is pruned to the keypoints
    present in the data (so an amputated limb is removed). Saves a PNG montage of
    ``n_frames`` evenly-spaced frames x cameras.

    Rendering needs an EGL/GPU context; on failure this returns None (and prints
    a note) rather than raising, so it never breaks preprocessing.

    Args:
        keypoints: (T, N, 3) aligned keypoints for one bout (model units).
        kp_names: keypoint names matching axis 1 of ``keypoints``.
        flybody_path: fly model MJCF path.
        floor_path: floor/arena MJCF path.
        out_path: output PNG path (Path).
        n_frames: number of evenly-spaced frames to render.
        cameras: camera names (``_fly`` suffix added if missing).
        height, width: render size per panel.
        floor_offset: z offset for attaching the fly above the floor.
        trunk_keypoints: rigid markers used for the Kabsch alignment.

    Returns:
        ``out_path`` on success, else ``None``.
    """
    import numpy as np
    import mujoco
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    out_path = _Path(out_path)
    try:
        kp = np.asarray(keypoints)
        T = kp.shape[0]
        names = list(kp_names)

        # Build pruned model + floor + colored mocap markers.
        spec = mujoco.MjSpec.from_file(str(flybody_path))
        # Apply the subject-specific per-segment morph (if provided) so markers
        # overlay the morphed mesh, matching what STAC solved on.
        if segment_scales:
            try:
                import sys as _sys
                from pathlib import Path as _P
                _sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "stac-mjx"))
                from stac_mjx.rescale import rescale_per_segment
                rescale_per_segment(spec, list(segment_scales))
            except Exception as _e:
                print(f"  [render] segment morph skipped: {_e}")
        prune_spec_to_keypoints(spec, names, verbose=False)
        floor_spec = mujoco.MjSpec.from_file(str(floor_path))
        floor_spec.worldbody.add_frame(
            pos=[0, 0, floor_offset], quat=[1, 0, 0, 0]
        ).attach_body(spec.body('thorax'), '', suffix='_fly')
        floor_spec = add_aligned_mocap_bodies(floor_spec, names, color_coded=True,
                                              prefix='aligned_')
        mj_model = floor_spec.compile()
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)
        mocap_idx = get_aligned_mocap_indices(mj_model, names, prefix='aligned_')

        # Drop the floor to just below the fly's lowest point so the body sits on
        # top of it (the rest-pose feet hang below the thorax; without this the
        # fly renders half-buried in the floor).
        floor_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        if floor_gid >= 0:
            zs = [mj_data.geom_xpos[g, 2] for g in range(mj_model.ngeom) if g != floor_gid]
            if zs:
                mj_model.geom_pos[floor_gid, 2] = float(min(zs)) - 0.02
                mujoco.mj_forward(mj_model, mj_data)

        # Model rest-pose tracking-site positions for present keypoints.
        ref = {}
        for n in names:
            sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f'tracking[{n}]_fly')
            if sid >= 0:
                ref[n] = mj_data.site_xpos[sid].copy()
        trunk = [n for n in trunk_keypoints if n in ref and n in names]

        fidx = [T // 2] if n_frames <= 1 else \
            sorted(set(int(round(x)) for x in np.linspace(0, T - 1, n_frames)))

        cams = []
        for c in cameras:
            cn = str(c) if str(c).endswith('_fly') else f'{c}_fly'
            if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cn) >= 0:
                cams.append(cn)
        if not cams:
            cams = [mj_model.camera(0).name]

        def kabsch(P, Q):
            """Rigid R, t mapping P -> Q (no scaling)."""
            Pc, Qc = P - P.mean(0), Q - Q.mean(0)
            U, _, Vt = np.linalg.svd(Pc.T @ Qc)
            D = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ np.diag([1.0, 1.0, D]) @ U.T
            return R, Q.mean(0) - R @ P.mean(0)

        so = mujoco.MjvOption()
        so.sitegroup[:] = [1, 1, 1, 1, 1, 0]
        so.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        nrows, ncols = len(fidx), len(cams)
        fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                                squeeze=False)
        with mujoco.Renderer(mj_model, height=height, width=width) as rnd:
            for r, t in enumerate(fidx):
                tk = [n for n in trunk
                      if np.all(np.isfinite(kp[t, names.index(n)]))]
                if len(tk) >= 3:
                    P = np.array([kp[t, names.index(n)] for n in tk])
                    Q = np.array([ref[n] for n in tk])
                    R, t_centroid = kabsch(P, Q)
                else:
                    R, t_centroid = np.eye(3), np.zeros(3)
                # Pin the thorax keypoint exactly onto the model's thorax site:
                # keep the trunk-derived rotation R but choose the translation so
                # anchor_keypoint maps onto its model site. Makes the thorax
                # coincide and turns any residual splay into a direct read on the
                # scale. Falls back to the Kabsch centroid translation if the
                # anchor is missing/NaN this frame.
                if (anchor_keypoint in names and anchor_keypoint in ref
                        and np.all(np.isfinite(kp[t, names.index(anchor_keypoint)]))):
                    a = kp[t, names.index(anchor_keypoint)]
                    tt = ref[anchor_keypoint] - R @ a
                else:
                    tt = t_centroid
                for i, mid in mocap_idx.items():
                    p = kp[t, i]
                    mj_data.mocap_pos[mid] = (R @ p + tt) if np.all(np.isfinite(p)) \
                        else np.array([0.0, 0.0, -100.0])  # hide NaN markers off-scene
                    mj_data.mocap_quat[mid] = [1, 0, 0, 0]
                mujoco.mj_forward(mj_model, mj_data)
                for c, cn in enumerate(cams):
                    rnd.update_scene(mj_data, camera=cn, scene_option=so)
                    rnd.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    axs[r][c].imshow(rnd.render())
                    axs[r][c].axis('off')
                    if r == 0:
                        axs[r][c].set_title(cn, fontsize=8)
                axs[r][0].text(-0.04, 0.5, f'frame {t}', rotation=90, va='center',
                               ha='right', transform=axs[r][0].transAxes, fontsize=8)
        fig.suptitle(out_path.stem, fontsize=9)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"  [render] skipped {out_path.name}: {e}")
        try:
            plt.close('all')
        except Exception:
            pass
        return None


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
