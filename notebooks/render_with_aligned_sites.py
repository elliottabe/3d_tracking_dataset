#!/usr/bin/env python3
"""
Modified rendering code to visualize aligned keypoints as sites on the MuJoCo model.
"""

import mujoco
import jax.numpy as jnp
import mediapy as media
from tqdm.auto import tqdm
from utils.add_aligned_keypoint_sites import add_aligned_keypoint_sites_to_model, get_aligned_site_indices

# Step 1: Add aligned keypoint sites to the model (run once)
print("=" * 60)
print("Adding aligned keypoint sites to MuJoCo model...")
print("=" * 60)

xml_path = '/home/eabe/Research/MyRepos/Fly_tracking/assets/fruitfly_v1/fruitfly_v1_free.xml'

# Add sites with color coding (antenna=red, wings=blue, legs=green)
add_aligned_keypoint_sites_to_model(
    xml_path,
    node_names=filtered_node_names,
    color_coded=True,  # Use different colors for different body parts
    backup=True
)

# Step 2: Load model with the new sites
print("\n" + "=" * 60)
print("Loading modified model...")
print("=" * 60)

spec = mujoco.MjSpec().from_file(xml_path)
mj_model = spec.compile()

# Get tracking sites
site_names = [site.name for site in spec.sites if 'tracking' in site.name]
site_idxs = jnp.array([site.id for site in spec.sites if 'tracking' in site.name])

# Verify wing tracking sites
wing_sites = [name for name in site_names if 'Wing' in name]
print(f"Found {len(wing_sites)} wing tracking sites:", wing_sites)
expected_wing_sites = ['tracking[WingL_base]', 'tracking[WingL_V12]', 'tracking[WingL_V13]',
                       'tracking[WingR_base]', 'tracking[WingR_V12]', 'tracking[WingR_V13]']
missing_sites = [s for s in expected_wing_sites if s not in site_names]
if missing_sites:
    print(f"WARNING: Missing wing sites: {missing_sites}")
else:
    print("✓ All expected wing tracking sites found")

# Get aligned keypoint sites
aligned_sites = [site.name for site in spec.sites if 'aligned[' in site.name]
print(f"\n✓ Found {len(aligned_sites)} aligned keypoint sites")
for site_name in aligned_sites:
    print(f"  - {site_name}")

print(f"\nTotal tracking sites: {len(site_names)}")
print(f"Total aligned sites: {len(aligned_sites)}")

# Step 3: Initialize model data
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)

# Get reference pose from tracking sites
site_subset = [skeleton_to_mujoco[name] for name in filtered_node_names if name in skeleton_to_mujoco]
ref_pose = mj_data.site_xpos[site_subset]
print(f'\nReference pose anchor: {ref_pose[0]} at site {site_names[0]}')

# Step 4: Get aligned site indices for updating
aligned_site_ids = get_aligned_site_indices(mj_model, filtered_node_names)
print(f"\n✓ Mapped {len(aligned_site_ids)} aligned sites for rendering")

# Step 5: Setup rendering options
scene_option = mujoco.MjvOption()
scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]  # Show all site groups
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

# Step 6: Render frames with aligned keypoints
print("\n" + "=" * 60)
print("Rendering frames with aligned keypoints...")
print("=" * 60)

frames = []
num_frames = len(aligned_test_clip)

# Render every Nth frame for faster preview (set to 1 for all frames)
frame_step = 2
frames_to_render = range(0, num_frames, frame_step)

print(f"Total frames: {num_frames}")
print(f"Rendering every {frame_step} frame(s): {len(frames_to_render)} frames")

with mujoco.Renderer(mj_model, height=512, width=512) as renderer:
    for t in tqdm(frames_to_render, desc="Rendering"):
        # Update aligned keypoint site positions from aligned_test_clip
        for i, site_id in aligned_site_ids.items():
            mj_data.site_xpos[site_id] = aligned_test_clip[t, i, :]

        # Forward kinematics to update model state
        mujoco.mj_forward(mj_model, mj_data)

        # Render frame
        renderer.update_scene(mj_data, camera='track1', scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

print(f"\n✓ Rendered {len(frames)} frames")

# Step 7: Display video
print("\n" + "=" * 60)
print("Displaying video...")
print("=" * 60)

# Calculate playback FPS (original is 800 fps, downsample for viewing)
playback_fps = 30

print(f"Playback FPS: {playback_fps}")
print(f"Video duration: {len(frames) / playback_fps:.2f} seconds")
print("\nColors:")
print("  🔴 Red sites = Antenna (aligned)")
print("  🔵 Blue sites = Wings (aligned)")
print("  🟢 Green sites = Legs (aligned)")
print("  ⚪ Other sites = Original tracking sites")

media.show_video(frames, fps=playback_fps)
