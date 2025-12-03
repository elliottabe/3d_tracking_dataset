import matplotlib.pyplot as plt
import numpy as np

def visualize_alignment(original_points, aligned_points, reference_pose, frame_idx=0,
					title="Procrustes Alignment Verification",
					fly_skel=None,
					end_eff_indices=None, floor_height=-0.125, contact_threshold=0.001):
	"""
	Enhanced visualization of Procrustes alignment results with ground plane and contact points

	Args:
		original_points: (T, N, 3) or (N, 3) original keypoints
		aligned_points: (T, N, 3) or (N, 3) aligned keypoints
		reference_pose: (N, 3) reference pose
		frame_idx: Frame index to visualize if input is temporal
		title: Plot title
		fly_skel: Skeleton edges array - can be flat (N, 2) or grouped by body part
		end_eff_indices: Indices of end effectors for contact detection
		floor_height: Height of the floor/ground plane
		contact_threshold: Distance threshold for contact detection
	"""
	# For courtship data with 13 nodes, use leg tip indices
	# Filtered skeleton order: Antenna, Wings (6), Leg tips (6)
	# Leg tips are at indices: 7, 8, 9, 10, 11, 12
	if end_eff_indices is None:
		end_eff_indices = np.array([7, 8, 9, 10, 11, 12])

	# Normalize fly_skel to be grouped format
	if fly_skel is not None:
		# Check if it's flat (2D array where each row is an edge)
		if isinstance(fly_skel, np.ndarray) and fly_skel.ndim == 2:
			# It's a flat skeleton, wrap it in a single group
			fly_skel = [fly_skel]
		elif not isinstance(fly_skel, (list, np.ndarray)):
			fly_skel = None

	# Use colors for each leg group
	fly_clrs = [
		[0.5, 0.5, 0.5],      # Default - gray
		[0.22, 0.46, 0.67],   # T1L - blue
		[0.76, 0.21, 0.17],   # T2L - red
		[0.92, 0.5, 0.18],    # T3L - orange
		[0.55, 0.41, 0.72],   # T1R - purple
		[0.32, 0.62, 0.24],   # T2R - green
		[0.51, 0.34, 0.30],   # T3R - brown
	]

	def draw_ground_plane(ax, points, floor_height, alpha=0.2):
		"""Draw ground plane based on the extent of the points"""
		x_min, x_max = points[:, 0].min(), points[:, 0].max()
		y_min, y_max = points[:, 1].min(), points[:, 1].max()

		# Expand the plane slightly beyond the points
		margin = 0.02
		x_range = x_max - x_min
		y_range = y_max - y_min
		x_min -= x_range * margin
		x_max += x_range * margin
		y_min -= y_range * margin
		y_max += y_range * margin

		# Create ground plane mesh
		xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
							np.linspace(y_min, y_max, 10))
		zz = np.full_like(xx, floor_height)

		ax.plot_surface(xx, yy, zz, color='gray', alpha=alpha)

	def draw_skeleton_with_contacts(ax, points, point_color, skeleton_alpha=0.6,
									show_contacts=False, floor_height=None, contact_threshold=None):
		"""Draw skeleton connections and highlight contact points"""
		# Draw points
		ax.scatter(points[:, 0], points[:, 1], points[:, 2],
					c=point_color, s=50, alpha=0.8)

		# Draw skeleton connections if provided
		if fly_skel is not None:
			for leg_idx, leg_connections in enumerate(fly_skel):
				color = fly_clrs[leg_idx % len(fly_clrs)]
				
				# Handle both grouped and flat formats
				if isinstance(leg_connections, np.ndarray) and leg_connections.ndim == 2:
					# It's a group of connections
					for connection in leg_connections:
						if len(connection) >= 2 and connection[1] < len(points):
							start_point = points[connection[0]]
							end_point = points[connection[1]]
							ax.plot([start_point[0], end_point[0]],
									[start_point[1], end_point[1]],
									[start_point[2], end_point[2]],
									color=color, alpha=skeleton_alpha, linewidth=2)
				elif len(leg_connections) == 2:
					# It's a single edge
					if leg_connections[1] < len(points):
						start_point = points[leg_connections[0]]
						end_point = points[leg_connections[1]]
						ax.plot([start_point[0], end_point[0]],
								[start_point[1], end_point[1]],
								[start_point[2], end_point[2]],
								color=color, alpha=skeleton_alpha, linewidth=2)

		# Highlight contact points if requested
		if show_contacts and floor_height is not None and contact_threshold is not None:
			contact_points = []
			for idx in end_eff_indices:
				if idx < len(points):
					end_eff_z = points[idx, 2]
					if abs(end_eff_z - floor_height) <= contact_threshold:
						contact_points.append(points[idx])

			if contact_points:
				contact_points = np.array(contact_points)
				ax.scatter(contact_points[:, 0], contact_points[:, 1], contact_points[:, 2],
							c='red', s=100, alpha=1.0, marker='*', edgecolors='black')

			return len(contact_points)
		return 0

	# Handle both single frame and temporal data
	if original_points.ndim == 3:
		orig_frame = original_points[frame_idx]
		aligned_frame = aligned_points[frame_idx]
	else:
		orig_frame = original_points
		aligned_frame = aligned_points

	# Calculate contact information for aligned points
	aligned_end_effs = aligned_frame[end_eff_indices]
	contact_mask = np.abs(aligned_end_effs[:, 2] - floor_height) <= contact_threshold
	n_contacts = np.sum(contact_mask)

	fig = plt.figure(figsize=(24, 8))

	# Plot 1: Original vs Reference
	ax1 = fig.add_subplot(141, projection='3d')
	draw_skeleton_with_contacts(ax1, orig_frame, 'red')
	draw_skeleton_with_contacts(ax1, reference_pose, 'blue')
	ax1.set_title('Original vs Reference')
	ax1.scatter([], [], [], c='red', s=50, label='Original')
	ax1.scatter([], [], [], c='blue', s=50, label='Reference')
	ax1.legend()
	ax1.set_xlabel('X')
	ax1.set_ylabel('Y')
	ax1.set_zlabel('Z')

	# Plot 2: Aligned vs Reference with ground plane
	ax2 = fig.add_subplot(142, projection='3d')
	draw_ground_plane(ax2, aligned_frame, floor_height)
	n_contacts_aligned = draw_skeleton_with_contacts(ax2, aligned_frame, 'green',
													show_contacts=True, floor_height=floor_height,
													contact_threshold=contact_threshold)
	draw_skeleton_with_contacts(ax2, reference_pose, 'blue')
	ax2.set_title(f'Aligned vs Reference\nContacts: {n_contacts_aligned}/{len(end_eff_indices)}')
	ax2.scatter([], [], [], c='green', s=50, label='Aligned')
	ax2.scatter([], [], [], c='blue', s=50, label='Reference')
	ax2.scatter([], [], [], c='red', s=100, marker='*', label='Contacts')
	ax2.legend()
	ax2.set_xlabel('X')
	ax2.set_ylabel('Y')
	ax2.set_zlabel('Z')

	# Plot 3: All together with ground plane
	ax3 = fig.add_subplot(143, projection='3d')
	draw_ground_plane(ax3, aligned_frame, floor_height)
	draw_skeleton_with_contacts(ax3, orig_frame, 'red', skeleton_alpha=0.4)
	draw_skeleton_with_contacts(ax3, aligned_frame, 'green', skeleton_alpha=0.7,
								show_contacts=True, floor_height=floor_height,
								contact_threshold=contact_threshold)
	draw_skeleton_with_contacts(ax3, reference_pose, 'blue', skeleton_alpha=0.9)
	ax3.set_title(f'All Together\nContacts: {n_contacts}/{len(end_eff_indices)}')
	ax3.scatter([], [], [], c='red', s=30, label='Original')
	ax3.scatter([], [], [], c='green', s=30, label='Aligned')
	ax3.scatter([], [], [], c='blue', s=50, label='Reference')
	ax3.scatter([], [], [], c='red', s=100, marker='*', label='Contacts')
	ax3.legend()
	ax3.set_xlabel('X')
	ax3.set_ylabel('Y')
	ax3.set_zlabel('Z')

	# Plot 4: End effector heights comparison
	ax4 = fig.add_subplot(144)

	# Original end effector heights
	orig_end_eff_heights = orig_frame[end_eff_indices, 2]
	aligned_end_eff_heights = aligned_frame[end_eff_indices, 2]
	ref_end_eff_heights = reference_pose[end_eff_indices, 2]

	x_pos = np.arange(len(end_eff_indices))
	width = 0.25

	bars1 = ax4.bar(x_pos - width, orig_end_eff_heights, width, label='Original', color='red', alpha=0.7)
	bars2 = ax4.bar(x_pos, aligned_end_eff_heights, width, label='Aligned', color='green', alpha=0.7)
	bars3 = ax4.bar(x_pos + width, ref_end_eff_heights, width, label='Reference', color='blue', alpha=0.7)

	# Mark contacts
	for i, (idx, is_contact) in enumerate(zip(end_eff_indices, contact_mask)):
		if is_contact:
			ax4.scatter(i, aligned_end_eff_heights[i], color='red', s=100, marker='*', zorder=10)

	ax4.axhline(y=floor_height, color='gray', linestyle='--', alpha=0.8, label=f'Floor ({floor_height})')
	ax4.axhline(y=floor_height + contact_threshold, color='orange', linestyle=':', alpha=0.6,
				label=f'Contact threshold')
	ax4.axhline(y=floor_height - contact_threshold, color='orange', linestyle=':', alpha=0.6)

	ax4.set_xlabel('End Effector Index')
	ax4.set_ylabel('Height (Z)')
	ax4.set_title(f'End Effector Heights\nContacts: {n_contacts}/{len(end_eff_indices)}')
	ax4.set_xticks(x_pos)
	# Label with leg names for courtship data (6 leg tips)
	leg_names = ['T1L', 'T2L', 'T3L', 'T1R', 'T2R', 'T3R']
	ax4.set_xticklabels(leg_names[:len(end_eff_indices)])
	ax4.legend()
	ax4.grid(True, alpha=0.3)

	plt.suptitle(f"{title} - Frame {frame_idx}", fontsize=16)
	plt.tight_layout()
	plt.show()

	# Print contact summary
	print(f"\nContact Summary for Frame {frame_idx}:")
	print(f"  Total contacts: {n_contacts}/{len(end_eff_indices)}")
	print(f"  Floor height: {floor_height}")
	print(f"  Contact threshold: {contact_threshold}")
	for i, (idx, height, is_contact) in enumerate(zip(end_eff_indices, aligned_end_eff_heights, contact_mask)):
		status = "✓ CONTACT" if is_contact else "  above"
		leg_name = leg_names[i] if i < len(leg_names) else f"E{idx}"
		print(f"  {leg_name} (node {idx}): {height:.4f} {status}")
def plot_skeleton_frame(points, frame_idx=0, title="Skeleton Visualization", 
                        fly_skel=None, figsize=(10, 10), view_angle=(20, 45), 
                        end_eff_indices=None, ax=None):
    """
    Plot a single frame with skeleton overlay
    
    Args:
        points: (T, N, 3) or (N, 3) keypoints array
        frame_idx: Frame index to visualize if input is temporal
        title: Plot title
        fly_skel: Skeleton edges array (grouped by body part)
        figsize: Figure size tuple (only used if ax is None)
        view_angle: (elevation, azimuth) for 3D view
        end_eff_indices: Optional list of end effector indices
        ax: Optional matplotlib 3D axes to plot on. If None, creates new figure.
    
    Returns:
        fig, ax: Figure and axes objects
    """
    # Handle both single frame and temporal data
    if points.ndim == 3:
        frame = points[frame_idx]
    else:
        frame = points
    
    # Define fly skeleton connections based on XML/walking_skeleton2.json order
    if fly_skel is None:
        # XML order: Scutellum(0), Antenna_Base(1), EyeL(2), EyeR(3), Abd_A4(4), Abd_tip(5)
        # T1L: ThxCx(6), Tro(7), FeTi(8), TiTa(9), TaT3(10), TaTip(11)
        # T1R: ThxCx(12), Tro(13), FeTi(14), TiTa(15), TaT3(16), TaTip(17)
        # T2L: Tro(18), FeTi(19), TiTa(20), TaT3(21), TaTip(22)
        # T2R: Tro(23), FeTi(24), TiTa(25), TaT3(26), TaTip(27)
        # T3L: Tro(28), FeTi(29), TiTa(30), TaT3(31), TaTip(32)
        # T3R: Tro(33), FeTi(34), TiTa(35), TaT3(36), TaTip(37)
        fly_skel = np.array([
            # Head/body chain
            [[0, 1], [0, 2], [0, 3], [0, 4], [4, 5]],
            # T1L leg (6 joints)
            [[6, 7], [7, 8], [8, 9], [9, 10], [10, 11]],
            # T1R leg (6 joints)
            [[12, 13], [13, 14], [14, 15], [15, 16], [16, 17]],
            # T2L leg (5 joints)
            [[18, 19], [19, 20], [20, 21], [21, 22]],
            # T2R leg (5 joints)
            [[23, 24], [24, 25], [25, 26], [26, 27]],
            # T3L leg (5 joints)
            [[28, 29], [29, 30], [30, 31], [31, 32]],
            # T3R leg (5 joints)
            [[33, 34], [34, 35], [35, 36], [36, 37]]
        ], dtype=object)
    
    # Colors for each leg group (head/body + 6 legs)
    fly_clrs = [
        [0.5, 0.5, 0.5],      # Head/body - gray
        [0.22, 0.46, 0.67],   # T1L - blue
        [0.55, 0.41, 0.72],   # T1R - purple
        [0.76, 0.21, 0.17],   # T2L - red
        [0.32, 0.62, 0.24],   # T2R - green
        [0.92, 0.5, 0.18],    # T3L - orange
        [0.51, 0.34, 0.30],   # T3R - brown
    ]
    
    # End effector indices: T1L_TaTip=11, T1R_TaTip=17, T2L_TaTip=22, T2R_TaTip=27, T3L_TaTip=32, T3R_TaTip=37
    if end_eff_indices is None:
        end_effector_indices = [11, 17, 22, 27, 32, 37]
    else:
        end_effector_indices = end_eff_indices
    
    # Directly map nodes to their leg group based on node ranges
    # head/body(0-5): leg_idx=0, T1L(6-11): leg_idx=1, T1R(12-17): leg_idx=2, 
    # T2L(18-22): leg_idx=3, T2R(23-27): leg_idx=4, T3L(28-32): leg_idx=5, T3R(33-37): leg_idx=6
    def get_leg_idx(node_idx):
        if node_idx <= 5:
            return 0  # head/body
        elif 6 <= node_idx <= 11:
            return 1  # T1L
        elif 12 <= node_idx <= 17:
            return 2  # T1R
        elif 18 <= node_idx <= 22:
            return 3  # T2L
        elif 23 <= node_idx <= 27:
            return 4  # T2R
        elif 28 <= node_idx <= 32:
            return 5  # T3L
        elif 33 <= node_idx <= 37:
            return 6  # T3R
        return 0  # default to gray
    
    # Assign colors to all nodes
    node_colors = [fly_clrs[get_leg_idx(i)] for i in range(len(frame))]
    
    # Make end effectors larger and more visible
    end_eff_mask = np.array([i in end_effector_indices for i in range(len(frame))])
    
    # Create new figure/axes if not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        created_new_fig = True
    else:
        # Use provided axes
        fig = ax.get_figure()
        created_new_fig = False
    
    # Draw regular keypoints
    regular_mask = ~end_eff_mask
    if np.any(regular_mask):
        regular_colors = [node_colors[i] for i in range(len(frame)) if regular_mask[i]]
        ax.scatter(frame[regular_mask, 0], frame[regular_mask, 1], frame[regular_mask, 2], 
                  c=regular_colors, s=50, alpha=0.8, edgecolors='white', linewidths=0.5)
    
    # Draw end effectors with leg colors (larger)
    if np.any(end_eff_mask):
        end_eff_colors = [node_colors[i] for i in range(len(frame)) if end_eff_mask[i]]
        ax.scatter(frame[end_eff_mask, 0], frame[end_eff_mask, 1], frame[end_eff_mask, 2], 
                  c=end_eff_colors, s=120, alpha=1.0, edgecolors='black', linewidths=1.5, marker='o')
    
    # Draw skeleton connections if provided
    for leg_idx, leg_connections in enumerate(fly_skel):
        color = fly_clrs[leg_idx % len(fly_clrs)]
        
        # Handle both grouped and flat formats
        if isinstance(leg_connections, np.ndarray) and leg_connections.ndim == 2:
            # It's a group of connections
            for connection in leg_connections:
                if len(connection) >= 2 and connection[1] < len(frame):
                    start_point = frame[connection[0]]
                    end_point = frame[connection[1]]
                    ax.plot([start_point[0], end_point[0]],
                            [start_point[1], end_point[1]],
                            [start_point[2], end_point[2]],
                            color=color, alpha=0.5, linewidth=2)
        elif len(leg_connections) == 2:
            # It's a single edge
            if leg_connections[1] < len(frame):
                start_point = frame[leg_connections[0]]
                end_point = frame[leg_connections[1]]
                ax.plot([start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        [start_point[2], end_point[2]],
                        color=color, alpha=0.5, linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title} - Frame {frame_idx}')
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Equal aspect ratio
    max_range = np.array([frame[:, 0].max()-frame[:, 0].min(),
                         frame[:, 1].max()-frame[:, 1].min(),
                         frame[:, 2].max()-frame[:, 2].min()]).max() / 2.0
    mid_x = (frame[:, 0].max()+frame[:, 0].min()) * 0.5
    mid_y = (frame[:, 1].max()+frame[:, 1].min()) * 0.5
    mid_z = (frame[:, 2].max()+frame[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Only call tight_layout and show if we created a new figure
    if created_new_fig:
        plt.tight_layout()
        plt.show()
    
    return fig, ax


def animate_skeleton(points, fly_skel=None, end_effector_indices=None, 
                     start_frame=0, end_frame=None, step=1, 
                     figsize=(10, 10), view_angle=(20, 45), 
                     fps=30, title="Skeleton Animation"):
    """
    Create an animation of skeleton keypoints over time
    
    Args:
        points: (T, N, 3) temporal keypoints array
        fly_skel: Skeleton edges array (grouped by body part or flat)
        end_effector_indices: List of end effector node indices
        start_frame: First frame to animate
        end_frame: Last frame to animate (None = all frames)
        step: Frame step size (use >1 to skip frames)
        figsize: Figure size tuple
        view_angle: (elevation, azimuth) for 3D view
        fps: Frames per second for animation
        title: Animation title
        
    Returns:
        matplotlib animation object
    """
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    # Ensure 3D data
    if points.ndim != 3:
        raise ValueError("Points must be (T, N, 3) array")
    
    # Normalize fly_skel to be grouped format
    if fly_skel is not None:
        # Check if it's flat (2D array where each row is an edge)
        if isinstance(fly_skel, np.ndarray) and fly_skel.ndim == 2:
            # It's a flat skeleton, wrap it in a single group
            fly_skel = [fly_skel]
        elif not isinstance(fly_skel, (list, np.ndarray)):
            fly_skel = None

    # Default end effector indices - infer from data shape
    if end_effector_indices is None:
        n_nodes = points.shape[1]
        if n_nodes == 13:
            # Courtship dataset: leg tips are last 6 nodes after reordering to XML order
            # XML order: Antenna(0), Wings(1-6), Leg tips(7-12)
            end_effector_indices = [7, 8, 9, 10, 11, 12]
        elif n_nodes == 38:
            # Walking dataset: TaTip indices
            end_effector_indices = [11, 16, 21, 27, 32, 37]
        else:
            # Default: assume last few nodes are end effectors
            end_effector_indices = list(range(max(0, n_nodes-6), n_nodes))

    # Define skeleton if not provided
    if fly_skel is None:
        n_nodes = points.shape[1]
        if n_nodes == 13:
            # Courtship dataset (13 nodes after XML reordering):
            # 0: Antenna_Base, 1-3: WingL (base, V12, V13), 4-6: WingR (base, V12, V13)
            # 7: T1L_TaTip, 8: T1R_TaTip, 9: T2L_TaTip, 10: T2R_TaTip, 11: T3L_TaTip, 12: T3R_TaTip
            fly_skel = [
                # Wing connections
                [[1, 2], [2, 3]],  # Left wing
                [[4, 5], [5, 6]],  # Right wing
            ]
        elif n_nodes == 38:
            # Walking dataset (38 nodes)
            fly_skel = [
                # Head/body
                [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5]],
                # T1L leg (6-11)
                [[6, 7], [7, 8], [8, 9], [9, 10], [10, 11]],
                # T2L leg (12-16)
                [[12, 13], [13, 14], [14, 15], [15, 16]],
                # T3L leg (17-21)
                [[17, 18], [18, 19], [19, 20], [20, 21]],
                # T1R leg (22-27)
                [[22, 23], [23, 24], [24, 25], [25, 26], [26, 27]],
                # T2R leg (28-32)
                [[28, 29], [29, 30], [30, 31], [31, 32]],
                # T3R leg (33-37)
                [[33, 34], [34, 35], [35, 36], [36, 37]]
            ]
        else:
            # No skeleton connections for unknown structure
            fly_skel = []
    
    # Colors for each leg/body part group
    fly_clrs = [
        [0.5, 0.5, 0.5],      # Default/body - gray
        [0.22, 0.46, 0.67],   # Group 1 - blue
        [0.76, 0.21, 0.17],   # Group 2 - red
        [0.92, 0.5, 0.18],    # Group 3 - orange
        [0.55, 0.41, 0.72],   # Group 4 - purple
        [0.32, 0.62, 0.24],   # Group 5 - green
        [0.51, 0.34, 0.30],   # Group 6 - brown
    ]
    
    # Node coloring function - simplified to just use index order
    def get_color_for_node(node_idx, n_nodes):
        if n_nodes == 13:
            # Courtship: Antenna(0)=gray, WingL(1-3)=blue, WingR(4-6)=red, legs(7-12)=various
            if node_idx == 0:
                return fly_clrs[0]  # Antenna - gray
            elif 1 <= node_idx <= 3:
                return fly_clrs[1]  # WingL - blue
            elif 4 <= node_idx <= 6:
                return fly_clrs[2]  # WingR - red
            elif node_idx == 7:
                return fly_clrs[1]  # T1L - blue
            elif node_idx == 8:
                return fly_clrs[4]  # T1R - purple
            elif node_idx == 9:
                return fly_clrs[2]  # T2L - red
            elif node_idx == 10:
                return fly_clrs[5]  # T2R - green
            elif node_idx == 11:
                return fly_clrs[3]  # T3L - orange
            elif node_idx == 12:
                return fly_clrs[6]  # T3R - brown
        elif n_nodes == 38:
            # Walking dataset
            if node_idx <= 5:
                return fly_clrs[0]  # head/body
            elif 6 <= node_idx <= 11:
                return fly_clrs[1]  # T1L
            elif 12 <= node_idx <= 16:
                return fly_clrs[2]  # T2L
            elif 17 <= node_idx <= 21:
                return fly_clrs[3]  # T3L
            elif 22 <= node_idx <= 27:
                return fly_clrs[4]  # T1R
            elif 28 <= node_idx <= 32:
                return fly_clrs[5]  # T2R
            elif 33 <= node_idx <= 37:
                return fly_clrs[6]  # T3R
        return fly_clrs[0]  # Default gray
    
    # Frame range
    if end_frame is None:
        end_frame = len(points)
    frames = range(start_frame, min(end_frame, len(points)), step)
    
    # Setup figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate global bounds for consistent axis limits
    all_points = points[start_frame:end_frame:step]
    x_min, x_max = all_points[:, :, 0].min(), all_points[:, :, 0].max()
    y_min, y_max = all_points[:, :, 1].min(), all_points[:, :, 1].max()
    z_min, z_max = all_points[:, :, 2].min(), all_points[:, :, 2].max()
    
    # Add margins
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Initialize plot elements
    n_nodes = points.shape[1]
    node_colors = [get_color_for_node(i, n_nodes) for i in range(n_nodes)]
    end_eff_mask = np.array([i in end_effector_indices for i in range(n_nodes)])
    
    # Create scatter plots
    regular_scatter = ax.scatter([], [], [], s=50, alpha=0.8, edgecolors='white', linewidths=0.5)
    end_eff_scatter = ax.scatter([], [], [], s=120, alpha=1.0, edgecolors='black', linewidths=1.5, marker='o')
    
    # Create line objects for skeleton
    skeleton_lines = []
    if fly_skel:
        for leg_idx, leg_connections in enumerate(fly_skel):
            color = fly_clrs[leg_idx % len(fly_clrs)]
            
            # Handle both grouped and flat formats
            if isinstance(leg_connections, np.ndarray) and leg_connections.ndim == 2:
                # It's a group of connections
                for connection in leg_connections:
                    if len(connection) >= 2:
                        line, = ax.plot([], [], [], color=color, alpha=0.7, linewidth=2)
                        skeleton_lines.append((line, connection))
            elif len(leg_connections) == 2:
                # It's a single edge
                line, = ax.plot([], [], [], color=color, alpha=0.7, linewidth=2)
                skeleton_lines.append((line, leg_connections))
    
    title_text = ax.set_title('')
    
    def update(frame_idx):
        frame = points[frame_idx]
        
        # Update regular nodes
        if np.any(~end_eff_mask):
            regular_points = frame[~end_eff_mask]
            regular_colors_list = [node_colors[i] for i in range(len(frame)) if not end_eff_mask[i]]
            regular_scatter._offsets3d = (regular_points[:, 0], regular_points[:, 1], regular_points[:, 2])
            regular_scatter.set_color(regular_colors_list)
        
        # Update end effectors
        if np.any(end_eff_mask):
            end_eff_points = frame[end_eff_mask]
            end_eff_colors_list = [node_colors[i] for i in range(len(frame)) if end_eff_mask[i]]
            end_eff_scatter._offsets3d = (end_eff_points[:, 0], end_eff_points[:, 1], end_eff_points[:, 2])
            end_eff_scatter.set_color(end_eff_colors_list)
        
        # Update skeleton lines
        for line, connection in skeleton_lines:
            if connection[1] < len(frame):
                start_point = frame[connection[0]]
                end_point = frame[connection[1]]
                line.set_data([start_point[0], end_point[0]], [start_point[1], end_point[1]])
                line.set_3d_properties([start_point[2], end_point[2]])
        
        title_text.set_text(f'{title} - Frame {frame_idx}/{len(points)-1}')
        
        return [regular_scatter, end_eff_scatter, title_text] + [line for line, _ in skeleton_lines]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)
    
    plt.close()  # Prevent static display
    return anim