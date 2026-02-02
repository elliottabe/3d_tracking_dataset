import mujoco
import cv2
import ray

import time as time_module
import numpy as np
import jax.numpy as jnp
import matplotlib as mpl
from tqdm.auto import tqdm

def make_vidoes(
    mj_model,
    mj_data,
    qposes_rollout,
    scene_option,
    camera="track1",
    height=512,
    width=512,
):
    """
    Make a video of the rollout and reference superimposed.
    """
    frames = []
    with mujoco.Renderer(mj_model, height=height, width=width) as renderer:
        for t in tqdm(range(len(qposes_rollout))):
            mj_data.qpos = qposes_rollout[t]
            mujoco.mj_forward(mj_model, mj_data)
            
            renderer.update_scene(
                mj_data, camera=f"{camera}", scene_option=scene_option
            )
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            pixels = renderer.render()
            frames.append(pixels)
    return frames

# @ray.remote
def render_frame(frame_idx, vid_ref, time_ref, config):
    """
    Render a single frame with video and flexible time series plots.
    Uses Ray remote function with object references for efficient data sharing.
    
    Parameters:
    -----------
    frame_idx : int
        Frame index to render
    vid_ref : np.ndarray or ray.ObjectRef
        Video frames array of shape (T, H, W, 3)
    time_ref : np.ndarray or ray.ObjectRef
        Time array of shape (T,)
    config : dict
        Configuration dictionary with keys:
        - figsize: tuple, default (15, 10)
        - dpi: int, default 100
        - trail: int, number of frames to highlight, default 5
        - plot_rows: list of dicts, each dict contains:
            - title: str, title for the plot group (appears above subplots)
            - height: int, number of grid rows for this plot group
            - subplots: list of dicts, each dict contains:
                - data_ref: ray.ObjectRef or np.ndarray, data array (T, ...)
                - traces: list of dicts, each dict contains:
                    - data_slice: tuple or callable, how to slice data (e.g., (slice(None), 0, 0) or lambda d: d[:, 0, 0])
                    - label: str, label for the trace
                    - color: str, matplotlib color
                - title: str, subplot title
                - ylabel: str, y-axis label
                - xlim: tuple, optional (min, max) for x-axis limits
                - ylim: tuple, optional (min, max) for y-axis limits
        - video_height: int, number of grid rows for video, default 4
    
    Example config:
    ```python
    config = {
        'figsize': (15, 12),
        'dpi': 100,
        'trail': 5,
        'plot_rows': [
            {
                'title': 'Wing V13',
                'height': 2,
                'subplots': [
                    {
                        'data_ref': wing_data_ref,  # Shape: (T, 2, 3)
                        'traces': [
                            {'data_slice': (slice(None), 0, 0), 'label': 'WingL', 'color': 'C0'},
                            {'data_slice': (slice(None), 1, 0), 'label': 'WingR', 'color': 'C1'},
                        ],
                        'title': 'X',
                        'ylabel': 'X (mm)'
                    },
                    # ... more subplots for Y, Z
                ]
            },
            # ... more plot rows
        ],
        'video_height': 4
    }
    ```
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    
    # Set matplotlib rcParams for consistent formatting in Ray workers
    mpl.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'pdf.fonttype': 42,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.facecolor': 'white',
        'pdf.use14corefonts': True,
        'svg.fonttype': 'none',
        'font.family': 'sans-serif',
        'font.serif': 'Arial',
    })
    
    # Extract config
    figsize = config.get('figsize', (15, 10))
    dpi = config.get('dpi', 100)
    trail = config.get('trail', 5)
    plot_rows = config.get('plot_rows', [])
    video_height = config.get('video_height', 4)
    
    # Dereference Ray ObjectRefs in plot_rows if needed
    import ray
    if plot_rows:
        for row in plot_rows:
            for subplot in row['subplots']:
                if isinstance(subplot['data_ref'], ray.ObjectRef):
                    subplot['data_ref'] = ray.get(subplot['data_ref'])
    
    # Calculate total grid height
    total_plot_height = sum(row['height'] for row in plot_rows)
    total_height = total_plot_height + video_height
    
    # Determine number of columns (max subplots in any row)
    n_cols = max(len(row['subplots']) for row in plot_rows) if plot_rows else 3
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = gridspec.GridSpec(total_height, n_cols, figure=fig, hspace=0.01, wspace=0.03)
    
    # Current time
    current_time = time_ref[frame_idx]
    
    # Render plot rows
    current_row = 0
    for row_config in plot_rows:
        row_height = row_config['height']
        subplots = row_config['subplots']
        
        for subplot_idx, subplot_config in enumerate(subplots):
            ax = fig.add_subplot(gs[current_row:current_row + row_height, subplot_idx])
            
            data_ref = subplot_config['data_ref']
            traces = subplot_config['traces']
            
            # Plot each trace
            for trace_idx, trace in enumerate(traces):
                data_slice = trace['data_slice']
                label = trace['label']
                color = trace.get('color', f'C{trace_idx}')
                
                # Extract data using slice or callable
                if callable(data_slice):
                    trace_data = data_slice(data_ref)
                else:
                    trace_data = data_ref[data_slice]
                
                # Plot full time series with low alpha
                ax.plot(time_ref, trace_data, alpha=0.3, linewidth=1, color=color)
                
                # Highlight trail
                start_idx = max(0, frame_idx - trail)
                end_idx = frame_idx + 1
                ax.plot(time_ref[start_idx:end_idx], trace_data[start_idx:end_idx],
                       alpha=1.0, linewidth=2, color=color, label=label)
            
            # Add vertical line at current time
            ax.axvline(current_time, color='r', linestyle='--', linewidth=1, alpha=0.7)
            
            # Set labels and title
            ax.set_title(subplot_config.get('title', ''), fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel(subplot_config.get('ylabel', ''), fontsize=8)
            ax.tick_params(labelsize=7)
            
            # Set axis limits if specified
            if 'xlim' in subplot_config:
                ax.set_xlim(subplot_config['xlim'])
            if 'ylim' in subplot_config:
                ax.set_ylim(subplot_config['ylim'])
            
            # Add legend to first subplot only
            if subplot_idx == 0:
                legend_config = subplot_config.get('legend', {})
                ncols = legend_config.get('ncols', 1)
                bbox = legend_config.get('bbox_to_anchor', (0.5, 1.))
                ax.legend(frameon=False, loc='upper left', bbox_to_anchor=bbox,
                         labelcolor='linecolor', handlelength=0, handleheight=0,
                         ncols=ncols, columnspacing=.1)
        
        current_row += row_height
    
    # Video frame (merged across all columns)
    ax_video = fig.add_subplot(gs[current_row:, :])
    ax_video.imshow(vid_ref[frame_idx])
    ax_video.axis('off')
    ax_video.set_title(f'Frame {frame_idx} (t={current_time:.3f}s)', fontsize=10)

    # Convert figure to numpy array
    fig.canvas.draw()
    width, height = int(fig.get_figwidth() * fig.dpi), int(fig.get_figheight() * fig.dpi)
    buf = fig.canvas.buffer_rgba()
    image = np.asarray(buf).reshape(height, width, 4)[:, :, :3]  # Remove alpha channel
    
    plt.close(fig)
    
    return image

# @ray.remote
# def render_frame_ray(frame_idx, vid_ref, time_ref, config): 
#     return render_frame(frame_idx, vid_ref, time_ref, config)


def build_plot_row_config(data_array, coord_indices, coord_names, trace_configs, row_height=2, row_title=''):
    """
    Helper function to build a plot row configuration for coordinates (x, y, z).
    
    Parameters:
    -----------
    data_array : np.ndarray
        Data array with shape (T, N, 3) where N is number of features
    coord_indices : list of int
        Indices to slice for each trace (e.g., [0, 1] for left/right wing)
    coord_names : list of str
        Names of coordinates (e.g., ['x', 'y', 'z'])
    trace_configs : list of dict
        List of trace configurations, each with:
        - label: str, label for the trace
        - color: str, matplotlib color
        - index: int, index into second dimension of data_array
    row_height : int, optional
        Height of this row in GridSpec units
    row_title : str, optional
        Title for this row of plots
        
    Returns:
    --------
    dict
        Plot row configuration dictionary
    """
    subplots = []
    for coord_idx, coord_name in enumerate(coord_names):
        traces = []
        for trace_config in trace_configs:
            trace_idx = trace_config['index']
            traces.append({
                'data_slice': (slice(None), trace_idx, coord_idx),
                'label': trace_config['label'],
                'color': trace_config['color']
            })
        
        subplots.append({
            'data_ref': data_array,
            'traces': traces,
            'title': coord_name.upper(),
            'ylabel': coord_name.upper(),
            'legend': {'ncols': 1}
        })
    
    return {
        'title': row_title,
        'height': row_height,
        'subplots': subplots
    }


def build_qpos_row_config(qpos_array, joint_groups, row_height=2, row_title=''):
    """
    Helper function to build a plot row configuration for joint angles.
    
    Parameters:
    -----------
    qpos_array : np.ndarray
        Joint angle data with shape (T, n_joints)
    joint_groups : list of dict
        List of joint group configurations, each with:
        - title: str, title for this subplot (e.g., 'Yaw')
        - ylabel: str, y-axis label (e.g., 'Angle (rad)')
        - joints: list of dict, each with:
            - index: int, index into qpos_array
            - label: str, label for the trace
            - color: str, matplotlib color
    row_height : int, optional
        Height of this row in GridSpec units
    row_title : str, optional
        Title for this row of plots
        
    Returns:
    --------
    dict
        Plot row configuration dictionary
    """
    subplots = []
    for group in joint_groups:
        traces = []
        for joint in group['joints']:
            traces.append({
                'data_slice': (slice(None), joint['index']),
                'label': joint['label'],
                'color': joint['color']
            })
        
        subplots.append({
            'data_ref': qpos_array,
            'traces': traces,
            'title': group['title'],
            'ylabel': group.get('ylabel', ''),
            'legend': {'ncols': 1}
        })
    
    return {
        'title': row_title,
        'height': row_height,
        'subplots': subplots
    }
    


# def render_animation_ray(vid, time, config=None, frames_to_render=None):
#     """
#     Render animation frames in parallel using Ray.
    
#     Parameters:
#     -----------
#     vid : np.ndarray
#         Video frames array of shape (T, H, W, 3)
#     time : np.ndarray
#         Time array of shape (T,)
#     config : dict, optional
#         Configuration dictionary with keys:
#         - figsize: tuple, default (15, 10)
#         - dpi: int, default 100
#         - trail: int, default 5
#         - plot_rows: list of dicts, each specifying a row of plots
#         - video_height: int, number of grid rows for video
#         See render_frame() docstring for full config format
#     frames_to_render : list, optional
#         List of frame indices to render. Default: all frames
        
#     Returns:
#     --------
#     np.ndarray
#         Rendered frames array of shape (T, H, W, 3)
#     """
#     if config is None:
#         config = {}
    
#     # Default to all frames
#     if frames_to_render is None:
#         frames_to_render = list(range(len(vid)))
    
#     print(f"Rendering {len(frames_to_render)} frames using Ray...")
#     print(f"Video shape: {vid.shape}")
#     start_time = time_module.time()
    
#     # Put large arrays into Ray's shared object store
#     vid_ref = ray.put(vid)
#     time_ref = ray.put(time)
    
#     # Put data arrays from plot_rows into Ray's shared object store
#     if 'plot_rows' in config:
#         for row in config['plot_rows']:
#             for subplot in row['subplots']:
#                 # If data_ref is not already a Ray ObjectRef, put it in store
#                 if not isinstance(subplot['data_ref'], ray.ObjectRef):
#                     subplot['data_ref'] = ray.put(subplot['data_ref'])
    
#     # Launch Ray tasks for all frames
#     result_refs = []
#     for frame_idx in frames_to_render:
#         result_refs.append(render_frame_ray.remote(frame_idx, vid_ref, time_ref, config))
    
#     # Get results with progress bar
#     results = []
#     for ref in tqdm(result_refs, desc="Rendering frames"):
#         results.append(ray.get(ref))
    
#     elapsed = time_module.time() - start_time
#     print(f"Rendering completed in {elapsed:.2f}s ({len(frames_to_render)/elapsed:.1f} fps)")
    
#     return np.stack(results)


def create_video(frames, output_path, fps=30, codec='mp4v'):
    """
    Create video from rendered frames.
    
    Parameters:
    -----------
    frames : np.ndarray
        Frames array of shape (T, H, W, 3) in RGB format
    output_path : str or Path
        Output video file path
    fps : int, optional
        Frames per second. Default: 30
    codec : str, optional
        Video codec. Default: 'mp4v'
    """
    from pathlib import Path
    
    output_path = Path(output_path)
    
    print(f"Creating video: {output_path}")
    
    # Get frame dimensions
    height, width = frames.shape[1:3]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write frames (convert RGB to BGR)
    for frame in tqdm(frames, desc="Writing video"):
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"Video saved: {output_path}")