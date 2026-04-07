"""Utility functions to load data from .mat .yaml and .h5 files."""

import os
import numpy as np
from jax import numpy as jnp
import yaml
import scipy.io as spio
import pickle
from typing import Text, Union
# from pynwb import NWBHDF5IO
# from ndx_pose import PoseEstimationSeries, PoseEstimation
import h5py
from pathlib import Path
from omegaconf import DictConfig
from omegaconf import OmegaConf
from dataclasses import dataclass, asdict, field
from typing import List, Dict


# Dataclasses for config files and stac-mjx outputs


@dataclass
class ModelConfig:
    """Configuration for body model."""

    MJCF_PATH: str
    FTOL: float
    ROOT_FTOL: float
    LIMB_FTOL: float
    N_ITERS: int
    KP_NAMES: List[str]
    KEYPOINT_MODEL_PAIRS: Dict[str, str]
    KEYPOINT_INITIAL_OFFSETS: Dict[str, str]
    ROOT_OPTIMIZATION_KEYPOINT: str
    TRUNK_OPTIMIZATION_KEYPOINTS: List[str]
    INDIVIDUAL_PART_OPTIMIZATION: Dict[str, List[str]]
    KEYPOINT_COLOR_PAIRS: Dict[str, str]
    SCALE_FACTOR: float
    MOCAP_SCALE_FACTOR: float
    SITES_TO_REGULARIZE: List[str]
    RENDER_FPS: int
    N_SAMPLE_FRAMES: int
    M_REG_COEF: int
    
    # Optional fields for newer configs
    N_ITER_Q: int = 800  # Number of iterations for q optimization
    N_ITER_M: int = 2000  # Number of iterations for m optimization
    name: str = ""  # Model name (optional)


@dataclass
class MujocoConfig:
    """Configuration for Mujoco."""

    solver: str
    iterations: int
    ls_iterations: int
    dt: float = 0.001  # Timestep for MuJoCo simulation


@dataclass
class StacConfig:
    """Configuration for STAC."""

    fit_offsets_path: str
    ik_only_path: str
    data_path: str
    num_clips: int
    n_fit_frames: int
    skip_fit_offsets: bool
    skip_ik_only: bool
    infer_qvels: bool
    n_frames_per_clip: int
    mujoco: MujocoConfig
    continuous: bool = False  # Whether data is continuous (for edge effect handling)
    
    # Optional fields used by pipeline scripts
    save_path: str = ""  # Base save directory
    xml_dir: str = ""  # Directory containing XML model files
    enable_padding: bool = False  # Whether to pad clips to same length


@dataclass
class Config:
    """Combined configuration for the model and STAC."""

    model: ModelConfig
    stac: StacConfig


@dataclass
class StacData:
    """Data structure for STAC output."""

    qpos: np.ndarray
    xpos: np.ndarray
    xquat: np.ndarray
    marker_sites: np.ndarray
    offsets: np.ndarray
    kp_data: np.ndarray
    names_qpos: List[str]
    names_xpos: List[str]
    kp_names: List[str]

    # Optional
    qvel: np.ndarray = field(default_factory=lambda: np.array([]))

    def as_dict(self) -> dict:
        """Convert the dataclass instance to a dictionary."""
        return asdict(self)


def load_mocap(cfg: DictConfig, base_path: Union[Path, None] = None):
    """Load mocap data based on file type.

    Loads mocap file based on filetype, and returns the data flattened
    for immediate consumption by stac_mjx algorithm.

    Args:
        cfg (DictConfig): Configs.
        base_path (Union[Path, None], optional): Base path for file paths in configs. Defaults to None.

    Returns:
        Mocap data flattened into an np array of shape [#frames, keypointXYZ],
        where 'keypointXYZ' represents the flattened 3D keypoint components.
        The data is also scaled by multiplication with "MOCAP_SCALE_FACTOR", e.g.
        if the mocap data is in mm and the model is in meters, this should be
        0.001.

    Raises:
        ValueError if an unsupported filetype is encountered.
        ValueError if ordered list of keypoint names is missing or
        does not match number of keypoints.
    """
    if base_path is None:
        base_path = Path.cwd()

    file_path = base_path / cfg.stac.data_path
    # using pathlib
    if file_path.suffix == ".mat":
        label3d_path = cfg.model.get("KP_NAMES_LABEL3D_PATH", None)
        data, kp_names = load_dannce(str(file_path), names_filename=label3d_path)
    elif file_path.suffix == ".nwb":
        data, kp_names = load_nwb(file_path)
    elif file_path.suffix == ".h5":
        data, kp_names = load_h5(file_path)
    else:
        raise ValueError(
            "Unsupported file extension. Please provide a .nwb or .mat file."
        )

    kp_names = kp_names or cfg.model.KP_NAMES

    if kp_names is None:
        raise ValueError(
            "Keypoint names not provided. Please provide an ordered list of keypoint names \
            corresponding to the keypoint data order."
        )

    if len(kp_names) != data.shape[2]:
        raise ValueError(
            f"Number of keypoint names ({len(kp_names)}) is not the same as the number of keypoints in data ({data.shape[1]})"
        )

    model_inds = [
        kp_names.index(src) for src, dst in cfg.model.KEYPOINT_MODEL_PAIRS.items()
    ]

    sorted_kp_names = [kp_names[i] for i in model_inds]

    # Scale mocap data to match model
    data = data * cfg.model.MOCAP_SCALE_FACTOR
    # Sort in kp_names order
    data = jnp.array(data[:, :, model_inds])
    # Flatten data from [#num frames, #keypoints, xyz]
    # into [#num frames, #keypointsXYZ]
    data = jnp.transpose(data, (0, 2, 1))
    data = jnp.reshape(data, (data.shape[0], -1))

    return data, sorted_kp_names


def load_dannce(filename, names_filename=None):
    """Load mocap data from .mat file.

    .mat file is presumed to be constructed by dannce:
    (https://github.com/spoonsso/dannce). In particular this means it relies on
    the data being in millimeters [num frames, num keypoints, xyz], and that we
    use the data stored in the "pred" key.
    """
    node_names = None
    if names_filename is not None:
        mat = spio.loadmat(names_filename)
        node_names = [item[0] for sublist in mat["joint_names"] for item in sublist]

    data = _check_keys(spio.loadmat(filename, struct_as_record=False, squeeze_me=True))[
        "pred"
    ]
    return data, node_names


def load_nwb(filename):
    """Load mocap data from .nwb file.

    Data is presumed [num frames, num keypoints, xyz].
    """
    data = []
    with NWBHDF5IO(filename, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        pose_est = nwbfile.processing["behavior"]["PoseEstimation"]
        node_names = pose_est.nodes[:].tolist()
        data = np.stack(
            [pose_est[node_name].data[:] for node_name in node_names], axis=-1
        )

    return data, node_names


def load_h5(filename):
    """Load .h5 file formatted as [frames, xyz, keypoints].

    Args:
        filename (str): Path to the .h5 file.

    Returns:
        dict: Dictionary containing the data from the .h5 file.
    """
    # TODO add track information
    data = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]

    data = np.array(data["tracks"])
    data = np.squeeze(data, axis=1)
    data = np.transpose(data, (0, 2, 1))
    return data, None


def _check_keys(dict):
    """Check if entries in dictionary are mat-objects.

    Mat-objects are changed to nested dictionaries.
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """A recursive function which constructs from matobjects nested dictionaries."""
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def _load_params(param_path):
    """Load parameters for the animal.

    :param param_path: Path to .yaml file specifying animal parameters.
    """
    with open(param_path, "r") as infile:
        try:
            params = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def save_dict_to_hdf5(group, dictionary):
    """Save a dictionary to an HDF5 group.

    Args:
        group (h5py.Group): HDF5 group to save the dictionary to.
        dictionary (dict): Dictionary to save.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            save_dict_to_hdf5(subgroup, value)
        else:
            group.attrs[key] = value


def save_data_to_h5(
    config: Config,
    kp_names: list,
    names_qpos: list,
    names_xpos: list,
    kp_data: np.ndarray,
    marker_sites: np.ndarray,
    offsets: np.ndarray,
    qpos: np.ndarray,
    xpos: np.ndarray,
    xquat: np.ndarray,
    qvel: np.ndarray,
    file_path: str,
):
    """Save configuration and STAC data to an HDF5 file.

    Args:
        config (Config): Configuration dataclass.
        kp_names (list): List of keypoint names.
        names_qpos (list): List of qpos names.
        names_xpos (list): List of xpos names.
        kp_data (np.ndarray): Keypoint data array.
        marker_sites (np.ndarray): Marker sites array.
        offsets (np.ndarray): Offsets array.
        qpos (np.ndarray): Qpos array.
        xpos (np.ndarray): Xpos array.
        xquat (np.ndarray): Xquat array.
        qvel (np.ndarray): Qvel array.
        file_path (str): Path to the HDF5 file.
    """
    with h5py.File(file_path, "w") as f:
        # Save config as a YAML string
        config_yaml = OmegaConf.to_yaml(OmegaConf.structured(config))
        f.create_dataset("config", data=np.string_(config_yaml))

        # Save stac output data
        f.create_dataset("kp_names", data=np.array(kp_names, dtype="S"))
        f.create_dataset("names_qpos", data=np.array(names_qpos, dtype="S"))
        f.create_dataset("names_xpos", data=np.array(names_xpos, dtype="S"))
        f.create_dataset("kp_data", data=kp_data, compression="gzip")
        f.create_dataset("marker_sites", data=marker_sites, compression="gzip")
        f.create_dataset("offsets", data=offsets, compression="gzip")
        f.create_dataset("qpos", data=qpos, compression="gzip")
        f.create_dataset("qvel", data=qvel, compression="gzip")
        f.create_dataset("xpos", data=xpos, compression="gzip")
        f.create_dataset("xquat", data=xquat, compression="gzip")


# Fuzzy matching function
def match_csv_to_skeleton(csv_names, skeleton_names):
    """
    Match CSV keypoint names to skeleton node names using fuzzy matching.
    
    Rules:
    1. Case-insensitive exact match
    2. Partial suffix match (e.g., 'Tip' matches 'TaTip')
    3. Underscore-flexible matching
    
    Returns:
        csv_to_skel_map: Dict mapping {csv_name: (skeleton_index, skeleton_name)}
    """
    import re
    
    def normalize_name(name):
        """Normalize name for matching: lowercase, underscores flexible"""
        return name.lower().replace('_', '')
    
    def match_score(csv_name, skel_name):
        """Calculate match score between two names"""
        csv_norm = normalize_name(csv_name)
        skel_norm = normalize_name(skel_name)
        
        # Exact match (after normalization)
        if csv_norm == skel_norm:
            return 100
        
        # Case-insensitive exact match (before normalization)
        if csv_name.lower() == skel_name.lower():
            return 99
        
        # Check if CSV name is prefix of skeleton name (handles abbreviations)
        # e.g., "T1L_Tip" matches "T1L_TaTip"
        csv_parts = csv_name.lower().split('_')
        skel_parts = skel_name.lower().split('_')
        
        if len(csv_parts) == len(skel_parts):
            # Check each part
            matches = 0
            for cp, sp in zip(csv_parts, skel_parts):
                if cp == sp:
                    matches += 1
                elif sp.endswith(cp) or cp.endswith(sp):
                    # Partial match like "Tip" vs "TaTip"
                    matches += 0.8
            
            if matches == len(csv_parts):
                return 90
            elif matches / len(csv_parts) > 0.5:
                return int(50 + 40 * matches / len(csv_parts))
        
        # Prefix match
        if skel_norm.startswith(csv_norm) or csv_norm.startswith(skel_norm):
            return 70
        
        return 0
    
    csv_to_skel_map = {}
    unmatched_csv = []
    
    for csv_name in csv_names:
        best_score = 0
        best_match = None
        best_idx = None
        
        for skel_idx, skel_name in enumerate(skeleton_names):
            score = match_score(csv_name, skel_name)
            if score > best_score:
                best_score = score
                best_match = skel_name
                best_idx = skel_idx
        
        if best_score >= 70:  # Threshold for acceptable match
            csv_to_skel_map[csv_name] = (best_idx, best_match)
        else:
            unmatched_csv.append(csv_name)
    
    return csv_to_skel_map, unmatched_csv

def load_stac_data(file_path) -> tuple[Config, StacData]:
    """Load configuration and STAC data from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        tuple: A tuple containing the Config and StacData dataclasses.
    """
    with h5py.File(file_path, "r") as f:
        # Load config from YAML string
        config_yaml = f["config"][()].decode("utf-8")
        config = OmegaConf.create(config_yaml)
        # Only extract model and stac fields for Config dataclass
        # (config may have additional fields like dataset_name, version, paths, etc.)
        config_filtered = {"model": config["model"], "stac": config["stac"]}
        config = OmegaConf.structured(Config(**config_filtered))

        # Load additional values
        kp_names = [name.decode("utf-8") for name in f["kp_names"]]
        names_qpos = [name.decode("utf-8") for name in f["names_qpos"]]
        names_xpos = [name.decode("utf-8") for name in f["names_xpos"]]
        kp_data = np.array(f["kp_data"])
        marker_sites = np.array(f["marker_sites"])
        offsets = np.array(f["offsets"])
        qpos = np.array(f["qpos"])
        qvel = np.array(f["qvel"])
        xpos = np.array(f["xpos"])
        xquat = np.array(f["xquat"])

        stac_data = StacData(
            kp_names=kp_names,
            names_qpos=names_qpos,
            names_xpos=names_xpos,
            kp_data=kp_data,
            marker_sites=marker_sites,
            offsets=offsets,
            qpos=qpos,
            qvel=qvel,
            xpos=xpos,
            xquat=xquat,
        )

    return config, stac_data


# Reorder kp_data array to match XML site order
def reorder_keypoints_array(keypoints, node_names, target_node_names):
    """
    Reorder keypoints array to match target node order.
    
    Args:
        keypoints: Array of shape (T, N, 3) or (N, 3) with keypoints
        node_names: Array or list of node names in current order
        target_node_names: List of node names in desired order
        
    Returns:
        Reordered keypoints array, reordered node names
    """
    # Create mapping from node name to current index
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    
    # Build reordering indices
    reorder_indices = []
    missing_nodes = []
    
    for target_name in target_node_names:
        if target_name in name_to_idx:
            reorder_indices.append(name_to_idx[target_name])
        else:
            missing_nodes.append(target_name)
    
    if missing_nodes:
        print(f"Warning: {len(missing_nodes)} nodes not found: {missing_nodes}")
    
    # Reorder the keypoints
    reorder_indices = np.array(reorder_indices)
    keypoints_reordered = keypoints[..., reorder_indices, :]
    
    print(f"Reordered {len(node_names)} nodes to match target order")
    print(f"Original: {node_names[:3]} ... {node_names[-3:]}")
    print(f"Target:   {target_node_names[:3]} ... {target_node_names[-3:]}")
    
    return keypoints_reordered, np.array(target_node_names)

# Function to reorder skeleton edges
def reorder_skeleton_edges(edges, node_names_old, node_names_new):
    """
    Update skeleton edges to use new node indices after reordering.
    
    Args:
        edges: Array of edges shape (E, 2)
        node_names_old: Original node names order
        node_names_new: New node names order (after reordering)
        
    Returns:
        New edges array with updated indices
    """
    # Create mapping from old index to new index
    old_to_new = {}
    for new_idx, new_name in enumerate(node_names_new):
        # Find this name in the old ordering
        try:
            old_idx = list(node_names_old).index(new_name)
            old_to_new[old_idx] = new_idx
        except ValueError:
            continue
    
    # Update edge indices
    new_edges = []
    for edge in edges:
        old_idx1, old_idx2 = edge
        if old_idx1 in old_to_new and old_idx2 in old_to_new:
            new_edges.append([old_to_new[old_idx1], old_to_new[old_idx2]])
    
    new_edges = np.array(new_edges)
    print(f"Reordered skeleton: {len(new_edges)} edges preserved")
    
    return new_edges
