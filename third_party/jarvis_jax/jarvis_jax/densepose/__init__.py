"""Continuous Surface Embedding (dense pose) support for the JARVIS-JAX fly stack.

Modules
-------
build_canonical_mesh : assemble the canonical fly surface from MuJoCo collision
                       geoms (one low-poly, segment-labelled mesh) + FPS subsets,
                       L/R symmetry, and a Laplacian edge graph.
mesh_assets          : load the canonical asset and re-pose its vertices under any
                       STAC qpos via MuJoCo forward kinematics (used to render
                       auto-labels and, later, to define dense IK targets).

A canonical vertex is treated as a "joint": the existing reproject -> V2VNet ->
soft_argmax_3d -> STAC-IK path consumes M selected vertices exactly as it
consumes the 50 keypoints, so dense pose is a tunable-density generalisation of
keypoint tracking (Design 1).
"""
