# Viz reference implementations (behavioral sources for the viz/ package)

These are the scratchpad one-off courtship-QC visualizations built during the
2026-07 debugging session. They are the **behavioral references** the viz
centralization plan (../2026-07-06-viz-centralization-plan.md) ports/promotes
onto `viz/core`:

- overlay view  : viz_both_flies.py, viz_mask_orient.py, viz_detector_headtail.py, viz_legmask_overlay.py, viz_bodyalign.py
- legskel view  : viz_legskel.py, viz_legcompare.py

They are rough (hardcoded paths, inline reprojection). Use them to reproduce
behavior on the shared core; delete this dir once the `viz/` views are built.
