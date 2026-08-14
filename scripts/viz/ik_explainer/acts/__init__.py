"""Per-act frame renderers for the IK explainer video.

Each module in this package renders one act's PNG sequence to
`<CLIP>/ik_explainer/frames/act*_*/f%05d.png`, independently of the others, so
any act can be re-rendered without repaying an upstream GPU pass. See
docs/specs/2026-08-13-ik-explainer-animation-design.md for the four-act arc.
"""
