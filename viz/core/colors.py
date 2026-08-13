"""Single visual language for all viz: BGR palette + keypoint semantics.
Colours are BGR (cv2). Groups/chains are derived from a KP_NAMES list so they
work for any keypoint ordering."""
import re

PALETTE = {
    "fly0": (255, 255, 0),   # cyan
    "fly1": (0, 165, 255),   # orange
    "head": (0, 0, 255),     # red
    "thorax": (0, 255, 255),  # yellow; completes head/thorax/abdomen alongside
    "tail": (255, 0, 0),     # blue
    "detector": (255, 255, 0),
    "fit": (0, 255, 0),      # green
    "mask": (200, 200, 200), # grey fill
    "mesh": (200, 200, 200),
}

def keypoint_groups(kp_names):
    g = {"head": [], "thorax": [], "abdomen": [], "legs": []}
    for i, n in enumerate(kp_names):
        if re.match(r"T[1-3][LR]_", n):        g["legs"].append(i)
        elif n.startswith(("Antenna", "Eye")): g["head"].append(i)
        elif n.startswith("Abd"):              g["abdomen"].append(i)
        elif n.startswith(("Scutellum", "Wing")): g["thorax"].append(i)
    return g

_SEGS = ["ThxCx", "Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip"]

def leg_chains(kp_names):
    idx = {n: i for i, n in enumerate(kp_names)}
    out = {}
    for leg in ("T1L", "T2L", "T3L", "T1R", "T2R", "T3R"):
        chain = [idx[f"{leg}_{s}"] for s in _SEGS if f"{leg}_{s}" in idx]
        if chain:
            out[leg] = chain
    return out
