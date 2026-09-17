"""Semantic class sets + a color palette."""
import colorsys

import numpy as np

S3DIS = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
         "table", "chair", "sofa", "bookcase", "board", "clutter"]

SCANNET = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
           "window", "bookshelf", "picture", "counter", "desk", "curtain",
           "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
           "otherfurniture"]

# Class ids the GeometricBaseline can actually assign (indices into the active set).
# Kept aligned to S3DIS names so swapping in a DL model later is consistent.
BASELINE_LABELS = {"ceiling": 0, "floor": 1, "wall": 2, "clutter": 12}


# Open-vocab class set for the industrial scans (edit freely -- Uni3D classifies
# against whatever text prompts you give). Mix of industrial objects + furniture.
INDUSTRIAL = [
    "pipe", "valve", "flange", "pump", "motor", "tank", "pressure vessel",
    "heat exchanger", "ladder", "stair", "handrail", "cable tray", "ducting",
    "electrical cabinet", "control panel", "junction box", "steel beam",
    "support frame", "pallet", "drum", "barrel", "gas cylinder", "toolbox",
    "table", "chair", "shelf", "cabinet", "box", "machine", "clutter",
]


def get_classes(name):
    return {"s3dis": S3DIS, "scannet": SCANNET,
            "industrial": INDUSTRIAL}.get(name, S3DIS)


def palette(n):
    return np.array(
        [colorsys.hsv_to_rgb((i * 0.6180339887) % 1.0, 0.65, 0.95) for i in range(n)],
        dtype=np.float32,
    )
