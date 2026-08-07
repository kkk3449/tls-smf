#!/usr/bin/env python3
"""fig18: real-scene photo strip (paper Sec. 5.1) — test room + cafeteria
panoramas, EXIF-rotated, downscaled, stacked with (a)/(b) labels.

  .venv/bin/python scripts/scene_photo_fig.py
"""
import os

from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = [("/home/caselab/Downloads/20260807_155756.jpg",
        "(a) test room, 4F (photographed in the robot-hall configuration)"),
       ("/home/caselab/Downloads/20260807_094617.jpg",
        "(b) cafeteria stress scene, 8F")]
FIGS = os.environ.get(
    "MANUSCRIPT_FIGS",
    "/home/caselab/blk360_ros2_ws/docs/electronics2026/figs")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "paper_figs")

imgs = []
for path, _ in SRC:
    im = ImageOps.exif_transpose(Image.open(path))
    if im.height > im.width:                      # EXIF tag missing: rotate
        im = im.transpose(Image.Transpose.ROTATE_270)
    im.thumbnail((2600, 2600))
    imgs.append(im)

h = sum(i.height / i.width for i in imgs)
fig, axes = plt.subplots(2, 1, figsize=(13.0, 13.0 * h + 0.8),
                         gridspec_kw={"hspace": 0.06})
for ax, im, (_, title) in zip(axes, imgs, SRC):
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=12, loc="left")
fig.tight_layout()
for d in (OUT, FIGS):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "fig18_scene_photos.jpg")
    fig.savefig(p, dpi=170, bbox_inches="tight", pil_kwargs={"quality": 82})
    print("->", p, os.path.getsize(p) // 1024, "KB")
