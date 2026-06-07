# Data preparation — wall/floor removal

Turns the raw BLK360 testroom scan into the wall/floor-removed cloud that Stage A
(`extract_objects.py`) consumes. Run from the directory holding `testroom260601.e57`.

## Pipeline

```
testroom260601.e57
   │  preprocess.py --stage 1     crop (AABB slice) + statistical outlier removal
   ▼
stage1_clean.ply
   │  preprocess.py --stage 2     RANSAC plane removal of ceiling/floor/walls
   ▼
stage2_no_wall.ply
   │  ply2e57.py                  PLY → E57 (keeps XYZ + RGB)
   ▼
stage2_no_wall.e57   ← the input used by blk360_seg (testroom_no_wall/)
```

One shot: `python preprocess.py --stage all` then `python ply2e57.py stage2_no_wall.ply stage2_no_wall.e57`.

## How `stage2` removes structure (the wall/floor algorithm)

Iterative RANSAC `segment_plane`, classifying each detected plane and acting per kind:

- **Ceiling** (horizontal normal, centroid above mid-height) → removed by *thickness*
  (`dist_ceil`), so stepped ceilings go too.
- **Floor** (horizontal, low) → registered as a boundary plane.
- **Outer wall** (vertical normal, centroid at the room edge) → registered as a
  boundary plane, with optional per-wall margins (`X+/X-/Y+/Y-`).
- **Interior vertical / tilted planes** → preserved (these are objects, not structure).

After detection, each registered floor/wall is applied as a **half-space slice**:
keep only points that sit at least `margin` *inside* the room from that plane. The
plane and everything outside it are cut cleanly (no wall residue), while furniture/
fixtures more than `margin` from the wall are fully kept. Re-detections of an
already-sliced direction are handled (low horizontal re-detect = floor remnant →
drop; higher = shelf/top → keep).

Key flags: `--wall-margin`, `--wall-margins 'Y+=0.05,X-=0.03'`, `--floor-margin`,
`--dist-ceil`, `--min-abs`, `--max-planes`.

`inspect_e57.py` is a helper to probe an .e57's fields/bounds (used to pick the crop box).
