"""Post-clustering structure-remnant filter.

Rooms with multi-level ceilings leave thin vertical 'soffit' faces at the
height transitions.  RANSAC structure removal only strips the large horizontal
ceiling planes, so those small vertical strips survive DBSCAN as phantom
"objects" (the showroom keyboard incident: four 3 cm-thin strips at 3.2 m
above the floor, verified by the VLM as keyboards from isolated renders).

The filter flags clusters that are simultaneously
  (a) thin vertical planar strips (plane normal nearly horizontal),
  (b) short (small z extent) and far above the floor,
  (c) attached to the ceiling: a reference cloud that still contains the
      ceiling has a dense layer of points directly above the cluster.

Flagged clusters are structural, not semantic objects, and are excluded from
the Stage-A object set (logged to structure_remnants.csv).
"""
import numpy as np


def estimate_floor_z(ref_xyz):
    """Densest 5 cm z-bin in the lower half of the cloud (robust to
    stairwells / windows that push the raw z-min far below the real floor)."""
    z = ref_xyz[:, 2]
    lower = z[z < np.median(z)]
    hist, edges = np.histogram(lower, bins=np.arange(lower.min(),
                                                     np.median(z) + 0.05, 0.05))
    k = int(hist.argmax())
    return float((edges[k] + edges[k + 1]) / 2)


def _plane_stats(pts):
    c = pts - pts.mean(0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    nz = abs(float(vt[2][2]))          # |z| of best-fit plane normal
    thick = float(s[2]) / np.sqrt(len(pts))  # rms thickness off the plane
    return nz, thick


def find_ceiling_remnants(objs, ref_xyz, max_nz=0.25, max_thick=0.03,
                          max_height=0.6, min_above_floor=1.5,
                          ceiling_band=0.45, min_ceiling_pts=40,
                          footprint_pad=0.3):
    """Flag ceiling-soffit remnants among Stage-A clusters.

    objs: dicts from objects.extract_objects (need xyz/bbox/id/n).
    ref_xyz: a cloud that still contains the ceiling (the pre-removal cloud
    when structure removal ran in-pipeline, else e.g. the raw scan).
    Returns a list of flag dicts (empty if none).
    """
    floor_z = estimate_floor_z(ref_xyz)
    flags = []
    for o in objs:
        p = o["xyz"]
        zmin, zmax = float(p[:, 2].min()), float(p[:, 2].max())
        if (zmax - zmin) > max_height or (zmin - floor_z) < min_above_floor:
            continue
        nz, thick = _plane_stats(p)
        if nz > max_nz or thick > max_thick:
            continue
        lo, hi = o["bbox_min"], o["bbox_max"]
        m = ((ref_xyz[:, 0] > lo[0] - footprint_pad) &
             (ref_xyz[:, 0] < hi[0] + footprint_pad) &
             (ref_xyz[:, 1] > lo[1] - footprint_pad) &
             (ref_xyz[:, 1] < hi[1] + footprint_pad) &
             (ref_xyz[:, 2] > zmax + 0.02) &
             (ref_xyz[:, 2] < zmax + ceiling_band))
        n_ceil = int(m.sum())
        if n_ceil < min_ceiling_pts:
            continue
        # the points above must form a CEILING (one thin dense horizontal
        # layer), not a continuing vertical face (wall/column remnant): most
        # band points concentrated in a 7 cm slab around the band's z-mode
        bz = ref_xyz[m, 2]
        hist, edges = np.histogram(bz, bins=np.arange(bz.min(),
                                                      bz.max() + 0.035, 0.035))
        k = int(hist.argmax())
        slab = (bz > edges[k] - 0.035) & (bz < edges[k] + 0.07)
        layer_frac = float(slab.sum()) / len(bz)
        if layer_frac < 0.6:
            continue
        # ...and that layer must be the topmost surface (a real ceiling has
        # nothing above it; a window sill / shelf edge has wall above)
        slab_top = float(edges[k]) + 0.07
        above = ((ref_xyz[:, 0] > lo[0] - footprint_pad) &
                 (ref_xyz[:, 0] < hi[0] + footprint_pad) &
                 (ref_xyz[:, 1] > lo[1] - footprint_pad) &
                 (ref_xyz[:, 1] < hi[1] + footprint_pad) &
                 (ref_xyz[:, 2] > slab_top + 0.1) &
                 (ref_xyz[:, 2] < slab_top + 1.5))
        if int(above.sum()) > 0.2 * int(slab.sum()):
            continue
        flags.append({
            "ceiling_layer_frac": round(layer_frac, 2),
            "id": o["id"], "n_points": o["n"],
            "zmin": round(zmin, 2), "zmax": round(zmax, 2),
            "height": round(zmax - zmin, 2),
            "above_floor": round(zmin - floor_z, 2),
            "normal_z": round(nz, 3), "rms_thickness": round(thick, 4),
            "ceiling_pts_above": n_ceil,
            "reason": "thin vertical strip attached to ceiling "
                      "(soffit at ceiling-height transition)",
        })
    return flags
