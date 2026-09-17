"""TOSM Place modeling from a 2D occupancy grid (deterministic).

Segments the mapped free space into Places (rooms / corridors) and builds the
place-level TOSM relations:

  isInsideOf   (object -> place)   which place each semantic object sits in
  isConnectedTo (place <-> place)  places joined through a doorway / opening

Method (morphological room segmentation, à la Bormann et al. 2016, deterministic):
  1. free-space mask from the occupancy grid;
  2. Euclidean distance transform (distance of each free cell to the nearest wall);
  3. erode by a door radius -> cells far enough from walls form room "cores"; this
     pinches off narrow doorways so adjacent rooms separate;
  4. label cores (drop tiny ones), then expand each core back over the full free
     space (nearest-core assignment) -> one room label per free cell;
  5. rooms sharing a border are isConnectedTo (the border sits at the doorway);
  6. an object is isInsideOf the place whose label contains its (x, y).

Pure geometry on the grid — no VLM. Needs scipy.ndimage (+ cv2 for polygons).
Grid convention: input is flipped so row 0 = map-frame bottom, y up, x right;
world_x = origin_x + col*res, world_y = origin_y + row*res.
"""
import numpy as np
from scipy import ndimage


def load_occupancy(pgm, yaml):
    """Read a ROS map (PGM + YAML) -> (grid, res, origin). grid: row0 = bottom."""
    with open(pgm, "rb") as f:
        assert f.readline().strip() == b"P5"
        d = f.readline()
        while d.startswith(b"#"):
            d = f.readline()
        w, h = map(int, d.split())
        f.readline()
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)
    g = np.flipud(img)                       # row0 -> bottom (map frame, y up)
    res, ox, oy = 0.05, 0.0, 0.0
    for line in open(yaml):
        s = line.strip()
        if s.startswith("resolution:"):
            res = float(s.split(":")[1])
        if s.startswith("origin:"):
            n = s.split("[")[1].split("]")[0].split(",")
            ox, oy = float(n[0]), float(n[1])
    return g, res, (ox, oy)


def _room_markers(dt, res, door_radius_m, peak_min_dist_m):
    """One marker per room: distance-transform local maxima (deeper than a
    doorway), with peaks of the same room merged. Robust to wide doorways."""
    fp = max(int(peak_min_dist_m / res), 1)
    local_max = ndimage.maximum_filter(dt, size=2 * fp + 1)
    peaks = (dt == local_max) & (dt > door_radius_m)
    # merge peaks belonging to the same room (within peak_min_dist)
    peaks = ndimage.binary_dilation(peaks, iterations=fp)
    markers, n = ndimage.label(peaks)
    return markers, n


def segment_places(grid, res, origin, door_radius_m=0.6, min_room_m2=2.0,
                   peak_min_dist_m=1.5, merge_width_m=2.0, free_above=250):
    """Segment free space into rooms by marker-seeded watershed.

    Seeds = distance-transform local maxima (one per room, deeper than any
    doorway); each free cell is assigned to the nearest seed by *geodesic*
    distance through free space (multi-source BFS), so the boundary falls at
    the doorway midpoint even for wide openings. Returns (labels, n_rooms);
    labels: 0 = non-room, 1..n = rooms (renumbered, sub-`min_room_m2` dropped).
    """
    from collections import deque
    free = grid >= free_above
    dt = ndimage.distance_transform_edt(free) * res          # metres to wall
    markers, n = _room_markers(dt, res, door_radius_m, peak_min_dist_m)
    if n == 0:
        return np.zeros_like(grid, dtype=np.int32), 0
    H, W = grid.shape
    labels = np.zeros((H, W), dtype=np.int32)
    q = deque()
    mr, mc = np.where(markers > 0)
    for r, c in zip(mr.tolist(), mc.tolist()):
        labels[r, c] = markers[r, c]
        q.append((r, c))
    # multi-source BFS over free space = geodesic Voronoi of the markers
    while q:
        r, c = q.popleft()
        lab = labels[r, c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free[nr, nc] and labels[nr, nc] == 0:
                labels[nr, nc] = lab
                q.append((nr, nc))
    labels = _merge_basins(labels, dt, door_width_m=merge_width_m)
    # drop rooms below the area floor, renumber 1..n_rooms
    cell = res * res
    n_rooms = 0
    out = np.zeros_like(labels)
    for k in range(1, int(labels.max()) + 1):
        m = labels == k
        if m.sum() * cell >= min_room_m2:
            n_rooms += 1
            out[m] = n_rooms
    return out, n_rooms


def _merge_basins(labels, dt, door_width_m):
    """Merge adjacent basins whose connection is wider than a doorway (no wall
    between them -> same room). Opening half-width ≈ max dt along the shared
    border; merge when 2·max_dt > door_width_m. Basins joined only by a
    doorway-width pinch stay separate (-> isConnectedTo)."""
    n = int(labels.max())
    if n <= 1:
        return labels
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    H, W = labels.shape
    border_max = {}
    for r in range(H):
        for c in range(W):
            a = labels[r, c]
            if a == 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if nr < H and nc < W:
                    b = labels[nr, nc]
                    if b > 0 and b != a:
                        key = (min(a, b), max(a, b))
                        border_max[key] = max(border_max.get(key, 0.0),
                                              dt[r, c], dt[nr, nc])
    for (a, b), mx in border_max.items():
        if 2.0 * mx > door_width_m:           # opening wider than a door -> merge
            parent[find(a)] = find(b)
    out = np.zeros_like(labels)
    for k in range(1, n + 1):
        out[labels == k] = find(k)
    return out


def _polygon(mask, res, origin, approx_eps_m=0.10):
    """Largest contour of a room mask -> world-coord polygon (or None)."""
    try:
        import cv2
    except ImportError:
        return None
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    eps = max(approx_eps_m / res, 1.0)
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return [[round(origin[0] + x * res, 3), round(origin[1] + y * res, 3)]
            for x, y in c]


def place_attributes(labels, n_rooms, res, origin, corridor_aspect=3.0,
                     corridor_width_m=2.0):
    """Per-room id/type/centroid/area/bbox/polygon."""
    cell = res * res
    places = []
    for k in range(1, n_rooms + 1):
        ys, xs = np.where(labels == k)
        area = len(xs) * cell
        cx = origin[0] + xs.mean() * res
        cy = origin[1] + ys.mean() * res
        x0 = origin[0] + xs.min() * res
        x1 = origin[0] + xs.max() * res
        y0 = origin[1] + ys.min() * res
        y1 = origin[1] + ys.max() * res
        w, h = x1 - x0, y1 - y0
        aspect = max(w, h) / max(min(w, h), 1e-6)
        short = min(w, h)
        ptype = ("corridor" if (aspect >= corridor_aspect
                                and short <= corridor_width_m) else "room")
        places.append({
            "id": f"place_{k}", "name": f"{ptype}_{k}", "type": ptype,
            "centroid": [float(round(cx,3)), float(round(cy,3))],
            "area_m2": float(round(area,2)),
            "bbox": [float(round(x0,3)), float(round(y0,3)), float(round(x1,3)), float(round(y1,3))],
            "polygon": _polygon(labels == k, res, origin),
        })
    return places


def place_connectivity(labels, n_rooms, res, origin, min_border_cells=3):
    """isConnectedTo between rooms that share a border (doorway)."""
    rels = []
    for a in range(1, n_rooms + 1):
        for b in range(a + 1, n_rooms + 1):
            # cells of room a that are 4-adjacent to room b
            ma, mb = labels == a, labels == b
            adj = np.zeros_like(ma)
            adj[1:, :] |= mb[:-1, :]
            adj[:-1, :] |= mb[1:, :]
            adj[:, 1:] |= mb[:, :-1]
            adj[:, :-1] |= mb[:, 1:]
            border = ma & adj
            nb = int(border.sum())
            if nb >= min_border_cells:
                ys, xs = np.where(border)
                rels.append({
                    "subject": f"place_{a}", "predicate": "isConnectedTo",
                    "object": f"place_{b}", "symmetric": True,
                    "doorway": [float(round(origin[0]+xs.mean()*res,3)),
                                float(round(origin[1]+ys.mean()*res,3))],
                    "border_cells": nb})
    return rels


def assign_objects(objects, labels, res, origin, places):
    """isInsideOf: map each object's (poseX, poseY) to a place label."""
    H, W = labels.shape
    id_by_k = {k + 1: places[k]["id"] for k in range(len(places))}
    out = []
    for o in objects:
        col = int(round((o["poseX"] - origin[0]) / res))
        row = int(round((o["poseY"] - origin[1]) / res))
        if 0 <= row < H and 0 <= col < W and labels[row, col] > 0:
            out.append({"subject": o["name"], "predicate": "isInsideOf",
                        "object": id_by_k[int(labels[row, col])]})
    return out


def build_place_message(pgm, yaml, objects=None, header=None, **seg_kw):
    """Full pipeline -> JSON-serializable dict (places + place/object relations)."""
    grid, res, origin = load_occupancy(pgm, yaml)
    seg = {k: seg_kw.pop(k) for k in ("door_radius_m", "min_room_m2", "merge_width_m", "peak_min_dist_m")
           if k in seg_kw}
    labels, n = segment_places(grid, res, origin, **seg)
    places = place_attributes(labels, n, res, origin)
    conn = place_connectivity(labels, n, res, origin)
    obj_place = assign_objects(objects, labels, res, origin, places) if objects else []
    header = header or {}
    return {
        "coordinateFrame": "map", "resolution": res,
        "manufacturer": header.get("manufacturer", "CASELAB"),
        "placeCounts": {"places": n, "isConnectedTo": len(conn),
                        "isInsideOf": len(obj_place)},
        "places": places,
        "placeRelations": conn,
        "objectPlaces": obj_place,
    }, labels
