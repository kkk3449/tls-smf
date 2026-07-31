#!/usr/bin/env python3
"""Resplit of electrical_cabinet_028 (owner GT: monitor wall = 10 monitors
+ desk + wall), 2026-07-31. Kept as provenance/reproduction record — the
run that produced monitor_028_m01..m10 / desk_028 / clutter_028_annex in
det4 + KG rev10 was executed inline with exactly this procedure.

Steps:
  1. obj_0028.ply -> vis_n2 frame; map wall-line band (d<0.08) dropped
     (12,973 wall pts; monitors protrude enough to survive the tight band).
  2. support-plane layering (desk slab peak z=-0.19): panel / slab / below,
     DBSCAN each layer in XY. Under-desk clusters merged into the desk;
     the off-wall annex block stays separate (not in owner GT -> clutter).
  3. panel points: principal along-wall coord u; density valleys in the
     monitor z-band (0.05-0.85, 3 cm bins, count<18, run>=3) -> 5 columns;
     forced upper/lower row cut at z=0.37 -> exactly 10 monitor units
     (matches the owner's count; bezel-adjacent pairs cannot be separated
     by DBSCAN/erosion because mounts connect them).
  4. registration: monitors + desk verified_owner (owner GT is the label
     source), annex unverified clutter; parent 028 node -> absent
     (superseded). KG update was SURGICAL (nodes via kg._node_from_record,
     edges recomputed): a full geometric re-upsert of flat record echoes
     revives old-epoch absent nodes as duplicate inserts — do not do that
     on cross-epoch graphs.
"""
print(__doc__)
