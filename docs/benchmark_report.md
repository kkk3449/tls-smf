# Benchmark report — segmentation × classification on BLK360 industrial scans

**Scene:** `testroom_no_wall/stage2_no_wall.e57` (walls/floor pre-removed), voxel-downsampled to **122,738 points**.
**Goal:** test the thesis that *ScanNet-pretrained closed-set SOTA underperforms on industrial objects* versus open-vocab / geometric methods.
**Hardware:** RTX PRO 4000 Blackwell (sm_120), torch 2.11.0+cu128.

> **Caveat — these are qualitative + proxy metrics, not rigorous mIoU/AP.** No
> ground-truth instance/semantic labels exist for this scan yet, so we cannot
> report IoU, AP, or precision/recall. Instead we use observable proxies:
> point **coverage**, **clutter rate** (fraction of objects the open-vocab
> classifier could not name → a boundary-quality proxy), class **diversity**,
> and mean top-1 **confidence**. Read these as directional evidence, not as a
> leaderboard.

---

## 1. Benchmark matrix

|                         | **Uni3D** (native-3D open-vocab) | **PointCLIP v2** (depth-proj + CLIP) |
|-------------------------|----------------------------------|--------------------------------------|
| **DBSCAN** (geometric)  | ✅ run — primary arm             | ✅ run (collapsed, see §4)           |
| **SPFormer** (ScanNet closed-set, AAAI'23) | ✅ run — closed-set arm | — (not run; PCv2 already shown degenerate) |

Plus a **semantic-only** arm — **PTv3** (Point Transformer v3, CVPR'24) ScanNet-
pretrained — reported separately in §3b (per-point class, no instances, so it does
not feed Stage-B classification).

Dropped: **Mask3D / Part2Object** — both require MinkowskiEngine (CUDA 11.3 /
torch 1.x), fundamentally incompatible with Blackwell sm_120 (requires
cu128/torch 2.x). SPFormer (spconv-based, same query-mask-transformer paradigm)
is the substitute.

---

## 2. Segmentation comparison (Stage A) — the headline result

| Segmenter | Objects | Point coverage | Clutter rate¹ | Distinct classes² | Mean top-1 conf³ |
|-----------|--------:|---------------:|--------------:|------------------:|-----------------:|
| **DBSCAN** (eps=0.30, geometric) | 30 | **98.3 %** | **23.3 %** (7/30) | **13** | **0.342** |
| **SPFormer** (ScanNet closed-set) | 74 | 51.1 % | 45.9 % (34/74) | 9 | 0.279 |

¹ fraction of objects Uni3D labels `clutter` — a boundary-quality proxy (worse cuts → more unrecognizable fragments).
² number of distinct top-1 industrial classes among the objects — a diversity/usefulness proxy.
³ mean Uni3D top-1 score across objects — a confidence proxy.

**Reading.** On every proxy, the **geometric DBSCAN beats the deep closed-set
SPFormer** on this out-of-distribution industrial scene:

- **Coverage 98.3 % vs 51.1 %.** SPFormer leaves ~half the points unassigned —
  it only fires on regions resembling its ScanNet training classes, so genuine
  industrial structure (pipes, frames, cable trays) falls through.
- **Clutter doubles, 23 % → 46 %.** SPFormer over-segments into 74 fragments,
  many of which are partial/cross-object cuts that the downstream classifier
  can no longer name. DBSCAN's geometry-driven boundaries yield coherent objects.
- **Lower diversity and confidence** despite 2.5× more objects — more pieces,
  but each less recognizable.

---

## 3. What SPFormer "saw" (closed-set hallucination)

SPFormer's own ScanNet labels on the 88 raw instances it produced:

```
chair:67  table:6  window:5  otherfurniture:3  cabinet:2  curtain:2  door:2  picture:1
```

A room full of **chairs and tables** — there are none. The model is forced to
project industrial geometry onto its 18 indoor-furniture classes. This is the
thesis in one line: **a methodological SOTA, trained closed-set on indoor rooms,
hallucinates furniture on industrial objects and recovers only half the scene.**

---

## 3b. PTv3 semantic segmentation — the same failure, even more stark

**PTv3** (Point Transformer v3, CVPR'24, Pointcept) is a *semantic* segmenter
(per-point class among 20 ScanNet classes — no instances), the current SOTA
backbone for indoor semantic segmentation. We ran the authors' ScanNet-pretrained
weights on the **wall/floor-removed** scene (122,738 pts, grid 0.02 m). Per-point
class distribution:

| ScanNet class | points | share | mean conf |
|---------------|-------:|------:|----------:|
| **wall** | 56,894 | **46.4 %** | 0.49 |
| **refrigerator** | 23,999 | 19.6 % | 0.38 |
| cabinet | 12,024 | 9.8 % | 0.30 |
| counter | 8,285 | 6.8 % | 0.57 |
| bathtub | 5,263 | 4.3 % | 0.33 |
| bookshelf | 4,679 | 3.8 % | 0.34 |
| shower curtain | 4,562 | 3.7 % | 0.31 |
| door / bed / chair / … | rest | <3 % each | 0.2–0.3 |

*(17 of 20 classes fire; mean confidence 0.43.)*

The single most damning number: **46 % of the points are labeled `wall` on a scene
where the walls were already removed.** Nearly 20 % become `refrigerator`, and the
rest scatter across bathroom/kitchen fixtures (counter, bathtub, shower curtain).
PTv3 has no concept of pipes, valves, frames, or cable trays, so it forces every
industrial surface onto the nearest indoor-furniture prior — at low confidence.
A semantic SOTA fails on this domain *even more visibly* than the instance SOTA:
it cannot even abstain, it relabels empty industrial space as room structure.

---

## 4. Classification comparison (Stage B) — Uni3D vs PointCLIP v2

Run on the **DBSCAN objects** (30), sequentially (one EVA02-E CLIP fits in 24 GB):

| Classifier | Behavior | Mean conf | Agreement w/ Uni3D |
|------------|----------|----------:|-------------------:|
| **Uni3D** (native 3D) | diverse, plausible labels (13 classes) | 0.342 | — |
| **PointCLIP v2** (depth-proj + CLIP) | **collapsed → `pipe` 23/30** | 0.495 | **0 %** |

PointCLIP v2 renders the point cloud to 10 depth views and classifies with CLIP.
On these sparse, partial BLK360 objects the depth projections are too degenerate
to be discriminative — it confidently (high mean conf, a known CLIP
mis-calibration) collapses to a single dominant class. **Uni3D, operating
natively in 3D, is the appropriate Stage-B classifier for this data.** (A visual
depth-view audit to separate "method weakness" from "render mis-calibration"
remains open.)

---

## 5. Conclusion

The best-performing pipeline on this industrial scan is **DBSCAN + Uni3D**, not
the methodologically-SOTA **SPFormer + Uni3D**:

- DBSCAN + Uni3D: 98 % coverage, 23 % clutter, 13 coherent industrial classes.
- SPFormer + Uni3D: 51 % coverage, 46 % clutter, furniture-biased fragments.

This empirically supports the thesis: **closed-set SOTA segmentation trained on
indoor ScanNet does not transfer to industrial point clouds** — a geometric
(DBSCAN) + open-vocab native-3D (Uni3D) pipeline is both higher-coverage and
more semantically useful. **Methodological SOTA ≠ domain performance.**

### Standing limitation
All numbers above are proxies. Rigorous **mIoU / AP** requires ground-truth
instance labels for `stage2_no_wall.e57`, which do not yet exist — the single
biggest blocker to turning this directional result into a publishable metric.

---

*Artifacts:* `outputs/stage2_no_wall_objects/` (DBSCAN), `outputs/stage2_no_wall_spformer/` (SPFormer), each with `classification.csv` + `scene_classified.ply`.
