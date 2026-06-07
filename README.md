# blk360_seg

Offline **semantic segmentation + open-vocabulary classification** of Leica
BLK360 (Cyclone360) point clouds, built to run and **compare two pipelines** on
an *industrial* environment (no fine-tuning, inference only):

| | Pipeline | Idea | Expectation |
|---|---|---|---|
| **Exp 1 (Baseline)** | **Mask3D + PTv3 + Uni3D**, ScanNet-pretrained | supervised SOTA applied as-is | weaker on industrial objects → shows SOTA's limits |
| **Exp 2 (Proposed)** | **Part2Object + Uni3D**, inference only | unsupervised discovery + open-vocab | better on industrial objects → unsup+open-vocab fits industry |

Metrics: **segmentation mIoU**, **classification Top-1**.
Class sets: **S3DIS-13 / ScanNet-20** (structure + furniture) by default — extend
with industrial classes (pipe, valve, tank, beam, rack, …) for the open-vocab part.

## ⚠️ The metrics need ground-truth labels

"No labels" only covers **training/inference** — both pipelines are inference-only.
But **mIoU and Top-1 are measured against ground truth**, so you must annotate a
small **evaluation set** (a few BLK360 scans labelled with the target classes,
including the industrial objects). Without it there's nothing to compute the
metrics against. Plan this first (or find a labelled industrial 3D set).

## Pipeline shape

```
load (csv/e57) → preprocess (downsample, normalize)
   ├─ Exp1: Mask3D (instances) + PTv3 (semantic)  ─┐
   │                                                ├─ Uni3D (open-vocab class) → labels
   └─ Exp2: Part2Object (unsup. objects)  ─────────┘
→ metrics vs GT (mIoU, Top-1) → compare Exp1 vs Exp2
```

## Models (separate repos, heavy, distinct deps)

- **PTv3 / Point Transformer V3** — semantic seg, via **Pointcept** (ScanNet/S3DIS ckpts).
- **Mask3D** — 3D instance seg (ScanNet ckpt).
- **Uni3D** (BAAI) — open-vocabulary 3D features aligned to CLIP; classify instances
  against text prompts (your class names, incl. industrial).
- **Part2Object** — unsupervised hierarchical 3D object discovery (no labels).

Each is its own repo with conflicting deps → run them as **adapters/steps**
(own sub-env, export cloud → run repo inference → import predictions) rather than
one monolithic env. This harness owns the common IO, the eval set, the metrics,
and the comparison.

## Recommended build order

1. **Env + CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
   (GPU exists per you; confirm the CUDA torch build works). Make a venv/conda env.
2. **Eval set**: annotate a few scans with the class set (the metric blocker).
3. **PTv3 (Pointcept)** semantic inference → first mIoU number. Most turnkey.
4. **Uni3D** open-vocab classification on instances → Top-1.
5. **Mask3D** instances (completes Exp 1).
6. **Part2Object** unsupervised (Exp 2), then the Exp1-vs-Exp2 comparison table.

## Current scaffold (runs today, no GPU)

Common infra is in place; a **geometric baseline** stands in for the DL pipelines
so IO / preprocessing / instance grouping / visualization can be validated now:

```
blk360seg/
├── io.py          .csv / .e57 -> (xyz, rgb)
├── preprocess.py  voxel downsample, normalize
├── classes.py     S3DIS-13 / ScanNet-20 names + palette
├── segmenter.py   Segmenter API; GeometricBaseline (runnable) + DL stubs
├── postprocess.py per-class DBSCAN -> instances
├── metrics.py     mIoU, Top-1  (need GT)
└── viz.py         color-by-label, save .ply/.csv, show
scripts/segment.py CLI (smoke test)
configs/default.yaml
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/segment.py --input ../testroom260601.e57 --viz
```

Next: replace the baseline with the Mask3D/PTv3/Uni3D and Part2Object adapters
(each behind the same `Segmenter` / a `Pipeline` interface), then wire `metrics.py`
against the annotated eval set.
