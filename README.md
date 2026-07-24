# blk360_seg — TLS → Queryable Robot Knowledge

Code for **"From Terrestrial Laser Scans to Queryable Robot Knowledge: A
VLM-Verified Framework for Incremental 3D Semantic Modeling"** (submitted to
*Electronics*).

The framework converts registered terrestrial laser scans (Leica BLK360 /
Cyclone360 exports) into a confidence-aware semantic knowledge graph and keeps
it current across repeated scans:

1. **Deterministic geometric backbone** — seeded-RANSAC structure removal,
   fixed-parameter DBSCAN clustering with a structure-remnant (ceiling-soffit)
   filter, explicit PCA-based object models. Re-runs are byte-identical on the
   same host.
2. **Multi-view VLM verification with escalation** — per-view forced-schema
   classification (Claude API), confidence-weighted late fusion, automatic
   escalation of split votes to a wider view set; unresolved objects stay
   explicitly `unverified`.
3. **Identity-preserving incremental update** — two-pass matching of re-scanned
   objects to mint-once node identities, confidence-gated node-level upserts,
   selective re-verification (unchanged geometry re-uses prior VLM verdicts).
4. **Derived layers** — verified-only place semantics (ring codes + LLM
   naming), structurally gated query views, VDA5050-style JSON export, Neo4j
   ingest, USD export for Isaac Sim.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # required for Stage B (VLM verification) only
```

The geometric backbone (Stage A) is CPU-only. The optional learned baselines
(Uni3D, PointCLIP V2, SPFormer, PTv3, Point-SAM) need `torch` + per-repo
dependencies and pretrained weights; see `patches/README.md` for the pinned
upstream commits. They are comparison baselines, not part of the proposed
pipeline.

## Pipeline quickstart

```bash
# Stage A: scan -> deterministic object instances (obj_*.ply + objects.csv)
python scripts/extract_objects.py --input scan.e57 --out outputs/scene_det

# Provisional open-vocab prior (optional; needs Uni3D weights) or skip
python scripts/classify_objects.py --objects-dir outputs/scene_det

# Explicit VDA5050 semantic-object records
python scripts/build_semantic_objects.py --objects-dir outputs/scene_det \
    --classification outputs/scene_det/classification.csv

# Multi-view render sets for verification
python scripts/render_object_views.py \
    --semantic outputs/scene_det/semanticObjects.json \
    --objects-dir outputs/scene_det --out outputs/scene_views

# Stage B: per-view VLM classification + late fusion + escalation (API cost)
python scripts/vlm_late_fusion.py \
    --input outputs/scene_det/semanticObjects.json \
    --views-dir outputs/scene_views --escalate \
    --out outputs/scene_det/semanticObjects.lf_esc.json

# Knowledge graph: first run creates, re-runs upsert (update/move/insert/absent)
python scripts/kg_upsert.py \
    --input outputs/scene_det/semanticObjects.lf_esc.json \
    --graph outputs/scene_kg.json --map-id scene \
    --timestamp 2026-07-24T12:00:00

# Re-scan economics: carry verdicts for byte-identical clusters, list the rest
python scripts/selective_reverify.py \
    --new outputs/rescan_det/semanticObjects.json \
    --new-objects-dir outputs/rescan_det \
    --prev outputs/scene_det/semanticObjects.lf_esc.json \
    --prev-objects-dir outputs/scene_det \
    --out outputs/rescan_det/semanticObjects.carried.json
```

Multi-epoch tooling: `scripts/register_epoch_scan.py` (FPFH+RANSAC → ICP
cross-epoch registration), `scripts/roomscope_transform.py` (frame transform +
room scoping), `scripts/synthetic_rescan.py` (controlled edit scenarios),
`scripts/place_layer.py` / `scripts/place_ring_naming.py` (place semantics).

Evaluation and paper artifacts: `scripts/analyze_escalation.py`,
`scripts/kg_query_benchmark.py`, `scripts/sensitivity_sweep.py`,
`scripts/make_paper_figs.py`.

## Layout

```
blk360seg/            library: io, preprocess, structure_filter, objects,
                      semantic_object, vlm_stage_b, kg, place_modeling,
                      spatial_relations, tosm_graph, usd_export, viz, ...
scripts/              pipeline CLIs (Stage A/B, KG, epochs, figures, baselines)
configs/              operating-point config + re-scan edit scenarios
data_prep/            e57/ply inspection and conversion helpers
patches/              pinned upstream commits + patches for baseline repos
```

## Data

The scans used in the paper were collected in a facility of CASELAB Co., Ltd.
and are subject to facility-owner restrictions; they are not distributed with
this repository. Point-cloud inputs are standard `.e57` (multi-setup files are
merged with per-setup registered poses applied) or `.csv`/`.ply`.

## Notes

- Stage B costs money (Anthropic API). Every VLM script prints per-run token
  usage and cost estimates.
- Determinism claims apply to Stage A only and are per-host (BLAS/compiler
  variation across hosts may flip borderline RANSAC/DBSCAN decisions).

## Citation

```bibtex
@article{kim2026tls2kg,
  author  = {Kim, Sangmin and Kim, Haryeong and Kuc, Tae-Yong},
  title   = {From Terrestrial Laser Scans to Queryable Robot Knowledge:
             A {VLM}-Verified Framework for Incremental 3{D} Semantic Modeling},
  journal = {Electronics},
  note    = {submitted},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
