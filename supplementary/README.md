# Supplementary materials

Assets that reproduce the paper's evaluations (paths are repo-relative).

## Ground truth (owner-audited)

| file | scene | contents |
|---|---|---|
| `showroom_gt.json` | showroom 57-cluster set | confirmed/plausible/unknown grades, structure-artifact flags |
| `showroom_gt_draft.json` | showroom | pre-owner render-audit draft (provenance) |
| `showroom_det_gt.json` | showroom det 38-set | GT transferred by bidirectional point overlap (5 cm, >=0.5) |
| `vis_n2_gt.json` | robot hall (room-scoped 51) | incl. owner answers + audit notes |
| `t3_owner_gt.json` | T3 re-scan (41 clusters) | exhaustive owner audit |
| `cafe8f_gt.json` | cafeteria stress scene (25 scored) | owner-corrected sheet; `_variantB` = fragment-excluded grading |

Grading practice and the anchoring limitation are described in Sec. 5.1 of
the paper; the audit notes embedded in the JSONs record every owner
override of a model-proposed label.

## Query benchmark

`query_bench_showroom_v5.json` / `query_bench_vis_n2.json`: the four-condition
benchmark (questions, per-condition answers, per-claim scoring, hallucination
traps). Generated and scored by `scripts/kg_query_benchmark.py`.

## Scoring

- Verification scoring incl. the synonym-tolerant matcher (the exact table
  used for every number in Sec. 5.3/5.8): `scripts/analyze_escalation.py`.
- Escalation protocol and forced answer schema (prompts, tool schema,
  vote fusion): `blk360seg/vlm_stage_b.py` + `scripts/vlm_late_fusion.py`.
- Render protocol (512 px, 8 azimuths + cardinal zooms): `scripts/render_object_views.py`.

## Configuration

`pipeline_config.yaml`: the fixed Stage-A parameters (3 cm voxel,
DBSCAN eps/min_pts, plane removal) used unchanged for every scene,
including the held-out cafeteria. Verifier models and evaluation dates:
claude-sonnet-4-6 (main campaign, July--August 2026), comparison models
and prices as footnoted in Table 8 of the paper.
