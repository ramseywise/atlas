---
name: segment-analysis
description: "Run and interpret customer segmentation results. Covers profiling, embedding choice, algorithm selection, evaluation, and Neo4j knowledge graph queries. Use after running the segmentation pipeline or when investigating segment quality."
disable-model-invocation: true
allowed-tools: Read Bash Grep Glob Write
---

Analyze customer segmentation for: `$ARGUMENTS` (describe the question or issue).

## Pipeline entry point

```bash
uv run python -m pipelines.segment
```

Runs: Profiler → Embedder → Clusterer → Evaluator → Labeler

## What each node produces

| Node | Output | Key file |
|------|--------|----------|
| Profiler | `CustomerProfile` feature vectors | `core/preprocessing/customer.py` |
| Embedder | float32 embedding matrix (tsfresh or Chronos) | `core/preprocessing/embeddings.py` |
| Clusterer | cluster labels + algorithm used | `core/segmentation/algorithms.py` |
| Evaluator | `SegmentEvalReport` (silhouette, CH, DB) | `core/segmentation/evaluation.py` |
| Labeler | human-readable names + descriptions per cluster | `core/segmentation/naming.py` |

## Investigating segment quality

### Read the eval report

```python
from core.segmentation.evaluation import evaluate_clusters
report = evaluate_clusters(embeddings, labels)
# report.silhouette, report.calinski_harabasz, report.davies_bouldin
# report.cluster_sizes — check for any < 3
```

### Check algorithm selection logic

The clusterer tries HDBSCAN first; falls back to KMeans if silhouette < 0.25:
- `core/segmentation/algorithms.py` → `select_best()`
- If KMeans was used: investigate whether feature quality is the root cause (not just algorithm choice)

### Diagnose poor silhouette scores

| Root cause | Diagnostic | Fix |
|------------|-----------|-----|
| Too many features, noise | Check `CustomerProfile` feature count | Add feature selection / PCA |
| Time-series without embedding | Embedder skipped | Use `embed_tsfresh()` or `embed_chronos()` |
| Wrong n_clusters (KMeans) | Try silhouette for k=2..10 | Grid over k |
| Outlier customers dominating | Check cluster size distribution | Set HDBSCAN `min_cluster_size` higher |

## Querying segments from Neo4j

```python
from core.knowledge.graph import AtlasGraph

g = AtlasGraph()

# All customers in a segment
g.query("MATCH (c:Customer)-[:BELONGS_TO]->(s:Segment {name: $name}) RETURN c", name="...")

# Customers in segment X that transact with merchant Y
g.query("""
  MATCH (c:Customer)-[:BELONGS_TO]->(s:Segment {name: $seg})
  MATCH (c)-[:TRANSACTS_WITH]->(m:Merchant {name: $merchant})
  RETURN c.id
""", seg="...", merchant="...")
```

Local dev Neo4j: `docker run -p 7474:7474 -p 7687:7687 neo4j:latest`
Env vars: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

## Re-seeding and refreshing

```bash
# Seed metric definitions
uv run python -m core.knowledge.seeds

# Trigger re-segmentation via API
curl -X POST http://localhost:8000/segments/refresh

# Or run pipeline directly
uv run python -m pipelines.segment
```

## Output from this skill

Produce:
1. **Segment summary table**: Name | Size | Silhouette contribution | Key features
2. **Quality verdict**: pass/fail per threshold, what's at risk
3. **Root cause** (if quality is poor): one hypothesis with evidence
4. **Recommended action**: one concrete next step
