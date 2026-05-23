# Ablation Study Results

## Image Retrieval (leave-one-out, on-the-fly re-embedding)

| Condition | N | Hits@1 | Hits@3 | MRR | Mean score | 95% CI |
|-----------|---|--------|--------|-----|------------|--------|
| Baseline CLIP | 148 | 141/148 (95%) | 147/148 (99%) | 0.973 | 0.926 | [0.91, 0.99] |
| Fine-tuned CLIP | 148 | 144/148 (97%) | 147/148 (99%) | 0.983 | 0.922 | [0.95, 0.99] |

**Fine-tuning Δ Hits@1: +2.0%**


## Text Retrieval - Strategy Comparison (landmark name queries)

| Strategy | N | Hits@1 | Hits@3 | MRR | 95% CI |
|----------|---|--------|--------|-----|--------|
| CLIP-only | 149 | 148/149 (99%) | 149/149 (100%) | 0.997 | [0.98, 1.00] |
| Hybrid (CLIP + BM25) | 149 | 147/149 (99%) | 149/149 (100%) | 0.992 | [0.97, 1.00] |

**BM25 Δ Hits@1: -0.7%**

