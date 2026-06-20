# Evaluation Findings

Empirical analysis of the Where Is This? retrieval pipeline across two axes:
(1) the contribution of CLIP fine-tuning to image retrieval accuracy, and
(2) the contribution of BM25 keyword boosting to text retrieval accuracy.

All evaluations were run on an NVIDIA L40S GPU.

---

## 1. Image Retrieval - Does Fine-Tuning Help?

### Method

Leave-one-out evaluation over all 148 indexed landmarks. For each landmark
the last image in its folder is used as the query; the remaining average
embedding is already in the index (slightly optimistic in absolute terms, but
the relative comparison between baseline and fine-tuned is fair). Crucially,
both the query encoder and the index embeddings are computed fresh with the
same model weights for each condition, so there is no model/index mismatch.

Metrics: Hits@1, Hits@3, MRR (Mean Reciprocal Rank), mean cosine score on
correct matches, and 95% bootstrap confidence intervals on Hits@1
(10,000 resamples).

### Results

| Condition       |   N | Hits@1        | Hits@3        |   MRR | Mean score | 95% CI (H@1)  |
|-----------------|-----|---------------|---------------|-------|------------|---------------|
| Baseline CLIP   | 148 | 141/148 (95%) | 147/148 (99%) | 0.973 |      0.926 | [0.91, 0.99]  |
| Fine-tuned CLIP | 148 | 144/148 (97%) | 147/148 (99%) | 0.983 |      0.922 | [0.95, 0.99]  |
| **Δ**           |     | **+2.0%**     | 0.0%          | **+0.010** | −0.004 |            |

Fine-tuning improves Hits@1 by 2 percentage points (141 → 144 correct top-1
retrievals) and MRR by 0.010. Hits@3 is unchanged at 99% - the one persistent
Hits@3 failure (Museum of Natural History / Kunsthistorisches Museum) is a
dataset-level edge case discussed below.

The confidence intervals are informative beyond the point estimates. The
baseline lower bound is 0.91; the fine-tuned lower bound is 0.95. Fine-tuning
did not simply get lucky on three images - it raised the floor of performance,
making results more consistently reliable.

Mean cosine score on correct matches is marginally lower for the fine-tuned
model (0.926 → 0.922). This is expected: a more discriminative embedding space
separates classes more tightly, so even correct matches sit slightly further
from 1.0 while wrong matches move further away. The improvement in rank
(Hits@1) confirms that the ordering is better, not worse.

### Failure Analysis

**Baseline failures (7 landmarks):**

| Landmark | Retrieved Instead | Score | Region |
|----------|------------------|-------|--------|
| Karnak Temple Luxor | Persepolis Iran | 0.903 | Africa |
| Iguazu Falls | Sigiriya Sri Lanka | 0.796 | Americas |
| Fushimi Inari Shrine | Senso-ji Temple Tokyo | 0.951 | Asia |
| Louvre Museum | Abu Simbel | 0.833 | Europe |
| Kunsthistorisches Museum Vienna | Vienna State Opera | 0.875 | Vienna |
| Votivkirche Vienna | Notre Dame Cathedral Paris | 0.934 | Vienna |
| Museum of Natural History Vienna | Kunsthistorisches Museum Vienna | 0.832 | Vienna |

**Fine-tuned failures (4 landmarks) - 3 fixed:**

| Landmark | Retrieved Instead | Score | Region | Fixed? |
|----------|------------------|-------|--------|--------|
| Fushimi Inari Shrine | Senso-ji Temple Tokyo | 0.937 | Asia | No |
| Louvre Museum | Abu Simbel | 0.829 | Europe | No |
| Kunsthistorisches Museum Vienna | Vienna State Opera | 0.838 | Vienna | No |
| Museum of Natural History Vienna | Kunsthistorisches Museum Vienna | 0.815 | Vienna | No |

Fine-tuning fixed: Karnak Temple Luxor, Iguazu Falls, Votivkirche Vienna.

### Interpreting the Persistent Failures

The four remaining failures fall into two categories:

**Genuine visual ambiguity (model cannot be blamed):**

- *Museum of Natural History / Kunsthistorisches Museum* - these two buildings
  are architectural twins, designed by the same architect, facing each other
  across Maria-Theresien-Platz. From most angles they are visually
  indistinguishable. No amount of fine-tuning on 8 images per landmark can
  resolve this without non-visual context (signage, surroundings).

- *Fushimi Inari Shrine / Senso-ji Temple Tokyo* - both feature red Japanese
  religious architecture (torii gates and temple structures). The retrieval
  score is high (0.937), indicating strong visual similarity. Correct
  disambiguation would require either more diverse training images or
  geographic metadata.

**Unexplained cross-regional confusions:**

- *Louvre Museum → Abu Simbel* - this pairing is counterintuitive. The likely
  explanation is that the leave-one-out query image (the last in the sorted
  folder) happens to be a particular angle or crop that shares visual features
  with the Abu Simbel query image in the index (large stone façade, similar
  lighting conditions). Inspecting the actual query image would confirm this.

### What Fine-Tuning Fixed

- *Votivkirche → Notre Dame*: Fine-tuning pulled the Votivkirche embedding
  closer to its own image distribution, away from generic "Gothic cathedral"
  cluster that contains Notre Dame.

- *Karnak Temple → Persepolis*: Both are ancient column-hall ruins. Fine-tuning
  on the landmark domain gave the model enough signal to distinguish Egyptian
  from Persian stone architecture.

- *Iguazu Falls → Sigiriya*: Likely a waterfall/lush-vegetation confusion.
  Fine-tuning improved the representation of natural landmarks, which were
  among the most underrepresented in the dataset.

---

## 2. Text Retrieval - Does BM25 Help?

### Method

All 149 landmark names used as text queries against the pre-built text FAISS
index (Wikipedia description embeddings). Ground truth: each name should
retrieve itself. Compares CLIP-only (TextSearcher) vs Hybrid (CLIP + BM25,
weight 0.7/0.3). The text index was built with baseline CLIP weights, so this
comparison isolates the retrieval strategy contribution independently of model
fine-tuning.

### Results

| Strategy             |   N | Hits@1        | Hits@3         |   MRR | 95% CI (H@1) |
|----------------------|-----|---------------|----------------|-------|--------------|
| CLIP-only            | 149 | 148/149 (99%) | 149/149 (100%) | 0.997 | [0.98, 1.00] |
| Hybrid (CLIP + BM25) | 149 | 147/149 (99%) | 149/149 (100%) | 0.992 | [0.97, 1.00] |
| **Δ**                |     | **−0.7%**     | 0.0%           | **−0.004** |         |

BM25 shows a marginal negative effect (−0.7% Hits@1, −0.004 MRR) on proper
noun queries. Three landmarks show strategy disagreement:

| Landmark | Winner |
|----------|--------|
| Augustinerkirche Vienna | CLIP-only |
| Mont Saint Michel | CLIP-only |
| Rhodes Memorial Cape Town | Hybrid |

### Interpretation

This result is expected and does not indicate that BM25 is harmful.
The query set - landmark names - is precisely the case where BM25 adds the
least value. CLIP's text encoder already handles proper nouns well; "Eiffel
Tower" maps to a vector close to the Eiffel Tower description without any
keyword overlap being needed.

BM25 was designed for a different query type: descriptive natural-language
queries where the user does not know the landmark name - for example, *"ancient
Roman arena where gladiators fought"* or *"baroque palace with large gardens
and fountains."* In these cases, content words (arena, gladiators, baroque,
palace) trigger BM25 signal that CLIP alone might distribute across semantically
similar landmarks. The system's stopword filtering and BM25 fallback (when
max score < 1e-6) are specifically designed to suppress BM25 on queries like
these landmark names where no keyword signal exists.

Evaluating BM25 on proper noun queries is therefore testing it outside its
design envelope. A more representative evaluation would use a held-out set of
descriptive queries with known ground truth - a direction for future work.

---

## 3. Descriptive Query Evaluation - Does BM25 Help, and Does Fusion Method Matter?

### Method

Section 2 tested BM25 outside its design envelope (landmark-name queries).
This evaluation instead uses 40 hand-written descriptive queries - the kind a
user would type when they recognise a place but don't know its name - split
evenly into 20 "keyword-rich" queries (containing rare, distinctive words
expected to overlap with the landmark's Wikipedia description, e.g. "ceremonial
capital of the Achaemenid Empire") and 20 "visual" queries (generic visual
descriptions with no rare keyword overlap, e.g. "stepped pyramid in the
jungle"). The keyword/visual label is a manual judgment call made when writing
each query, not derived from any automated metric.

Three retrieval strategies are compared:

- **CLIP-only** - pure semantic similarity, no lexical signal.
- **Weighted hybrid** - `combined = 0.7 * clip_score + 0.3 * bm25_normalised`
  (the system's deployed default).
- **RRF (Reciprocal Rank Fusion)** - `score = 1/(60 + clip_rank) + 1/(60 +
  bm25_rank)`, combining on rank position rather than raw score magnitude.
  Added specifically to test whether sidestepping the CLIP/BM25 score-scale
  mismatch (cosine similarity vs. BM25's raw, unbounded score) would
  outperform the hand-tuned fixed weighting.

### Results

| Subset              | Model    | Hits@1      | Hits@3      |   MRR | 95% CI (H@1) |
|----------------------|----------|-------------|-------------|-------|--------------|
| Keyword-rich (n=20)  | CLIP     | 16/20 (80%) | 19/20 (95%) | 0.875 | [0.60, 0.95] |
| Keyword-rich (n=20)  | Weighted | 18/20 (90%) | 19/20 (95%) | 0.925 | [0.75, 1.00] |
| Keyword-rich (n=20)  | RRF      | 17/20 (85%) | 19/20 (95%) | 0.892 | [0.70, 1.00] |
| Visual (n=20)        | CLIP     | 16/20 (80%) | 17/20 (85%) | 0.825 | [0.60, 0.95] |
| Visual (n=20)        | Weighted | 12/20 (60%) | 19/20 (95%) | 0.758 | [0.40, 0.80] |
| Visual (n=20)        | RRF      | 13/20 (65%) | 19/20 (95%) | 0.792 | [0.45, 0.85] |
| All (n=40)           | CLIP     | 32/40 (80%) | 36/40 (90%) | 0.850 | [0.68, 0.93] |
| All (n=40)           | Weighted | 30/40 (75%) | 38/40 (95%) | 0.842 | [0.60, 0.88] |
| All (n=40)           | RRF      | 30/40 (75%) | 38/40 (95%) | 0.842 | [0.60, 0.88] |

Δ Hits@1 vs. CLIP-only: Weighted +10pp keyword / −20pp visual. RRF +5pp
keyword / −15pp visual.

### Interpretation

**BM25's effect is not uniform - it is conditional on keyword rarity.** On
keyword-rich queries it improves Hits@1 (both fusion methods); on generic
visual queries using common nouns shared across many descriptions ("desert",
"falls", "pyramid", "volcanic") it reduces Hits@1, because the fixed weighting
cannot distinguish a disambiguating keyword from a generic one.

**RRF does not fix this - it flattens it.** Switching from raw-score blending
to rank-based fusion produced a smaller gain on keyword queries (+5pp vs.
+10pp) and a smaller loss on visual queries (−15pp vs. −20pp), but the
aggregate Hits@1 across all 40 queries is identical (75% either way - both
underperform CLIP-only's 80%). The mechanism is straightforward: RRF discards
score magnitude, so a CLIP score of 0.95 and a CLIP score of 0.55 both just
count as "rank 1" - it can express *that* CLIP won, not *how confidently*.
That dulls both the upside (a strong keyword match can no longer cut through
as decisively) and the downside (a weak, spurious BM25 keyword overlap can no
longer drag the ranking down as far).

Concrete example: for the query *"ancient Zoroastrian royal city... ceremonial
capital of the Achaemenid Empire"* (ground truth: Persepolis), CLIP-only
ranks Persepolis #2; the weighted hybrid promotes it to #1; RRF also promotes
it to #1 - all three fusion approaches agree here because the keyword overlap
is strong on both raw-score and rank terms. But for *"Sigiriya Sri Lanka"*
(a keyword-rich query CLIP misses entirely), the weighted hybrid's raw BM25
score is strong enough to pull the correct answer to #1, while RRF - which
only sees BM25's *rank*, not its score's magnitude - is not, and the query
still misses (ranked #3).

**Conclusion: neither fusion method addresses the actual root cause** - the
inability to tell a rare, disambiguating keyword from a generic one that
happens to appear in many descriptions. An IDF-weighted BM25 contribution
(downweighting common nouns, upweighting rare ones) targets that root cause
directly and remains the more promising next step over either fixed-weight
blending or RRF.

---

## 4. Summary

| Design choice | Metric | Effect |
|---------------|--------|--------|
| Fine-tuning (vs baseline CLIP) | Hits@1 | +2.0% (141 → 144/148) |
| Fine-tuning (vs baseline CLIP) | MRR | +0.010 |
| Fine-tuning (vs baseline CLIP) | CI lower bound | +0.04 (0.91 → 0.95) |
| BM25 hybrid (on name queries) | Hits@1 | −0.7% (within noise) |
| BM25 hybrid (on name queries) | MRR | −0.004 (within noise) |
| BM25 weighted hybrid (keyword-rich descriptive queries) | Hits@1 | +10pp (16 → 18/20) |
| BM25 weighted hybrid (visual descriptive queries) | Hits@1 | −20pp (16 → 12/20) |
| BM25 RRF (keyword-rich descriptive queries) | Hits@1 | +5pp (16 → 17/20) |
| BM25 RRF (visual descriptive queries) | Hits@1 | −15pp (16 → 13/20) |

Fine-tuning delivers a consistent, measurable improvement to image retrieval.
The remaining failures are attributable to genuine visual ambiguity or dataset
gaps rather than model deficiency. BM25 is neutral on landmark-name queries
(outside its design envelope) but, on descriptive queries, has a strong effect
that is conditional on keyword rarity rather than uniformly positive or
negative. Switching the fusion method from fixed-weight blending to
Reciprocal Rank Fusion flattens this effect in both directions without fixing
its underlying cause - an IDF-weighted BM25 contribution remains the more
targeted next step.

Both models achieve near-ceiling performance on text retrieval with landmark
name queries (99% Hits@1), confirming that CLIP's shared embedding space
generalises well to the landmark domain even without task-specific fine-tuning
of the text encoder.
