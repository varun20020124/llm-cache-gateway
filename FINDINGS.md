# Findings

## Summary

Semantic caches report hit rate. Hit rate says nothing about whether the cached answers were correct.

This project measures the false-hit rate of an embedding-similarity cache on financial document questions, and finds that at a common threshold of 0.90, **44% of cache hits return the wrong answer**. Raising the threshold makes this worse, not better. The failure is concentrated almost entirely in one dimension — fiscal year — which similarity thresholding cannot see.

A deterministic entity-check layered over the similarity match cuts the false-hit rate from **44% to 5% with no loss in hit rate**.

---

## Benchmark

300 labeled query pairs over SEC 10-K financial questions.

**Question space:** 3 companies (Apple, Microsoft, Nvidia) × 3 fiscal years (FY2022–FY2024) × 8 metrics (revenue, net income, operating income, R&D expense, gross margin, EPS, operating cash flow, total assets). Each metric has 3 phrasings that ask for the same number.

| Bucket | n | Label | Meaning |
|---|---|---|---|
| paraphrase | 100 | same | Same company/year/metric, different wording. Should hit. |
| hard_negative | 120 | different | Same wording, exactly one field changed. Must not hit. |
| unrelated | 80 | different | Company, year, and metric all differ. Easy miss. |

Hard negatives split evenly across three perturbation types: 40 year swaps, 40 company swaps, 40 metric swaps.

### Construction decisions

**Hard negatives hold the phrasing template fixed.** Only the perturbed field differs between the two strings. If both wording and entity varied, a cache miss could not be attributed to either. This produces a minimal pair that isolates the variable under test.

**Half the paraphrase pairs vary the year format** — `FY2023`, `FY 2023`, `fiscal 2023`, `fiscal year 2023`. Without this, the entity-check below would show zero over-blocking by construction rather than because it works.

**Every hard negative records which field was perturbed**, enabling results to be reported by perturbation type rather than as a single aggregate. This breakdown turned out to be the most important result.

**Deterministic seed (20260901).** Re-running the generator produces byte-identical output.

---

## Method

Queries are embedded with OpenAI `text-embedding-3-small` (1536 dimensions) and compared by cosine similarity:

```
cos(a, b) = (a · b) / (||a|| · ||b||)
```

Cosine is used rather than Euclidean distance because it measures direction while ignoring magnitude — two texts on the same subject should be considered similar regardless of length, and vector magnitude correlates with properties like text length that are not semantically meaningful.

Note that cosine similarity has a high floor within a single domain. Completely unrelated questions in this benchmark score around 0.5, not 0, because they share vocabulary, structure, and topic. Absolute values should be read against the distributions below, not against an intuition that 0.5 means "half similar."

A cache hit occurs when `similarity >= threshold`.

### Metric definitions

- **Hit rate** — of pairs that *should* hit, the fraction that did. This is what caching systems report.
- **False-hit rate** — of pairs that *did* hit, the fraction that were wrong.

The second is conditioned on hits, not on all pairs, because it answers the question a user actually has: given that I was served a cached answer, what is the chance it is wrong?

---

## Result 1: changing the year makes questions *more* similar than rephrasing them

| Group | n | mean | min | max |
|---|---|---|---|---|
| paraphrase | 100 | 0.904 | 0.820 | 0.967 |
| **year swap** | 40 | **0.976** | 0.964 | 0.989 |
| metric swap | 40 | 0.783 | 0.642 | 0.915 |
| company swap | 40 | 0.751 | 0.681 | 0.853 |
| unrelated | 80 | 0.505 | 0.418 | 0.644 |

The minimum year-swap similarity (0.964) exceeds the mean paraphrase similarity (0.904). The negative class sits almost entirely *above* the positive class — the distributions do not merely overlap, they are inverted.

Example, scored at 0.9884:

```
"What was Nvidia's gross margin in FY2023?"
"What was Nvidia's gross margin in FY2022?"
```

Nvidia's gross margin was roughly 57% in FY2023 and 73% in FY2024. Completely different answers.

The mechanism is mechanical: `FY2023` → `FY2022` changes one character in a ~50-character string. Both questions concern the same company and metric with identical sentence structure. The embedding model encodes that topical and structural sameness. It has no representation of the fact that the year is the semantically load-bearing token.

---

## Result 2: no threshold works

| threshold | hit rate | false-hit rate | wrong answers served |
|---|---|---|---|
| 0.80 | 100.0% | 39.4% | 65 |
| 0.84 | 89.0% | 36.0% | 50 |
| 0.88 | 70.0% | 38.1% | 43 |
| **0.90** | **55.0%** | **43.9%** | **43** |
| 0.92 | 34.0% | 54.7% | 41 |
| 0.94 | 15.0% | 72.7% | 40 |
| 0.96 | 1.0% | 97.6% | 40 |
| 1.00 | 0.0% | 0.0% | 0 |

**Raising the threshold increases the false-hit rate.** This follows directly from Result 1: tightening the threshold filters out paraphrases faster than it filters out year swaps, because year swaps are more similar than paraphrases. At 0.96 the cache almost never hits, and when it does it is wrong 97.6% of the time.

No threshold achieves under 1% false hits at any non-zero hit rate. The tradeoff curve has no shippable operating point.

### Leak rate by perturbation type (threshold 0.90)

| perturbation | leaked | rate |
|---|---|---|
| **year** | 40/40 | **100.0%** |
| metric | 3/40 | 7.5% |
| company | 0/40 | 0.0% |
| unrelated | 0/80 | 0.0% |

Every year swap leaks. The embedding model handles entity and topic changes correctly and is blind to temporal qualifiers.

This reframes the problem: the failure is not distributed noise that better tuning could reduce, it is concentrated in a single dimension that thresholding cannot see. The fix therefore cannot be a better threshold.

---

## Result 3: entity-aware invalidation

A deterministic check extracts fiscal year and company from both queries and forces a cache miss when either differs, regardless of embedding similarity.

Design decisions:

- **Biased toward blocking.** A false block costs one API call; a false allow costs a wrong answer. When extraction is uncertain, block.
- **Asymmetric on missing fields.** If neither query mentions a year, allow. If one does and the other does not, block.
- **No metric extraction.** Metrics have many synonyms ("net sales", "revenue", "top line"), so a lookup table would be brittle. The measured metric leak rate was low enough not to justify the complexity. This is a deliberate scope decision, and it is where the residual error lives.

### Guard activation

```
blocked 160/200 pairs that should not hit   (correct blocks)
blocked   0/100 pairs that should hit       (over-blocking)
```

The 160 correct blocks are 40 year swaps + 40 company swaps + 80 unrelated pairs (which also differ in company).

Zero over-blocking despite half the paraphrase pairs using different year formats on each side — the extractor handles `FY2023`, `FY 2023`, `fiscal 2023`, and `fiscal year 2023`.

### Baseline vs. guarded

| threshold | hit (base) | false (base) | hit (guard) | false (guard) |
|---|---|---|---|---|
| 0.80 | 100.0% | 39.4% | 100.0% | 12.3% |
| 0.84 | 89.0% | 36.0% | 89.0% | 9.2% |
| 0.88 | 70.0% | 38.1% | 70.0% | 4.1% |
| **0.90** | **55.0%** | **43.9%** | **55.0%** | **5.2%** |
| 0.92 | 34.0% | 54.7% | 34.0% | 2.9% |
| 0.94 | 15.0% | 72.7% | 15.0% | 0.0% |

**At threshold 0.90: false-hit rate falls from 43.9% to 5.2% with the hit rate unchanged at 55.0%.**

The hit rate is unchanged because the guard blocks only pairs the embedding was wrong about. It does not shift the tradeoff curve — it removes the failure mode the curve was measuring.

---

## The residual 5%

All remaining false hits at 0.90 are metric swaps:

```
0.9037  "How much cash did Nvidia generate from operations in FY2024?"
        "How much revenue did Nvidia generate in FY2024?"

0.9035  "What was Nvidia's total revenue in FY2023?"
        "What was Nvidia's net income in FY2023?"

0.9237  "How much operating profit did Nvidia report in FY2022?"
        "How much profit did Nvidia report for FY2022?"
```

Three false hits against roughly 55 true hits. With n=3, the guarded false-hit rate is noisy across thresholds — a single pair crossing the line moves it by nearly two points, which is why the guarded column is not monotonic.

The third pair is the interesting one: operating income vs. net income, differing by a single word. It behaves like the year swaps do.

---

## Known limitation

The metric-swap bucket is **easier than intended**. Pairs like "total revenue" vs. "diluted earnings per share" are lexically distant, so the embedding model separates them without difficulty — which is why the metric leak rate is 7.5% rather than something closer to the year swaps' 100%.

A harder version would pair lexically similar but numerically distinct metrics: gross margin vs. operating margin, total assets vs. total liabilities, operating cash flow vs. free cash flow. Under that benchmark the metric dimension would likely need its own guard, and the residual 5% would be larger.

This was found by reading generated output by hand, which is why the labeled set is manually reviewed rather than accepted as generated. It is documented rather than hidden, and results are reported by perturbation type so it stays visible.

---

## Reproducing

```bash
python scripts/build_pairs.py         # regenerate the benchmark
python scripts/embed_pairs.py         # compute similarities (needs OPENAI_API_KEY)
python scripts/run_sweep.py           # baseline threshold sweep
python scripts/run_guarded_sweep.py   # sweep with entity-aware invalidation
pytest tests/
```

Embeddings are cached to disk keyed by SHA-256 of model and text, so only the first run makes API calls. Full cost is under one cent.
