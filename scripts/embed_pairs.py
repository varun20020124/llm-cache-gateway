"""Compute cosine similarity for every labeled pair.

Usage: python scripts/embed_pairs.py
"""

import json
from pathlib import Path

from benchmark.pairs import load_pairs, unique_queries
from cache.embeddings import cosine_similarity, embed

OUT_PATH = Path("results/similarities.jsonl")


def main():
    pairs = load_pairs()
    print(f"Loaded {len(pairs)} pairs")

    queries = unique_queries(pairs)
    vectors = embed(queries)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for p in pairs:
            sim = cosine_similarity(vectors[p.q1], vectors[p.q2])
            f.write(json.dumps({
                "id": p.id,
                "bucket": p.bucket,
                "perturbed": p.perturbed,
                "label": p.label,
                "similarity": round(sim, 6),
            }) + "\n")

    print(f"Wrote {len(pairs)} similarities to {OUT_PATH}")


if __name__ == "__main__":
    main()