"""Sweep the similarity threshold and measure hit rate against false hits."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

SIMILARITIES_PATH = Path("results/similarities.jsonl")


@dataclass
class Result:
    threshold: float
    hit_rate: float          # of should-hit pairs, fraction that hit
    false_hit_rate: float    # of all hits, fraction that were wrong
    true_hits: int
    false_hits: int
    missed: int              # should have hit, didn't


def load_similarities(path: Path = SIMILARITIES_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/embed_pairs.py"
        )
    return [json.loads(line) for line in path.open()]


def evaluate_at(records: list[dict], threshold: float) -> Result:
    true_hits = false_hits = missed = 0

    for r in records:
        hit = r["similarity"] >= threshold
        should_hit = r["label"] == "same"

        if hit and should_hit:
            true_hits += 1
        elif hit and not should_hit:
            false_hits += 1
        elif not hit and should_hit:
            missed += 1

    total_should_hit = true_hits + missed
    total_hits = true_hits + false_hits

    return Result(
        threshold=threshold,
        hit_rate=true_hits / total_should_hit if total_should_hit else 0.0,
        false_hit_rate=false_hits / total_hits if total_hits else 0.0,
        true_hits=true_hits,
        false_hits=false_hits,
        missed=missed,
    )


def sweep(records: list[dict], start=0.70, stop=1.00, step=0.01) -> list[Result]:
    thresholds = [start + i * step for i in range(int((stop - start) / step) + 1)]
    return [evaluate_at(records, round(t, 4)) for t in thresholds]


def false_hits_by_type(records: list[dict], threshold: float) -> dict[str, dict]:
    """Which perturbation types leak through at this threshold."""
    breakdown = {}
    for r in records:
        if r["label"] == "same":
            continue
        key = r["perturbed"] or r["bucket"]
        entry = breakdown.setdefault(key, {"total": 0, "leaked": 0})
        entry["total"] += 1
        if r["similarity"] >= threshold:
            entry["leaked"] += 1
    for entry in breakdown.values():
        entry["leak_rate"] = entry["leaked"] / entry["total"]
    return breakdown