"""Run the threshold sweep and report the operating points.

Usage: python scripts/run_sweep.py
"""

import json
from dataclasses import asdict
from pathlib import Path

from benchmark.evaluate import (
    evaluate_at,
    false_hits_by_type,
    load_similarities,
    sweep,
)

OUT_PATH = Path("results/sweep.jsonl")


def main():
    records = load_similarities()
    results = sweep(records)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    print(f"{'thresh':>7} {'hit rate':>9} {'false hits':>11} {'n_false':>8}")
    print("-" * 40)
    for r in results:
        if round(r.threshold * 100) % 2 == 0:
            print(f"{r.threshold:>7.2f} {r.hit_rate:>8.1%} "
                  f"{r.false_hit_rate:>10.1%} {r.false_hits:>8}")

    # Best threshold holding false hits under 1%.
    clean = [r for r in results if r.false_hit_rate <= 0.01]
    if clean:
        best = max(clean, key=lambda r: r.hit_rate)
        print(f"\nBest under 1% false hits: threshold {best.threshold:.2f}, "
              f"hit rate {best.hit_rate:.1%}")
    else:
        print("\nNo threshold achieves under 1% false hits.")

    # A typical production default.
    default = evaluate_at(records, 0.90)
    print(f"\nAt a common default of 0.90:")
    print(f"  hit rate       {default.hit_rate:.1%}")
    print(f"  false-hit rate {default.false_hit_rate:.1%} "
          f"({default.false_hits} wrong answers served)")

    print("\n  leak rate by perturbation type:")
    for key, stats in sorted(false_hits_by_type(records, 0.90).items()):
        print(f"    {key:<12} {stats['leaked']:>3}/{stats['total']:<3} "
              f"{stats['leak_rate']:>6.1%}")


if __name__ == "__main__":
    main()