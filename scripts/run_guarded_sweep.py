"""Re-run the threshold sweep with entity-aware invalidation applied.

Usage: python scripts/run_guarded_sweep.py
"""

import json
from pathlib import Path

from benchmark.pairs import load_pairs
from cache.invalidation import blocked_reason

SIMILARITIES_PATH = Path("results/similarities.jsonl")
OUT_PATH = Path("results/guarded_sweep.jsonl")


def evaluate(records, threshold, guarded):
    true_hits = false_hits = missed = 0
    for r in records:
        hit = r["similarity"] >= threshold
        if guarded and r["blocked"]:
            hit = False
        should_hit = r["label"] == "same"
        if hit and should_hit:
            true_hits += 1
        elif hit:
            false_hits += 1
        elif should_hit:
            missed += 1

    total_should = true_hits + missed
    total_hits = true_hits + false_hits
    return {
        "threshold": threshold,
        "hit_rate": true_hits / total_should if total_should else 0.0,
        "false_hit_rate": false_hits / total_hits if total_hits else 0.0,
        "false_hits": false_hits,
    }


def main():
    pairs = {p.id: p for p in load_pairs()}
    records = []
    for line in SIMILARITIES_PATH.open():
        r = json.loads(line)
        p = pairs[r["id"]]
        r["blocked"] = blocked_reason(p.q1, p.q2) is not None
        records.append(r)

    # How often does the guard fire, and on what?
    blocked_by_label = {"same": 0, "different": 0}
    for r in records:
        if r["blocked"]:
            blocked_by_label[r["label"]] += 1
    n_same = sum(1 for r in records if r["label"] == "same")
    n_diff = len(records) - n_same

    print("Guard activation:")
    print(f"  blocked {blocked_by_label['different']}/{n_diff} pairs that "
          f"should not hit (correct blocks)")
    print(f"  blocked {blocked_by_label['same']}/{n_same} pairs that "
          f"should hit (over-blocking)")

    thresholds = [0.80 + i * 0.02 for i in range(11)]
    print(f"\n{'thresh':>7} {'hit (base)':>11} {'false (base)':>13} "
          f"{'hit (guard)':>12} {'false (guard)':>14}")
    print("-" * 62)

    rows = []
    for t in thresholds:
        t = round(t, 2)
        base = evaluate(records, t, guarded=False)
        guard = evaluate(records, t, guarded=True)
        rows.append({"threshold": t, "baseline": base, "guarded": guard})
        print(f"{t:>7.2f} {base['hit_rate']:>10.1%} {base['false_hit_rate']:>12.1%} "
              f"{guard['hit_rate']:>11.1%} {guard['false_hit_rate']:>13.1%}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    base90 = evaluate(records, 0.90, guarded=False)
    guard90 = evaluate(records, 0.90, guarded=True)
    print(f"\nAt threshold 0.90:")
    print(f"  baseline: {base90['hit_rate']:.1%} hits, "
          f"{base90['false_hit_rate']:.1%} false-hit rate")
    print(f"  guarded:  {guard90['hit_rate']:.1%} hits, "
          f"{guard90['false_hit_rate']:.1%} false-hit rate")


if __name__ == "__main__":
    main()