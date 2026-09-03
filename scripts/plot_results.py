"""Generate figures for the README.

Usage: python scripts/plot_results.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

SIMILARITIES_PATH = Path("results/similarities.jsonl")
GUARDED_PATH = Path("results/guarded_sweep.jsonl")
DOCS_DIR = Path("docs")

LABELS = {
    "paraphrase": "Paraphrase (should hit)",
    "year": "Year swap",
    "metric": "Metric swap",
    "company": "Company swap",
    "unrelated": "Unrelated",
}
COLORS = {
    "paraphrase": "#2E7D32",
    "year": "#C62828",
    "metric": "#EF6C00",
    "company": "#1565C0",
    "unrelated": "#616161",
}


def plot_distributions():
    groups = defaultdict(list)
    for line in SIMILARITIES_PATH.open():
        r = json.loads(line)
        groups[r["perturbed"] or r["bucket"]].append(r["similarity"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["unrelated", "company", "metric", "paraphrase", "year"]

    for key in order:
        ax.hist(groups[key], bins=25, range=(0.4, 1.0), alpha=0.65,
                label=LABELS[key], color=COLORS[key])

    ax.axvline(0.90, color="black", linestyle="--", linewidth=1)
    ax.text(0.902, ax.get_ylim()[1] * 0.92, "threshold 0.90", fontsize=9)

    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Pairs")
    ax.set_title("Year swaps are more similar than genuine paraphrases")
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = DOCS_DIR / "similarity_distributions.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def plot_tradeoff():
    thresholds, base_false, guard_false, hit_rates = [], [], [], []
    for line in GUARDED_PATH.open():
        row = json.loads(line)
        thresholds.append(row["threshold"])
        base_false.append(row["baseline"]["false_hit_rate"] * 100)
        guard_false.append(row["guarded"]["false_hit_rate"] * 100)
        hit_rates.append(row["baseline"]["hit_rate"] * 100)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(thresholds, base_false, "o-", color="#C62828",
            label="False-hit rate (similarity only)")
    ax.plot(thresholds, guard_false, "o-", color="#2E7D32",
            label="False-hit rate (with entity check)")
    ax.plot(thresholds, hit_rates, "s--", color="#9E9E9E",
            label="Hit rate (both)", alpha=0.7)

    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Percent")
    ax.set_title("Raising the threshold does not reduce wrong answers")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = DOCS_DIR / "false_hit_tradeoff.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    plot_distributions()
    plot_tradeoff()


if __name__ == "__main__":
    main()