# scripts/build_pairs.py
"""Generate labeled query pairs for semantic cache evaluation.

Three buckets:
  paraphrase    - same question, different wording        -> should hit cache
  hard_negative - same wording, one entity changed        -> must not hit cache
  unrelated     - different company, year, and metric     -> easy miss

Hard negatives hold the phrasing template fixed so the only textual
difference is the perturbed field. Usage: python scripts/build_pairs.py
"""

import json
import random
from itertools import combinations, product
from pathlib import Path

SEED = 20260901
OUT_PATH = Path("data/pairs.jsonl")

N_PARAPHRASE = 100
N_HARD_NEGATIVE = 120
N_UNRELATED = 80

COMPANIES = ["Apple", "Microsoft", "Nvidia"]
YEARS = ["FY2022", "FY2023", "FY2024"]
# Ways a user might write the same fiscal year.
YEAR_FORMATS = {
    "FY2022": ["FY2022", "fiscal 2022", "fiscal year 2022", "FY 2022"],
    "FY2023": ["FY2023", "fiscal 2023", "fiscal year 2023", "FY 2023"],
    "FY2024": ["FY2024", "fiscal 2024", "fiscal year 2024", "FY 2024"],
}

# Each metric maps to phrasings that ask for the SAME number.
METRICS = {
    "revenue": [
        "What was {company}'s total revenue in {year}?",
        "How much revenue did {company} generate in {year}?",
        "What were {company}'s net sales for {year}?",
    ],
    "net_income": [
        "What was {company}'s net income in {year}?",
        "How much profit did {company} report for {year}?",
        "What were {company}'s earnings in {year}?",
    ],
    "operating_income": [
        "What was {company}'s operating income in {year}?",
        "How much operating profit did {company} report in {year}?",
        "What was {company}'s income from operations for {year}?",
    ],
    "rd_expense": [
        "What was {company}'s R&D expense in {year}?",
        "How much did {company} spend on research and development in {year}?",
        "What were {company}'s research and development costs for {year}?",
    ],
    "gross_margin": [
        "What was {company}'s gross margin in {year}?",
        "What gross margin did {company} report for {year}?",
        "How much gross profit margin did {company} have in {year}?",
    ],
    "eps": [
        "What was {company}'s diluted earnings per share in {year}?",
        "What diluted EPS did {company} report for {year}?",
        "How much did {company} earn per diluted share in {year}?",
    ],
    "operating_cash_flow": [
        "What was {company}'s operating cash flow in {year}?",
        "How much cash did {company} generate from operations in {year}?",
        "What was {company}'s cash flow from operating activities in {year}?",
    ],
    "total_assets": [
        "What were {company}'s total assets in {year}?",
        "How much in total assets did {company} report for {year}?",
        "What was {company}'s total asset balance at the end of {year}?",
    ],
}

METRIC_KEYS = list(METRICS)


def render(company, year, metric, template_idx):
    return METRICS[metric][template_idx].format(company=company, year=year)


def make_pair(pair_id, q1, q2, label, bucket, perturbed=None, meta=None):
    record = {
        "id": pair_id,
        "q1": q1,
        "q2": q2,
        "label": label,
        "bucket": bucket,
        "perturbed": perturbed,
    }
    if meta:
        record["meta"] = meta
    return record


def build_paraphrases(rng, n):
    """Same (company, year, metric); different phrasing, sometimes
    a different way of writing the year."""
    candidates = []
    for company, year, metric in product(COMPANIES, YEARS, METRIC_KEYS):
        n_templates = len(METRICS[metric])
        for i, j in combinations(range(n_templates), 2):
            candidates.append((company, year, metric, i, j))

    rng.shuffle(candidates)
    pairs = []
    for k, (company, year, metric, i, j) in enumerate(candidates[:n], start=1):
        # Half the pairs also vary how the year is written.
        if k % 2 == 0:
            fmt1, fmt2 = rng.sample(YEAR_FORMATS[year], 2)
            year_variant = "format"
        else:
            fmt1 = fmt2 = year
            year_variant = "identical"

        pairs.append(
            make_pair(
                f"p{k:03d}",
                render(company, fmt1, metric, i),
                render(company, fmt2, metric, j),
                label="same",
                bucket="paraphrase",
                meta={"company": company, "year": year, "metric": metric,
                      "year_variant": year_variant},
            )
        )
    return pairs


def build_hard_negatives(rng, n):
    """Same phrasing template; exactly one field perturbed."""
    candidates = []
    for company, year, metric in product(COMPANIES, YEARS, METRIC_KEYS):
        for t_idx in range(len(METRICS[metric])):
            base = render(company, year, metric, t_idx)

            for other in YEARS:
                if other != year:
                    candidates.append(
                        (base, render(company, other, metric, t_idx), "year",
                         {"company": company, "metric": metric,
                          "from": year, "to": other})
                    )

            for other in COMPANIES:
                if other != company:
                    candidates.append(
                        (base, render(other, year, metric, t_idx), "company",
                         {"year": year, "metric": metric,
                          "from": company, "to": other})
                    )

            for other in METRIC_KEYS:
                if other != metric and t_idx < len(METRICS[other]):
                    candidates.append(
                        (base, render(company, year, other, t_idx), "metric",
                         {"company": company, "year": year,
                          "from": metric, "to": other})
                    )

    # Balance across the three perturbation types.
    by_type = {"year": [], "company": [], "metric": []}
    for c in candidates:
        by_type[c[2]].append(c)
    for bucket in by_type.values():
        rng.shuffle(bucket)

    per_type = n // 3
    selected = []
    for ptype in ("year", "company", "metric"):
        selected.extend(by_type[ptype][:per_type])
    selected.extend(by_type["year"][per_type:per_type + n - len(selected)])
    rng.shuffle(selected)

    pairs = []
    for k, (q1, q2, ptype, meta) in enumerate(selected, start=1):
        pairs.append(
            make_pair(f"n{k:03d}", q1, q2, "different", "hard_negative",
                      perturbed=ptype, meta=meta)
        )
    return pairs


def build_unrelated(rng, n):
    """Company, year, and metric all differ."""
    pairs = []
    seen = set()
    k = 0
    while len(pairs) < n:
        c1, c2 = rng.sample(COMPANIES, 2)
        y1, y2 = rng.sample(YEARS, 2)
        m1, m2 = rng.sample(METRIC_KEYS, 2)
        t1 = rng.randrange(len(METRICS[m1]))
        t2 = rng.randrange(len(METRICS[m2]))

        q1 = render(c1, y1, m1, t1)
        q2 = render(c2, y2, m2, t2)
        if (q1, q2) in seen:
            continue
        seen.add((q1, q2))

        k += 1
        pairs.append(
            make_pair(f"u{k:03d}", q1, q2, "different", "unrelated")
        )
    return pairs


def main():
    rng = random.Random(SEED)

    pairs = (
        build_paraphrases(rng, N_PARAPHRASE)
        + build_hard_negatives(rng, N_HARD_NEGATIVE)
        + build_unrelated(rng, N_UNRELATED)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    counts = {}
    for p in pairs:
        key = p["bucket"] if not p["perturbed"] else f"{p['bucket']}:{p['perturbed']}"
        counts[key] = counts.get(key, 0) + 1

    print(f"Wrote {len(pairs)} pairs to {OUT_PATH}")
    for key in sorted(counts):
        print(f"  {key:<26} {counts[key]}")


if __name__ == "__main__":
    main()