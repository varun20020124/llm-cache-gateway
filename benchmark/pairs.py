"""Load and validate the labeled query pairs."""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("data/pairs.jsonl")

VALID_LABELS = {"same", "different"}
VALID_BUCKETS = {"paraphrase", "hard_negative", "unrelated"}


@dataclass(frozen=True)
class Pair:
    id: str
    q1: str
    q2: str
    label: str
    bucket: str
    perturbed: str | None = None

    @property
    def should_hit(self) -> bool:
        """True if a correct cache would serve q1's answer for q2."""
        return self.label == "same"


def load_pairs(path: Path = DEFAULT_PATH) -> list[Pair]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/build_pairs.py"
        )

    pairs = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            if record["label"] not in VALID_LABELS:
                raise ValueError(f"line {line_no}: bad label {record['label']}")
            if record["bucket"] not in VALID_BUCKETS:
                raise ValueError(f"line {line_no}: bad bucket {record['bucket']}")
            pairs.append(
                Pair(
                    id=record["id"],
                    q1=record["q1"],
                    q2=record["q2"],
                    label=record["label"],
                    bucket=record["bucket"],
                    perturbed=record.get("perturbed"),
                )
            )

    ids = [p.id for p in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pair ids")

    return pairs


def unique_queries(pairs: list[Pair]) -> list[str]:
    """Every distinct question string, so each is embedded only once."""
    seen = {}
    for p in pairs:
        seen.setdefault(p.q1, None)
        seen.setdefault(p.q2, None)
    return list(seen)