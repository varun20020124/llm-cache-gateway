"""In-memory semantic cache with entity-aware invalidation."""

from dataclasses import dataclass, field

import numpy as np

from cache.embeddings import cosine_similarity, embed
from cache.invalidation import blocked_reason

DEFAULT_THRESHOLD = 0.90


@dataclass
class Entry:
    query: str
    response: str
    vector: np.ndarray


@dataclass
class Stats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    blocked: int = 0  # similarity cleared the bar, entity check refused

    def as_dict(self) -> dict:
        hit_rate = self.hits / self.lookups if self.lookups else 0.0
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "blocked_by_entity_check": self.blocked,
            "hit_rate": round(hit_rate, 4),
        }


class SemanticCache:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, guard: bool = True):
        self.threshold = threshold
        self.guard = guard
        self.entries: list[Entry] = []
        self.stats = Stats()

    def lookup(self, query: str) -> str | None:
        self.stats.lookups += 1

        if not self.entries:
            self.stats.misses += 1
            return None

        vector = embed([query])[query]

        best: Entry | None = None
        best_score = -1.0
        for entry in self.entries:
            score = cosine_similarity(vector, entry.vector)
            if score > best_score:
                best_score, best = score, entry

        if best_score < self.threshold:
            self.stats.misses += 1
            return None

        if self.guard and blocked_reason(query, best.query):
            self.stats.blocked += 1
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        return best.response

    def store(self, query: str, response: str) -> None:
        vector = embed([query])[query]
        self.entries.append(Entry(query=query, response=response, vector=vector))