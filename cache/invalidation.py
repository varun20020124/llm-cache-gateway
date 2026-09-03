"""Entity-aware cache invalidation.

Embedding similarity is blind to the fiscal year in a financial query
(see results/FINDINGS.md). This module extracts structured fields from
both queries and blocks the match when a field differs, regardless of
how similar the embeddings are.

Design bias: when extraction is uncertain, block. A false block costs one
API call; a false allow costs a wrong answer.
"""

import re

# FY2023, FY 2023, fiscal 2023, fiscal year 2023, in 2023
_YEAR_PATTERNS = [
    re.compile(r"\bFY\s?(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bfiscal\s+year\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bfiscal\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(19|20)(\d{2})\b"),
]

_COMPANIES = {
    "apple": "apple",
    "microsoft": "microsoft",
    "nvidia": "nvidia",
}


def extract_year(text: str) -> int | None:
    """Return the fiscal year, or None if the query doesn't mention one."""
    for pattern in _YEAR_PATTERNS[:3]:
        match = pattern.search(text)
        if match:
            return int(match.group(1))

    match = _YEAR_PATTERNS[3].search(text)
    if match:
        return int(match.group(0))

    return None


def extract_company(text: str) -> str | None:
    lowered = text.lower()
    found = {name for token, name in _COMPANIES.items() if token in lowered}
    if len(found) == 1:
        return found.pop()
    return None  # zero or ambiguous


def blocked_reason(q1: str, q2: str) -> str | None:
    """Return why the match is blocked, or None if it's allowed."""
    y1, y2 = extract_year(q1), extract_year(q2)
    if y1 != y2:
        return "year"

    c1, c2 = extract_company(q1), extract_company(q2)
    if c1 != c2:
        return "company"

    return None


def allows_match(q1: str, q2: str) -> bool:
    return blocked_reason(q1, q2) is None