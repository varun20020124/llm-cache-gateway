# tests/test_invalidation.py
from cache.invalidation import allows_match, extract_year


def test_extracts_fy_format():
    assert extract_year("What was Apple's revenue in FY2023?") == 2023


def test_returns_none_when_no_year():
    assert extract_year("What is Apple's revenue?") is None


def test_blocks_year_mismatch():
    assert not allows_match(
        "What was Nvidia's gross margin in FY2023?",
        "What was Nvidia's gross margin in FY2022?",
    )


def test_allows_true_paraphrase():
    assert allows_match(
        "What was Apple's total revenue in FY2023?",
        "How much revenue did Apple generate in FY2023?",
    )


def test_blocks_when_one_query_has_no_year():
    assert not allows_match(
        "What was Apple's revenue in FY2023?",
        "What was Apple's revenue?",
    )