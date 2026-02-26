# tests/test_agent6.py
# Tests for Agent 6 composite scoring and ranking.

import pytest
from agents.quant.agent5_factors import FactorScores
from agents.scoring.agent6_ranker import rank_stocks


def make_score(ticker, momentum, quality, low_vol):
    return FactorScores(ticker=ticker, momentum=momentum, quality=quality, low_vol=low_vol)


# ── Basic ranking tests ───────────────────────────────────────────────────────

def test_returns_top_n_stocks():
    scores = [make_score(f"T{i}", 0.1 * i, 0.5, 0.2) for i in range(10)]
    result = rank_stocks(scores, top_n=3)
    assert len(result) == 3


def test_rank_1_has_highest_composite_score():
    scores = [
        make_score("HIGH", 0.5,  0.9, 0.1),   # excellent on all factors
        make_score("LOW",  -0.3, 0.1, 0.4),   # poor on all factors
        make_score("MID",  0.1,  0.5, 0.2),
    ]
    result = rank_stocks(scores, top_n=3)
    assert result[0].ticker == "HIGH"
    assert result[0].rank == 1


def test_lower_volatility_scores_higher():
    # Two identical stocks except volatility
    scores = [
        make_score("LOW_VOL",  0.2, 0.6, 0.10),   # low vol = good
        make_score("HIGH_VOL", 0.2, 0.6, 0.50),   # high vol = bad
    ]
    result = rank_stocks(scores, top_n=2)
    assert result[0].ticker == "LOW_VOL"


def test_handles_empty_input():
    result = rank_stocks([], top_n=3)
    assert result == []


def test_handles_all_none_scores():
    scores = [make_score("BAD", None, None, None)]
    # Should not crash — returns empty (all nulls dropped) or score of 0
    result = rank_stocks(scores, top_n=3)
    # Acceptable: either returned with neutral score or dropped
    assert isinstance(result, list)


def test_handles_partial_none_scores():
    scores = [
        make_score("PARTIAL", 0.3, None, 0.15),   # quality missing
        make_score("FULL",    0.2, 0.7, 0.20),    # all present
    ]
    result = rank_stocks(scores, top_n=2)
    assert len(result) == 2
    # Neither should crash
    tickers = [r.ticker for r in result]
    assert "PARTIAL" in tickers
    assert "FULL" in tickers


def test_composite_scores_are_deterministic():
    scores = [make_score(f"T{i}", 0.05 * i, 0.5, 0.2) for i in range(5)]
    result_a = rank_stocks(scores, top_n=3)
    result_b = rank_stocks(scores, top_n=3)
    assert [r.ticker for r in result_a] == [r.ticker for r in result_b]


def test_returns_fewer_than_n_if_universe_small():
    scores = [make_score("ONLY", 0.2, 0.5, 0.15)]
    result = rank_stocks(scores, top_n=3)
    assert len(result) == 1
