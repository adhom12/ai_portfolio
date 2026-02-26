# tests/test_agent5.py
# Tests for Agent 5 factor computation.
# Uses synthetic price and fundamentals data.

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agents.quant.agent5_factors import (
    _compute_momentum,
    _compute_quality,
    _compute_volatility,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_price_df(n_days: int, start_price: float = 100.0, drift: float = 0.001) -> pd.DataFrame:
    """Generate synthetic price history with a given drift."""
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq="B")
    np.random.seed(42)
    returns = np.random.normal(drift, 0.015, n_days)
    prices = start_price * np.cumprod(1 + returns)
    return pd.DataFrame({
        "open":   prices * 0.99,
        "high":   prices * 1.01,
        "low":    prices * 0.98,
        "close":  prices,
        "volume": np.full(n_days, 1_000_000),
    }, index=dates)


# ── Momentum tests ────────────────────────────────────────────────────────────

def test_momentum_positive_for_trending_stock():
    df = make_price_df(300, drift=0.003)  # strong uptrend
    momentum = _compute_momentum(df)
    assert momentum is not None
    assert momentum > 0


def test_momentum_negative_for_declining_stock():
    df = make_price_df(300, drift=-0.003)  # downtrend
    momentum = _compute_momentum(df)
    assert momentum is not None
    assert momentum < 0


def test_momentum_returns_none_for_short_history():
    df = make_price_df(50)  # not enough history
    momentum = _compute_momentum(df)
    assert momentum is None


# ── Quality tests ─────────────────────────────────────────────────────────────

def test_quality_high_roe_low_debt():
    fundamentals = {"return_on_equity": 0.30, "debt_to_equity": 0.5}
    score = _compute_quality(fundamentals)
    assert score is not None
    assert score > 0.7  # should be a high quality score


def test_quality_negative_roe_high_debt():
    fundamentals = {"return_on_equity": -0.20, "debt_to_equity": 2.8}
    score = _compute_quality(fundamentals)
    assert score is not None
    assert score < 0.3  # should be a low quality score


def test_quality_uses_available_fields():
    # Only ROE available — should still return a score
    fundamentals = {"return_on_equity": 0.25, "debt_to_equity": None}
    score = _compute_quality(fundamentals)
    assert score is not None


def test_quality_returns_none_when_all_missing():
    fundamentals = {"return_on_equity": None, "debt_to_equity": None}
    score = _compute_quality(fundamentals)
    assert score is None


def test_quality_score_between_zero_and_one():
    for roe, dte in [(0.5, 0.0), (0.0, 1.5), (-0.5, 3.0)]:
        score = _compute_quality({"return_on_equity": roe, "debt_to_equity": dte})
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for roe={roe}, dte={dte}"


# ── Volatility tests ──────────────────────────────────────────────────────────

def test_volatility_returns_positive_float():
    df = make_price_df(100)
    vol = _compute_volatility(df)
    assert vol is not None
    assert vol > 0


def test_higher_vol_stock_scores_higher():
    low_vol_df  = make_price_df(100, drift=0.001)
    high_vol_df = make_price_df(100, drift=0.001)

    # Inject extra noise into high vol version
    np.random.seed(99)
    high_vol_df["close"] = high_vol_df["close"] * (1 + np.random.normal(0, 0.03, len(high_vol_df)))

    low_vol  = _compute_volatility(low_vol_df)
    high_vol = _compute_volatility(high_vol_df)

    assert high_vol > low_vol


def test_volatility_returns_none_for_short_history():
    df = make_price_df(10)  # not enough
    vol = _compute_volatility(df)
    assert vol is None
