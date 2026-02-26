# tests/test_agent4.py
# Tests for Agent 4 hard screening.
# Uses a MockDataProvider with synthetic data — no network calls, always fast.

import pytest
from data.skeleton.base_provider import DataProvider, DataProviderError
from agents.quant.agent4_screener import screen_universe
import pandas as pd


class MockDataProvider(DataProvider):
    """Synthetic data provider for testing. Configured per test."""

    def __init__(self, fundamentals_map: dict):
        self._fundamentals = fundamentals_map

    def get_price_history(self, ticker, start_date, end_date):
        raise NotImplementedError("Not used in Agent 4 tests")

    def get_fundamentals(self, ticker):
        if ticker not in self._fundamentals:
            raise DataProviderError(f"No mock data for {ticker}")
        return self._fundamentals[ticker]

    def get_tickers_for_sector(self, sector):
        raise NotImplementedError("Not used in Agent 4 tests")


# ── Fixtures ─────────────────────────────────────────────────────────────────

GOOD_FUNDAMENTALS = {
    "market_cap":        5_000_000_000,   # $5B ✓
    "avg_daily_volume":  10_000_000,      # $10M/day ✓
    "debt_to_equity":    1.0,             # D/E 1.0 ✓
    "return_on_equity":  0.20,
    "earnings_per_share": 3.50,
    "price":             100.0,           # $100 ✓
}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_good_stock_passes():
    provider = MockDataProvider({"AAPL": GOOD_FUNDAMENTALS})
    passed, results = screen_universe(["AAPL"], provider)
    assert "AAPL" in passed
    assert results[0].passed is True


def test_low_market_cap_fails():
    bad = {**GOOD_FUNDAMENTALS, "market_cap": 100_000_000}  # $100M — too small
    provider = MockDataProvider({"SMOL": bad})
    passed, results = screen_universe(["SMOL"], provider)
    assert "SMOL" not in passed
    assert "market_cap" in results[0].reason.lower() or "Market cap" in results[0].reason


def test_low_price_fails():
    bad = {**GOOD_FUNDAMENTALS, "price": 2.0}  # $2 — penny stock
    provider = MockDataProvider({"PENY": bad})
    passed, _ = screen_universe(["PENY"], provider)
    assert "PENY" not in passed


def test_high_debt_to_equity_fails():
    bad = {**GOOD_FUNDAMENTALS, "debt_to_equity": 5.0}  # D/E 5 — too leveraged
    provider = MockDataProvider({"LEVR": bad})
    passed, _ = screen_universe(["LEVR"], provider)
    assert "LEVR" not in passed


def test_missing_data_fails():
    provider = MockDataProvider({})  # No data for any ticker
    passed, results = screen_universe(["UNKN"], provider)
    assert "UNKN" not in passed
    assert results[0].passed is False


def test_mixed_universe():
    data = {
        "GOOD": GOOD_FUNDAMENTALS,
        "BAD1": {**GOOD_FUNDAMENTALS, "market_cap": 50_000_000},
        "BAD2": {**GOOD_FUNDAMENTALS, "price": 1.0},
        "ALSO_GOOD": {**GOOD_FUNDAMENTALS, "market_cap": 10_000_000_000},
    }
    provider = MockDataProvider(data)
    passed, _ = screen_universe(list(data.keys()), provider)
    assert set(passed) == {"GOOD", "ALSO_GOOD"}


def test_none_market_cap_fails():
    bad = {**GOOD_FUNDAMENTALS, "market_cap": None}
    provider = MockDataProvider({"NONE": bad})
    passed, _ = screen_universe(["NONE"], provider)
    assert "NONE" not in passed
