# agents/agent4_screener.py
# Agent 4 — Hard Screening
#
# Purpose: Binary pass/fail filtering of the raw sector universe.
# Takes a list of tickers, eliminates untradeable / too-risky candidates,
# returns a clean list for Agent 5 to score.
#
# Rules here are HARD limits — not scored, just pass or fail.
# If a stock fails any single check, it's out.

import logging
from dataclasses import dataclass

from data.skeleton.base_provider import DataProvider, DataProviderError
from config.settings import (
    MIN_MARKET_CAP,
    MIN_AVG_DAILY_VOLUME_USD,
    MAX_DEBT_TO_EQUITY,
    MIN_PRICE,
)

logger = logging.getLogger(__name__)


@dataclass
class ScreeningResult:
    """Result for a single ticker after hard screening."""
    ticker: str
    passed: bool
    reason: str  # Why it passed or the reason it was rejected


def screen_universe(
    tickers: list[str],
    data_provider: DataProvider,
) -> tuple[list[str], list[ScreeningResult]]:
    """
    Run hard screening filters on a list of tickers.

    Args:
        tickers:        Raw universe from Agent 3 sector selection
        data_provider:  DataProvider instance

    Returns:
        passed_tickers: List of tickers that cleared all filters
        results:        Full ScreeningResult list for auditing/logging
    """
    results: list[ScreeningResult] = []

    for ticker in tickers:
        result = _screen_single(ticker, data_provider)
        results.append(result)

        if result.passed:
            logger.debug(f"PASS  {ticker}: {result.reason}")
        else:
            logger.debug(f"FAIL  {ticker}: {result.reason}")

    passed = [r.ticker for r in results if r.passed]
    logger.info(f"Screening complete: {len(passed)}/{len(tickers)} passed")

    return passed, results


def _screen_single(ticker: str, data_provider: DataProvider) -> ScreeningResult:
    """Apply all hard filters to a single ticker. Returns on first failure."""

    # ── Fetch fundamentals ──────────────────────────────────────────────────
    try:
        fundamentals = data_provider.get_fundamentals(ticker)
    except DataProviderError as e:
        return ScreeningResult(ticker, False, f"Data unavailable: {e}")

    # ── Filter 1: Minimum price ──────────────────────────────────────────────
    price = fundamentals.get("price")
    if price is None:
        return ScreeningResult(ticker, False, "Price unavailable")
    if price < MIN_PRICE:
        return ScreeningResult(ticker, False, f"Price ${price:.2f} below minimum ${MIN_PRICE}")

    # ── Filter 2: Minimum market cap ─────────────────────────────────────────
    market_cap = fundamentals.get("market_cap")
    if market_cap is None:
        return ScreeningResult(ticker, False, "Market cap unavailable")
    if market_cap < MIN_MARKET_CAP:
        return ScreeningResult(
            ticker, False,
            f"Market cap ${market_cap/1e9:.1f}B below minimum ${MIN_MARKET_CAP/1e9:.1f}B"
        )

    # ── Filter 3: Minimum average daily volume ───────────────────────────────
    avg_vol = fundamentals.get("avg_daily_volume")
    if avg_vol is None:
        return ScreeningResult(ticker, False, "Average volume unavailable")
    if avg_vol < MIN_AVG_DAILY_VOLUME_USD:
        return ScreeningResult(
            ticker, False,
            f"Avg daily volume ${avg_vol/1e6:.1f}M below minimum ${MIN_AVG_DAILY_VOLUME_USD/1e6:.1f}M"
        )

    # ── Filter 4: Maximum debt-to-equity ─────────────────────────────────────
    dte = fundamentals.get("debt_to_equity")
    if dte is not None and dte > MAX_DEBT_TO_EQUITY:
        return ScreeningResult(
            ticker, False,
            f"Debt/equity {dte:.1f} exceeds maximum {MAX_DEBT_TO_EQUITY}"
        )

    # ── All filters passed ───────────────────────────────────────────────────
    return ScreeningResult(ticker, True, "All filters passed")


# ── NOTE: Earnings exclusion filter ─────────────────────────────────────────
# Stocks near earnings announcements carry gap risk on daily rebalancing.
# This filter is stubbed here — implement once your data provider exposes
# an earnings calendar endpoint.
#
# def _has_upcoming_earnings(ticker, data_provider) -> bool:
#     calendar = data_provider.get_earnings_calendar(ticker)
#     days_to_earnings = (calendar.next_date - datetime.today()).days
#     return days_to_earnings <= EARNINGS_EXCLUSION_DAYS
