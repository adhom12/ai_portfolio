# agents/agent5_factors.py
# Agent 5 — Factor Scoring
#
# Purpose: Compute continuous factor scores for each ticker that passed Agent 4.
# Returns a FactorScores object per ticker — no ranking here, just raw scores.
#
# Factors:
#   1. Momentum     — 12-1 month price return (excludes last month to avoid reversal)
#   2. Quality      — composite of ROE and debt-to-equity
#   3. Low Vol      — inverse of realised volatility (lower vol = higher score)
#
# All scores are raw (unscaled). Agent 6 handles normalisation and weighting.

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.skeleton.base_provider import DataProvider, DataProviderError
from config.settings import (
    MOMENTUM_LONG_WINDOW_DAYS,
    MOMENTUM_SHORT_WINDOW_DAYS,
    VOLATILITY_WINDOW_DAYS,
)

logger = logging.getLogger(__name__)


@dataclass
class FactorScores:
    """Raw factor scores for a single ticker."""
    ticker: str
    momentum: float | None   # 12-1 month return, e.g. 0.18 = 18%
    quality:  float | None   # Composite quality score, higher = better
    low_vol:  float | None   # Annualised realised vol — lower = less volatile


def compute_factor_scores(
    tickers: list[str],
    data_provider: DataProvider,
    as_of_date: str | None = None,
) -> list[FactorScores]:
    """
    Compute factor scores for a list of tickers.

    Args:
        tickers:        Tickers that passed Agent 4 screening
        data_provider:  DataProvider instance
        as_of_date:     'YYYY-MM-DD' — score as of this date.
                        Defaults to today. Set explicitly for backtesting.

    Returns:
        List of FactorScores, one per ticker (None scores where data was insufficient)
    """
    end_date = as_of_date or datetime.today().strftime("%Y-%m-%d")
    # We need enough history for the longest lookback
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d")
        - timedelta(days=MOMENTUM_LONG_WINDOW_DAYS + 30)  # buffer
    ).strftime("%Y-%m-%d")

    scores = []
    for ticker in tickers:
        score = _score_single(ticker, data_provider, start_date, end_date)
        scores.append(score)
        logger.debug(
            f"{ticker}: momentum={score.momentum}, "
            f"quality={score.quality}, low_vol={score.low_vol}"
        )

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_single(
    ticker: str,
    data_provider: DataProvider,
    start_date: str,
    end_date: str,
) -> FactorScores:
    """Compute all factor scores for a single ticker."""
    try:
        prices = data_provider.get_price_history(ticker, start_date, end_date)
        fundamentals = data_provider.get_fundamentals(ticker)
    except DataProviderError as e:
        logger.warning(f"Data fetch failed for {ticker}: {e}")
        return FactorScores(ticker, None, None, None)

    return FactorScores(
        ticker=ticker,
        momentum=_compute_momentum(prices),
        quality=_compute_quality(fundamentals),
        low_vol=_compute_volatility(prices),
    )


def _compute_momentum(prices: pd.DataFrame) -> float | None:
    """
    12-1 month price momentum.

    Returns the total return from 12 months ago to 1 month ago,
    excluding the most recent month to avoid short-term reversal.

    Returns None if insufficient price history.
    """
    close = prices["close"].dropna()

    if len(close) < MOMENTUM_LONG_WINDOW_DAYS:
        return None

    # Price at start of momentum window (~12 months ago)
    price_12m_ago = close.iloc[-(MOMENTUM_LONG_WINDOW_DAYS)]

    # Price at end of momentum window (~1 month ago, excluding recent reversal period)
    price_1m_ago = close.iloc[-(MOMENTUM_SHORT_WINDOW_DAYS)]

    if price_12m_ago <= 0:
        return None

    return (price_1m_ago - price_12m_ago) / price_12m_ago


def _compute_quality(fundamentals: dict) -> float | None:
    """
    Composite quality score combining ROE and debt-to-equity.

    Higher ROE = better quality.
    Lower D/E = better quality (so we invert it).

    Returns a single score; None if both inputs are unavailable.
    """
    roe = fundamentals.get("return_on_equity")  # e.g. 0.35 = 35% ROE
    dte = fundamentals.get("debt_to_equity")     # e.g. 1.5

    if roe is None and dte is None:
        return None

    # Each component scores between 0 and 1 where available
    # We average whatever we have
    components = []

    if roe is not None:
        # Clip ROE to [-0.5, 1.0] range — extreme values are likely noise
        roe_clipped = max(-0.5, min(1.0, roe))
        # Rescale to [0, 1]: 0% ROE → 0.33, 50% ROE → 1.0
        components.append((roe_clipped + 0.5) / 1.5)

    if dte is not None:
        # Invert D/E so lower debt = higher score
        # D/E of 0 → score 1.0, D/E of 3 → score 0.0
        dte_clipped = max(0.0, min(3.0, dte))
        components.append(1.0 - (dte_clipped / 3.0))

    return sum(components) / len(components)


def _compute_volatility(prices: pd.DataFrame) -> float | None:
    """
    Annualised realised volatility over the configured window.

    This is a RAW volatility value — Agent 6 inverts it so that
    lower volatility produces a higher composite score.

    Returns None if insufficient price history.
    """
    close = prices["close"].dropna()

    if len(close) < VOLATILITY_WINDOW_DAYS:
        return None

    recent_close = close.iloc[-VOLATILITY_WINDOW_DAYS:]
    daily_returns = recent_close.pct_change().dropna()

    if len(daily_returns) < 10:
        return None

    # Annualise: multiply daily std by sqrt(252 trading days)
    annualised_vol = daily_returns.std() * np.sqrt(252)

    return float(annualised_vol)
