# backtesting/backtest.py
# Simple backtesting framework skeleton.
# Iterates over historical dates, runs the full pipeline at each rebalance,
# and tracks portfolio returns.
#
# This is a SKELETON — the core loop and result structure are in place,
# but you'll flesh out the portfolio tracking once Agent 7 is wired in.

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from data.base_provider import DataProvider
from agents.agent4_screener import screen_universe
from agents.agent5_factors import compute_factor_scores
from agents.agent6_ranker import rank_stocks, RankedStock
from config.settings import BACKTEST_START_DATE, BACKTEST_END_DATE, REBALANCE_FREQUENCY

logger = logging.getLogger(__name__)


@dataclass
class BacktestSnapshot:
    """State of the portfolio at a single rebalance date."""
    date: str
    sector: str
    selected_tickers: list[str]
    ranked_stocks: list[RankedStock]
    # TODO: add forward returns, portfolio weights, P&L once Agent 7 is integrated


@dataclass
class BacktestResult:
    """Aggregated results across the full backtest period."""
    sector: str
    start_date: str
    end_date: str
    snapshots: list[BacktestSnapshot] = field(default_factory=list)

    # TODO: add performance metrics (Sharpe, max drawdown, CAGR etc.)

    def summary(self) -> str:
        return (
            f"Backtest: {self.sector} | {self.start_date} → {self.end_date} | "
            f"{len(self.snapshots)} rebalance periods"
        )


def run_backtest(
    sector: str,
    data_provider: DataProvider,
    start_date: str | None = None,
    end_date: str | None = None,
    frequency: str | None = None,
) -> BacktestResult:
    """
    Run a historical backtest of the quant engine for a given sector.

    Args:
        sector:         GICS sector to test
        data_provider:  DataProvider instance
        start_date:     'YYYY-MM-DD', defaults to settings
        end_date:       'YYYY-MM-DD', defaults to settings
        frequency:      pandas offset alias ('D', 'W', 'M'), defaults to settings

    Returns:
        BacktestResult with snapshot history
    """
    start  = start_date or BACKTEST_START_DATE
    end    = end_date   or BACKTEST_END_DATE
    freq   = frequency  or REBALANCE_FREQUENCY

    logger.info(f"Starting backtest: {sector} | {start} → {end} | freq={freq}")

    result = BacktestResult(sector=sector, start_date=start, end_date=end)

    # Generate rebalance dates
    rebalance_dates = pd.date_range(start=start, end=end, freq=freq)
    logger.info(f"{len(rebalance_dates)} rebalance periods")

    for date in rebalance_dates:
        date_str = date.strftime("%Y-%m-%d")
        logger.debug(f"Processing {date_str}...")

        try:
            snapshot = _run_single_period(sector, data_provider, date_str)
            result.snapshots.append(snapshot)
        except Exception as e:
            # Log and continue — don't let one bad date break the whole backtest
            logger.warning(f"Period {date_str} failed: {e}")
            continue

    logger.info(f"Backtest complete: {len(result.snapshots)} successful periods")
    return result


def _run_single_period(
    sector: str,
    data_provider: DataProvider,
    as_of_date: str,
) -> BacktestSnapshot:
    """Run the full agent pipeline for a single historical date."""

    # Agent 4
    raw_tickers = data_provider.get_tickers_for_sector(sector)
    passed, _ = screen_universe(raw_tickers, data_provider)

    # Agent 5
    factor_scores = compute_factor_scores(passed, data_provider, as_of_date=as_of_date)

    # Agent 6
    ranked = rank_stocks(factor_scores)

    return BacktestSnapshot(
        date=as_of_date,
        sector=sector,
        selected_tickers=[r.ticker for r in ranked],
        ranked_stocks=ranked,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TODO: Forward return computation
# ─────────────────────────────────────────────────────────────────────────────
# Once the core loop is working, add:
#
# def compute_forward_returns(snapshots, data_provider, holding_period_days):
#     """
#     For each snapshot, fetch the actual returns of selected_tickers
#     over the holding period and attach to each BacktestSnapshot.
#     This gives you the raw material for Sharpe ratio, drawdown, etc.
#     """
#     pass
