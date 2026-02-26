# main.py
# End-to-end pipeline runner for the quant engine.
# Run this to test the full Agent 4 → 5 → 6 flow for a given sector.
#
# Usage:
#   python main.py
#   python main.py --sector Technology --date 2023-06-01

import argparse
import logging
from datetime import datetime

from data.yfinance_provider import YFinanceProvider
from agents.agent4_screener import screen_universe
from agents.agent5_factors import compute_factor_scores
from agents.agent6_ranker import rank_stocks

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def run_pipeline(sector: str, as_of_date: str | None = None) -> None:
    """
    Full quant engine pipeline for a single sector.

    Args:
        sector:     GICS sector name e.g. 'Technology'
        as_of_date: 'YYYY-MM-DD' — run as of this date. Defaults to today.
    """
    date_str = as_of_date or datetime.today().strftime("%Y-%m-%d")
    logger.info(f"━━━ Quant Engine | Sector: {sector} | As of: {date_str} ━━━")

    # ── Initialise data provider ──────────────────────────────────────────────
    # Swap YFinanceProvider for your production provider here when ready.
    data_provider = YFinanceProvider()

    # ── Agent 4: Hard Screening ───────────────────────────────────────────────
    logger.info("Agent 4 — Fetching sector universe and screening...")
    raw_tickers = data_provider.get_tickers_for_sector(sector)
    logger.info(f"Raw universe: {len(raw_tickers)} tickers")

    passed_tickers, screening_results = screen_universe(raw_tickers, data_provider)

    if not passed_tickers:
        logger.error("Agent 4: No tickers passed screening. Halting.")
        return

    logger.info(f"Agent 4 complete: {len(passed_tickers)} tickers passed screening")

    # ── Agent 5: Factor Scoring ───────────────────────────────────────────────
    logger.info("Agent 5 — Computing factor scores...")
    factor_scores = compute_factor_scores(passed_tickers, data_provider, as_of_date=date_str)

    scored_count = sum(
        1 for s in factor_scores
        if any(v is not None for v in [s.momentum, s.quality, s.low_vol])
    )
    logger.info(f"Agent 5 complete: {scored_count}/{len(passed_tickers)} stocks scored")

    if scored_count == 0:
        logger.error("Agent 5: No stocks could be scored. Halting.")
        return

    # ── Agent 6: Composite Ranking ────────────────────────────────────────────
    logger.info("Agent 6 — Ranking stocks by composite score...")
    ranked = rank_stocks(factor_scores)

    if not ranked:
        logger.error("Agent 6: Ranking produced no output. Halting.")
        return

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n" + "━" * 55)
    print(f"  TOP PICKS — {sector.upper()} | {date_str}")
    print("━" * 55)
    print(f"  {'Rank':<6} {'Ticker':<8} {'Score':>8}  {'Mom':>7}  {'Qual':>7}  {'Vol':>7}")
    print("  " + "─" * 50)
    for stock in ranked:
        print(
            f"  {stock.rank:<6} {stock.ticker:<8} "
            f"{stock.composite_score:>8.3f}  "
            f"{stock.momentum_z:>7.3f}  "
            f"{stock.quality_z:>7.3f}  "
            f"{stock.low_vol_z:>7.3f}"
        )
    print("━" * 55)
    print(f"  → Passing to Agent 7: {[s.ticker for s in ranked]}")
    print("━" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Engine — Agent 4/5/6 Pipeline")
    parser.add_argument("--sector", default="Technology", help="GICS sector name")
    parser.add_argument("--date",   default=None,         help="As-of date YYYY-MM-DD")
    args = parser.parse_args()

    run_pipeline(sector=args.sector, as_of_date=args.date)
