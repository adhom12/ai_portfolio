# agents/agent6_ranker.py
# Agent 6 — Composite Scoring & Ranking
#
# Purpose: Take raw factor scores from Agent 5, normalise them,
# apply weights, and return the top N stocks per sector.
#
# Steps:
#   1. Drop tickers with too many missing factor scores
#   2. Cross-sectionally normalise each factor (z-score within the sector universe)
#   3. Apply configured weights from settings.py
#   4. Compute composite score
#   5. Return top N tickers with their scores
#
# Output is what gets handed to Agent 7 (portfolio construction).

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agents.quant.agent5_factors import FactorScores
from config.settings import FACTOR_WEIGHTS, TOP_N_PER_SECTOR

logger = logging.getLogger(__name__)


@dataclass
class RankedStock:
    """A single stock in the final ranked output."""
    ticker: str
    composite_score: float
    momentum_z: float | None   # normalised momentum score
    quality_z:  float | None   # normalised quality score
    low_vol_z:  float | None   # normalised low-vol score (inverted vol)
    rank: int                  # 1 = best


def rank_stocks(
    factor_scores: list[FactorScores],
    top_n: int | None = None,
) -> list[RankedStock]:
    """
    Normalise, weight, and rank stocks by composite factor score.

    Args:
        factor_scores:  Output from Agent 5
        top_n:          Number of top stocks to return.
                        Defaults to TOP_N_PER_SECTOR from settings.

    Returns:
        List of RankedStock, sorted best-first, length <= top_n
    """
    n = top_n or TOP_N_PER_SECTOR

    if not factor_scores:
        logger.warning("No factor scores provided to ranker")
        return []

    # Build DataFrame for vectorised operations
    df = pd.DataFrame([{
        "ticker":   s.ticker,
        "momentum": s.momentum,
        "quality":  s.quality,
        "low_vol":  s.low_vol,    # raw volatility — will be INVERTED below
    } for s in factor_scores])

    # ── Step 1: Invert volatility so lower vol = higher score ────────────────
    df["low_vol"] = df["low_vol"].apply(
        lambda v: -v if v is not None else None
    )

    # ── Step 2: Drop tickers with ALL factor scores missing ──────────────────
    factor_cols = ["momentum", "quality", "low_vol"]
    before = len(df)
    df = df.dropna(subset=factor_cols, how="all")
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} tickers with no factor data")

    if df.empty:
        logger.error("No scoreable stocks remaining after dropping nulls")
        return []

    # ── Step 3: Cross-sectional z-score normalisation ────────────────────────
    # Each factor is normalised within this universe so scores are comparable.
    # Tickers with missing individual factors get a neutral score of 0.
    for col in factor_cols:
        col_z = col + "_z"
        series = df[col].astype(float)
        mean = series.mean(skipna=True)
        std  = series.std(skipna=True)

        if std == 0 or pd.isna(std):
            # All values identical — assign neutral z-score
            df[col_z] = 0.0
        else:
            df[col_z] = (series - mean) / std

        # Fill remaining NaN (stocks missing this factor) with 0 (neutral)
        df[col_z] = df[col_z].fillna(0.0)

    # ── Step 4: Weighted composite score ────────────────────────────────────
    w_mom = FACTOR_WEIGHTS["momentum"]
    w_qua = FACTOR_WEIGHTS["quality"]
    w_vol = FACTOR_WEIGHTS["low_vol"]

    df["composite_score"] = (
        w_mom * df["momentum_z"] +
        w_qua * df["quality_z"]  +
        w_vol * df["low_vol_z"]
    )

    # ── Step 5: Rank and return top N ────────────────────────────────────────
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    top = df.head(n)

    results = [
        RankedStock(
            ticker          = row["ticker"],
            composite_score = round(row["composite_score"], 4),
            momentum_z      = round(row["momentum_z"],      4),
            quality_z       = round(row["quality_z"],       4),
            low_vol_z       = round(row["low_vol_z"],       4),
            rank            = row["rank"],
        )
        for _, row in top.iterrows()
    ]

    for r in results:
        logger.info(
            f"Rank {r.rank}: {r.ticker} "
            f"(composite={r.composite_score:.3f}, "
            f"mom={r.momentum_z:.3f}, "
            f"qual={r.quality_z:.3f}, "
            f"vol={r.low_vol_z:.3f})"
        )

    return results
