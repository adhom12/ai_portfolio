# utils/validators.py
# Data validation helpers.
# Call these after fetching data to catch silent failures early.
# The system should halt on bad data, not trade on it.

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def validate_price_history(df: pd.DataFrame, ticker: str, min_rows: int = 30) -> bool:
    """
    Validate a price history DataFrame.
    Logs specific issues and returns False if data is not fit for use.
    """
    if df is None or df.empty:
        logger.error(f"{ticker}: Price history is empty")
        return False

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"{ticker}: Missing columns: {missing}")
        return False

    if len(df) < min_rows:
        logger.error(f"{ticker}: Only {len(df)} rows, need at least {min_rows}")
        return False

    null_pct = df["close"].isna().mean()
    if null_pct > 0.05:
        logger.error(f"{ticker}: {null_pct:.1%} of close prices are null")
        return False

    if (df["close"] <= 0).any():
        logger.error(f"{ticker}: Non-positive close prices detected")
        return False

    return True


def validate_fundamentals(fundamentals: dict, ticker: str) -> bool:
    """
    Validate a fundamentals dict.
    Returns False if critical fields are missing entirely.
    """
    if not fundamentals:
        logger.error(f"{ticker}: Fundamentals dict is empty")
        return False

    critical_fields = ["market_cap", "price"]
    for field in critical_fields:
        if fundamentals.get(field) is None:
            logger.warning(f"{ticker}: Critical field '{field}' is None")
            # Warning only — agents handle None gracefully via their own filters

    return True
