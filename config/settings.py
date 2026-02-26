# config/settings.py
# All tunable parameters for the quant engine live here.
# Change behaviour by editing this file — not by hunting through agent code.

# ─────────────────────────────────────────
# Agent 4 — Hard Screening Thresholds
# ─────────────────────────────────────────

# Minimum market cap in USD (filters out micro/nano caps)
MIN_MARKET_CAP = 2_000_000_000  # $2B — large/mid cap only

# Minimum average daily dollar volume over the lookback window
# Ensures we can actually enter/exit positions without moving the market
MIN_AVG_DAILY_VOLUME_USD = 5_000_000  # $5M/day

# Number of trading days to compute average volume over
VOLUME_LOOKBACK_DAYS = 20

# Maximum debt-to-equity ratio — hard quality floor
MAX_DEBT_TO_EQUITY = 3.0

# Exclude stocks with earnings announcements within this many days
# Earnings create gap risk that can blow up a daily-rebalanced portfolio
EARNINGS_EXCLUSION_DAYS = 2

# Minimum share price — filters out penny stocks
MIN_PRICE = 5.0


# ─────────────────────────────────────────
# Agent 5 — Factor Lookback Windows
# ─────────────────────────────────────────

# Momentum: 12-month return excluding most recent month
# Standard academic definition (Jegadeesh & Titman)
MOMENTUM_LONG_WINDOW_DAYS = 252   # ~12 months of trading days
MOMENTUM_SHORT_WINDOW_DAYS = 21   # ~1 month — excluded from signal to avoid reversal

# Low volatility: realised volatility window
VOLATILITY_WINDOW_DAYS = 63       # ~3 months of trading days

# Quality metrics pulled from fundamentals (no lookback — point-in-time)
# ROE, D/E, earnings stability — defined in agent5_factors.py


# ─────────────────────────────────────────
# Agent 6 — Composite Score Weights
# ─────────────────────────────────────────
# Must sum to 1.0
# Adjust these as you validate factor contributions in backtesting

FACTOR_WEIGHTS = {
    "momentum": 0.40,
    "quality":  0.35,
    "low_vol":  0.25,
}

# Number of top stocks to return per sector
TOP_N_PER_SECTOR = 3


# ─────────────────────────────────────────
# Backtesting
# ─────────────────────────────────────────

BACKTEST_START_DATE = "2019-01-01"
BACKTEST_END_DATE   = "2023-12-31"
REBALANCE_FREQUENCY = "D"   # 'D' daily, 'W' weekly, 'M' monthly
