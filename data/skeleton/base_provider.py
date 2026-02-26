# data/base_provider.py
# Abstract interface for all data providers.
# Your quant agents ONLY import from this — never from yfinance, polygon, etc. directly.
# To swap providers: write a new class that inherits DataProvider, swap in main.py.

from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    """
    Contract that every data provider must fulfil.
    Agents 4, 5, 6 only ever call these methods.
    """

    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Returns daily OHLCV data for a ticker.

        Expected columns: ['open', 'high', 'low', 'close', 'volume']
        Index: DatetimeIndex (UTC or tz-naive), ascending
        No gaps — caller should handle missing dates.

        Args:
            ticker:     e.g. 'AAPL'
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'

        Returns:
            pd.DataFrame with columns [open, high, low, close, volume]

        Raises:
            DataProviderError if data unavailable or malformed
        """
        ...

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> dict:
        """
        Returns latest point-in-time fundamental metrics.

        Expected keys (all float, None if unavailable):
            market_cap          — total market capitalisation in USD
            avg_daily_volume    — average daily dollar volume (20-day)
            debt_to_equity      — total debt / total equity
            return_on_equity    — net income / shareholders equity (TTM)
            earnings_per_share  — diluted EPS (TTM)
            price               — latest close price

        Args:
            ticker: e.g. 'AAPL'

        Returns:
            dict with keys above

        Raises:
            DataProviderError if data unavailable
        """
        ...

    @abstractmethod
    def get_tickers_for_sector(self, sector: str) -> list[str]:
        """
        Returns list of ticker symbols for a given sector.

        Args:
            sector: e.g. 'Technology', 'Healthcare' — GICS sector names

        Returns:
            list of ticker strings

        Raises:
            DataProviderError if sector unknown
        """
        ...


class DataProviderError(Exception):
    """Raised when a data provider cannot fulfil a request."""
    pass
