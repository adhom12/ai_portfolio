# data/yfinance_provider.py
# Concrete DataProvider backed by yfinance.
# FOR PROTOTYPING ONLY — yfinance is unreliable for production use.
# When your friend lands on a production provider (FMP, Polygon etc.),
# write a new class inheriting DataProvider and swap this out in main.py.

import yfinance as yf
import pandas as pd
from data.skeleton.base_provider import DataProvider, DataProviderError

# Hardcoded sector → tickers map for prototyping.
# In production this will come from your data provider's universe endpoint.
SECTOR_TICKERS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL",
        "CSCO", "ADBE", "AMD", "INTC", "QCOM",
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "ABBV", "MRK",
        "TMO", "ABT", "DHR", "BMY", "AMGN",
    ],
    "Financials": [
        "BRK-B", "JPM", "V", "MA", "BAC",
        "WFC", "GS", "MS", "BLK", "AXP",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE",
        "SBUX", "TJX", "BKNG", "LOW", "CMG",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB",
        "MPC", "PSX", "VLO", "PXD", "OXY",
    ],
}


class YFinanceProvider(DataProvider):
    """
    yfinance-backed DataProvider.
    Sufficient for factor logic development and backtesting on historical data.
    Not suitable for production — replace before live trading.
    """

    def get_price_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            raise DataProviderError(f"yfinance failed for {ticker}: {e}")

        if raw.empty:
            raise DataProviderError(f"No price data returned for {ticker}")

        # Normalise column names to lowercase
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"

        return df

    def get_fundamentals(self, ticker: str) -> dict:
        try:
            info = yf.Ticker(ticker).info
        except Exception as e:
            raise DataProviderError(f"yfinance fundamentals failed for {ticker}: {e}")

        if not info:
            raise DataProviderError(f"Empty fundamentals for {ticker}")

        # Map yfinance keys to our standard schema
        # Some fields may be None — agents must handle this gracefully
        return {
            "market_cap":        info.get("marketCap"),
            "avg_daily_volume":  info.get("averageDailyVolume10Day"),
            "debt_to_equity":    info.get("debtToEquity"),
            "return_on_equity":  info.get("returnOnEquity"),
            "earnings_per_share": info.get("trailingEps"),
            "price":             info.get("currentPrice") or info.get("regularMarketPrice"),
        }

    def get_tickers_for_sector(self, sector: str) -> list[str]:
        if sector not in SECTOR_TICKERS:
            raise DataProviderError(
                f"Unknown sector '{sector}'. "
                f"Available: {list(SECTOR_TICKERS.keys())}"
            )
        return SECTOR_TICKERS[sector]
