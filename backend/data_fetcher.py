"""
data_fetcher.py (INDIA VERSION)
-------------------------------
Fetches historical stock data using yFinance.
Supports Indian stocks (NSE/BSE) automatically.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 1


# -------------------------------
# 🔹 Helper: Normalize ticker
# -------------------------------
def normalize_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()

    # If already has exchange suffix → keep it
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker

    # Default to NSE
    return ticker + ".NS"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe_name = ticker.replace("/", "_")
    return CACHE_DIR / f"{safe_name}_{start}_{end}.pkl"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() < CACHE_TTL_HOURS * 3600


# -------------------------------
# 🔹 Main Data Fetcher
# -------------------------------
def fetch_stock_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    period: str = "2y",
    use_cache: bool = True,
) -> pd.DataFrame:

    ticker = normalize_ticker(ticker)

    # Resolve dates
    if start_date and end_date:
        start_str = start_date
        end_str = end_date
    else:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=730)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

    cache_file = _cache_path(ticker, start_str, end_str)

    # Load cache
    if use_cache and _is_cache_valid(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_str, end=end_str, auto_adjust=True)

        # 🔥 Fallback: try BSE if NSE fails
        if df.empty and ticker.endswith(".NS"):
            ticker_bse = ticker.replace(".NS", ".BO")
            stock = yf.Ticker(ticker_bse)
            df = stock.history(start=start_str, end=end_str, auto_adjust=True)

        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'")

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        # Save cache
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {str(e)}")


# -------------------------------
# 🔹 Stock Info (India Supported)
# -------------------------------
def get_stock_info(ticker: str) -> dict:
    try:
        ticker = normalize_ticker(ticker)
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", None),
            "currency": info.get("currency", "INR"),
            "exchange": info.get("exchange", "NSE/BSE"),
            "pe_ratio": info.get("trailingPE", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", None),
            "beta": info.get("beta", None),
            "dividend_yield": info.get("dividendYield", None),
        }

    except Exception:
        return {"name": ticker, "sector": "N/A", "industry": "N/A"}


# -------------------------------
# 🔹 Validate Ticker
# -------------------------------
def validate_ticker(ticker: str) -> bool:
    try:
        df = fetch_stock_data(ticker, use_cache=False)
        return len(df) > 10
    except Exception:
        return False


# -------------------------------
# 🔹 Multi Stock Fetch (Portfolio)
# -------------------------------
def get_multiple_stocks(tickers: list, period: str = "1y") -> dict:
    result = {}

    for t in tickers:
        try:
            df = fetch_stock_data(t, period=period)
            result[t] = df["Close"]
        except Exception:
            print(f"Skipping {t}")

    return result