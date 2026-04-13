"""
helpers.py
----------
Utility functions shared across the app.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def format_currency(value: float, prefix: str = "$") -> str:
    """Format a number as currency string."""
    if value is None:
        return "N/A"
    if abs(value) >= 1e12:
        return f"{prefix}{value/1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"{prefix}{value/1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{prefix}{value/1e6:.2f}M"
    return f"{prefix}{value:,.2f}"


def format_pct(value: float, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def get_date_range_options():
    """Return preset date range options for the UI."""
    today = datetime.today()
    return {
        "1 Month": today - timedelta(days=30),
        "3 Months": today - timedelta(days=90),
        "6 Months": today - timedelta(days=180),
        "1 Year": today - timedelta(days=365),
        "2 Years": today - timedelta(days=730),
        "5 Years": today - timedelta(days=1825),
    }


POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
    "BABA", "V", "JPM", "JNJ", "WMT", "PG", "XOM", "BAC", "MA", "DIS",
    "PYPL", "ADBE", "CRM", "INTC", "AMD", "QCOM", "TXN", "AVGO",
    "ORCL", "IBM", "CSCO", "UBER", "LYFT", "SPOT", "SNAP", "TWTR",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS",
    "^NSEI", "^BSESN", "^GSPC", "^DJI", "^IXIC",
    "BTC-USD", "ETH-USD",
]


def risk_label(volatility_pct: float) -> str:
    """Map annualized volatility to risk label."""
    if volatility_pct < 15:
        return "🟢 Low Risk"
    elif volatility_pct < 30:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"


def signal_badge(signal: str) -> str:
    """Return emoji badge for signal."""
    return {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}.get(signal, signal)


def truncate_name(name: str, max_len: int = 30) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"
