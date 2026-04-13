"""
predictor.py
------------
High-level prediction orchestration.
Combines model output with signal generation and risk metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from backend.data_fetcher import fetch_stock_data, get_stock_info
from backend.model import StockPredictor


# ─────────────────────────────────────────────
# Signal thresholds
# ─────────────────────────────────────────────
BUY_THRESHOLD = 0.03    # +3% expected return → BUY
SELL_THRESHOLD = -0.03  # -3% expected return → SELL


@dataclass
class PredictionResult:
    ticker: str
    current_price: float
    predictions: dict            # {'1d': price, '1w': price, '1m': price}
    signals: dict                # {'1d': 'BUY'/'SELL'/'HOLD', ...}
    expected_returns: dict       # {'1d': 0.025, ...}
    confidence: dict             # {'1d': 'High'/'Medium'/'Low', ...}
    metrics: dict                # model metrics per horizon
    stock_info: dict
    historical_chart: Optional[dict] = field(default_factory=dict)


def _generate_signal(current: float, predicted: float) -> tuple:
    """Return (signal_str, expected_return_float)."""
    ret = (predicted - current) / (current + 1e-10)
    if ret > BUY_THRESHOLD:
        signal = "BUY"
    elif ret < SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"
    return signal, ret


def _estimate_confidence(r2: float, mape: float) -> str:
    """Map model quality metrics to a confidence label."""
    if r2 > 0.85 and mape < 5:
        return "High"
    elif r2 > 0.65 and mape < 10:
        return "Medium"
    else:
        return "Low"


def run_prediction(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    force_retrain: bool = False,
    progress_callback=None,
) -> PredictionResult:
    """
    Full end-to-end prediction pipeline for a ticker.

    Args:
        ticker: Stock symbol
        start_date / end_date: Data range (optional)
        force_retrain: Skip cache and retrain
        progress_callback: Optional callable(step:str, pct:int)

    Returns:
        PredictionResult dataclass
    """

    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    _progress("Fetching market data...", 10)
    raw_df = fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    stock_info = get_stock_info(ticker)
    current_price = float(raw_df["Close"].iloc[-1])

    _progress("Training XGBoost models...", 40)
    predictor = StockPredictor(ticker)

    if force_retrain:
        predictor.train(raw_df)
    else:
        loaded = predictor.load()
        if not loaded:
            predictor.train(raw_df)

    _progress("Generating predictions...", 75)
    raw_predictions = predictor.predict(raw_df)

    signals = {}
    expected_returns = {}
    confidence = {}

    for horizon, predicted_price in raw_predictions.items():
        sig, ret = _generate_signal(current_price, predicted_price)
        signals[horizon] = sig
        expected_returns[horizon] = ret

        # Use model metrics for confidence
        m = predictor.metrics.get(horizon, {})
        confidence[horizon] = _estimate_confidence(
            m.get("r2", 0), m.get("mape", 100)
        )

    _progress("Building chart data...", 90)
    historical_chart = {}
    for horizon in ["1d", "1w", "1m"]:
        try:
            chart_df = predictor.predict_historical(raw_df, horizon=horizon)
            # Keep last 120 rows for chart performance
            chart_df = chart_df.tail(120)
            historical_chart[horizon] = chart_df
        except Exception:
            historical_chart[horizon] = pd.DataFrame()

    _progress("Done.", 100)

    return PredictionResult(
        ticker=ticker,
        current_price=current_price,
        predictions=raw_predictions,
        signals=signals,
        expected_returns=expected_returns,
        confidence=confidence,
        metrics=predictor.metrics,
        stock_info=stock_info,
        historical_chart=historical_chart,
    )


def get_signal_color(signal: str) -> str:
    """Map signal to Streamlit-compatible color."""
    return {"BUY": "#00e676", "SELL": "#ff1744", "HOLD": "#ffd600"}.get(signal, "#ffffff")


def format_return(ret: float) -> str:
    """Format return as colored percentage string."""
    pct = ret * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"
