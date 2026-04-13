"""
feature_engineering.py
-----------------------
Creates technical indicators and ML-ready features from raw OHLCV data.
Includes: Moving Averages, RSI, MACD, Bollinger Bands, Volatility, Lag features.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# Core Indicator Functions
# ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple:
    """MACD Line, Signal Line, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple:
    """Bollinger Bands: upper, middle (SMA), lower."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR) — volatility measure."""
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (OBV)."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
    """Stochastic Oscillator (%K and %D)."""
    lowest_low = df["Low"].rolling(k_period).min()
    highest_high = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


# ─────────────────────────────────────────────
# Main Feature Engineering Pipeline
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Input: Raw OHLCV DataFrame
    Output: DataFrame with all technical features added
    """
    feat = df.copy()

    # ── Price-derived features ──
    feat["returns"] = feat["Close"].pct_change()
    feat["log_returns"] = np.log(feat["Close"] / feat["Close"].shift(1))

    # ── Moving Averages ──
    for w in [5, 10, 20, 50, 200]:
        feat[f"sma_{w}"] = feat["Close"].rolling(w).mean()
        feat[f"ema_{w}"] = feat["Close"].ewm(span=w, adjust=False).mean()

    # MA crossover signals (binary)
    feat["golden_cross"] = (feat["sma_50"] > feat["sma_200"]).astype(int)
    feat["ma_ratio_20_50"] = feat["sma_20"] / (feat["sma_50"] + 1e-10)
    feat["ma_ratio_50_200"] = feat["sma_50"] / (feat["sma_200"] + 1e-10)

    # Price relative to MAs
    feat["price_to_sma20"] = feat["Close"] / (feat["sma_20"] + 1e-10)
    feat["price_to_sma50"] = feat["Close"] / (feat["sma_50"] + 1e-10)
    feat["price_to_sma200"] = feat["Close"] / (feat["sma_200"] + 1e-10)

    # ── Volatility ──
    feat["volatility_10"] = feat["returns"].rolling(10).std()
    feat["volatility_20"] = feat["returns"].rolling(20).std()
    feat["volatility_30"] = feat["returns"].rolling(30).std()
    feat["atr"] = compute_atr(df)
    feat["atr_pct"] = feat["atr"] / (feat["Close"] + 1e-10)

    # ── RSI ──
    feat["rsi_14"] = compute_rsi(feat["Close"], 14)
    feat["rsi_7"] = compute_rsi(feat["Close"], 7)
    feat["rsi_21"] = compute_rsi(feat["Close"], 21)

    # ── MACD ──
    feat["macd"], feat["macd_signal"], feat["macd_hist"] = compute_macd(feat["Close"])
    feat["macd_pct"] = feat["macd"] / (feat["Close"] + 1e-10)

    # ── Bollinger Bands ──
    feat["bb_upper"], feat["bb_mid"], feat["bb_lower"] = compute_bollinger_bands(feat["Close"])
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / (feat["bb_mid"] + 1e-10)
    feat["bb_pct"] = (feat["Close"] - feat["bb_lower"]) / (
        feat["bb_upper"] - feat["bb_lower"] + 1e-10
    )

    # ── Stochastic ──
    feat["stoch_k"], feat["stoch_d"] = compute_stochastic(df)

    # ── Volume features ──
    feat["volume_sma_20"] = feat["Volume"].rolling(20).mean()
    feat["volume_ratio"] = feat["Volume"] / (feat["volume_sma_20"] + 1e-10)
    feat["obv"] = compute_obv(df)
    feat["obv_sma"] = feat["obv"].rolling(20).mean()

    # ── Candlestick body / shadow ──
    feat["body"] = (feat["Close"] - feat["Open"]).abs() / (feat["Open"] + 1e-10)
    feat["upper_shadow"] = (feat["High"] - feat[["Open", "Close"]].max(axis=1)) / (
        feat["Open"] + 1e-10
    )
    feat["lower_shadow"] = (feat[["Open", "Close"]].min(axis=1) - feat["Low"]) / (
        feat["Open"] + 1e-10
    )
    feat["hl_range"] = (feat["High"] - feat["Low"]) / (feat["Open"] + 1e-10)

    # ── Lag features (past N close prices) ──
    for lag in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        feat[f"lag_close_{lag}"] = feat["Close"].shift(lag)
        feat[f"lag_return_{lag}"] = feat["returns"].shift(lag)

    # ── Rolling statistics ──
    for w in [5, 10, 20]:
        feat[f"rolling_max_{w}"] = feat["Close"].rolling(w).max()
        feat[f"rolling_min_{w}"] = feat["Close"].rolling(w).min()
        feat[f"rolling_mean_{w}"] = feat["Close"].rolling(w).mean()
        feat[f"rolling_std_{w}"] = feat["Close"].rolling(w).std()

    # ── Calendar features ──
    feat["day_of_week"] = feat.index.dayofweek
    feat["month"] = feat.index.month
    feat["quarter"] = feat.index.quarter
    feat["day_of_year"] = feat.index.dayofyear

    return feat


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create prediction targets:
      - target_1d  : next 1 day close
      - target_5d  : next 5 trading days (≈1 week)
      - target_21d : next 21 trading days (≈1 month)
    """
    df = df.copy()
    df["target_1d"] = df["Close"].shift(-1)
    df["target_5d"] = df["Close"].shift(-5)
    df["target_21d"] = df["Close"].shift(-21)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excludes OHLCV and targets)."""
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "target_1d", "target_5d", "target_21d",
    }
    return [c for c in df.columns if c not in exclude]


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: engineer features + add targets + drop NaN rows."""
    df = engineer_features(df)
    df = create_targets(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df
