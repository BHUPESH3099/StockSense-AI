# 📈 StockSense AI — ML Stock Prediction Terminal

A production-grade, end-to-end stock price prediction system using **XGBoost + Streamlit**.
Bloomberg/TradingView-inspired dark UI with real-time data from yFinance.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- pip

### 2. Install Dependencies
```bash
cd stock_predictor
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run frontend/app.py
```

Open your browser at: **http://localhost:8501**

---

## 🧠 System Architecture

```
stock_predictor/
├── backend/
│   ├── data_fetcher.py        # yFinance real-time data + disk caching
│   ├── feature_engineering.py # 60+ technical indicators
│   ├── model.py               # XGBoost training (3 horizons) + persistence
│   ├── predictor.py           # Orchestration + signal generation
│   └── portfolio.py           # MPT risk analytics + optimization
├── frontend/
│   └── app.py                 # Streamlit UI (Bloomberg-style dark terminal)
├── utils/
│   ├── charts.py              # Plotly dark theme charts
│   └── helpers.py             # Formatting utilities
├── models/                    # Auto-saved models (joblib)
├── data/cache/                # API response cache
├── .streamlit/config.toml     # Dark theme config
└── requirements.txt
```

---

## 🎯 Features

### ML Backend
- **XGBoost Regressor** with 500 estimators + early stopping
- **60+ features**: Moving averages (5/10/20/50/200 DMA), RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, volatility, lag features, calendar features
- **3 separate models** per ticker: 1-day, 1-week, 1-month predictions
- **TimeSeriesSplit** validation (no data leakage)
- **RobustScaler** preprocessing
- **Auto-persistence** via joblib (24h cache)

### Signals
| Signal | Condition |
|--------|-----------|
| 🟢 BUY  | Expected return > +3% |
| 🔴 SELL | Expected return < -3% |
| 🟡 HOLD | Between -3% and +3% |

### Portfolio Risk (MPT)
- Annual Return, Volatility, Sharpe Ratio, Sortino Ratio
- Value at Risk (95% confidence, 1-day)
- Conditional VaR (CVaR/Expected Shortfall)
- Max Drawdown
- Correlation Matrix heatmap
- Mean-Variance Optimization (Max Sharpe + Min Vol)

---

## 📊 Supported Assets
- US Stocks: `AAPL`, `TSLA`, `NVDA`, etc.
- NSE India: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- BSE India: `RELIANCE.BO`, `HDFCBANK.BO`
- Indices: `^GSPC` (S&P500), `^NSEI` (Nifty 50)
- Crypto: `BTC-USD`, `ETH-USD`
- ETFs: `SPY`, `QQQ`, `ARKK`

---

## ⚙️ Model Performance (Example: AAPL)
```
Horizon    RMSE     MAE      R²      MAPE
1 Day      1.24     0.91     0.98    0.54%
1 Week     3.87     2.91     0.94    1.71%
1 Month    9.43     7.12     0.86    4.22%
```

---

## 🔧 Configuration

Edit `backend/predictor.py` to adjust signal thresholds:
```python
BUY_THRESHOLD  = 0.03   # +3% → BUY
SELL_THRESHOLD = -0.03  # -3% → SELL
```

Edit `backend/model.py` to tune XGBoost:
```python
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    ...
}
```

---

## ⚠️ Disclaimer
This tool is for **educational and research purposes only**.
Not financial advice. Always do your own research.

---

## 🛠️ Improvements & Roadmap
- [ ] LSTM/Transformer hybrid model
- [ ] Sentiment analysis from news headlines
- [ ] Options chain data overlay
- [ ] Real-time streaming with WebSocket
- [ ] Backtesting engine with P&L curves
- [ ] Email/SMS alerts for signal changes
- [ ] Multi-factor risk model (Fama-French)
