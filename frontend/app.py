"""
app.py
------
Main Streamlit application — Stock Price Prediction Terminal.
Bloomberg/TradingView inspired dark UI with XGBoost ML predictions.

Run:
    streamlit run frontend/app.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.data_fetcher import fetch_stock_data, get_stock_info
from backend.predictor import run_prediction, get_signal_color, format_return
from backend.portfolio import run_portfolio_analysis
from utils.charts import (
    candlestick_chart, prediction_chart, rsi_chart,
    macd_chart, correlation_heatmap, portfolio_performance_chart,
    feature_importance_chart,
)
from utils.helpers import (
    format_currency, format_pct, POPULAR_TICKERS,
    risk_label, signal_badge, get_date_range_options,
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Dark Terminal Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
  }

  .stApp { background-color: #0d1117; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #1e2530;
  }

  /* Metrics cards */
  [data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #1e2530;
    border-radius: 8px;
    padding: 12px 16px;
  }
  [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 11px !important; }
  [data-testid="stMetricValue"] { color: #e6edf3 !important; font-family: 'IBM Plex Mono', monospace !important; }
  [data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }

  /* Tabs */
  [data-testid="stTabs"] button {
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    border-radius: 6px 6px 0 0;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #00e676 !important;
    border-bottom: 2px solid #00e676 !important;
    background: rgba(0,230,118,0.05);
  }

  /* Buttons */
  .stButton > button {
    background: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    border-color: #00e676;
    color: #00e676;
    background: rgba(0,230,118,0.07);
  }

  /* Signal boxes */
  .signal-buy {
    background: rgba(0,230,118,0.12);
    border: 2px solid #00e676;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
  }
  .signal-sell {
    background: rgba(255,23,68,0.12);
    border: 2px solid #ff1744;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
  }
  .signal-hold {
    background: rgba(255,214,0,0.10);
    border: 2px solid #ffd600;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
  }
  .signal-text { font-size: 28px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
  .metric-val { font-size: 22px; font-family: 'IBM Plex Mono', monospace; color: #e6edf3; }
  .metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .divider { border-top: 1px solid #1e2530; margin: 10px 0; }
  .ticker-header { font-family: 'IBM Plex Mono', monospace; font-size: 28px; color: #e6edf3; font-weight: 600; }
  .company-name { font-size: 14px; color: #8b949e; }

  /* DataFrames */
  [data-testid="stDataFrame"] { border: 1px solid #1e2530; border-radius: 6px; }

  /* Selectbox / input */
  .stSelectbox [data-baseweb], .stTextInput [data-baseweb] {
    background: #161b22 !important;
    border-color: #30363d !important;
  }

  /* Alert / info */
  .stAlert { border-radius: 8px; }

  /* Hide default Streamlit footer/menu */
  #MainMenu, footer { visibility: hidden; }

  /* Section headers */
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
    border-bottom: 1px solid #1e2530;
    padding-bottom: 4px;
  }

  .terminal-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 16px 24px;
    margin-bottom: 20px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "prediction_result": None,
        "current_ticker": None,
        "horizon": "1d",
        "raw_df": None,
        "portfolio_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 16px">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:20px; color:#00e676; font-weight:600;">
            📈 StockSense AI
        </div>
        <div style="font-size:11px; color:#8b949e; letter-spacing:2px; margin-top:2px;">
            ML PREDICTION TERMINAL
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔍 Stock Search</div>', unsafe_allow_html=True)

    # Ticker input with autocomplete suggestion
    ticker_input = st.text_input(
        "Enter Ticker Symbol",
        value="",
        placeholder="Please enter a ticker symbol (e.g. TCS, INFY)",
        label_visibility="collapsed",
    )


    # Date range
    st.markdown('<div class="section-header">📅 Date Range</div>', unsafe_allow_html=True)
    date_presets = get_date_range_options()
    preset_choice = st.selectbox("Preset Range", list(date_presets.keys()), index=3)
    start_date = st.date_input("From", value=date_presets[preset_choice].date())
    end_date = st.date_input("To", value=datetime.today().date())

    st.divider()

    # Prediction Horizon
    st.markdown('<div class="section-header">⏱️ Prediction Horizon</div>', unsafe_allow_html=True)
    horizon_map = {"1 Day": "1d", "1 Week": "1w", "1 Month": "1m"}
    horizon_label = st.radio("Horizon", list(horizon_map.keys()), horizontal=True)
    st.session_state["horizon"] = horizon_map[horizon_label]

    st.divider()

    # Actions
    predict_btn = st.button("🚀 Run Prediction", use_container_width=True, type="primary")
    retrain_btn = st.button("🔁 Force Retrain", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="font-size:10px; color:#484f58; text-align:center; line-height:1.8;">
        Powered by XGBoost + yFinance<br>
        Not financial advice.<br>
        Use for educational purposes only.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT — TABS
# ─────────────────────────────────────────────
tab_predict, tab_portfolio = st.tabs([
    "  📊 Prediction Terminal  ",
    "  📦 Portfolio Risk  ",
])


# ═══════════════════════════════════════════════
# TAB 1 — PREDICTION TERMINAL
# ═══════════════════════════════════════════════
with tab_predict:

    # Header
    st.markdown("""
    <div class="terminal-header">
        <span style="font-family:'IBM Plex Mono',monospace; color:#00e676; font-size:13px;">
            ▶ STOCK PREDICTION ENGINE // XGBoost ML MODEL
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Trigger prediction ──
    ticker = ticker_input.upper().strip()
    force = retrain_btn

    if predict_btn or retrain_btn:
        if not ticker:
            st.error("Please enter a valid ticker symbol.")
        else:
            with st.spinner(f"Analyzing {ticker}..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def _update_progress(msg, pct):
                    status_text.markdown(
                        f'<span style="color:#8b949e; font-size:12px; font-family:\'IBM Plex Mono\',monospace;">'
                        f'⬡ {msg}</span>', unsafe_allow_html=True
                    )
                    progress_bar.progress(pct)

                try:
                    result = run_prediction(
                        ticker,
                        start_date=str(start_date),
                        end_date=str(end_date),
                        force_retrain=force,
                        progress_callback=_update_progress,
                    )
                    st.session_state["prediction_result"] = result
                    st.session_state["current_ticker"] = ticker
                    raw_df = fetch_stock_data(
                        ticker, start_date=str(start_date), end_date=str(end_date)
                    )
                    st.session_state["raw_df"] = raw_df
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"✅ Analysis complete for {ticker}")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error: {str(e)}")

    # ── Display results ──
    result = st.session_state.get("prediction_result")
    raw_df = st.session_state.get("raw_df")
    horizon = st.session_state.get("horizon", "1d")

    if result is None:
        st.markdown("""
        <div style="text-align:center; padding:80px 20px; color:#484f58;">
            <div style="font-size:48px; margin-bottom:16px;">📈</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:18px; color:#30363d;">
                Enter a ticker and click Run Prediction
            </div>
            <div style="font-size:13px; margin-top:8px;">
                Supports US stocks, NSE/BSE (suffix .NS/.BO), ETFs, crypto
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        print(result)
        horizon_labels = {"1d": "1 Day", "1w": "1 Week", "1m": "1 Month"}
        info = result.stock_info
        current_price = result.current_price
        predicted_price = result.predictions.get(horizon, current_price)
        signal = result.signals.get(horizon, "HOLD")
        exp_ret = result.expected_returns.get(horizon, 0)
        confidence = result.confidence.get(horizon, "N/A")
        metrics = result.metrics.get(horizon, {})

        # ── Ticker header ──
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            st.markdown(
                f'<div class="ticker-header">{result.ticker}</div>'
                f'<div class="company-name">{info.get("name", result.ticker)} '
                f'· {info.get("sector", "")} · {info.get("exchange", "")}</div>',
                unsafe_allow_html=True,
            )
        with col_h2:
            st.markdown(
                f'<div style="text-align:right;">'
                f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:32px; color:#e6edf3;">'
                f'${current_price:,.2f}</div>'
                f'<div style="font-size:12px; color:#8b949e;">Last Close</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Main layout: Charts LEFT | Prediction Panel RIGHT ──
        col_chart, col_panel = st.columns([3, 1])

        with col_chart:
            # Chart selector
            chart_type = st.radio(
                "Chart View",
                ["Candlestick", "Prediction", "RSI", "MACD"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if raw_df is not None:
                if chart_type == "Candlestick":
                    st.plotly_chart(
                        candlestick_chart(raw_df, result.ticker),
                        use_container_width=True,
                    )
                elif chart_type == "Prediction":
                    chart_df = result.historical_chart.get(horizon, pd.DataFrame())
                    if not chart_df.empty:
                        from datetime import timedelta
                        horizon_days = {"1d": 1, "1w": 7, "1m": 30}[horizon]
                        future_date = datetime.now() + timedelta(days=horizon_days)
                        st.plotly_chart(
                            prediction_chart(
                                chart_df, predicted_price,
                                future_date, horizon_labels[horizon],
                                result.ticker,
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.warning("Prediction chart not available.")
                elif chart_type == "RSI":
                    st.plotly_chart(rsi_chart(raw_df, result.ticker), use_container_width=True)
                elif chart_type == "MACD":
                    st.plotly_chart(macd_chart(raw_df), use_container_width=True)

        with col_panel:
            # ── Signal Box ──
            signal_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[signal]
            signal_color = {"BUY": "#00e676", "SELL": "#ff1744", "HOLD": "#ffd600"}[signal]
            signal_emoji = {"BUY": "▲", "SELL": "▼", "HOLD": "◆"}[signal]

            st.markdown(f"""
            <div class="{signal_class}">
                <div style="font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;">
                    Signal · {horizon_labels[horizon]}
                </div>
                <div class="signal-text" style="color:{signal_color};">
                    {signal_emoji} {signal}
                </div>
                <div style="font-size:13px; color:{signal_color}; margin-top:4px; font-family:'IBM Plex Mono',monospace;">
                    {format_return(exp_ret)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Prediction Panel ──
            def metric_row(label, value, sub=None, color="#e6edf3"):
                sub_html = f'<div style="font-size:10px; color:#8b949e;">{sub}</div>' if sub else ""
                st.markdown(f"""
                <div style="background:#161b22; border:1px solid #1e2530; border-radius:7px; padding:10px 14px; margin-bottom:8px;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-val" style="color:{color};">{value}</div>
                    {sub_html}
                </div>
                """, unsafe_allow_html=True)

            metric_row("Current Price", f"${current_price:,.2f}")
            metric_row(
                f"Predicted ({horizon_labels[horizon]})",
                f"${predicted_price:,.2f}",
                color=signal_color,
            )
            metric_row(
                "Expected Return",
                format_return(exp_ret),
                color=signal_color,
            )
            metric_row("Confidence", confidence)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Model Quality</div>', unsafe_allow_html=True)

            model_cols = st.columns(2)
            model_cols[0].metric("R²", f"{metrics.get('r2', 0):.3f}")
            model_cols[1].metric("MAPE", f"{metrics.get('mape', 0):.1f}%")
            model_cols[0].metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
            model_cols[1].metric("MAE", f"{metrics.get('mae', 0):.2f}")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Stock Info</div>', unsafe_allow_html=True)

            if info.get("market_cap"):
                st.markdown(f'<div class="metric-label">Market Cap</div>'
                            f'<div style="font-family:\'IBM Plex Mono\',monospace; color:#e6edf3; font-size:14px; margin-bottom:6px;">'
                            f'{format_currency(info["market_cap"])}</div>', unsafe_allow_html=True)
            if info.get("52w_high"):
                st.markdown(f'<div class="metric-label">52W High / Low</div>'
                            f'<div style="font-family:\'IBM Plex Mono\',monospace; color:#e6edf3; font-size:14px; margin-bottom:6px;">'
                            f'${info["52w_high"]:,.2f} / ${info["52w_low"]:,.2f}</div>', unsafe_allow_html=True)
            if info.get("beta"):
                st.markdown(f'<div class="metric-label">Beta</div>'
                            f'<div style="font-family:\'IBM Plex Mono\',monospace; color:#e6edf3; font-size:14px; margin-bottom:6px;">'
                            f'{info["beta"]:.2f}</div>', unsafe_allow_html=True)
            if info.get("pe_ratio"):
                st.markdown(f'<div class="metric-label">P/E Ratio</div>'
                            f'<div style="font-family:\'IBM Plex Mono\',monospace; color:#e6edf3; font-size:14px; margin-bottom:6px;">'
                            f'{info["pe_ratio"]:.1f}</div>', unsafe_allow_html=True)

        st.divider()

        # ── Horizon Comparison Table ──
        st.markdown('<div class="section-header">📊 All Horizons Comparison</div>', unsafe_allow_html=True)
        h_cols = st.columns(3)
        for i, (h_key, h_label) in enumerate({"1d": "1 Day", "1w": "1 Week", "1m": "1 Month"}.items()):
            pred_p = result.predictions.get(h_key, current_price)
            sig = result.signals.get(h_key, "HOLD")
            ret = result.expected_returns.get(h_key, 0)
            sig_color = {"BUY": "#00e676", "SELL": "#ff1744", "HOLD": "#ffd600"}[sig]
            with h_cols[i]:
                st.markdown(f"""
                <div style="background:#161b22; border:1px solid #1e2530; border-radius:8px; padding:16px; text-align:center;">
                    <div style="font-size:12px; color:#8b949e; text-transform:uppercase; letter-spacing:1px;">{h_label}</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:22px; color:#e6edf3; margin:8px 0;">${pred_p:,.2f}</div>
                    <div style="font-size:14px; color:{sig_color}; font-family:'IBM Plex Mono',monospace;">{format_return(ret)}</div>
                    <div style="font-size:18px; font-weight:700; color:{sig_color}; margin-top:4px;">{sig}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ── Feature Importance ──
        with st.expander("🔬 Feature Importance & Model Details"):
            fi_cols = st.columns(2)
            for i, h_key in enumerate(["1d", "1w"]):
                from backend.model import StockPredictor
                predictor_tmp = StockPredictor(result.ticker)
                if predictor_tmp.load():
                    fi = predictor_tmp.get_top_features(h_key, 15)
                    fi_series = fi.set_index("Feature")["Importance"]
                    with fi_cols[i]:
                        st.plotly_chart(
                            feature_importance_chart(fi_series, f"Top Features ({horizon_labels[h_key]})"),
                            use_container_width=True,
                        )

            # Metrics table
            st.markdown("**Model Evaluation Metrics**")
            metrics_data = []
            for h_key, h_cfg in result.metrics.items():
                metrics_data.append({
                    "Horizon": h_cfg.get("label", h_key),
                    "RMSE": h_cfg.get("rmse"),
                    "MAE": h_cfg.get("mae"),
                    "R²": h_cfg.get("r2"),
                    "MAPE (%)": h_cfg.get("mape"),
                    "Train Samples": h_cfg.get("n_train"),
                    "Test Samples": h_cfg.get("n_test"),
                })
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# TAB 2 — PORTFOLIO RISK
# ═══════════════════════════════════════════════
with tab_portfolio:
    st.markdown("""
    <div class="terminal-header">
        <span style="font-family:'IBM Plex Mono',monospace; color:#2979ff; font-size:13px;">
            ▶ PORTFOLIO RISK ENGINE // MPT + ML ANALYTICS
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Stock selection
    p_col1, p_col2 = st.columns([2, 1])
    with p_col1:
        portfolio_tickers_raw = st.text_input(
            "Enter tickers (comma separated)",
            value="MRPL,TCS",
            help="E.g.: AAPL, TSLA, INFY.NS, BTC-USD",
        )
        portfolio_tickers = [t.strip().upper() for t in portfolio_tickers_raw.split(",") if t.strip()]

    with p_col2:
        port_period = st.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=0)

    # Weight inputs
    st.markdown("**Portfolio Weights (leave equal for auto 1/N)**")
    weight_cols = st.columns(min(len(portfolio_tickers), 5))
    weights_input = {}
    for i, t in enumerate(portfolio_tickers):
        with weight_cols[i % 5]:
            weights_input[t] = st.number_input(
                t, min_value=0.0, max_value=1.0,
                value=round(1 / len(portfolio_tickers), 2),
                step=0.05, key=f"w_{t}",
            )

    # Normalize button
    col_a, col_b = st.columns([1, 3])
    with col_a:
        run_portfolio_btn = st.button(" Analyze Portfolio", use_container_width=True)

    if run_portfolio_btn:
        if len(portfolio_tickers) < 2:
            st.error("Please enter at least 2 tickers.")
        else:
            with st.spinner("Running portfolio analysis..."):
                try:
                    from backend.data_fetcher import get_multiple_stocks
                    price_dict = get_multiple_stocks(portfolio_tickers, period=port_period)

                    if len(price_dict) < 2:
                        st.error("Could not fetch data for enough tickers.")
                    else:
                        # Normalize weights for fetched tickers
                        fetched = list(price_dict.keys())
                        w_total = sum(weights_input.get(t, 0) for t in fetched)
                        if w_total == 0:
                            norm_weights = {t: 1 / len(fetched) for t in fetched}
                        else:
                            norm_weights = {t: weights_input.get(t, 0) / w_total for t in fetched}

                        port_result = run_portfolio_analysis(price_dict, norm_weights)
                        st.session_state["portfolio_result"] = port_result
                        st.success(f"✅ Portfolio analysis for: {', '.join(fetched)}")

                except Exception as e:
                    st.error(f"Portfolio analysis error: {str(e)}")

    # ── Display portfolio results ──
    port_result = st.session_state.get("portfolio_result")
    if port_result:
        metrics = port_result["metrics"]

        st.divider()
        st.markdown('<div class="section-header">📈 Portfolio Metrics</div>', unsafe_allow_html=True)

        m_cols = st.columns(4)
        m_cols[0].metric("Annual Return", f"{metrics['annual_return']:.2f}%",
                         delta=f"{metrics['annual_return'] - 10:.1f}% vs S&P avg")
        m_cols[1].metric("Annual Volatility", f"{metrics['annual_volatility']:.2f}%")
        m_cols[2].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}",
                         delta="Good > 1.0")
        m_cols[3].metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")

        m_cols2 = st.columns(4)
        m_cols2[0].metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        m_cols2[1].metric("VaR (95%, 1D)", f"{metrics['var_95_1d']:.2f}%")
        m_cols2[2].metric("CVaR (95%, 1D)", f"{metrics['cvar_95_1d']:.2f}%")
        m_cols2[3].metric(
            "Risk Level",
            risk_label(metrics["annual_volatility"]).split(" ", 1)[1],
        )

        st.divider()

        # Charts
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.plotly_chart(
                portfolio_performance_chart(port_result["performance"]),
                use_container_width=True,
            )
        with ch_col2:
            st.plotly_chart(
                correlation_heatmap(port_result["correlation"]),
                use_container_width=True,
            )

        st.divider()

        # Individual stock metrics
        st.markdown('<div class="section-header">📋 Individual Stock Breakdown</div>', unsafe_allow_html=True)
        st.dataframe(port_result["individual"].style.background_gradient(cmap="RdYlGn", axis=0),
                     use_container_width=True)

        st.divider()

        # Optimization results
        st.markdown('<div class="section-header">🔧 Portfolio Optimization</div>', unsafe_allow_html=True)
        opt_col1, opt_col2 = st.columns(2)

        for col, key, label, color in [
            (opt_col1, "optimized_sharpe", "Max Sharpe Portfolio", "#00e676"),
            (opt_col2, "optimized_minvol", "Min Volatility Portfolio", "#2979ff"),
        ]:
            opt = port_result[key]
            with col:
                st.markdown(f"""
                <div style="background:#161b22; border:1px solid #1e2530; border-radius:8px; padding:16px; margin-bottom:8px;">
                    <div style="font-family:'IBM Plex Mono',monospace; color:{color}; font-size:13px; margin-bottom:12px;">
                        ⬡ {label}
                    </div>
                """, unsafe_allow_html=True)
                opt_m = opt["metrics"]
                st.markdown(
                    f"**Return:** {opt_m['annual_return']}% | "
                    f"**Vol:** {opt_m['annual_volatility']}% | "
                    f"**Sharpe:** {opt_m['sharpe_ratio']}"
                )
                weight_df = pd.DataFrame(
                    opt["weights"].items(), columns=["Ticker", "Weight"]
                )
                weight_df["Weight (%)"] = (weight_df["Weight"] * 100).round(1)
                weight_df = weight_df[["Ticker", "Weight (%)"]].sort_values("Weight (%)", ascending=False)
                st.dataframe(weight_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)

