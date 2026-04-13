"""
charts.py
---------
Plotly chart builders for the Streamlit frontend.
Produces Bloomberg/TradingView-style dark charts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DARK_BG = "#0d1117"
GRID_COLOR = "#1e2530"
TEXT_COLOR = "#c9d1d9"
GREEN = "#00e676"
RED = "#ff1744"
YELLOW = "#ffd600"
BLUE = "#2979ff"
PURPLE = "#aa00ff"
ORANGE = "#ff6d00"

LAYOUT_BASE = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color=TEXT_COLOR, family="'IBM Plex Mono', monospace"),
    xaxis=dict(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        showspikes=True,
        spikecolor=TEXT_COLOR,
        spikethickness=1,
    ),
    yaxis=dict(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        showspikes=True,
        spikecolor=TEXT_COLOR,
        spikethickness=1,
    ),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(13,17,23,0.8)",
        bordercolor=GRID_COLOR,
        borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=10, r=10, t=40, b=10),
)


def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Full OHLCV candlestick chart with volume bars."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
    )

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            increasing_fillcolor=GREEN,
            decreasing_fillcolor=RED,
        ),
        row=1, col=1,
    )

    # 20 & 50 DMA overlays
    if len(df) >= 20:
        sma20 = df["Close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=sma20, name="SMA 20",
                       line=dict(color=YELLOW, width=1.2), opacity=0.9),
            row=1, col=1,
        )
    if len(df) >= 50:
        sma50 = df["Close"].rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=sma50, name="SMA 50",
                       line=dict(color=ORANGE, width=1.2), opacity=0.9),
            row=1, col=1,
        )

    # Volume bars (green/red)
    colors = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color=colors, opacity=0.7),
        row=2, col=1,
    )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"  {ticker} — Price History", font=dict(size=16, color=TEXT_COLOR)),
        xaxis_rangeslider_visible=False,
        height=520,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def prediction_chart(
    hist_df: pd.DataFrame,
    predicted_price: float,
    current_date,
    horizon_label: str,
    ticker: str,
) -> go.Figure:
    """Actual vs Predicted price chart with future prediction marker."""
    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(
        x=hist_df.index,
        y=hist_df["Actual"],
        name="Actual",
        line=dict(color=BLUE, width=2),
        mode="lines",
    ))

    # In-sample predicted
    fig.add_trace(go.Scatter(
        x=hist_df.index,
        y=hist_df["Predicted"],
        name="Predicted",
        line=dict(color=PURPLE, width=1.5, dash="dot"),
        mode="lines",
        opacity=0.85,
    ))

    # Future prediction point
    last_date = hist_df.index[-1]
    last_actual = float(hist_df["Actual"].iloc[-1])

    # Bridge from last actual to predicted
    fig.add_trace(go.Scatter(
        x=[last_date, current_date],
        y=[last_actual, predicted_price],
        name=f"Forecast ({horizon_label})",
        line=dict(color=GREEN, width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=[6, 14], color=[BLUE, GREEN], symbol=["circle", "star"]),
    ))

    # Shaded uncertainty band (±2% around prediction)
    band_pct = 0.02
    fig.add_trace(go.Scatter(
        x=[current_date, current_date],
        y=[predicted_price * (1 - band_pct), predicted_price * (1 + band_pct)],
        fill=None, mode="lines",
        line_color="rgba(0,230,118,0)",
        showlegend=False,
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"  {ticker} — Actual vs Predicted ({horizon_label})",
            font=dict(size=15, color=TEXT_COLOR),
        ),
        height=420,
    )
    return fig


def volume_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Standalone volume bar chart."""
    colors = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
    fig = go.Figure(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, opacity=0.8, name="Volume"
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"  {ticker} — Volume", font=dict(size=14, color=TEXT_COLOR)),
        height=220,
    )
    return fig


def rsi_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """RSI chart with overbought/oversold bands."""
    from backend.feature_engineering import compute_rsi
    rsi = compute_rsi(df["Close"])

    fig = go.Figure()
    fig.add_hline(y=70, line_color=RED, line_dash="dot", opacity=0.6, annotation_text="Overbought")
    fig.add_hline(y=30, line_color=GREEN, line_dash="dot", opacity=0.6, annotation_text="Oversold")
    fig.add_hline(y=50, line_color=TEXT_COLOR, line_dash="dot", opacity=0.3)

    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name="RSI (14)",
        line=dict(color=YELLOW, width=1.8),
        fill="tozeroy", fillcolor="rgba(255,214,0,0.05)",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="  RSI (14)", font=dict(size=13, color=TEXT_COLOR)),
        height=220,
    )

    fig.update_yaxes(range=[0, 100])


def macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD chart with signal line and histogram."""
    from backend.feature_engineering import compute_macd
    macd, signal, hist = compute_macd(df["Close"])

    hist_colors = [GREEN if h >= 0 else RED for h in hist]

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                         marker_color=hist_colors, opacity=0.65))
    fig.add_trace(go.Scatter(x=df.index, y=macd, name="MACD",
                             line=dict(color=BLUE, width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=signal, name="Signal",
                             line=dict(color=ORANGE, width=1.5)))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="  MACD (12, 26, 9)", font=dict(size=13, color=TEXT_COLOR)),
        height=220,
    )
    return fig


def correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Portfolio correlation heatmap."""
    fig = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1, zmax=1,
        text=corr_df.round(2).values,
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar=dict(title="Correlation", tickfont=dict(color=TEXT_COLOR)),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="  Correlation Matrix", font=dict(size=14, color=TEXT_COLOR)),
        height=400,
    )
    return fig


def portfolio_performance_chart(perf_df: pd.DataFrame) -> go.Figure:
    """Cumulative portfolio vs benchmark chart."""
    fig = go.Figure()
    colors_map = {"Portfolio": GREEN, "Equal Weight": YELLOW}
    for col in perf_df.columns:
        fig.add_trace(go.Scatter(
            x=perf_df.index, y=perf_df[col],
            name=col,
            line=dict(color=colors_map.get(col, BLUE), width=2),
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="  Portfolio Cumulative Performance", font=dict(size=14, color=TEXT_COLOR)),
        height=320,
        yaxis_title="Growth of $1",
    )
    return fig


def feature_importance_chart(fi_series: pd.Series, title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart for top feature importances."""
    fi_sorted = fi_series.sort_values()
    fig = go.Figure(go.Bar(
        x=fi_sorted.values,
        y=fi_sorted.index,
        orientation="h",
        marker=dict(
            color=fi_sorted.values,
            colorscale="Viridis",
        ),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"  {title}", font=dict(size=13, color=TEXT_COLOR)),
        height=400,
        xaxis_title="Importance Score",
    )
    return fig
