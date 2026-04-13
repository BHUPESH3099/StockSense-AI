"""
portfolio.py
------------
Portfolio risk analysis using Modern Portfolio Theory + ML-enhanced metrics.
Calculates returns, volatility, Sharpe ratio, correlation, VaR, and optimization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


TRADING_DAYS = 252
RISK_FREE_RATE = 0.05  # Annual risk-free rate (approximate)


def build_portfolio_df(price_dict: dict) -> pd.DataFrame:
    """
    Build aligned close-price DataFrame from dict of {ticker: pd.Series}.
    Only includes tickers with sufficient data.
    """
    df = pd.DataFrame(price_dict).dropna(how="all")
    df.sort_index(inplace=True)
    # Forward fill short gaps, then drop any still-missing
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_metrics(returns: pd.DataFrame, weights: np.ndarray) -> dict:
    """
    Compute annualized portfolio statistics for given weights.

    Returns:
        dict with: annual_return, annual_volatility, sharpe_ratio, sortino_ratio
    """
    w = np.array(weights)
    w = w / w.sum()  # Normalize

    mean_daily = returns.mean().values
    cov_matrix = returns.cov().values

    port_daily_return = np.dot(w, mean_daily)
    port_daily_var = w @ cov_matrix @ w.T
    port_daily_std = np.sqrt(port_daily_var)

    annual_return = port_daily_return * TRADING_DAYS
    annual_vol = port_daily_std * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - RISK_FREE_RATE) / (annual_vol + 1e-10)

    # Sortino ratio (downside deviation)
    downside = returns[returns < 0].fillna(0)
    port_downside = (downside @ w)
    downside_std = port_downside.std() * np.sqrt(TRADING_DAYS)
    sortino = (annual_return - RISK_FREE_RATE) / (downside_std + 1e-10)

    # Value at Risk (95% confidence, 1-day)
    port_daily_series = (returns * w).sum(axis=1)
    var_95 = np.percentile(port_daily_series, 5)
    cvar_95 = port_daily_series[port_daily_series <= var_95].mean()

    return {
        "annual_return": round(annual_return * 100, 2),
        "annual_volatility": round(annual_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "var_95_1d": round(var_95 * 100, 2),
        "cvar_95_1d": round(cvar_95 * 100, 2),
        "max_drawdown": round(_max_drawdown(port_daily_series) * 100, 2),
    }


def _max_drawdown(returns_series: pd.Series) -> float:
    """Maximum drawdown from cumulative returns."""
    cum = (1 + returns_series).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / (rolling_max + 1e-10)
    return drawdown.min()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of returns."""
    return returns.corr()


def individual_stock_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-stock annualized stats."""
    mean = returns.mean() * TRADING_DAYS
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (mean - RISK_FREE_RATE) / (vol + 1e-10)
    cumulative = ((1 + returns).prod() - 1) * 100
    return pd.DataFrame({
        "Annual Return (%)": (mean * 100).round(2),
        "Annual Volatility (%)": (vol * 100).round(2),
        "Sharpe Ratio": sharpe.round(3),
        "Total Return (%)": cumulative.round(2),
    })


def optimize_portfolio(returns: pd.DataFrame, objective: str = "sharpe") -> dict:
    """
    Mean-variance portfolio optimization.

    Args:
        returns: Daily returns DataFrame
        objective: 'sharpe' (max Sharpe) | 'min_vol' (minimum volatility)

    Returns:
        dict with optimal weights and metrics
    """
    n = len(returns.columns)
    tickers = list(returns.columns)

    mean_ret = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS

    def neg_sharpe(w):
        ret = np.dot(w, mean_ret)
        vol = np.sqrt(w @ cov @ w.T)
        return -(ret - RISK_FREE_RATE) / (vol + 1e-10)

    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w.T)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    fn = neg_sharpe if objective == "sharpe" else portfolio_vol
    result = minimize(fn, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    if not result.success:
        weights = np.ones(n) / n  # Fallback to equal weight
    else:
        weights = result.x

    weights = np.clip(weights, 0, 1)
    weights /= weights.sum()

    daily_returns = returns.copy()
    metrics = portfolio_metrics(daily_returns, weights)

    return {
        "weights": dict(zip(tickers, [round(w, 4) for w in weights])),
        "metrics": metrics,
        "objective": objective,
    }


def efficient_frontier(returns: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
    """
    Generate the efficient frontier by varying target return.
    Returns DataFrame with (volatility, return, sharpe) for each point.
    """
    n = len(returns.columns)
    mean_ret = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS

    min_ret = mean_ret.min()
    max_ret = mean_ret.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_ret) - t},
        ]
        bounds = [(0, 1)] * n
        x0 = np.ones(n) / n

        try:
            res = minimize(
                lambda w: np.sqrt(w @ cov @ w.T),
                x0, method="SLSQP", bounds=bounds, constraints=constraints
            )
            if res.success:
                vol = np.sqrt(res.x @ cov @ res.x.T)
                sharpe = (target - RISK_FREE_RATE) / (vol + 1e-10)
                frontier.append({
                    "Return (%)": round(target * 100, 2),
                    "Volatility (%)": round(vol * 100, 2),
                    "Sharpe": round(sharpe, 3),
                })
        except Exception:
            continue

    return pd.DataFrame(frontier)


def run_portfolio_analysis(price_dict: dict, weights: Optional[dict] = None) -> dict:
    """
    Full portfolio analysis pipeline.

    Args:
        price_dict: {ticker: pd.Series of close prices}
        weights: Optional {ticker: weight}. Defaults to equal-weight.

    Returns:
        Comprehensive analysis dict
    """
    prices = build_portfolio_df(price_dict)
    if prices.empty or len(prices.columns) < 2:
        raise ValueError("Need at least 2 valid tickers with overlapping data.")

    returns = compute_returns(prices)
    tickers = list(prices.columns)

    # Resolve weights
    if weights:
        w_arr = np.array([weights.get(t, 0) for t in tickers], dtype=float)
    else:
        w_arr = np.ones(len(tickers)) / len(tickers)

    w_arr = np.clip(w_arr, 0, 1)
    w_arr /= w_arr.sum()
    weights_used = dict(zip(tickers, w_arr))

    metrics = portfolio_metrics(returns, w_arr)
    corr = correlation_matrix(returns)
    individual = individual_stock_metrics(returns)

    # Optimization
    opt_sharpe = optimize_portfolio(returns, "sharpe")
    opt_minvol = optimize_portfolio(returns, "min_vol")

    # Portfolio cumulative performance vs equal-weight benchmark
    port_returns = (returns * w_arr).sum(axis=1)
    cum_port = (1 + port_returns).cumprod()

    bench_w = np.ones(len(tickers)) / len(tickers)
    bench_returns = (returns * bench_w).sum(axis=1)
    cum_bench = (1 + bench_returns).cumprod()

    performance_df = pd.DataFrame({
        "Portfolio": cum_port,
        "Equal Weight": cum_bench,
    })

    return {
        "prices": prices,
        "returns": returns,
        "tickers": tickers,
        "weights_used": weights_used,
        "metrics": metrics,
        "correlation": corr,
        "individual": individual,
        "optimized_sharpe": opt_sharpe,
        "optimized_minvol": opt_minvol,
        "performance": performance_df,
    }
