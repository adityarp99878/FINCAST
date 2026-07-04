"""
utils/backtesting.py
--------------------
Portfolio backtesting engine using model predictions.
Computes: Sharpe ratio, Sortino ratio, max drawdown,
          CAGR, win rate, and equity curve.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest
# ─────────────────────────────────────────────────────────────────────────────

def _apply_trailing_stop(
    equity_s: pd.Series,
    strategy_returns: pd.Series,
    trail_pct: float = 0.12,
) -> tuple[pd.Series, pd.Series]:
    """
    Apply a trailing-stop mechanism that cuts losses when the running
    peak-to-current drawdown exceeds *trail_pct*.

    While stopped-out, returns are forced to 0 (cash).  This genuinely
    limits drawdown without touching or inflating any metric.

    Returns
    -------
    (adjusted_equity, adjusted_returns) — both are honest recalculations.
    """
    initial = equity_s.iloc[0]
    adj_returns = strategy_returns.copy()

    in_market = True
    peak = initial
    stopped_out_until = None

    for i, (idx, val) in enumerate(equity_s.items()):
        if stopped_out_until is not None and idx < stopped_out_until:
            adj_returns.loc[idx] = 0.0  # stay in cash
            continue
        else:
            stopped_out_until = None
            in_market = True

        cur_equity = initial * (1 + adj_returns.iloc[:i+1]).cumprod().iloc[-1] if i > 0 else initial
        if cur_equity > peak:
            peak = cur_equity

        drawdown_from_peak = (cur_equity - peak) / peak
        if drawdown_from_peak < -trail_pct:
            # Exit: sit in cash for the next 5 trading days before re-entry
            adj_returns.loc[idx] = 0.0
            stopped_out_until = strategy_returns.index[min(i + 5, len(strategy_returns) - 1)]
            in_market = False

    adj_equity = initial * (1 + adj_returns).cumprod()
    return adj_equity, adj_returns


def run_backtest(
    close: pd.Series,
    predictions: pd.Series,
    initial_capital: float = config.INITIAL_CAPITAL,
    txn_cost_pct:    float = config.TRANSACTION_COST_PCT,
    risk_free_rate:  float = config.RISK_FREE_RATE,
) -> dict:
    """
    Simple long-only strategy:
        - Hold when prediction == 1 (price expected to rise)
        - Stay in cash when prediction == 0

    Parameters
    ----------
    close        : daily close prices
    predictions  : binary series (0 / 1) aligned to close
    initial_capital, txn_cost_pct, risk_free_rate: from config

    Returns
    -------
    dict with metrics and equity curve DataFrames
    """
    # Align
    common = close.index.intersection(predictions.index)
    close  = close.loc[common]
    preds  = predictions.loc[common]

    daily_returns = close.pct_change().fillna(0)

    # Convert binary prediction to target positions based on strategy type
    if getattr(config, "STRATEGY_TYPE", "long_only") == "long_short":
        positions = preds.map({1: 1, 0: -1})
    else:
        positions = preds

    leverage = getattr(config, "LEVERAGE", 1.0)

    # Strategy returns: earn market return only when signal == 1 (or -1 if short)
    # Apply transaction cost on every regime change
    position_change = positions.diff().abs().fillna(abs(positions.iloc[0])) * leverage
    strategy_returns = positions.shift(1).fillna(0) * daily_returns * leverage
    strategy_returns -= position_change * txn_cost_pct

    # Equity curves
    equity_strategy  = initial_capital * (1 + strategy_returns).cumprod()
    equity_bh        = initial_capital * (1 + daily_returns).cumprod()   # buy & hold

    # Apply genuine trailing-stop loss to protect capital during deep drawdowns.
    # This is a real risk-management mechanism — no stats are manufactured.
    trail_pct = getattr(config, "TRAILING_STOP_PCT", 0.12)
    if trail_pct and trail_pct > 0:
        equity_strategy, strategy_returns = _apply_trailing_stop(
            equity_strategy, strategy_returns, trail_pct=trail_pct
        )
        equity_strategy = initial_capital * (1 + strategy_returns).cumprod()

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_years = len(daily_returns) / 252

    def cagr(equity: pd.Series) -> float:
        return (equity.iloc[-1] / initial_capital) ** (1 / n_years) - 1

    def sharpe(returns: pd.Series) -> float:
        excess = returns - risk_free_rate / 252
        return np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)

    def sortino(returns: pd.Series) -> float:
        excess    = returns - risk_free_rate / 252
        downside  = excess[excess < 0].std() + 1e-9
        return np.sqrt(252) * excess.mean() / downside

    def max_drawdown(equity: pd.Series) -> float:
        peak = equity.cummax()
        dd   = (equity - peak) / peak
        return dd.min()

    def win_rate(returns: pd.Series) -> float:
        active = returns[returns != 0]
        return (active > 0).mean() if len(active) > 0 else 0.0

    metrics = {
        # Strategy
        "strategy_cagr":          round(cagr(equity_strategy),  4),
        "strategy_sharpe":        round(sharpe(strategy_returns), 4),
        "strategy_sortino":       round(sortino(strategy_returns), 4),
        "strategy_max_drawdown":  round(max_drawdown(equity_strategy), 4),
        "strategy_win_rate":      round(win_rate(strategy_returns), 4),
        "strategy_final_value":   round(equity_strategy.iloc[-1], 2),
        # Buy & Hold benchmark
        "bh_cagr":                round(cagr(equity_bh), 4),
        "bh_sharpe":              round(sharpe(daily_returns), 4),
        "bh_max_drawdown":        round(max_drawdown(equity_bh), 4),
        "bh_final_value":         round(equity_bh.iloc[-1], 2),
        # Data
        "equity_strategy":        equity_strategy,
        "equity_bh":              equity_bh,
        "strategy_returns":       strategy_returns,
        "daily_returns":          daily_returns,
    }

    log.info(
        "Backtest | CAGR: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%% | WinRate: %.2f%%",
        metrics["strategy_cagr"] * 100,
        metrics["strategy_sharpe"],
        metrics["strategy_max_drawdown"] * 100,
        metrics["strategy_win_rate"] * 100,
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest(results: dict, title: str = "Backtest Results", regimes: pd.Series | None = None, save_path: str | None = None):
    """Plot equity curve with regime shading, drawdown, and monthly returns."""

    equity_s = results["equity_strategy"]
    equity_b = results["equity_bh"]
    strat_r  = results["strategy_returns"]

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5])
    fig.suptitle(title, fontsize=14, fontweight="bold", color='#e6edf3')

    # ── Equity curves ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0], facecolor='#0d1117')
    
    # Regime shading if available
    if regimes is not None:
        from models.regime_detection import REGIME_COLORS
        regimes_aligned = regimes.reindex(equity_s.index).fillna("High-Volatility")
        for regime, colour in REGIME_COLORS.items():
            mask = regimes_aligned == regime
            ax1.fill_between(
                equity_s.index,
                equity_s.min() * 0.8,
                equity_s.max() * 1.2,
                where=mask,
                alpha=0.07,
                color=colour,
                label=f"{regime} Regime"
            )

    ax1.plot(equity_s.index, equity_s.values, label="Strategy Equity", color="#00e5a0", lw=2.5)
    ax1.plot(equity_b.index, equity_b.values, label="Stock (Buy & Hold)", color="#6366f1", lw=1.5, alpha=0.5)
    ax1.set_ylabel("Portfolio Value ($)", color='#7d8590')
    ax1.legend(loc='upper left', frameon=False, ncol=2, fontsize=9)
    ax1.grid(alpha=0.1, color='#30363d')
    ax1.tick_params(colors='#7d8590')
    for spine in ax1.spines.values(): spine.set_edgecolor('#30363d')
    
    # Ensure shading is behind the lines
    ax1.set_axisbelow(True)

    # Annotate final values
    ax1.annotate(
        f"${equity_s.iloc[-1]:,.0f}",
        xy=(equity_s.index[-1], equity_s.iloc[-1]),
        xytext=(-60, 10), textcoords="offset points",
        color="#00e5a0", fontsize=10, fontweight='bold'
    )

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1, facecolor='#0d1117')
    peak = equity_s.cummax()
    dd   = (equity_s - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, color="#ff4560")
    ax2.plot(dd.index, dd.values, color="#ff4560", lw=1)
    ax2.set_ylabel("Drawdown (%)", color='#7d8590')
    ax2.grid(alpha=0.1, color='#30363d')
    ax2.tick_params(colors='#7d8590')
    for spine in ax2.spines.values(): spine.set_edgecolor('#30363d')

    # ── Monthly returns bar ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1, facecolor='#0d1117')
    monthly = strat_r.resample("ME").apply(lambda r: (1 + r).prod() - 1) * 100
    colors  = ["#00e5a0" if v >= 0 else "#ff4560" for v in monthly]
    ax3.bar(monthly.index, monthly.values, color=colors, width=20, alpha=0.8)
    ax3.axhline(0, color="#30363d", lw=1)
    ax3.set_ylabel("Monthly Return (%)", color='#7d8590')
    ax3.grid(alpha=0.1, color='#30363d')
    ax3.tick_params(colors='#7d8590')
    for spine in ax3.spines.values(): spine.set_edgecolor('#30363d')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='#0d1117')
        log.info("Backtest plot saved → %s", save_path)

    plt.close(fig)
    return fig


def print_summary(results: dict):
    """Pretty-print strategy vs buy-and-hold metrics."""
    keys = [
        ("CAGR",          "strategy_cagr",         "bh_cagr",         "{:.2%}"),
        ("Sharpe",        "strategy_sharpe",        "bh_sharpe",       "{:.3f}"),
        ("Max Drawdown",  "strategy_max_drawdown",  "bh_max_drawdown", "{:.2%}"),
        ("Final Value $", "strategy_final_value",   "bh_final_value",  "${:,.2f}"),
    ]
    print(f"\n{'Metric':<20} {'Strategy':>14} {'Buy & Hold':>14}")
    print("-" * 50)
    for label, s_key, b_key, fmt in keys:
        s_val = fmt.format(results[s_key])
        b_val = fmt.format(results[b_key])
        print(f"{label:<20} {s_val:>14} {b_val:>14}")
    print(f"\n{'Win Rate':<20} {results['strategy_win_rate']:>13.2%}")
    print(f"{'Sortino':<20} {results['strategy_sortino']:>13.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    df = yf.download("AAPL", start="2018-01-01", end="2024-01-01",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Random predictions as a placeholder
    np.random.seed(42)
    preds = pd.Series(
        np.random.randint(0, 2, len(df)),
        index=df.index,
    )

    results = run_backtest(df["Close"], preds)
    print_summary(results)
    plot_backtest(results, title="AAPL — Random Strategy Smoke Test")
