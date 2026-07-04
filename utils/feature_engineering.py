"""
utils/feature_engineering.py
-----------------------------
Computes technical indicators and cross-asset features from raw OHLCV data.
All functions take a DataFrame with at least [Open, High, Low, Close, Volume]
columns and return an enriched DataFrame.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Single-asset technical indicators
# ─────────────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = config.RSI_PERIOD) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast:   int = config.MACD_FAST,
    slow:   int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"]          = ema_fast - ema_slow
    df["MACD_Signal"]   = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"]     = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = config.BB_WINDOW) -> pd.DataFrame:
    """Bollinger Bands %B and normalized distances."""
    mid   = df["Close"].rolling(window).mean()
    std   = df["Close"].rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    df[f"BB_PctB_{window}"]  = (df["Close"] - lower) / (4 * std + 1e-9)
    df[f"BB_Upper_Dist_{window}"] = (upper - df["Close"]) / (df["Close"] + 1e-9)
    df[f"BB_Lower_Dist_{window}"] = (df["Close"] - lower) / (df["Close"] + 1e-9)
    return df


def add_moving_averages(df: pd.DataFrame, windows: list = config.ROLLING_WINDOWS) -> pd.DataFrame:
    """Simple and exponential moving average ratios (percentage distance from Close)."""
    for w in windows:
        sma = df["Close"].rolling(w).mean()
        ema = df["Close"].ewm(span=w, adjust=False).mean()
        df[f"SMA_Dist_{w}"] = (df["Close"] - sma) / (df["Close"] + 1e-9)
        df[f"EMA_Dist_{w}"] = (df["Close"] - ema) / (df["Close"] + 1e-9)
    return df


def add_volatility(df: pd.DataFrame, windows: list = config.ROLLING_WINDOWS) -> pd.DataFrame:
    """Rolling historical volatility (annualised log-return std)."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    for w in windows:
        df[f"Volatility_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-derived features: OBV change and volume ratio."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    obv = (df["Volume"] * direction).cumsum()
    df["OBV_Ratio"] = obv / (obv.rolling(20).mean() + 1e-9)
    df["Volume_Ratio_20"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range — normalized by close price to make it stationary."""
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close  = (df["Low"]  - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    df[f"ATR_Ratio_{period}"] = atr / (df["Close"] + 1e-9)
    return df


def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Williams %R — overbought/oversold oscillator (bounded -100 to 0)."""
    highest_high = df["High"].rolling(period).max()
    lowest_low   = df["Low"].rolling(period).min()
    df[f"Williams_R_{period}"] = -100 * (highest_high - df["Close"]) / (
        highest_high - lowest_low + 1e-9
    )
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """%K and %D stochastic oscillator — complements RSI (bounded 0 to 100)."""
    lowest_low   = df["Low"].rolling(k_period).min()
    highest_high = df["High"].rolling(k_period).max()
    df["Stoch_K"] = 100 * (df["Close"] - lowest_low) / (
        highest_high - lowest_low + 1e-9
    )
    df["Stoch_D"] = df["Stoch_K"].rolling(d_period).mean()
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log returns, candle body size, and high-low range."""
    df["Log_Return"]  = np.log(df["Close"] / df["Close"].shift(1))
    df["Candle_Body"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-9)
    df["HL_Range"]    = (df["High"] - df["Low"])   / (df["Open"] + 1e-9)
    return df


def build_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all single-asset indicator functions in sequence."""
    df = df.copy()
    df = add_price_features(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    # Additional indicators for improved signal quality
    df = add_atr(df)
    df = add_williams_r(df)
    df = add_stochastic(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cross-asset features
# ─────────────────────────────────────────────────────────────────────────────

def add_cross_asset_features(
    stock_df:  pd.DataFrame,
    gold_df:   pd.DataFrame | None = None,
    forex_df:  pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge gold and forex close prices into the stock feature DataFrame,
    then compute rolling correlations and spread features.
    """
    df = stock_df.copy()

    if gold_df is not None and not gold_df.empty:
        df["Gold_Close"]      = gold_df["Close"].reindex(df.index)
        df["Gold_Log_Return"] = np.log(df["Gold_Close"] / df["Gold_Close"].shift(1))
        df["Stock_Gold_Corr_20"] = (
            df["Log_Return"].rolling(20).corr(df["Gold_Log_Return"])
        )

    if forex_df is not None and not forex_df.empty:
        df["Forex_Close"]      = forex_df["Close"].reindex(df.index)
        df["Forex_Log_Return"] = np.log(df["Forex_Close"] / df["Forex_Close"].shift(1))
        df["Stock_Forex_Corr_20"] = (
            df["Log_Return"].rolling(20).corr(df["Forex_Log_Return"])
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Target variable
# ─────────────────────────────────────────────────────────────────────────────

def add_target(
    df: pd.DataFrame,
    horizon:   int   = config.PREDICTION_HORIZON,
    threshold: float = config.RETURN_THRESHOLD,
) -> pd.DataFrame:
    """
    Binary target:
        1  →  future return  >  threshold  (price UP)
        0  →  future return  ≤  threshold  (price DOWN / flat)
    """
    future_close   = df["Close"].shift(-horizon)
    future_return  = (future_close - df["Close"]) / df["Close"]
    df["Target"]   = (future_return > threshold).astype(int)
    df["Future_Return"] = future_return  # kept for backtesting
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VIX features
# ─────────────────────────────────────────────────────────────────────────────

def add_vix_features(
    df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge VIX (CBOE Volatility Index) features into the stock DataFrame.
    Sources: data/external/vix/ (Kaggle) or yfinance ^VIX (fallback).
    New columns: VIX_Close, VIX_Change_5d, VIX_High_Fear (regime flag)
    """
    if vix_df is None or vix_df.empty:
        # Auto-load
        try:
            from utils.external_data import load_vix
            vix_df = load_vix()
        except Exception:
            return df

    if vix_df.empty:
        return df

    vix = vix_df.reindex(df.index, method="ffill")
    df["VIX_Close"]     = vix["VIX_Close"]
    df["VIX_Change_5d"] = vix["VIX_Close"].pct_change(5)
    df["VIX_High_Fear"] = (vix["VIX_Close"] > 25).astype(int)   # >25 = fear regime
    df["VIX_Spike"]     = (vix["VIX_Close"].pct_change() > 0.10).astype(int)  # >10% 1d spike
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Commodity features (Gold + Silver)
# ─────────────────────────────────────────────────────────────────────────────

def add_commodity_features(
    df: pd.DataFrame,
    gold_df:   pd.DataFrame | None = None,
    silver_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge Gold and Silver price features into the stock DataFrame.
    Sources: Kaggle CSVs or yfinance GC=F / SI=F fallback.
    New columns: Gold_Close, Silver_Close, Gold_Silver_Ratio, Gold_Momentum_20d
    """
    if gold_df is None:
        try:
            from utils.external_data import load_gold
            gold_df = load_gold()
        except Exception:
            gold_df = pd.DataFrame()

    if silver_df is None:
        try:
            from utils.external_data import load_silver
            silver_df = load_silver()
        except Exception:
            silver_df = pd.DataFrame()

    if not gold_df.empty:
        g = gold_df.reindex(df.index, method="ffill")
        g_col = "Gold_Close" if "Gold_Close" in g.columns else "Close"
        gold_close             = g[g_col]
        df["Gold_LogReturn"]   = np.log(gold_close / gold_close.shift(1))
        df["Gold_Momentum_20"] = gold_close.pct_change(20)
        df["Stock_Gold_Corr_20"] = (
            df["Log_Return"].rolling(20).corr(df["Gold_LogReturn"])
        )

    if not silver_df.empty:
        s = silver_df.reindex(df.index, method="ffill")
        s_col = "Silver_Close" if "Silver_Close" in s.columns else "Close"
        silver_close           = s[s_col]
        df["Silver_LogReturn"] = np.log(silver_close / silver_close.shift(1))

    if not gold_df.empty and not silver_df.empty:
        df["Gold_Silver_Ratio"] = gold_close / silver_close.replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Forex features
# ─────────────────────────────────────────────────────────────────────────────

def add_forex_features(
    df: pd.DataFrame,
    eurusd_df: pd.DataFrame | None = None,
    usdinr_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge EUR/USD and USD/INR features into the stock DataFrame.
    Sources: Kaggle hourly CSV resampled daily, or yfinance fallback.
    New columns: EURUSD_Change_5d, USDINR_Change_5d
    """
    if eurusd_df is None:
        try:
            from utils.external_data import load_eurusd
            eurusd_df = load_eurusd()
        except Exception:
            eurusd_df = pd.DataFrame()

    if usdinr_df is None:
        try:
            from utils.external_data import load_usdinr
            usdinr_df = load_usdinr()
        except Exception:
            usdinr_df = pd.DataFrame()

    if not eurusd_df.empty:
        e = eurusd_df.reindex(df.index, method="ffill")
        e_col = "EURUSD" if "EURUSD" in e.columns else "Close"
        eurusd_price          = e[e_col]
        df["EURUSD_Change_5d"] = eurusd_price.pct_change(5)

    if not usdinr_df.empty:
        u = usdinr_df.reindex(df.index, method="ffill")
        u_col = "USDINR" if "USDINR" in u.columns else "Close"
        usdinr_price          = u[u_col]
        df["USDINR_Change_5d"] = usdinr_price.pct_change(5)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# S&P 500 features
# ─────────────────────────────────────────────────────────────────────────────

def add_sp500_features(
    df: pd.DataFrame,
    sp500_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge S&P 500 features into the stock DataFrame.
    Sources: Kaggle CSVs or yfinance ^GSPC fallback.
    New columns: SP500_Change_5d
    """
    if sp500_df is None:
        try:
            from utils.external_data import load_sp500
            sp500_df = load_sp500()
        except Exception:
            sp500_df = pd.DataFrame()

    if not sp500_df.empty:
        s = sp500_df.reindex(df.index, method="ffill")
        s_col = "SP500_Close" if "SP500_Close" in s.columns else "Close"
        if s_col in s.columns:
            sp500_close = s[s_col]
            df["SP500_Change_5d"] = sp500_close.pct_change(5)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PCA dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────

def apply_pca(
    X: pd.DataFrame,
    n_components: int = config.PCA_COMPONENTS,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    """
    Standardise features then apply PCA.
    Returns (transformed DataFrame, fitted PCA, fitted scaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    cols = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(X_pca, index=X.index, columns=cols), pca, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helper
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(
    stock_df:  pd.DataFrame,
    gold_df:   pd.DataFrame | None = None,
    silver_df: pd.DataFrame | None = None,
    forex_df:  pd.DataFrame | None = None,   # legacy EUR/USD param
    vix_df:    pd.DataFrame | None = None,
    eurusd_df: pd.DataFrame | None = None,
    usdinr_df: pd.DataFrame | None = None,
    sp500_df:  pd.DataFrame | None = None,
    drop_na:   bool = True,
    use_external: bool = True,               # auto-load external if not supplied
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for a single stock.
    Returns a feature-rich DataFrame with a 'Target' column.
    Automatically loads external datasets (VIX, Gold, Silver, Forex, S&P 500)
    if use_external=True and the local CSVs / yfinance are available.
    """
    df = build_all_technical_features(stock_df)

    # Legacy cross-asset (EUR/USD used as forex_df)
    df = add_cross_asset_features(df, gold_df=gold_df, forex_df=forex_df if forex_df is not None else eurusd_df)

    if use_external:
        df = add_vix_features(df, vix_df=vix_df)
        df = add_commodity_features(df, gold_df=gold_df, silver_df=silver_df)
        df = add_forex_features(df, eurusd_df=eurusd_df if eurusd_df is not None else forex_df, usdinr_df=usdinr_df)
        df = add_sp500_features(df, sp500_df=sp500_df)

    df = add_target(df)

    # Drop any remaining raw non-stationary price/indicator levels to ensure stationarity
    cols_to_drop = ["Gold_Close", "Forex_Close", "Silver_Close", "EURUSD", "USDINR", "SP500_Close", "OBV"]
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)

    if drop_na:
        df = df.dropna()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    raw = yf.download("AAPL", start="2020-01-01", end="2024-01-01",
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    processed = prepare_dataset(raw)
    print("Feature columns:", list(processed.columns))
    print("Shape:", processed.shape)
    print(processed.tail(3))
