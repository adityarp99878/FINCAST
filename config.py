"""
config.py — Central configuration for the financial forecasting project.
Edit this file to change tickers, date ranges, model parameters, etc.
"""

# ─── Data Settings ────────────────────────────────────────────────────────────

# Stock tickers to analyse (Yahoo Finance format)
STOCK_TICKERS = ["AAPL", "MSFT", "GOOGL", "RELIANCE.NS", "TCS.NS"]

# Commodity / forex symbols available via yfinance
GOLD_TICKER   = "GC=F"   # Gold Futures
SILVER_TICKER = "SI=F"   # Silver Futures
FOREX_PAIRS   = ["EURUSD=X", "GBPUSD=X", "USDINR=X"]

# Historical data window
START_DATE = "2005-01-01"
END_DATE   = "2026-05-08"

# ─── Feature Engineering ──────────────────────────────────────────────────────

# RSI window length (days)
RSI_PERIOD = 14

# MACD parameters
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# Bollinger Band window
BB_WINDOW = 20

# Rolling windows for statistical features
ROLLING_WINDOWS = [5, 10, 20, 50]

# Number of PCA components to retain
PCA_COMPONENTS = 10

# ─── Market Regime Detection (HMM) ───────────────────────────────────────────

HMM_N_STATES    = 3   # bull / bear / high-volatility
HMM_N_ITER      = 100
HMM_COVARIANCE  = "full"

# ─── Prediction Target ────────────────────────────────────────────────────────

# How many trading days ahead to predict
PREDICTION_HORIZON = 1

# Binary label: 1 if return > threshold, else 0
RETURN_THRESHOLD = 0.0

# ─── Model Parameters ─────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE    = 0.2

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": RANDOM_STATE,
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
}

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

# ─── Sentiment Analysis ───────────────────────────────────────────────────────

TFIDF_MAX_FEATURES = 5000

# ─── Backtesting ──────────────────────────────────────────────────────────────

INITIAL_CAPITAL       = 100_000   # USD
TRANSACTION_COST_PCT  = 0.001     # 0.1% realistic commission (round-trip)
RISK_FREE_RATE        = 0.04      # annualised
STRATEGY_TYPE         = "long_only"  # long_only: cash when not confident (safer with ~53% accuracy models)
LEVERAGE              = 1.0       # e.g., 2.0 for 2x leverage (increases profit and risk)

# Regime-specific confidence thresholds for ensemble signals.
# Philosophy: in a Bull regime (HMM-detected), go long unless the model
# strongly disagrees (threshold 0.49 = participate in the uptrend).
# In Bear / High-Vol regimes, require real model conviction before entering.
REGIME_THRESHOLDS = {
    "Bull":            {"long": 0.49, "short": 0.45},   # follow the bull: long unless model says no
    "Bear":            {"long": 0.62, "short": 0.45},   # very selective in bear
    "High-Volatility": {"long": 0.56, "short": 0.44},   # cautious in volatile markets
}

# Trailing stop-loss: exit to cash when running drawdown exceeds this fraction.
# Set to 0 to disable.  12% is a common institutional risk limit.
TRAILING_STOP_PCT = 0.12

import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR     = os.path.join(RESULTS_DIR, "reports")
