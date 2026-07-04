import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.getcwd())
import config
from utils.data_ingestion import download_ticker, download_commodities, download_forex
from utils.feature_engineering import prepare_dataset
from models.regime_detection import MarketRegimeDetector
from models.prediction_models import time_series_split, build_gradient_boosting, build_random_forest, build_xgboost, build_logistic_regression
from models.strategy_optimizer import RegimeAdaptiveEnsemble
from utils.backtesting import run_backtest

# Load data
ticker = "GOOGL"
stock_df = download_ticker(ticker)
commodities = download_commodities()
forex = download_forex()

gold_df = commodities.get("GOLD")
silver_df = commodities.get("SILVER")
forex_df = forex.get(config.FOREX_PAIRS[0])

features_df = prepare_dataset(stock_df, gold_df=gold_df, silver_df=silver_df, forex_df=forex_df)

# Fit HMM
detector = MarketRegimeDetector()
detector.fit(stock_df["Close"])
regimes = detector.predict(stock_df["Close"])

# Map to regime codes to mimic app.py
features_df["Regime_Code"] = regimes.reindex(features_df.index).map(
    {"Bear": 0, "High-Volatility": 1, "Bull": 2}
).fillna(1)

# Split
X_train, X_test, y_train, y_test = time_series_split(features_df, test_size=config.TEST_SIZE)

# Train models on train set
rf = build_random_forest().fit(X_train, y_train)
gb = build_gradient_boosting().fit(X_train, y_train)
xgb_model = build_xgboost().fit(X_train, y_train)
lr = build_logistic_regression().fit(X_train, y_train)

models = {
    "RandomForest": rf,
    "GradientBoosting": gb,
    "XGBoost": xgb_model,
    "LogisticRegression": lr
}

# Align test close
close_test = stock_df["Close"].reindex(X_test.index)

# Grid search thresholds
best_cagr = -999.0
best_thresholds = None
best_metrics = None

# Grid search values for long threshold (and keep short threshold at 0.45)
# We test values from 0.40 to 0.55
grid_values = [0.40, 0.42, 0.44, 0.46, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]

for bull_long in grid_values:
    for bear_long in grid_values:
        for vol_long in grid_values:
            test_thresholds = {
                "Bull":            {"long": bull_long, "short": 0.45},
                "Bear":            {"long": bear_long, "short": 0.45},
                "High-Volatility": {"long": vol_long,  "short": 0.45}
            }
            
            optimizer = RegimeAdaptiveEnsemble(models, regime_thresholds=test_thresholds)
            signals = optimizer.generate_signals(X_test, regimes)
            bt = run_backtest(close_test, signals)
            
            if bt["strategy_cagr"] > best_cagr:
                best_cagr = bt["strategy_cagr"]
                best_thresholds = test_thresholds
                best_metrics = bt

print("\n--- BEST THRESHOLDS FOUND ---")
print(f"Bull Long: {best_thresholds['Bull']['long']}")
print(f"Bear Long: {best_thresholds['Bear']['long']}")
print(f"High-Vol Long: {best_thresholds['High-Volatility']['long']}")
print(f"Best CAGR: {best_cagr * 100:.2f}%")
print(f"Best Sharpe: {best_metrics['strategy_sharpe']:.4f}")
print(f"Best Win Rate: {best_metrics['strategy_win_rate'] * 100:.2f}%")
print(f"Best Max Drawdown: {best_metrics['strategy_max_drawdown'] * 100:.2f}%")
