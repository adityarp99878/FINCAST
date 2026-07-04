"""
main.py
-------
End-to-end pipeline for the Explainable Multi-Asset Financial Forecasting system.

Run:
    python main.py
"""

import os, logging
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")   # use a windowed backend (swap to "Qt5Agg" if TkAgg not installed)
import matplotlib.pyplot as plt
plt.ion()                 # interactive mode: show() no longer blocks

import config
from utils.data_ingestion      import download_ticker, download_commodities, download_forex
from utils.feature_engineering import prepare_dataset
from models.regime_detection   import detect_regimes
from models.prediction_models  import (
    time_series_split, train_and_evaluate_all,
    feature_importance_plot, explain_with_shap,
)
from utils.backtesting         import run_backtest, print_summary, plot_backtest
from models.strategy_optimizer import calculate_strategy_signals

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def step1_download_data(ticker: str) -> dict:
    """Download stock, commodity, and forex data."""
    log.info("=== STEP 1 - Data Download (%s) ===", ticker)
    commodities = download_commodities()
    forex       = download_forex()

    stock_df = download_ticker(ticker)

    return {
        "ticker":     ticker,
        "stock_df":   stock_df,
        "gold_df":    commodities.get("GOLD"),
        "forex_df":   forex.get(config.FOREX_PAIRS[0]),
    }


def step2_feature_engineering(data: dict) -> pd.DataFrame:
    """Build features and target variable."""
    log.info("=== STEP 2 - Feature Engineering ===")
    df = prepare_dataset(
        stock_df = data["stock_df"],
        gold_df  = data["gold_df"],
        forex_df = data["forex_df"],
    )
    log.info("Feature matrix: %s", df.shape)

    # Save processed data
    os.makedirs(config.DATA_PROC_DIR, exist_ok=True)
    safe = data["ticker"].replace(".", "_")
    df.to_csv(os.path.join(config.DATA_PROC_DIR, f"{safe}_features.csv"))
    return df


def step3_regime_detection(data: dict, features_df: pd.DataFrame) -> pd.Series:
    """Detect bull / bear / high-volatility regimes."""
    log.info("=== STEP 3 - Market Regime Detection ===")
    close = data["stock_df"]["Close"].dropna()

    regimes, detector = detect_regimes(close)
    log.info("Regime distribution:\n%s", regimes.value_counts().to_string())

    detector.save()
    save_path = os.path.join(config.PLOTS_DIR, f"{data['ticker']}_regimes.png")
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    detector.plot_regimes(close, title=f"{data['ticker']} Market Regimes", save_path=save_path)

    # Attach regime as a feature
    features_df["Regime"] = regimes.reindex(features_df.index)
    features_df["Regime_Code"] = features_df["Regime"].map(
        {"Bear": 0, "High-Volatility": 1, "Bull": 2}
    ).fillna(1)
    return features_df


def step4_train_models(features_df: pd.DataFrame) -> dict:
    """Train all three classifiers."""
    log.info("=== STEP 4 - Model Training ===")

    # Drop non-numeric columns before splitting
    df = features_df.select_dtypes(include="number").dropna()
    X_train, X_test, y_train, y_test = time_series_split(df)

    results = train_and_evaluate_all(X_train, X_test, y_train, y_test)

    log.info("=== Model Comparison ===")
    for name, r in results.items():
        m = r["metrics"]
        log.info("%-22s  Accuracy: %.4f  ROC-AUC: %.4f", name, m["accuracy"], m["roc_auc"])

    return {"results": results, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}


def step5_explainability(model_data: dict, ticker: str):
    """Generate SHAP and feature importance plots."""
    log.info("=== STEP 5 - Explainability ===")
    X_train = model_data["X_train"]
    X_test  = model_data["X_test"]

    for name, r in model_data["results"].items():
        model = r["model"]

        # Feature importance (tree models)
        fi_path = os.path.join(config.PLOTS_DIR, f"{ticker}_{name}_importance.png")
        feature_importance_plot(name, model, list(X_train.columns), save_path=fi_path)

        # SHAP (Random Forest only — others are slow)
        if name == "RandomForest":
            shap_path = os.path.join(config.PLOTS_DIR, f"{ticker}_SHAP.png")
            explain_with_shap(model, X_train, X_test, save_path=shap_path)


def step6_backtest(data: dict, model_data: dict, features_df: pd.DataFrame, ticker: str):
    """Backtest the Regime-Adaptive Ensemble strategy vs buy-and-hold."""
    log.info("=== STEP 6 - Portfolio Backtesting (Adaptive Ensemble) ===")

    # Extract all trained models
    models = {name: r["model"] for name, r in model_data["results"].items()}
    X_test = model_data["X_test"]
    regimes = features_df["Regime"]

    # Generate signals using the ensemble strategy
    signals = calculate_strategy_signals(models, X_test, regimes)
    
    log.info("Signal distribution:\n%s", signals.value_counts().to_string())

    close_test = data["stock_df"]["Close"].reindex(X_test.index)
    bt_results = run_backtest(close_test, signals)

    print_summary(bt_results)
    bt_path = os.path.join(config.PLOTS_DIR, f"{ticker}_backtest.png")
    plot_backtest(bt_results, title=f"{ticker} — Adaptive Ensemble Strategy", regimes=regimes, save_path=bt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Explainable Multi-Asset Financial Forecasting ===")
    log.info("Team: Aditya R Prasanth, T Anandha Krishnan, Shiv Nandan J")

    for ticker in config.STOCK_TICKERS:
        log.info("\n\n" + "="*50)
        log.info("Running pipeline for: %s", ticker)
        log.info("="*50)
        
        data        = step1_download_data(ticker)
        features_df = step2_feature_engineering(data)
        features_df = step3_regime_detection(data, features_df)
        model_data  = step4_train_models(features_df)
        step5_explainability(model_data, data["ticker"])
        step6_backtest(data, model_data, features_df, data["ticker"])

    log.info("=== All pipelines complete. Results saved to: %s ===", config.RESULTS_DIR)
    input("\nAll graphs are open. Press ENTER to close them and exit…")
    plt.close("all")


if __name__ == "__main__":
    main()
