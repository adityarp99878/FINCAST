# Explainable Multi-Asset Financial Forecasting

**Course:** 22AIE304 – Machine Learning  
**Team:** Aditya R Prasanth · T Anandha Krishnan · Shiv Nandan J

---

## Project Structure

```
financial_forecasting/
├── config.py                      # Central configuration (tickers, params, paths)
├── main.py                        # Full end-to-end pipeline runner
├── requirements.txt
│
├── data/
│   ├── raw/                       # Cached CSV downloads
│   └── processed/                 # Feature-engineered datasets
│
├── models/
│   ├── regime_detection.py        # HMM-based market regime classifier
│   └── prediction_models.py       # RF / GBM / LR + SHAP explainability
│
├── utils/
│   ├── data_ingestion.py          # yfinance downloader + alignment helpers
│   ├── feature_engineering.py     # Technical indicators, cross-asset features, PCA
│   ├── sentiment_analysis.py      # TF-IDF + Naive Bayes sentiment pipeline
│   └── backtesting.py             # Portfolio simulator (Sharpe, drawdown, equity curve)
│
└── results/
    ├── plots/                     # Auto-saved PNG figures
    └── reports/                   # Saved metrics and summaries
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

This will:
1. Download AAPL + Gold + EURUSD data via `yfinance` (cached to `data/raw/`)
2. Engineer 40+ technical and cross-asset features
3. Detect bull / bear / high-volatility regimes using an HMM
4. Train Random Forest, Gradient Boosting, and Logistic Regression
5. Generate SHAP explainability plots
6. Run a portfolio backtest vs buy-and-hold

---

## Configuration

All parameters live in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `STOCK_TICKERS` | `["AAPL", ...]` | Stocks to analyse |
| `START_DATE` | `2005-01-01` | Data start |
| `PREDICTION_HORIZON` | `1` | Days ahead to predict |
| `HMM_N_STATES` | `3` | Regime states (Bull/Bear/HighVol) |
| `PCA_COMPONENTS` | `10` | PCA output dimensions |
| `INITIAL_CAPITAL` | `$100,000` | Backtest starting capital |

---

## Module Overview

### `utils/data_ingestion.py`
Downloads and caches OHLCV data. Call `download_all()` for everything or
`download_ticker("AAPL")` for a single asset.

### `utils/feature_engineering.py`
Builds RSI, MACD, Bollinger Bands, moving averages, volatility, OBV,
cross-asset correlations, and the binary target variable.

### `models/regime_detection.py`
`MarketRegimeDetector` wraps `hmmlearn.GaussianHMM`. Automatically maps
raw HMM states to Bear / High-Volatility / Bull by mean return ranking.

### `models/prediction_models.py`
Three sklearn `Pipeline` objects (scaler + classifier). Includes
`explain_with_shap()` for SHAP summary plots and `feature_importance_plot()`
for tree-model importances.

### `utils/sentiment_analysis.py`
TF-IDF + Multinomial Naive Bayes. Load the FNSPID dataset via
`train_from_csv("path/to/fnspid.csv")`, or call `aggregate_daily_sentiment()`
to merge news scores into your feature DataFrame.

### `utils/backtesting.py`
Long-only strategy: hold on signal=1, cash on signal=0.
Reports CAGR, Sharpe, Sortino, max drawdown, win rate vs buy-and-hold.

---

## Adding News Sentiment Data

1. Download the [FNSPID dataset](https://github.com/Zdong104/FNSPID).
2. Run:
```python
from utils.sentiment_analysis import train_from_csv, aggregate_daily_sentiment
clf = train_from_csv("data/raw/fnspid.csv")
daily_sentiment = aggregate_daily_sentiment(news_df, clf)
# Then merge daily_sentiment into your features DataFrame
```

---

## Literature References

| # | Citation |
|---|---|
| 1 | Lopez de Prado (2018). *Advances in Financial Machine Learning.* |
| 2 | Hamilton (1989). HMM regime switching. *Econometrica.* |
| 3 | Breiman (2001). Random Forests. *Machine Learning.* |
| 4 | Lundberg & Lee (2017). SHAP values. *NeurIPS.* |
