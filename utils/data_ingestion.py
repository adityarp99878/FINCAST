"""
utils/data_ingestion.py
-----------------------
Downloads and caches raw financial data from Yahoo Finance.
Covers stocks, gold/silver futures, and forex pairs.
"""

import os
import logging
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core downloader
# ─────────────────────────────────────────────────────────────────────────────

def download_ticker(
    ticker: str,
    start: str = config.START_DATE,
    end: str   = config.END_DATE,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker.
    Results are cached as CSV inside data/raw/ to avoid repeated API calls.
    """
    safe_name = ticker.replace("=", "_").replace(".", "_")
    cache_path = os.path.join(config.DATA_RAW_DIR, f"{safe_name}.csv")

    if cache and os.path.exists(cache_path):
        log.info("Loading cached data for %s", ticker)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    log.info("Downloading %s  [%s → %s]", ticker, start, end)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        log.warning("No data returned for %s", ticker)
        return df

    # Flatten multi-level columns if present (yfinance ≥ 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    df.to_csv(cache_path)
    log.info("Saved %d rows for %s", len(df), ticker)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Batch downloaders
# ─────────────────────────────────────────────────────────────────────────────

def download_stocks() -> dict[str, pd.DataFrame]:
    """Download all stock tickers defined in config."""
    return {t: download_ticker(t) for t in config.STOCK_TICKERS}


def download_commodities() -> dict[str, pd.DataFrame]:
    """Download gold and silver futures."""
    tickers = {
        "GOLD":   config.GOLD_TICKER,
        "SILVER": config.SILVER_TICKER,
    }
    return {name: download_ticker(sym) for name, sym in tickers.items()}


def download_forex() -> dict[str, pd.DataFrame]:
    """Download forex pairs."""
    return {pair: download_ticker(pair) for pair in config.FOREX_PAIRS}


def download_all() -> dict[str, pd.DataFrame]:
    """Download every asset class and return a combined dict."""
    log.info("=== Starting full data download ===")
    data = {}
    data.update(download_stocks())
    data.update(download_commodities())
    data.update(download_forex())
    log.info("=== Download complete — %d datasets ===", len(data))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def align_dates(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Reindex all DataFrames to a common business-day calendar
    (inner join on dates present in ALL series).
    """
    if not datasets:
        return datasets

    common_idx = None
    for df in datasets.values():
        if df.empty:
            continue
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    if common_idx is None:
        return datasets

    log.info("Aligning %d datasets to %d common dates", len(datasets), len(common_idx))
    return {k: df.reindex(common_idx) for k, df in datasets.items()}


def get_close_prices(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract the 'Close' column from every dataset and merge into
    a single wide DataFrame (one column per asset).
    """
    closes = {}
    for name, df in datasets.items():
        if df.empty or "Close" not in df.columns:
            continue
        closes[name] = df["Close"]

    return pd.DataFrame(closes).dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    datasets = download_all()
    aligned  = align_dates(datasets)
    closes   = get_close_prices(aligned)
    print("\nClose price matrix shape:", closes.shape)
    print(closes.tail())
