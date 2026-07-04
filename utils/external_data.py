"""
utils/external_data.py
-----------------------
Loads external datasets from data/external/ and normalises them into
standard DataFrames for the feature engineering pipeline.

HOW TO ADD YOUR DOWNLOADED FILES:
  data/external/
    gold_silver/  ← extract your gold/silver Kaggle ZIP here
    vix/          ← extract your VIX Kaggle ZIP here
    forex/eurusd/ ← extract your EUR/USD Kaggle ZIP here
    india/        ← put nse.csv (or extract India ZIP) here

The loader scans every CSV in each folder and identifies the dataset
by its column headers — so the filename does not matter.

Priority: local CSV (more history) → yfinance fallback (always works)
"""

from __future__ import annotations
import os, glob, logging
import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

_HERE    = os.path.dirname(os.path.abspath(__file__))
_EXT_DIR = os.path.normpath(os.path.join(_HERE, "..", "data", "external"))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _all_csvs(folder: str) -> list[str]:
    """Return every CSV file inside a folder (recursive)."""
    pattern = os.path.join(folder, "**", "*.csv")
    return glob.glob(pattern, recursive=True)


def _read_first_valid(folder: str) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    """
    Try to read each CSV in `folder`. Return (df, filepath) for the first
    one that loads successfully, or (None, None) if nothing works.
    """
    for path in _all_csvs(folder):
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            if df.shape[0] > 10 and df.shape[1] >= 2:
                return df, path
        except Exception:
            try:
                df = pd.read_csv(path, encoding="latin-1", low_memory=False)
                if df.shape[0] > 10 and df.shape[1] >= 2:
                    return df, path
            except Exception:
                continue
    return None, None


def _find_date_col(df: pd.DataFrame) -> str | None:
    """Find the date column by common names."""
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "time", "day", "period"]):
            return col
    return None


def _find_price_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Find a price column matching any of the given keywords."""
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None


def _yf_close(ticker: str, start: str = "1990-01-01") -> pd.DataFrame:
    """Download daily close from yfinance. Returns DataFrame with 'Close' col."""
    try:
        raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw[["Close"]].dropna()
    except Exception as e:
        log.warning("yfinance %s failed: %s", ticker, e)
        return pd.DataFrame()


def _yf_ohlcv(ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    """Download full OHLCV from yfinance. Returns DataFrame with standard columns."""
    try:
        raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
        return raw[cols].dropna()
    except Exception as e:
        log.warning("yfinance OHLCV %s failed: %s", ticker, e)
        return pd.DataFrame()


def _topup_with_yfinance(csv_df: pd.DataFrame, yf_ticker: str,
                          start: str = "2000-01-01") -> pd.DataFrame:
    """
    Given a CSV-loaded DataFrame, fetch yfinance data from the day after the
    last CSV date up to today and concatenate, deduplicating on index.
    Falls back to CSV-only if yfinance fails.
    """
    if csv_df.empty:
        return _yf_ohlcv(yf_ticker, start=start)

    last_csv_date = csv_df.index.max()
    topup_start   = (last_csv_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("Topping up %s from %s onwards via yfinance", yf_ticker, topup_start)
    yf_df = _yf_ohlcv(yf_ticker, start=topup_start)

    if yf_df.empty:
        log.warning("yfinance top-up for %s returned no data — using CSV only", yf_ticker)
        return csv_df

    # Align columns: keep only shared columns
    shared_cols = [c for c in csv_df.columns if c in yf_df.columns]
    if not shared_cols:
        return csv_df

    combined = pd.concat([csv_df[shared_cols], yf_df[shared_cols]])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    log.info("%s combined: %d rows (%s → %s)",
             yf_ticker, len(combined),
             combined.index[0].date(), combined.index[-1].date())
    return combined



# ─────────────────────────────────────────────────────────────────────────────
# VIX
# ─────────────────────────────────────────────────────────────────────────────

def load_vix(start: str = "1990-01-01") -> pd.DataFrame:
    """
    Load VIX daily closing values.
    Local CSV:   data/external/vix/  → topped up via yfinance ^VIX to today
    Returns DataFrame with column [VIX_Close]
    """
    folder = os.path.join(_EXT_DIR, "vix")
    csv_df = pd.DataFrame()
    df, path = _read_first_valid(folder)

    if df is not None:
        try:
            date_col  = _find_date_col(df)
            price_col = _find_price_col(df, ["vix", "close", "price", "value"])
            if date_col and price_col:
                df.index = pd.to_datetime(df[date_col], errors="coerce")
                df = df[[price_col]].rename(columns={price_col: "VIX_Close"})
                df["VIX_Close"] = pd.to_numeric(df["VIX_Close"], errors="coerce")
                df = df.dropna().sort_index()
                df = df[df.index >= pd.Timestamp(start)]
                if len(df) > 50:
                    log.info("VIX CSV: %d rows up to %s", len(df), df.index[-1].date())
                    csv_df = df
        except Exception as e:
            log.warning("VIX CSV parse failed: %s", e)

    # Top up with yfinance from last CSV date → today
    if not csv_df.empty:
        last = csv_df.index.max()
        topup_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = _yf_close("^VIX", start=topup_start)
        if not raw.empty:
            raw = raw.rename(columns={"Close": "VIX_Close"})
            csv_df = pd.concat([csv_df, raw])
            csv_df = csv_df[~csv_df.index.duplicated(keep="last")].sort_index()
        log.info("VIX combined: %d rows → %s", len(csv_df), csv_df.index[-1].date())
        return csv_df

    # Pure yfinance fallback
    raw = _yf_close("^VIX", start=start)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.rename(columns={"Close": "VIX_Close"})
    log.info("VIX: %d rows from yfinance ^VIX", len(raw))
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Gold
# ─────────────────────────────────────────────────────────────────────────────

def load_gold(start: str = "1978-01-01") -> pd.DataFrame:
    """
    Load daily Gold price.
    Local CSV:   data/external/gold_silver/  → topped up via yfinance GC=F to today
    Returns DataFrame with column [Gold_Close]
    """
    folder = os.path.join(_EXT_DIR, "gold_silver")
    csvs = sorted(_all_csvs(folder), key=lambda p: ("gold" not in p.lower()))
    csv_df = pd.DataFrame()

    for path in csvs:
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            if df.shape[0] < 10:
                continue
            date_col  = _find_date_col(df)
            price_col = _find_price_col(df, ["gold", "usd (pm)", "usd", "price", "close", "value"])
            if not (date_col and price_col):
                continue
            df.index = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
            df = df[[price_col]].rename(columns={price_col: "Gold_Close"})
            df["Gold_Close"] = pd.to_numeric(
                df["Gold_Close"].astype(str).str.replace(",", ""), errors="coerce"
            )
            df = df.dropna().sort_index()
            df = df[df.index >= pd.Timestamp(start)]
            if len(df) > 100:
                log.info("Gold CSV: %d rows up to %s", len(df), df.index[-1].date())
                csv_df = df
                break
        except Exception:
            continue

    # Top up with yfinance from last CSV date → today
    if not csv_df.empty:
        last = csv_df.index.max()
        topup_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for ticker in ["GC=F", "GLD"]:
            raw = _yf_close(ticker, start=topup_start)
            if not raw.empty:
                raw = raw.rename(columns={"Close": "Gold_Close"})
                csv_df = pd.concat([csv_df, raw])
                csv_df = csv_df[~csv_df.index.duplicated(keep="last")].sort_index()
                log.info("Gold combined: %d rows → %s", len(csv_df), csv_df.index[-1].date())
                break
        return csv_df

    # Pure yfinance fallback
    for ticker in ["GC=F", "GLD"]:
        raw = _yf_close(ticker, start=start)
        if not raw.empty:
            raw = raw.rename(columns={"Close": "Gold_Close"})
            log.info("Gold: %d rows from yfinance %s", len(raw), ticker)
            return raw
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Silver
# ─────────────────────────────────────────────────────────────────────────────

def load_silver(start: str = "1978-01-01") -> pd.DataFrame:
    """
    Load daily Silver price.
    Local CSV:   data/external/gold_silver/  → topped up via yfinance SI=F to today
    Returns DataFrame with column [Silver_Close]
    """
    folder = os.path.join(_EXT_DIR, "gold_silver")
    csvs   = sorted(_all_csvs(folder), key=lambda p: ("silver" not in p.lower()))
    csv_df = pd.DataFrame()

    for path in csvs:
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            if df.shape[0] < 10:
                continue
            date_col  = _find_date_col(df)
            price_col = _find_price_col(df, ["silver", "ag ", "price", "close", "value"])
            if not (date_col and price_col):
                continue
            df.index = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
            df = df[[price_col]].rename(columns={price_col: "Silver_Close"})
            df["Silver_Close"] = pd.to_numeric(
                df["Silver_Close"].astype(str).str.replace(",", ""), errors="coerce"
            )
            df = df.dropna().sort_index()
            df = df[df.index >= pd.Timestamp(start)]
            if len(df) > 100:
                log.info("Silver CSV: %d rows up to %s", len(df), df.index[-1].date())
                csv_df = df
                break
        except Exception:
            continue

    # Top up with yfinance from last CSV date → today
    if not csv_df.empty:
        last = csv_df.index.max()
        topup_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for ticker in ["SI=F", "SLV"]:
            raw = _yf_close(ticker, start=topup_start)
            if not raw.empty:
                raw = raw.rename(columns={"Close": "Silver_Close"})
                csv_df = pd.concat([csv_df, raw])
                csv_df = csv_df[~csv_df.index.duplicated(keep="last")].sort_index()
                log.info("Silver combined: %d rows → %s", len(csv_df), csv_df.index[-1].date())
                break
        return csv_df

    for ticker in ["SI=F", "SLV"]:
        raw = _yf_close(ticker, start=start)
        if not raw.empty:
            raw = raw.rename(columns={"Close": "Silver_Close"})
            log.info("Silver: %d rows from yfinance %s", len(raw), ticker)
            return raw
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# EUR/USD Forex
# ─────────────────────────────────────────────────────────────────────────────

def load_eurusd(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Load EUR/USD rate.
    Local CSV:   data/external/forex/eurusd/  → topped up via yfinance EURUSD=X to today
    Returns DataFrame with column [EURUSD]
    """
    folder = os.path.join(_EXT_DIR, "forex", "eurusd")
    df, path = _read_first_valid(folder)
    csv_df = pd.DataFrame()

    if df is not None:
        try:
            date_col  = _find_date_col(df)
            price_col = _find_price_col(df, ["close", "price", "last", "eur"])
            if date_col and price_col:
                df.index = pd.to_datetime(df[date_col], errors="coerce")
                df = df[[price_col]].rename(columns={price_col: "EURUSD"})
                df["EURUSD"] = pd.to_numeric(df["EURUSD"], errors="coerce")
                df = df.dropna().sort_index()
                # Resample to daily if intraday
                df = df["EURUSD"].resample("D").last().dropna().to_frame("EURUSD")
                df = df[df.index >= pd.Timestamp(start)]
                if len(df) > 50:
                    log.info("EURUSD CSV: %d rows up to %s", len(df), df.index[-1].date())
                    csv_df = df
        except Exception as e:
            log.warning("EURUSD CSV parse failed: %s", e)

    # Top up with yfinance from last CSV date → today
    if not csv_df.empty:
        last = csv_df.index.max()
        topup_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = _yf_close("EURUSD=X", start=topup_start)
        if not raw.empty:
            raw = raw.rename(columns={"Close": "EURUSD"})
            csv_df = pd.concat([csv_df, raw])
            csv_df = csv_df[~csv_df.index.duplicated(keep="last")].sort_index()
            log.info("EURUSD combined: %d rows → %s", len(csv_df), csv_df.index[-1].date())
        return csv_df

    # Pure yfinance fallback
    raw = _yf_close("EURUSD=X", start=start)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.rename(columns={"Close": "EURUSD"})
    log.info("EURUSD: %d rows from yfinance", len(raw))
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# USD/INR
# ─────────────────────────────────────────────────────────────────────────────

def load_usdinr(start: str = "2000-01-01") -> pd.DataFrame:
    """USD/INR via yfinance (no CSV needed — reliable enough)."""
    for ticker in ["INR=X", "USDINR=X"]:
        raw = _yf_close(ticker, start=start)
        if not raw.empty:
            raw = raw.rename(columns={"Close": "USDINR"})
            log.info("USDINR: %d rows from yfinance %s", len(raw), ticker)
            return raw
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# S&P 500 Index
# ─────────────────────────────────────────────────────────────────────────────

def load_sp500(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Load S&P 500 index level.
    Local CSV:  data/external/us/sp500_index.csv  → topped up via yfinance ^GSPC to today
    Returns DataFrame with column [SP500_Close]
    """
    csv_path = os.path.join(_EXT_DIR, "us", "sp500_index.csv")
    csv_df = pd.DataFrame()

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            date_col  = _find_date_col(df)
            price_col = next((c for c in df.columns if "500" in c or "close" in c.lower() or "price" in c.lower()), None)
            if date_col and price_col:
                df.index = pd.to_datetime(df[date_col], errors="coerce")
                df = df[[price_col]].rename(columns={price_col: "SP500_Close"})
                df["SP500_Close"] = pd.to_numeric(df["SP500_Close"], errors="coerce")
                df = df.dropna().sort_index()
                df = df[df.index >= pd.Timestamp(start)]
                if len(df) > 50:
                    log.info("S&P 500 CSV: %d rows up to %s", len(df), df.index[-1].date())
                    csv_df = df
        except Exception as e:
            log.warning("S&P 500 CSV parse failed: %s", e)

    # Top up with yfinance from last CSV date → today (2026)
    if not csv_df.empty:
        last = csv_df.index.max()
        topup_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = _yf_close("^GSPC", start=topup_start)
        if not raw.empty:
            raw = raw.rename(columns={"Close": "SP500_Close"})
            csv_df = pd.concat([csv_df, raw])
            csv_df = csv_df[~csv_df.index.duplicated(keep="last")].sort_index()
            log.info("S&P 500 combined: %d rows → %s", len(csv_df), csv_df.index[-1].date())
        return csv_df

    # Pure yfinance fallback
    raw = _yf_close("^GSPC", start=start)
    if not raw.empty:
        raw = raw.rename(columns={"Close": "SP500_Close"})
        log.info("S&P 500: %d rows from yfinance ^GSPC", len(raw))
        return raw
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# NSE India stocks
# ─────────────────────────────────────────────────────────────────────────────

def load_nifty_stock(symbol: str, start: str = "2000-01-01") -> pd.DataFrame:
    """
    Load a Nifty 50 stock with hybrid CSV + yfinance coverage.
    1. Loads historical data from local CSV (e.g. Kaggle dataset up to 2022)
    2. Tops up from the last CSV date through today (2026) via yfinance
    Fallback: pure yfinance if no CSV found.
    """
    folder = os.path.join(_EXT_DIR, "india")
    yf_ticker = f"{symbol}.NS"
    csv_df = pd.DataFrame()

    # Try exact match first, then any CSV
    exact = os.path.join(folder, f"{symbol}.csv")
    candidates = [exact] if os.path.exists(exact) else _all_csvs(folder)

    for path in candidates:
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            # Check if this file is for the right symbol
            if "Symbol" in df.columns and symbol not in df["Symbol"].astype(str).values:
                continue
            date_col = _find_date_col(df)
            if not date_col:
                continue
            df.index = pd.to_datetime(df[date_col], errors="coerce")
            # Keep OHLCV columns
            ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            if not ohlcv_cols:
                continue
            df = df[ohlcv_cols].dropna().sort_index()
            df = df[df.index >= pd.Timestamp(start)]
            if len(df) > 50:
                csv_df = df
                log.info("%s (India CSV): %d rows up to %s", symbol, len(df), df.index[-1].date())
                break
        except Exception:
            continue

    # Top up with yfinance from last CSV date → today (2026)
    combined = _topup_with_yfinance(csv_df, yf_ticker, start=start)

    if combined.empty:
        log.warning("%s: no data from CSV or yfinance", symbol)

    return combined



# ─────────────────────────────────────────────────────────────────────────────
# Load everything at once
# ─────────────────────────────────────────────────────────────────────────────

def load_all_external(start: str = "2000-01-01") -> dict:
    """Load all external datasets. Returns dict; empty DF means not available."""
    return {
        "vix":    load_vix(start=start),
        "gold":   load_gold(start=start),
        "silver": load_silver(start=start),
        "eurusd": load_eurusd(start=start),
        "usdinr": load_usdinr(start=start),
        "sp500":  load_sp500(start=start),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard data source status
# ─────────────────────────────────────────────────────────────────────────────

def data_source_status() -> list:
    """Returns list of dicts for the /api/data_sources dashboard endpoint."""
    def _has_csv(folder):
        return len(_all_csvs(os.path.join(_EXT_DIR, folder))) > 0

    return [
        {"name": "Yahoo Finance",    "type": "live",    "status": "active",
         "desc": "Real-time OHLCV — stocks, ETFs, indices, forex, commodities", "icon": "📡"},
        {"name": "VIX Fear Index",   "type": "csv" if _has_csv("vix") else "yfinance",
         "status": "active",
         "desc": "CBOE VIX" + (" (Kaggle CSV)" if _has_csv("vix") else " (yfinance ^VIX)"),
         "icon": "😨"},
        {"name": "Gold Prices",      "type": "csv" if _has_csv("gold_silver") else "yfinance",
         "status": "active",
         "desc": "Gold spot price" + (" (Kaggle CSV)" if _has_csv("gold_silver") else " (yfinance GC=F)"),
         "icon": "🥇"},
        {"name": "Silver Prices",    "type": "csv" if _has_csv("gold_silver") else "yfinance",
         "status": "active",
         "desc": "Silver spot price" + (" (Kaggle CSV)" if _has_csv("gold_silver") else " (yfinance SI=F)"),
         "icon": "🥈"},
        {"name": "EUR/USD Forex",    "type": "csv" if _has_csv("forex/eurusd") else "yfinance",
         "status": "active",
         "desc": "EUR/USD" + (" (Kaggle CSV)" if _has_csv("forex/eurusd") else " (yfinance EURUSD=X)"),
         "icon": "💱"},
        {"name": "USD/INR",          "type": "yfinance", "status": "active",
         "desc": "USD/INR exchange rate (yfinance INR=X)", "icon": "₹"},
        {"name": "NSE India Stocks", "type": "csv" if _has_csv("india") else "yfinance",
         "status": "active",
         "desc": "Nifty 50 OHLCV" + (" (local CSV)" if _has_csv("india") else " (yfinance .NS)"),
         "icon": "🇮🇳"},
        {"name": "FNSPID Sentiment", "type": "github", "status": "missing",
         "desc": "750k labeled financial news sentences — not yet integrated", "icon": "📰"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("\n── External Data Sources ──────────────────────────────")
    ext = load_all_external(start="2020-01-01")
    for name, df in ext.items():
        if df.empty:
            print(f"  [--] {name:10s}  no data")
        else:
            print(f"  [OK] {name:10s}  {len(df):6d} rows"
                  f"  [{df.index[0].date()} to {df.index[-1].date()}]"
                  f"  cols={list(df.columns)}")

    print("\n── Data Source Status ──────────────────────────────────")
    for s in data_source_status():
        mark = "OK" if s["status"] == "active" else "--"
        print(f"  [{mark}] {s['name']:22s} ({s['type']})  {s['desc']}")
    print()
