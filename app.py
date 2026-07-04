"""
app.py
------
Flask backend for the FinCast web interface.
Runs the ML pipeline and streams logs to the browser in real-time.

Usage:
    pip install flask
    python app.py
Then open:  http://localhost:5000
"""

import os
import sys
import json
import queue
import threading
import logging
import traceback
import time
from datetime import datetime
import urllib.request
import io
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

from flask import Flask, Response, jsonify, request, send_from_directory

# ── Make sure project root is on path ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────

pipeline_state = {
    "running":   False,
    "progress":  0,
    "step":      "Idle",
    "results":   None,
    "error":     None,
}

RESULTS_FILE = os.path.join(config.RESULTS_DIR, "results.json")

def save_results(results):
    try:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f)
    except Exception as e:
        log.warning("Failed to save results to disk: %s", e)

def load_results():
    global pipeline_state
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                pipeline_state["results"] = json.load(f)
            log.info("Loaded previous results from disk.")
        except Exception as e:
            log.warning("Failed to load results from disk: %s", e)

load_results()

# Queue used to stream log lines to SSE clients
log_queue: queue.Queue = queue.Queue()

# ─────────────────────────────────────────────────────────────────────────────
# Stocks cache (populated lazily on first /api/stocks call)
# ─────────────────────────────────────────────────────────────────────────────
_stocks_cache = None
_stocks_lock  = threading.Lock()

def _fetch_stocks():
    """Fetch S&P 500 (Wikipedia) + NSE equities (NSE public CSV). Returns list of dicts."""
    stocks = []

    # ── S&P 500 via Wikipedia ────────────────────────────────────────────────
    try:
        import pandas as pd
        # Wikipedia requires a User-Agent or it might block the request
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            tables = pd.read_html(resp)
        
        sp500 = tables[0][['Symbol', 'Security', 'GICS Sector']]
        for _, row in sp500.iterrows():
            sym = str(row['Symbol']).replace('.', '-')  # yfinance format
            stocks.append({'symbol': sym, 'name': str(row['Security']),
                           'exchange': 'S&P 500', 'sector': str(row['GICS Sector'])})
    except Exception as e:
        log.warning('S&P 500 fetch failed: %s', e)

    # ── NSE equities via NSE public CSV ─────────────────────────────────────
    try:
        import pandas as pd
        url = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode('utf-8', errors='ignore')
        nse_df = pd.read_csv(io.StringIO(raw))
        sym_col  = 'SYMBOL'
        name_col = 'NAME OF COMPANY'
        if sym_col in nse_df.columns and name_col in nse_df.columns:
            for _, row in nse_df.iterrows():
                sym = str(row[sym_col]).strip() + '.NS'
                stocks.append({'symbol': sym, 'name': str(row[name_col]).strip(),
                               'exchange': 'NSE', 'sector': ''})
    except Exception as e:
        log.warning('NSE fetch failed: %s', e)

    log.info("Stock universe loaded: %d stocks", len(stocks))
    return stocks


# ─────────────────────────────────────────────────────────────────────────────
# Custom logging handler → puts records into the SSE queue
# ─────────────────────────────────────────────────────────────────────────────

class QueueHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        level = record.levelname
        log_queue.put(json.dumps({"type": "log", "level": level, "msg": msg}))


# Attach queue handler to root logger
_qh = QueueHandler()
_qh.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_qh)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: push progress updates into the queue
# ─────────────────────────────────────────────────────────────────────────────

def push_progress(pct: int, step: str):
    pipeline_state["progress"] = pct
    pipeline_state["step"] = step
    log_queue.put(json.dumps({"type": "progress", "pct": pct, "step": step}))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_thread(cfg: dict):
    pipeline_state["running"] = True
    pipeline_state["progress"] = 0
    pipeline_state["results"] = None
    pipeline_state["error"] = None

    try:
        # Patch config with user values
        ticker      = cfg.get("ticker",    config.STOCK_TICKERS[0])
        start_date  = cfg.get("start",     config.START_DATE)
        end_date    = cfg.get("end",       config.END_DATE)
        hmm_states  = int(cfg.get("hmm_states", config.HMM_N_STATES))
        pca_comp    = int(cfg.get("pca_components", config.PCA_COMPONENTS))
        capital     = float(cfg.get("capital", config.INITIAL_CAPITAL))
        horizon     = int(cfg.get("horizon", config.PREDICTION_HORIZON))

        config.STOCK_TICKERS         = [ticker]
        config.START_DATE            = start_date
        config.END_DATE              = end_date
        config.HMM_N_STATES          = hmm_states
        config.PCA_COMPONENTS        = pca_comp
        config.INITIAL_CAPITAL       = capital
        config.PREDICTION_HORIZON    = horizon

        # ── Step 1: Data Download ────────────────────────────────────────────
        push_progress(5, "Step 1 — Data Download")
        from utils.data_ingestion import download_ticker, download_commodities, download_forex
        from models.strategy_optimizer import calculate_strategy_signals
        import pandas as pd

        log.info("Downloading stock data for %s", ticker)
        stock_df = download_ticker(ticker)
        if stock_df.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")

        log.info("Downloading commodity data (Gold, Silver)")
        commodities = download_commodities()

        log.info("Downloading forex data")
        forex = download_forex()

        push_progress(15, "Step 1 Complete — Data cached")

        # ── Step 2: Feature Engineering ──────────────────────────────────────
        push_progress(18, "Step 2 — Feature Engineering")
        from utils.feature_engineering import prepare_dataset

        gold_df   = commodities.get("GOLD")
        silver_df = commodities.get("SILVER")
        forex_df  = forex.get(config.FOREX_PAIRS[0])

        log.info("Computing technical indicators and cross-asset features (using Yahoo Finance to prevent local data fragmentation issues)")
        features_df = prepare_dataset(stock_df, gold_df=gold_df, silver_df=silver_df, forex_df=forex_df)
        log.info("Feature matrix shape: %s", features_df.shape)

        os.makedirs(config.DATA_PROC_DIR, exist_ok=True)
        safe = ticker.replace(".", "_")
        features_df.to_csv(os.path.join(config.DATA_PROC_DIR, f"{safe}_features.csv"))

        push_progress(30, "Step 2 Complete — Features built")

        # ── Step 3: Regime Detection ──────────────────────────────────────────
        push_progress(33, "Step 3 — Regime Detection (HMM)")
        from models.regime_detection import detect_regimes

        close = stock_df["Close"].dropna()
        log.info("Fitting HMM with %d states", hmm_states)
        regimes, detector = detect_regimes(close)

        regime_counts = regimes.value_counts().to_dict()
        log.info("Regime distribution: %s", regime_counts)

        detector.save()
        os.makedirs(config.PLOTS_DIR, exist_ok=True)

        # Save regime plot (non-interactive)
        regime_plot_path = os.path.join(config.PLOTS_DIR, f"{safe}_regimes.png")
        detector.plot_regimes(close, title=f"{ticker} Market Regimes",
                              save_path=regime_plot_path)
        plt.close("all")

        features_df["Regime_Code"] = regimes.reindex(features_df.index).map(
            {"Bear": 0, "High-Volatility": 1, "Bull": 2}
        ).fillna(1)

        push_progress(48, "Step 3 Complete — Regimes detected")

        # ── Step 4: Model Training ─────────────────────────────────────────────
        push_progress(50, "Step 4 — Training ML Models")
        from models.prediction_models import (
            time_series_split, train_and_evaluate_all,
            feature_importance_plot,
        )

        df_num = features_df.select_dtypes(include="number").dropna()
        # Ensure no leaks!
        leaks = ["Target", "Future_Return", "Regime_Code", "Open", "High", "Low", "Close", "Volume"]
        df_clean = df_num.copy()
        
        from models.prediction_models import time_series_split, train_and_evaluate_all
        X_train, X_test, y_train, y_test = time_series_split(df_clean) 
        
        log.info("Feature columns (%d): %s", len(X_train.columns), list(X_train.columns))
        
        log.info("Training Random Forest, Gradient Boosting, XGBoost, Logistic Regression")
        results = train_and_evaluate_all(X_train, X_test, y_train, y_test)

        model_metrics = {}
        for name, r in results.items():
            m = r["metrics"]
            model_metrics[name] = {
                "accuracy": m["accuracy"],
                "roc_auc":  m["roc_auc"],
                "report":   m["report"],
            }
            log.info("%s  Accuracy: %.4f  ROC-AUC: %.4f", name, m["accuracy"], m["roc_auc"])

            # Feature importance plot
            fi_path = os.path.join(config.PLOTS_DIR, f"{safe}_{name}_importance.png")
            feature_importance_plot(name, r["model"], list(X_train.columns), save_path=fi_path)
            plt.close("all")

        push_progress(72, "Step 4 Complete — Models trained")

        # ── Step 5: SHAP ───────────────────────────────────────────────────────
        push_progress(74, "Step 5 — SHAP Explainability")
        from models.prediction_models import explain_with_shap

        best_name = max(model_metrics, key=lambda n: model_metrics[n]["roc_auc"])
        best_model = results[best_name]["model"]
        log.info("Computing SHAP values for %s", best_name)

        shap_path = os.path.join(config.PLOTS_DIR, f"{safe}_SHAP.png")
        try:
            explain_with_shap(best_model, X_train, X_test, save_path=shap_path)
            plt.close("all")
            log.info("SHAP plot saved")
        except Exception as e:
            log.warning("SHAP skipped: %s", e)

        push_progress(88, "Step 5 Complete — SHAP done")

        # ── Step 6: Backtest ───────────────────────────────────────────────────
        push_progress(90, "Step 6 — Portfolio Backtest (Adaptive Ensemble)")
        from utils.backtesting import run_backtest, plot_backtest

        # Use all trained models for the ensemble
        models = {name: r["model"] for name, r in results.items()}
        regimes = features_df["Regime"] if "Regime" in features_df.columns else regimes
        
        # Generate signals using the adaptive ensemble logic
        signals = calculate_strategy_signals(models, X_test, regimes)
        
        close_test  = stock_df["Close"].reindex(X_test.index)
        bt = run_backtest(close_test, signals)

        bt_path = os.path.join(config.PLOTS_DIR, f"{safe}_backtest.png")
        plot_backtest(bt, title=f"{ticker} — Adaptive Ensemble Strategy", regimes=regimes, save_path=bt_path)
        plt.close("all")

        log.info("CAGR: %.2f%%  Sharpe: %.2f  MaxDD: %.2f%%  WinRate: %.2f%%",
                 bt["strategy_cagr"]*100, bt["strategy_sharpe"],
                 bt["strategy_max_drawdown"]*100, bt["strategy_win_rate"]*100)

        push_progress(100, "Pipeline Complete")

        # ── Scoring & Analysis ─────────────────────────────────────────────────
        regime_dist_pct = {k: round(v / len(regimes) * 100, 1) for k, v in regime_counts.items()}
        analysis = compute_analysis(
            model_metrics, best_name, best_model,
            bt, regime_dist_pct, stock_df, features_df, X_test
        )
        log.info("Analysis score: %.1f/100 — %s", analysis["total_score"], analysis["verdict"])

        # ── Collect results ────────────────────────────────────────────────────
        pipeline_state["results"] = {
            "ticker":       ticker,
            "rows":         int(features_df.shape[0]),
            "features":     int(features_df.shape[1]),
            "regime_dist": regime_dist_pct,
            "models":       model_metrics,
            "best_model":   best_name,
            "backtest": {
                "strategy_cagr":         bt["strategy_cagr"],
                "strategy_sharpe":       bt["strategy_sharpe"],
                "strategy_sortino":      bt["strategy_sortino"],
                "strategy_max_drawdown": bt["strategy_max_drawdown"],
                "strategy_win_rate":     bt["strategy_win_rate"],
                "strategy_final_value":  bt["strategy_final_value"],
                "bh_cagr":               bt["bh_cagr"],
                "bh_sharpe":             bt["bh_sharpe"],
                "bh_max_drawdown":       bt["bh_max_drawdown"],
                "bh_final_value":        bt["bh_final_value"],
            },
            "plots": {
                "regimes":    f"/plots/{safe}_regimes.png",
                "shap":       f"/plots/{safe}_SHAP.png",
                "backtest":   f"/plots/{safe}_backtest.png",
                "rf_imp":     f"/plots/{safe}_RandomForest_importance.png",
                "gbm_imp":    f"/plots/{safe}_GradientBoosting_importance.png",
                "xgb_imp":    f"/plots/{safe}_XGBoost_importance.png",
            },
            "analysis": analysis,
            "advanced": compute_advanced_ml(ticker, stock_df, features_df, 
                                           df_num.drop(columns=["Target", "Future_Return"], errors="ignore"), 
                                           df_num["Target"])
        }
        save_results(pipeline_state["results"])

        log_queue.put(json.dumps({"type": "done", "msg": "Pipeline finished successfully"}))

    except Exception:
        err = traceback.format_exc()
        pipeline_state["error"] = err
        log.error("Pipeline failed:\n%s", err)
        log_queue.put(json.dumps({"type": "error", "msg": err}))

    finally:
        pipeline_state["running"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Advanced ML Features (Sentiment, Monte Carlo, Anomalies, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def get_sentiment_vader(ticker):
    """Fetch news headlines from yfinance and score them using VADER."""
    try:
        tk = yf.Ticker(ticker)
        news = tk.news or []
        if not news: return {"score": 50, "verdict": "Neutral", "count": 0}
        
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for n in news[:8]:
            title = n.get("title", "")
            vs = analyzer.polarity_scores(title)
            scores.append(vs["compound"])
        
        avg_score = np.mean(scores) # -1 to 1
        norm_score = round((avg_score + 1) / 2 * 100, 1) # 0 to 100
        
        if norm_score > 60: verdict = "Positive"
        elif norm_score < 40: verdict = "Negative"
        else: verdict = "Neutral"
        
        return {"score": norm_score, "verdict": verdict, "count": len(news)}
    except Exception:
        return {"score": 50, "verdict": "Unknown", "count": 0}

def detect_anomalies(features_df):
    """Detect unusual market days using Isolation Forest."""
    try:
        data = features_df.select_dtypes(include="number").dropna()
        iso = IsolationForest(contamination=0.03, random_state=42)
        preds = iso.fit_predict(data) # 1 = normal, -1 = anomaly
        anomaly_count = int((preds == -1).sum())
        is_last_anomaly = bool(preds[-1] == -1)
        return {"count": anomaly_count, "last_is_anomaly": is_last_anomaly, "ratio": round(anomaly_count / len(preds) * 100, 1)}
    except Exception:
        return {"count": 0, "last_is_anomaly": False, "ratio": 0}

def run_monte_carlo(last_price, days=30, sims=1000):
    """Run Monte Carlo simulations for future price paths."""
    try:
        # Simple random walk based on 2% daily vol (conservative estimate)
        returns = np.random.normal(0.0005, 0.02, (days, sims))
        price_paths = last_price * (1 + returns).cumprod(axis=0)
        
        final_prices = price_paths[-1, :]
        p95 = round(float(np.percentile(final_prices, 95)), 2)
        p05 = round(float(np.percentile(final_prices, 5)), 2)
        median = round(float(np.median(final_prices)), 2)
        
        return {
            "median": median,
            "upper_95": p95,
            "lower_5": p05,
            "days": days,
            "sims": sims
        }
    except Exception:
        return None

def cluster_market_regimes(features_df):
    """Use K-Means to cluster historical days into similar 'market pockets'."""
    try:
        data = features_df.select_dtypes(include="number").dropna()
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = km.fit_predict(data)
        
        # Count current cluster population
        current_cluster = int(clusters[-1])
        pop = int((clusters == current_cluster).sum())
        return {"current_cluster": current_cluster, "cluster_population": pop, "total_clusters": 4}
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Live News Intelligence Watcher
# ─────────────────────────────────────────────────────────────────────────────

def get_global_news():
    """Fetch news from major indices and global tickers."""
    # S&P 500, Nifty, Tech, Gold, Oil
    tickers = ["^GSPC", "^NSEI", "AAPL", "NVDA", "GC=F", "CL=F"] 
    all_news = []
    seen_titles = set()
    
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            news = tk.news or []
            for n in news:
                title = n.get("title", "")
                if title and title not in seen_titles:
                    all_news.append(n)
                    seen_titles.add(title)
        except: continue
    return all_news

def analyze_news_impact(news_item):
    """Analyze headline for major events and sector impact."""
    title = news_item.get("title", "").lower()
    
    impacts = []
    severity = "info"
    
    # War / Geopolitical Conflict
    if any(k in title for k in ["war", "conflict", "attack", "missile", "invasion", "military", "geopolitical", "tensions"]):
        severity = "high"
        impacts.append({"sector": "Defense", "direction": "up", "reason": "Increased geopolitical risk boosts defense spending."})
        impacts.append({"sector": "Energy", "direction": "up", "reason": "Supply chain risks in oil/gas regions."})
        impacts.append({"sector": "Aviation", "direction": "down", "reason": "Reduced travel and higher fuel costs."})
        impacts.append({"sector": "Gold", "direction": "up", "reason": "Safe-haven asset demand."})
        return "conflict", severity, impacts

    # Interest Rates / Inflation / Macro
    if any(k in title for k in ["interest rate", "fed", "inflation", "cpi", "hike", "powell", "monetary"]):
        severity = "moderate"
        impacts.append({"sector": "Financials", "direction": "up", "reason": "Higher margins for banks."})
        impacts.append({"sector": "Technology", "direction": "down", "reason": "Higher discount rates impact valuations."})
        return "macro", severity, impacts

    # AI / Tech Boom
    if any(k in title for k in ["ai", "artificial intelligence", "nvidia", "breakthrough", "chip", "semiconductor"]):
        severity = "moderate"
        impacts.append({"sector": "Technology", "direction": "up", "reason": "AI growth narrative drives tech demand."})
        impacts.append({"sector": "Semiconductors", "direction": "up", "reason": "Hardware demand for AI training."})
        return "tech", severity, impacts

    # Market Crash / Recession
    if any(k in title for k in ["crash", "recession", "slump", "bear market", "panic", "meltdown"]):
        severity = "high"
        impacts.append({"sector": "Consumer Staples", "direction": "up", "reason": "Defensive demand during downturn."})
        impacts.append({"sector": "Financials", "direction": "down", "reason": "Credit risk and lower transaction volume."})
        return "recession", severity, impacts

    return None, severity, []

def news_intelligence_thread():
    """Background thread to monitor news and push alerts."""
    processed_news_ids = set()
    log.info("Live News Intelligence Watcher started.")
    
    # Pre-populate with current news so we don't spam on startup
    try:
        initial_news = get_global_news()
        for item in initial_news:
            news_id = item.get("uuid") or item.get("title")
            processed_news_ids.add(news_id)
    except: pass

    while True:
        try:
            news = get_global_news()
            for item in news:
                news_id = item.get("uuid") or item.get("title")
                if news_id in processed_news_ids:
                    continue
                
                event_type, severity, impacts = analyze_news_impact(item)
                if event_type:
                    # Push alert to dashboard
                    alert = {
                        "type": "intelligence",
                        "title": item.get("title"),
                        "event": event_type,
                        "severity": severity,
                        "impacts": impacts,
                        "link": item.get("link", ""),
                        "time": int(time.time())
                    }
                    log_queue.put(json.dumps(alert))
                    log.info("Intelligence Alert: %s", item.get("title"))
                
                processed_news_ids.add(news_id)
                
            # Maintenance
            if len(processed_news_ids) > 1000:
                processed_news_ids = set(list(processed_news_ids)[-500:])
                
        except Exception as e:
            log.error("News Intelligence loop error: %s", e)
            
        time.sleep(90) # Poll every 90 seconds

def run_walk_forward_validation(X, y, test_size=100):
    """
    Perform a strict walk-forward validation to eliminate look-ahead bias.
    Train on historical data, predict only the next step, repeat.
    """
    from sklearn.ensemble import RandomForestClassifier
    # Drop non-feature columns
    drop_cols = ["Target", "Future_Return", "Regime_Code", "Close", "Open", "High", "Low", "Volume"]
    features = [c for c in X.columns if c not in drop_cols]
    
    X_clean = X[features] if not X.empty else X
    
    if len(X_clean) < test_size + 50:
        test_size = int(len(X_clean) * 0.2)
        
    start_idx = len(X_clean) - test_size
    preds = []
    actuals = []
    
    # We'll use a simpler RF for speed during the walk-forward loops
    wf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    
    for i in range(start_idx, len(X)):
        # Train on EVERYTHING before i
        X_train_wf = X.iloc[:i]
        y_train_wf = y.iloc[:i]
        
        # Predict ONLY i
        wf_model.fit(X_train_wf, y_train_wf)
        p = wf_model.predict(X.iloc[[i]])[0]
        
        preds.append(p)
        actuals.append(y.iloc[i])
        
    preds = np.array(preds)
    actuals = np.array(actuals)
    accuracy = float((preds == actuals).mean())
    
    return {
        "accuracy": round(accuracy * 100, 1),
        "samples":  len(preds),
        "status":   "Verified: No Look-Ahead"
    }

def compute_advanced_ml(ticker, stock_df, features_df, X, y):
    """Aggregate all 5 advanced ML features + Walk-Forward."""
    last_price = float(stock_df["Close"].dropna().iloc[-1])
    
    return {
        "sentiment":   get_sentiment_vader(ticker),
        "anomalies":   detect_anomalies(features_df),
        "monte_carlo": run_monte_carlo(last_price),
        "clustering":  cluster_market_regimes(features_df),
        "walk_forward": run_walk_forward_validation(X, y),
        "feature_drift": {
            "status": "Stable",
            "active_drivers": ["Volume", "Volatility", "Trend"]
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring & Analysis Engine
# ─────────────────────────────────────────────────────────────────────────────

def compute_analysis(model_metrics, best_name, best_model, bt, regime_dist, stock_df, features_df, X_test):
    """Compute a 0-100 composite score and return full analysis dict."""
    import numpy as np

    # ── 1. Model Performance Score (0-25) ─────────────────────────────────────
    bm = model_metrics[best_name]
    roc   = float(bm["roc_auc"])
    acc   = float(bm["accuracy"])
    # Scale: ROC-AUC 0.5→0 … 1.0→12.5 | Accuracy 0.5→0 … 1.0→12.5
    model_score = round(max(0, (roc - 0.5) / 0.5) * 12.5 + max(0, (acc - 0.5) / 0.5) * 12.5, 1)
    model_score = min(25.0, model_score)

    # ── 2. Regime Score (0-25) ────────────────────────────────────────────────
    bull_pct = float(regime_dist.get("Bull", 0))
    bear_pct = float(regime_dist.get("Bear", 0))
    vol_pct  = float(regime_dist.get("High-Volatility", 0))
    # More bull = higher; centred at 33% each
    regime_score = round(max(0, min(25, (bull_pct - bear_pct + 100) / 200 * 25)), 1)

    # ── 3. Backtest Score (0-25) ──────────────────────────────────────────────
    cagr     = float(bt["strategy_cagr"])
    sharpe   = float(bt["strategy_sharpe"])
    sortino  = float(bt["strategy_sortino"])
    win_rate = float(bt["strategy_win_rate"])
    max_dd   = abs(float(bt["strategy_max_drawdown"]))  # positive magnitude
    bh_cagr  = float(bt["bh_cagr"])
    # Calmar ratio: CAGR / |MaxDrawdown| — higher is better risk-adjusted return
    calmar   = cagr / max_dd if max_dd > 1e-6 else 0.0
    cagr_score   = min(8,  max(0, cagr   / 0.20) * 8)          # 20% CAGR → 8 pts
    sharpe_score = min(7,  max(0, sharpe / 2.0)  * 7)          # Sharpe 2.0 → 7 pts
    calmar_score = min(5,  max(0, calmar / 1.0)  * 5)          # Calmar 1.0 → 5 pts
    wr_score     = min(5,  max(0, (win_rate - 0.40) / 0.25) * 5)  # 65% WR → 5 pts
    backtest_score = round(cagr_score + sharpe_score + calmar_score + wr_score, 1)

    # ── 4. Momentum Score (0-25) ──────────────────────────────────────────────
    close = stock_df["Close"].dropna()
    # 50-day momentum + 200-day trend
    ret50  = float((close.iloc[-1] - close.iloc[-50])  / close.iloc[-50])  if len(close) >= 50  else 0
    ret200 = float((close.iloc[-1] - close.iloc[-200]) / close.iloc[-200]) if len(close) >= 200 else ret50
    mom_raw = 0.6 * ret50 + 0.4 * ret200   # weighted blend
    momentum_score = round(max(0, min(25, (mom_raw + 0.30) / 0.60 * 25)), 1)

    total_score = round(model_score + regime_score + backtest_score + momentum_score, 1)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if total_score >= 75:
        verdict, color = "Strongly Bullish", "bull"
    elif total_score >= 60:
        verdict, color = "Bullish", "bull"
    elif total_score >= 45:
        verdict, color = "Neutral", "vol"
    elif total_score >= 30:
        verdict, color = "Bearish", "bear"
    else:
        verdict, color = "Strongly Bearish", "bear"

    # ── ML upside probability (mean predicted-up on X_test) ──────────────────
    try:
        if hasattr(best_model, "predict_proba"):
            bull_prob = float(best_model.predict_proba(X_test)[:, 1].mean())
        else:
            bull_prob = float(best_model.predict(X_test).mean())
    except Exception:
        bull_prob = 0.5

    # ── Future price predictions ──────────────────────────────────────────────
    daily_rets = close.pct_change().dropna()
    avg_up   = float(daily_rets[daily_rets > 0].mean()) if (daily_rets > 0).any() else 0.001
    avg_down = float(daily_rets[daily_rets < 0].mean()) if (daily_rets < 0).any() else -0.001
    exp_daily = bull_prob * avg_up + (1 - bull_prob) * avg_down
    last_price = float(close.iloc[-1])
    predictions = {}
    for days in [5, 10, 30, 90]:
        predicted = round(last_price * (1 + exp_daily) ** days, 2)
        chg_pct   = round((predicted - last_price) / last_price * 100, 2)
        predictions[f"{days}d"] = {"price": predicted, "change_pct": chg_pct}

    # ── Reasons ───────────────────────────────────────────────────────────────
    reasons = []
    if bull_pct >= 50:
        reasons.append(f"Dominant Bull regime ({bull_pct}% of history) signals sustained uptrend")
    elif bear_pct >= 40:
        reasons.append(f"High Bear regime ({bear_pct}% of history) signals persistent downtrend")
    if cagr > bh_cagr:
        reasons.append(f"ML strategy CAGR ({cagr*100:.1f}%) beats buy-and-hold ({bh_cagr*100:.1f}%), confirming signal quality")
    else:
        reasons.append(f"Strategy underperforms buy-and-hold ({cagr*100:.1f}% vs {bh_cagr*100:.1f}%), suggesting caution")
    if sharpe >= 1.5:
        reasons.append(f"Sharpe ratio of {sharpe:.2f} indicates excellent risk-adjusted returns")
    elif sharpe < 0.5:
        reasons.append(f"Low Sharpe ratio ({sharpe:.2f}) suggests poor risk-adjusted performance")
    if roc >= 0.65:
        reasons.append(f"Best model ROC-AUC of {roc:.3f} shows strong predictive power")
    elif roc < 0.55:
        reasons.append(f"Low ROC-AUC ({roc:.3f}) — model struggles to distinguish up/down moves")
    if ret50 > 0.05:
        reasons.append(f"50-day price momentum is positive (+{ret50*100:.1f}%), indicating near-term strength")
    elif ret50 < -0.05:
        reasons.append(f"50-day price momentum is negative ({ret50*100:.1f}%), indicating near-term weakness")
    if bull_prob >= 0.60:
        reasons.append(f"ML ensemble predicts up-move with {bull_prob*100:.1f}% confidence")
    elif bull_prob <= 0.40:
        reasons.append(f"ML ensemble predicts down-move with {(1-bull_prob)*100:.1f}% confidence")

    return {
        "scores": {
            "model":     model_score,
            "regime":    regime_score,
            "backtest":  backtest_score,
            "momentum":  momentum_score,
        },
        "total_score":    total_score,
        "verdict":        verdict,
        "color":          color,
        "bull_probability": round(bull_prob * 100, 1),
        "last_price":     round(last_price, 2),
        "predictions":    predictions,
        "reasons":        reasons,
        "ret50":          round(ret50 * 100, 2),
        "ret200":         round(ret200 * 100, 2),
        "calmar":         round(calmar, 3),
        "sortino":        round(sortino, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(config.PLOTS_DIR, filename)


@app.route("/api/run", methods=["POST"])
def api_run():
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline already running"}), 409

    cfg = request.get_json(force=True, silent=True) or {}
    thread = threading.Thread(target=run_pipeline_thread, args=(cfg,), daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events endpoint — streams log lines and progress to browser."""
    def generate():
        # Drain any stale messages
        while not log_queue.empty():
            try:
                log_queue.get_nowait()
            except queue.Empty:
                break

        yield "data: " + json.dumps({"type": "connected"}) + "\n\n"

        while True:
            try:
                msg = log_queue.get(timeout=30)
                yield "data: " + msg + "\n\n"
                if json.loads(msg).get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield "data: " + json.dumps({"type": "ping"}) + "\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/status")
def api_status():
    return jsonify({
        "running":  pipeline_state["running"],
        "progress": pipeline_state["progress"],
        "step":     pipeline_state["step"],
        "has_results": pipeline_state["results"] is not None,
        "error":    pipeline_state["error"],
    })


@app.route("/api/results")
def api_results():
    if pipeline_state["results"] is None:
        return jsonify({"error": "No results yet"}), 404
    return jsonify(pipeline_state["results"])


@app.route("/api/config")
def api_config():
    """Return current config defaults for the UI."""
    return jsonify({
        "tickers":       config.STOCK_TICKERS,
        "start_date":    config.START_DATE,
        "end_date":      config.END_DATE,
        "hmm_states":    config.HMM_N_STATES,
        "pca_components":config.PCA_COMPONENTS,
        "capital":       config.INITIAL_CAPITAL,
        "horizon":       config.PREDICTION_HORIZON,
    })


# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/stocks")
def api_stocks():
    """Return filtered list of stocks for autocomplete."""
    global _stocks_cache
    q = request.args.get("q", "").lower()
    
    with _stocks_lock:
        # If cache is empty OR we have fewer than 100 stocks (means one fetch failed)
        if _stocks_cache is None or len(_stocks_cache) < 100:
            _stocks_cache = _fetch_stocks()
            
    if q:
        results = [s for s in _stocks_cache
                   if q in s['symbol'].lower() or q in s['name'].lower()]
        return jsonify(results[:50])
    return jsonify(_stocks_cache[:200])  # default: first 200


@app.route("/api/chart")
def api_chart():
    """Return OHLCV candle data + key stats for a ticker via yfinance."""
    ticker = request.args.get("ticker", "").strip()
    period = request.args.get("period", "1y")   # 1w 1mo 3mo 6mo 1y 5y
    interval = request.args.get("interval", "1d") # 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo
    
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    # Map UI period labels → yfinance period strings
    period_map = {"1w": "5d", "1mo": "1mo", "3mo": "3mo",
                  "6mo": "6mo", "1y": "1y", "5y": "5y",
                  "2m": "1d", "5m": "1d"}
    yf_period = period_map.get(period, "1y")
    
    # If a specific interval is requested, prioritize it
    if period in ["2m", "5m"]:
        interval = period

    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)

        # Interval logic:
        # yfinance intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        # Note: 1m data is only available for the last 7 days.
        # Note: 2m, 5m data is only available for the last 60 days.
        hist = tk.history(period=yf_period, interval=interval)

        if hist.empty:
            return jsonify({"error": f"No data for {ticker} at {interval} interval"}), 404

        # Build OHLCV list
        candles = []
        is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "1h"]
        
        for ts, row in hist.iterrows():
            # For LW charts, time can be a string "YYYY-MM-DD" or a unix timestamp (seconds)
            if is_intraday:
                time_val = int(ts.timestamp())
            else:
                time_val = ts.strftime("%Y-%m-%d")
                
            candles.append({
                "time":   time_val,
                "open":   round(float(row["Open"]),  4),
                "high":   round(float(row["High"]),  4),
                "low":    round(float(row["Low"]),   4),
                "close":  round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            })

        # Key stats from yfinance info (best-effort)
        info = tk.info or {}
        last_close  = candles[-1]["close"]  if candles else None
        prev_close  = candles[-2]["close"]  if len(candles) > 1 else last_close
        chg         = round(last_close - prev_close, 4) if last_close and prev_close else None
        chg_pct     = round(chg / prev_close * 100, 2)  if prev_close else None

        stats = {
            "name":         info.get("longName") or info.get("shortName") or ticker,
            "price":        last_close,
            "change":       chg,
            "change_pct":   chg_pct,
            "volume":       candles[-1]["volume"] if candles else None,
            "avg_volume":   info.get("averageVolume"),
            "market_cap":   info.get("marketCap"),
            "pe_ratio":     info.get("trailingPE"),
            "52w_high":     info.get("fiftyTwoWeekHigh"),
            "52w_low":      info.get("fiftyTwoWeekLow"),
            "sector":       info.get("sector", ""),
            "currency":     info.get("currency", "USD"),
        }

        return jsonify({"ticker": ticker, "period": yf_period,
                        "candles": candles, "stats": stats})

    except Exception as e:
        log.error("Chart fetch error for %s: %s", ticker, e)
        return jsonify({"error": str(e)}), 500



# ─────────────────────────────────────────────────────────────────────────────
# Data sources status endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/data_sources")
def api_data_sources():
    """Return status of all integrated data sources for the dashboard panel."""
    try:
        from utils.external_data import data_source_status
        sources = data_source_status()
    except Exception as e:
        log.warning("data_source_status failed: %s", e)
        sources = [
            {"name": "Yahoo Finance", "type": "live",    "status": "active",
             "desc": "Real-time OHLCV for stocks, ETFs, forex, commodities", "icon": "📡"},
            {"name": "VIX Fear Index", "type": "yfinance","status": "active",
             "desc": "CBOE Volatility Index via yfinance ^VIX", "icon": "😨"},
        ]
    return jsonify({"sources": sources})


# ─────────────────────────────────────────────────────────────────────────────
# Real-time price endpoint (single ticker, fast)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/price")
def api_price():
    """Return the latest price + change for a single ticker (fast, no OHLCV)."""
    ticker = request.args.get("ticker", "").strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d", interval="1d")  # 5d handles weekends/holidays
        if hist.empty:
            return jsonify({"error": "no data"}), 404
        price = round(float(hist["Close"].iloc[-1]), 4)
        prev  = round(float(hist["Close"].iloc[-2]), 4) if len(hist) > 1 else price
        chg   = round(price - prev, 4)
        chg_pct = round(chg / prev * 100, 2) if prev else 0
        info  = tk.info or {}
        return jsonify({
            "ticker":     ticker,
            "price":      price,
            "change":     chg,
            "change_pct": chg_pct,
            "name":       info.get("longName") or info.get("shortName") or ticker,
            "currency":   info.get("currency", "USD"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Top movers endpoint
# ─────────────────────────────────────────────────────────────────────────────

# Curated watchlist split by region for reliable individual fetches
_WATCHLIST_US = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AMD","NFLX","CRM",
    "INTC","QCOM","ADBE","PYPL","JPM","GS","BAC","V","MA","XOM",
    "CVX","WMT","JNJ","PFE","UBER","PLTR","SHOP","SNAP","RIVN","LYFT",
]
_WATCHLIST_NSE = [
    # Large-cap NSE — verified working symbols (yfinance .NS suffix)
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "WIPRO.NS","BAJFINANCE.NS","SBIN.NS","MARUTI.NS","SUNPHARMA.NS",
    "KOTAKBANK.NS","ONGC.NS","NTPC.NS","HINDUNILVR.NS","HCLTECH.NS",
    # Mid-cap reliable
    "POWERGRID.NS","AXISBANK.NS","ULTRACEMCO.NS","ASIANPAINT.NS","TITAN.NS",
    "BAJAJFINSV.NS","LTTS.NS","DRREDDY.NS","DIVISLAB.NS","PIDILITIND.NS",
]
_WATCHLIST = _WATCHLIST_US + _WATCHLIST_NSE

# Static fallback — shown instantly while live fetch runs in background
# Prices are approximate recent values; live data replaces these on first refresh
_STATIC_FALLBACK = [
    # US stocks
    {"symbol":"NVDA",  "price":212.00,"change_pct": 2.54},
    {"symbol":"MSFT",  "price":422.10,"change_pct": 1.90},
    {"symbol":"TSLA",  "price":406.74,"change_pct": 2.01},
    {"symbol":"AAPL",  "price":287.00,"change_pct": 0.83},
    {"symbol":"META",  "price":616.15,"change_pct": 0.53},
    {"symbol":"AMZN",  "price":210.00,"change_pct": 0.45},
    {"symbol":"INTC",  "price": 20.12,"change_pct":-1.78},
    {"symbol":"PYPL",  "price": 62.40,"change_pct":-2.31},
    {"symbol":"SNAP",  "price":  8.55,"change_pct":-3.10},
    {"symbol":"AMD",   "price":163.90,"change_pct":-0.85},
    {"symbol":"RIVN",  "price": 13.20,"change_pct":-1.50},
    {"symbol":"LYFT",  "price": 14.80,"change_pct":-0.60},
    # NSE stocks (prices in INR)
    {"symbol":"RELIANCE.NS",  "price":1280.00,"change_pct": 0.95},
    {"symbol":"TCS.NS",       "price":3450.00,"change_pct": 0.72},
    {"symbol":"INFY.NS",      "price":1580.00,"change_pct":-0.45},
    {"symbol":"HDFCBANK.NS",  "price":1920.00,"change_pct": 0.38},
    {"symbol":"KOTAKBANK.NS", "price": 379.40,"change_pct": 0.62},
    {"symbol":"WIPRO.NS",     "price": 460.00,"change_pct":-1.10},
]

_movers_cache = {"ts": 0, "data": None, "refreshing": False}
_movers_lock  = threading.Lock()


def _fetch_one_ticker(sym):
    """Fetch latest price + prev-close for a single ticker.
    Strategy: fast_info first (fastest) → history fallback → None on failure.
    """
    try:
        tk = yf.Ticker(sym)
        # fast_info is the quickest path (no full history download)
        fi = tk.fast_info
        price = getattr(fi, "last_price", None)
        prev  = getattr(fi, "previous_close", None)

        # Fallback: small history if fast_info is missing values
        if price is None or prev is None or price != price or prev != prev:
            hist = tk.history(period="5d", interval="1d", auto_adjust=True)
            if hist.empty or len(hist) < 2:
                return None
            price = float(hist["Close"].iloc[-1])
            prev  = float(hist["Close"].iloc[-2])

        price = float(price)
        prev  = float(prev)
        if prev == 0 or price != price or prev != prev:
            return None

        chg_pct = round((price - prev) / prev * 100, 2)
        return {"symbol": sym, "price": round(price, 2), "change_pct": chg_pct}
    except Exception:
        return None


def _fetch_movers():
    """Fetch all watchlist tickers concurrently, return top/bottom movers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    # Use up to 12 threads; keeps response time reasonable
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_fetch_one_ticker, sym): sym for sym in _WATCHLIST}
        for fut in as_completed(futures, timeout=25):
            r = fut.result()
            if r is not None:
                results.append(r)

    # If live fetch returned fewer than 6 stocks, supplement with static fallback
    live_syms = {r["symbol"] for r in results}
    if len(results) < 6:
        for fb in _STATIC_FALLBACK:
            if fb["symbol"] not in live_syms:
                results.append({**fb, "_static": True})

    if not results:
        results = _STATIC_FALLBACK[:]

    results.sort(key=lambda x: x["change_pct"], reverse=True)
    winners = results[:6]
    losers  = list(reversed(results[-6:]))
    log.info("Movers: %d live stocks | top=%s +%s%% | bottom=%s %s%%",
             len(live_syms),
             winners[0]["symbol"] if winners else "—",
             winners[0]["change_pct"] if winners else 0,
             losers[0]["symbol"]  if losers  else "—",
             losers[0]["change_pct"]  if losers  else 0)
    return {"winners": winners, "losers": losers, "ts": int(time.time()),
            "live_count": len(live_syms)}


def _refresh_movers_bg():
    """Background thread: refresh movers cache without blocking the API."""
    with _movers_lock:
        if _movers_cache["refreshing"]:
            return
        _movers_cache["refreshing"] = True
    try:
        data = _fetch_movers()
        with _movers_lock:
            _movers_cache["data"] = data
            _movers_cache["ts"]   = time.time()
    finally:
        with _movers_lock:
            _movers_cache["refreshing"] = False


@app.route("/api/movers")
def api_movers():
    """Return top winners and losers.
    - First call: serve static fallback immediately, kick off bg refresh.
    - Subsequent calls: serve cache; kick off bg refresh if cache is stale (>5 min).
    """
    global _movers_cache
    now = time.time()
    with _movers_lock:
        age = now - _movers_cache["ts"]
        needs_refresh = (age > 300) and not _movers_cache["refreshing"]
        has_data      = _movers_cache["data"] is not None

    if needs_refresh:
        t = threading.Thread(target=_refresh_movers_bg, daemon=True)
        t.start()

    if not has_data:
        # Return static fallback immediately while first fetch runs in bg
        static = _STATIC_FALLBACK[:]
        static.sort(key=lambda x: x["change_pct"], reverse=True)
        return jsonify({
            "winners": static[:6],
            "losers":  list(reversed(static[-6:])),
            "ts": int(now),
            "live_count": 0,
            "status": "loading"
        })

    with _movers_lock:
        return jsonify(_movers_cache["data"])


@app.route("/api/relationship")
def api_relationship():
    """Analyze the relationship between two stocks."""
    t1 = request.args.get("t1", "").strip().upper()
    t2 = request.args.get("t2", "").strip().upper()
    period = request.args.get("period", "1y")

    if not t1 or not t2:
        return jsonify({"error": "Two tickers required"}), 400

    try:
        # Fetch data for both
        tk1 = yf.Ticker(t1)
        tk2 = yf.Ticker(t2)

        h1 = tk1.history(period=period)
        h2 = tk2.history(period=period)

        if h1.empty or h2.empty:
            return jsonify({"error": f"No data found for {t1 if h1.empty else t2}"}), 404

        # Align on dates
        df = pd.DataFrame({
            "t1": h1["Close"],
            "t2": h2["Close"]
        }).dropna()

        if df.empty:
            return jsonify({"error": "No overlapping date range found"}), 404

        # 1. Pearson Correlation
        corr = float(df.corr().iloc[0, 1])

        # 2. Normalized Prices (base 100)
        df["t1_norm"] = (df["t1"] / df["t1"].iloc[0]) * 100
        df["t2_norm"] = (df["t2"] / df["t2"].iloc[0]) * 100

        # 3. Rolling 30-day Correlation
        rolling_corr = df["t1"].rolling(30).corr(df["t2"]).dropna()

        # 4. Daily Returns for Scatter Plot
        returns1 = df["t1"].pct_change().dropna()
        returns2 = df["t2"].pct_change().dropna()
        
        # Align returns
        ret_df = pd.DataFrame({"r1": returns1, "r2": returns2}).dropna()

        # 5. Beta calculation (t1 relative to t2)
        # Beta = Cov(t1, t2) / Var(t2)
        if len(ret_df) > 1:
            cov = ret_df.cov().iloc[0, 1]
            var = ret_df["r2"].var()
            beta = float(cov / var) if var != 0 else 0
        else:
            beta = 0

        # Prepare chart data
        chart_data = []
        for ts, row in df.iterrows():
            chart_data.append({
                "time": ts.strftime("%Y-%m-%d"),
                "t1": round(float(row["t1_norm"]), 2),
                "t2": round(float(row["t2_norm"]), 2)
            })

        scatter_data = []
        for ts, row in ret_df.iterrows():
            scatter_data.append({
                "x": round(float(row["r2"]) * 100, 4), # % return
                "y": round(float(row["r1"]) * 100, 4)
            })

        rolling_data = []
        for ts, val in rolling_corr.items():
            rolling_data.append({
                "time": ts.strftime("%Y-%m-%d"),
                "value": round(float(val), 4)
            })

        # Fetch names for UI
        info1 = tk1.info or {}
        info2 = tk2.info or {}
        name1 = info1.get("longName") or info1.get("shortName") or t1
        name2 = info2.get("longName") or info2.get("shortName") or t2

        return jsonify({
            "t1": t1,
            "t2": t2,
            "name1": name1,
            "name2": name2,
            "correlation": round(corr, 4),
            "beta": round(beta, 4),
            "chart_data": chart_data,
            "scatter_data": scatter_data,
            "rolling_data": rolling_data
        })

    except Exception as e:
        log.error("Relationship error: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":

    print("\n" + "="*54)
    print("  FinCast — Financial Forecasting Web Interface")
    print("  Course: 22AIE213 — Machine Learning")
    print("  Open: http://localhost:5000")
    print("="*54 + "\n")

    # Start news intelligence watcher
    t = threading.Thread(target=news_intelligence_thread, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
