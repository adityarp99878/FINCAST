"""
models/regime_detection.py
---------------------------
Hidden Markov Model (HMM) for financial market regime classification.
Identifies latent states: Bull Market / Bear Market / High Volatility.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib, os, logging

from hmmlearn.hmm import GaussianHMM

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Regime Detector
# ─────────────────────────────────────────────────────────────────────────────

REGIME_LABELS = {0: "Bear", 1: "High-Volatility", 2: "Bull"}
REGIME_COLORS = {"Bear": "#ff4560", "High-Volatility": "#f5a623", "Bull": "#00e5a0"}


class MarketRegimeDetector:
    """
    Wraps hmmlearn's GaussianHMM to detect market regimes from
    log-returns and rolling volatility.
    """

    def __init__(
        self,
        n_states:   int = config.HMM_N_STATES,
        n_iter:     int = config.HMM_N_ITER,
        covariance: str = config.HMM_COVARIANCE,
        random_state: int = config.RANDOM_STATE,
    ):
        self.n_states   = n_states
        self.model      = GaussianHMM(
            n_components    = n_states,
            covariance_type = covariance,
            n_iter          = n_iter,
            random_state    = random_state,
        )
        self._fitted     = False
        self._state_map  = {}  # raw HMM state → semantic label

    # ── Feature prep ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_hmm_features(close: pd.Series, vol_window: int = 20) -> np.ndarray:
        """
        Returns a 2-column observation matrix:
            col-0 : daily log-return
            col-1 : rolling std of log-returns (proxy for volatility)
        """
        log_ret  = np.log(close / close.shift(1)).dropna()
        rolling_vol = log_ret.rolling(vol_window).std().dropna()

        # align both series
        common   = log_ret.index.intersection(rolling_vol.index)
        X = np.column_stack([log_ret.loc[common], rolling_vol.loc[common]])
        return X, common

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, close: pd.Series) -> "MarketRegimeDetector":
        X, _ = self._build_hmm_features(close)
        self.model.fit(X)
        self._fitted = True
        self._assign_semantic_labels()
        log.info("HMM fitted — states: %s", self._state_map)
        return self

    def _assign_semantic_labels(self):
        """
        Map HMM states to Bear / High-Vol / Bull based on the mean return
        of each state (lowest return → Bear, highest → Bull, middle → High-Vol).
        """
        means = self.model.means_[:, 0]          # mean log-return per state
        order = np.argsort(means)                 # ascending
        labels = ["Bear", "High-Volatility", "Bull"]
        self._state_map = {int(order[i]): labels[i] for i in range(self.n_states)}

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, close: pd.Series) -> pd.Series:
        """Return a Series of regime labels aligned to the close price index."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")

        X, idx = self._build_hmm_features(close)
        raw_states = self.model.predict(X)
        labels = [self._state_map[s] for s in raw_states]
        return pd.Series(labels, index=idx, name="Regime")

    def predict_proba(self, close: pd.Series) -> pd.DataFrame:
        """Return posterior probabilities for each regime state."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")

        X, idx = self._build_hmm_features(close)
        proba = self.model.predict_proba(X)
        cols  = [self._state_map[i] for i in range(self.n_states)]
        return pd.DataFrame(proba, index=idx, columns=cols)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | None = None):
        path = path or os.path.join(config.MODELS_DIR, "regime_detector.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        log.info("Regime detector saved → %s", path)

    @classmethod
    def load(cls, path: str | None = None) -> "MarketRegimeDetector":
        path = path or os.path.join(config.MODELS_DIR, "regime_detector.pkl")
        return joblib.load(path)

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_regimes(
        self,
        close: pd.Series,
        title: str = "Market Regime Detection",
        save_path: str | None = None,
    ):
        regimes = self.predict(close)
        close_aligned = close.reindex(regimes.index)

        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, facecolor='#0d1117')

        # Price panel with regime shading
        ax1.set_facecolor('#0d1117')
        ax1.plot(close_aligned.index, close_aligned.values, color="#e6edf3", lw=1.5)
        ax1.set_ylabel("Close Price", color='#7d8590')
        ax1.set_title(title, color='#e6edf3', fontweight='bold')
        ax1.tick_params(colors='#7d8590')
        for spine in ax1.spines.values(): spine.set_edgecolor('#30363d')

        for regime, colour in REGIME_COLORS.items():
            mask = regimes == regime
            ax1.fill_between(
                close_aligned.index,
                close_aligned.min(),
                close_aligned.max(),
                where=mask.reindex(close_aligned.index).fillna(False),
                alpha=0.15,
                color=colour,
                label=regime,
            )
        ax1.legend(loc="upper left", frameon=False)
        ax1.grid(alpha=0.05, color='#30363d')

        # Regime state over time
        ax2.set_facecolor('#0d1117')
        regime_num = regimes.map({v: k for k, v in self._state_map.items()})
        ax2.step(regime_num.index, regime_num.values, color="#6366f1", lw=2, where='post')
        ax2.set_yticks(list(self._state_map.keys()))
        ax2.set_yticklabels([self._state_map[k] for k in self._state_map.keys()])
        ax2.set_ylabel("Regime", color='#7d8590')
        ax2.set_xlabel("Date", color='#7d8590')
        ax2.tick_params(colors='#7d8590')
        ax2.grid(alpha=0.05, color='#30363d')
        for spine in ax2.spines.values(): spine.set_edgecolor('#30363d')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='#0d1117')
            log.info("Regime plot saved → %s", save_path)

        plt.close(fig)
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def detect_regimes(close: pd.Series) -> tuple[pd.Series, MarketRegimeDetector]:
    """Fit and return (regime labels, fitted detector) in one call."""
    detector = MarketRegimeDetector()
    detector.fit(close)
    regimes = detector.predict(close)
    return regimes, detector


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    df = yf.download("AAPL", start="2010-01-01", end="2024-01-01",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    regimes, detector = detect_regimes(df["Close"])
    print(regimes.value_counts())
    detector.plot_regimes(df["Close"], title="AAPL Market Regimes")
