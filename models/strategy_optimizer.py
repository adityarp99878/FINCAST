"""
models/strategy_optimizer.py
----------------------------
Logic for ensembling multiple ML models and applying regime-based filters
to generate trading signals.
"""

import pandas as pd
import numpy as np
import logging
import config

log = logging.getLogger(__name__)

class RegimeAdaptiveEnsemble:
    """
    Ensembles multiple classifiers and adjusts entry/exit thresholds
    based on the current market regime.
    """

    def __init__(self, models: dict, regime_thresholds: dict = config.REGIME_THRESHOLDS):
        self.models = models  # dict of name -> fitted pipeline
        self.regime_thresholds = regime_thresholds

    def get_ensemble_probabilities(self, X: pd.DataFrame) -> pd.Series:
        """
        Calculate average probability (soft voting) across all models.
        """
        all_probas = []
        for name, model in self.models.items():
            try:
                # Get probability for class 1 (price up)
                proba = model.predict_proba(X)[:, 1]
                all_probas.append(proba)
            except Exception as e:
                log.warning("Model %s failed to predict_proba: %s", name, e)

        if not all_probas:
            return pd.Series(0.5, index=X.index)

        avg_proba = np.mean(all_probas, axis=0)
        return pd.Series(avg_proba, index=X.index, name="Ensemble_Proba")

    def generate_signals(self, X: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        """
        Generate -1, 0, 1 signals based on ensemble probability and regime-specific thresholds.
        """
        probas = self.get_ensemble_probabilities(X)
        
        # Align regimes with X index
        regimes = regimes.reindex(X.index).fillna("High-Volatility")
        
        signals = pd.Series(0, index=X.index, name="Signal")
        
        for i, (idx, proba) in enumerate(probas.items()):
            regime = regimes.loc[idx]
            thresholds = self.regime_thresholds.get(regime, self.regime_thresholds["High-Volatility"])
            
            # Logic: 
            # 1 (Long) if proba > threshold_long
            # -1 (Short) if proba < threshold_short
            if proba > thresholds["long"]:
                signals.loc[idx] = 1
            elif proba < thresholds["short"]:
                signals.loc[idx] = -1 if config.STRATEGY_TYPE == "long_short" else 0
            else:
                signals.loc[idx] = 0
                
        return signals

def calculate_strategy_signals(models: dict, X: pd.DataFrame, regimes: pd.Series) -> pd.Series:
    """Convenience function to get signals from a set of models and regimes."""
    optimizer = RegimeAdaptiveEnsemble(models)
    return optimizer.generate_signals(X, regimes)
