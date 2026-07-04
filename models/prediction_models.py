"""
models/prediction_models.py
----------------------------
Train, evaluate and persist the three classical ML classifiers:
    - Random Forest
    - Gradient Boosting (XGBoost)
    - Logistic Regression

Also includes time-series-aware train/test splitting and SHAP explainability.
"""

from __future__ import annotations
import os, logging, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Train / Test split (time-series safe)
# ─────────────────────────────────────────────────────────────────────────────

def time_series_split(
    df: pd.DataFrame,
    target_col: str = "Target",
    test_size:  float = config.TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronological split — NO shuffling to prevent look-ahead bias.
    Drops non-feature columns before returning X.
    """
    drop_cols = [target_col, "Future_Return", "Regime_Code", "Open", "High", "Low", "Close", "Volume"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df[target_col]

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_train) == 0:
        raise ValueError("Not enough data to train. Feature matrix is empty after dropping NaNs. Check if external datasets overlap with the stock's date range.")

    log.info(
        "Train: %d rows (%s → %s) | Test: %d rows (%s → %s)",
        len(X_train), X_train.index[0].date(), X_train.index[-1].date(),
        len(X_test),  X_test.index[0].date(),  X_test.index[-1].date() if len(X_test) > 0 else "N/A",
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_random_forest() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)),
    ])


def build_gradient_boosting() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(**config.GRADIENT_BOOSTING_PARAMS)),
    ])


def build_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)),
    ])


def build_xgboost() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ])


MODEL_REGISTRY = {
    "RandomForest":        build_random_forest,
    "GradientBoosting":    build_gradient_boosting,
    "XGBoost":             build_xgboost,
    "LogisticRegression":  build_logistic_regression,
}


PARAM_GRIDS = {
    "RandomForest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_split": [2, 5, 10]
    },
    "GradientBoosting": {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7]
    },
    "XGBoost": {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7],
        "clf__subsample": [0.8, 1.0]
    },
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "saga"]
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True
) -> Pipeline:
    """Build and fit a named model with automated hyperparameter tuning."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")

    log.info("Training %s …", name)
    pipeline = MODEL_REGISTRY[name]()
    
    if tune and name in PARAM_GRIDS:
        log.info("Tuning %s via RandomizedSearchCV...", name)
        # Random search to avoid full exhaustive search taking too long
        search = RandomizedSearchCV(
            pipeline, 
            param_distributions=PARAM_GRIDS[name], 
            n_iter=10, 
            cv=3, 
            scoring="roc_auc", 
            random_state=42, 
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        log.info("Best parameters for %s: %s", name, search.best_params_)
        pipeline = search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        
    log.info("%s training complete.", name)
    return pipeline


def evaluate_model(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Return a dict of evaluation metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":    name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_proba), 4),
        "report":   classification_report(y_test, y_pred),
    }

    log.info(
        "%s | Accuracy: %.4f | ROC-AUC: %.4f",
        name, metrics["accuracy"], metrics["roc_auc"],
    )
    return metrics


def plot_confusion_matrix(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str | None = None,
):
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Down", "Up"],
        cmap="Blues",
    )
    disp.ax_.set_title(f"{name} — Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_shap(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_display: int = 15,
    save_path: str | None = None,
) -> shap.Explanation:
    """
    Compute SHAP values for the test set using a TreeExplainer
    (falls back to KernelExplainer for non-tree models).
    """
    # Extract the raw classifier and the scaler from the pipeline
    scaler = model.named_steps["scaler"]
    clf    = model.named_steps["clf"]

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    plt.style.use('dark_background')
    try:
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer(X_test_scaled)
    except Exception:
        log.warning("TreeExplainer failed — falling back to KernelExplainer (slow).")
        explainer   = shap.KernelExplainer(clf.predict_proba, shap.sample(X_train_scaled, 100))
        shap_values = explainer(X_test_scaled)

    # Summary plot
    fig = plt.figure(facecolor='#0d1117')
    # Random Forest returns (samples, features, classes), XGBoost returns (samples, features)
    if len(shap_values.shape) == 3:
        plot_values = shap_values[:, :, 1]
    else:
        plot_values = shap_values
        
    shap.summary_plot(plot_values, X_test_scaled, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance — Class 1 (Price Up)", color='#e6edf3', fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='#0d1117')
        log.info("SHAP plot saved → %s", save_path)

    plt.close(fig)
    return shap_values


def feature_importance_plot(
    name: str,
    model: Pipeline,
    feature_names: list[str],
    top_n: int = 20,
    save_path: str | None = None,
):
    """Bar chart of feature importance for tree-based models."""
    clf = model.named_steps["clf"]

    if not hasattr(clf, "feature_importances_"):
        log.warning("%s does not expose feature_importances_", name)
        return

    importances = pd.Series(clf.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    importances.plot(kind="barh", ax=ax, color="#00e5a0", alpha=0.8)
    ax.set_title(f"{name} — Top {top_n} Feature Importances", color='#e6edf3', fontweight='bold')
    ax.set_xlabel("Importance", color='#7d8590')
    ax.tick_params(colors='#7d8590')
    ax.grid(alpha=0.1, color='#30363d')
    for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='#0d1117')
    
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: Pipeline, name: str, path: str | None = None):
    path = path or os.path.join(config.MODELS_DIR, f"{name}.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log.info("Model saved → %s", path)


def load_model(name: str, path: str | None = None) -> Pipeline:
    path = path or os.path.join(config.MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Train all models — convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate_all(
    X_train, X_test, y_train, y_test
) -> dict[str, dict]:
    """Train + evaluate all three models and return results dict."""
    results = {}
    for name in MODEL_REGISTRY:
        model   = train_model(name, X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        save_model(model, name)
        results[name] = {"model": model, "metrics": metrics}
    return results
