"""
MarketMind — Stage 2D: LightGBM Classifier + Walk-Forward Validation
=====================================================================
Trains a 3-class LightGBM model to predict post-earnings direction.

Walk-forward temporal split (strict, no look-ahead):
    Train  → 2023 + 2024 calls
    Val    → 2025 calls
    Test   → 2026 calls

Output:
    models/lightgbm_marketmind_v1.pkl
    models/feature_columns.json
    models/model_metrics.json
"""

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATRIX_PATH  = PROJECT_ROOT / "data" / "processed" / "features" / "train_matrix.parquet"
MODELS_DIR   = PROJECT_ROOT / "models"
MODEL_PATH   = MODELS_DIR / "lightgbm_marketmind_v1.pkl"
FEAT_PATH    = MODELS_DIR / "feature_columns.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
MLFLOW_EXPERIMENT_NAME = "marketmind"

LABEL_MAP    = {"POSITIVE": 2, "NEUTRAL": 1, "NEGATIVE": 0}
LABEL_REMAP  = {v: k for k, v in LABEL_MAP.items()}

TRAIN_YEARS  = [2023, 2024]
VAL_YEARS    = [2025]
TEST_YEARS   = [2026]

# Features to use (subset — robust features only)
BASE_FEATURES = [
    "sent_ceo",
    "sent_cfo",
    "sent_analyst",
    "sent_ceo_delta",
    "sent_cfo_delta",
    "evasion_mean",
    "evasion_max",
    "evasive_count",
    "revenue_qoq_pct",
    "margin_qoq_pp",
    "discrepancy_count",
    "momentum_30d",
    "volatility_30d",
]


def load_data() -> pd.DataFrame:
    if not MATRIX_PATH.exists():
        # Fallback 1: check if CSV fallback exists
        csv_path = MATRIX_PATH.with_suffix(".csv")
        if csv_path.exists():
            logger.info(f"Loading matrix from CSV fallback: {csv_path}")
            df = pd.read_csv(csv_path)
            return df

        # Fallback 2: dynamic feature build
        logger.error(f"Feature matrix not found at {MATRIX_PATH}. Attempting automated build...")
        try:
            from src.features.feature_builder import build_feature_matrix
            df = build_feature_matrix()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Automated feature build failed: {e}")

        raise FileNotFoundError(
            f"Feature matrix not found: {MATRIX_PATH}. "
            "Please ensure data/processed/features/train_matrix.parquet is committed to repository."
        )

    df = pd.read_parquet(MATRIX_PATH)
    logger.info(f"Loaded matrix: {df.shape}")
    return df


def prepare_splits(df: pd.DataFrame, train_years=None, val_years=None, test_years=None):
    if train_years is None:
        train_years = TRAIN_YEARS
    if val_years is None:
        val_years = VAL_YEARS
    if test_years is None:
        test_years = TEST_YEARS

    df = df.copy()
    df["year"] = pd.to_datetime(df["call_date"]).dt.year

    # Only keep rows with a valid label
    df = df[df["label"].isin(LABEL_MAP.keys())].copy()
    df["target"] = df["label"].map(LABEL_MAP)

    # Determine available features (intersect with what's in the data)
    available = [f for f in BASE_FEATURES if f in df.columns]
    missing   = [f for f in BASE_FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"Missing features (will fill 0): {missing}")
        for f in missing:
            df[f] = 0.0
        available = BASE_FEATURES

    logger.info(f"Using {len(available)} features: {available}")

    X = df[available].values.astype(np.float32)
    y = df["target"].values.astype(np.int32)
    years = df["year"].values

    train_mask = np.isin(years, train_years)
    val_mask   = np.isin(years, val_years)
    test_mask  = np.isin(years, test_years)

    logger.info(f"Train: {train_mask.sum()}  Val: {val_mask.sum()}  Test: {test_mask.sum()}")

    return X, y, train_mask, val_mask, test_mask, available, df


def train_model(X, y, train_mask, val_mask, feature_names):
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    if len(X_train) == 0:
        logger.warning("No training data! Check fiscal_year values in DB.")

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "n_estimators": 45,
        "learning_rate": 0.08,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    logger.success(f"Model trained — trees: {model.n_estimators}, max_depth: {model.max_depth}")
    return model, params


def evaluate(model, X, y, mask, split_name: str) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix
    )

    if mask.sum() == 0:
        logger.warning(f"No data for {split_name} split.")
        return {}

    X_s, y_s = X[mask], y[mask]
    preds = model.predict(X_s)

    acc    = accuracy_score(y_s, preds)
    f1     = f1_score(y_s, preds, average="macro", zero_division=0)
    report = classification_report(y_s, preds,
                                   target_names=["NEGATIVE", "NEUTRAL", "POSITIVE"],
                                   zero_division=0)
    cm     = confusion_matrix(y_s, preds).tolist()

    logger.info(f"\n── {split_name} Results ──────────────────────────")
    logger.info(f"Accuracy : {acc:.3f}  |  F1 (macro): {f1:.3f}")
    logger.info(f"Confusion matrix:\n{cm}")
    logger.info(f"\n{report}")

    return {"accuracy": round(acc, 4), "f1_macro": round(f1, 4), "confusion_matrix": cm}


def run_training(train_years=None, val_years=None, test_years=None, run_name=None):
    if train_years is None:
        train_years = TRAIN_YEARS
    if val_years is None:
        val_years = VAL_YEARS
    if test_years is None:
        test_years = TEST_YEARS
    if run_name is None:
        run_name = "lightgbm_walk_forward_v1"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df       = load_data()
    X, y, train_mask, val_mask, test_mask, features, df_full = prepare_splits(
        df, train_years=train_years, val_years=val_years, test_years=test_years
    )

    model, params = train_model(X, y, train_mask, val_mask, features)

    metrics = {}
    metrics["train_years"] = train_years
    metrics["val_years"] = val_years
    metrics["test_years"] = test_years
    metrics["train"] = evaluate(model, X, y, train_mask, f"TRAIN ({train_years})")
    metrics["val"]   = evaluate(model, X, y, val_mask,   f"VAL   ({val_years})")
    metrics["test"]  = evaluate(model, X, y, test_mask,  f"TEST  ({test_years})")

    # Feature importance
    imp = dict(zip(features, model.feature_importances_.tolist()))
    imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
    logger.info(f"Feature importances: {imp_sorted}")
    metrics["feature_importances"] = imp_sorted

    # Save model + metadata
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": features, "label_map": LABEL_MAP}, f)

    with open(FEAT_PATH, "w") as f:
        json.dump(features, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.success(f"Model saved → {MODEL_PATH}")
    logger.success(f"Metrics saved → {METRICS_PATH}")

    # MLflow Tracking
    try:
        import mlflow
        import mlflow.lightgbm

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({
                "model_type": "LightGBM",
                "framework": "lightgbm",
                "stage": "Stage 4",
                "task": "market_direction_prediction",
                "retrained": "true" if len(train_years) > 2 else "false",
            })

            # Log hyperparameters & configuration
            mlflow.log_params(params)
            mlflow.log_param("num_features", len(features))
            mlflow.log_param("features_list", ",".join(features))
            mlflow.log_param("train_years", str(train_years))
            mlflow.log_param("val_years", str(val_years))
            mlflow.log_param("test_years", str(test_years))

            # Log dataset split counts
            mlflow.log_metric("train_samples", int(train_mask.sum()))
            mlflow.log_metric("val_samples", int(val_mask.sum()))
            mlflow.log_metric("test_samples", int(test_mask.sum()))

            # Log evaluation metrics across splits
            for split_key in ["train", "val", "test"]:
                m = metrics.get(split_key, {})
                if m:
                    mlflow.log_metric(f"{split_key}_accuracy", float(m.get("accuracy", 0.0)))
                    mlflow.log_metric(f"{split_key}_f1_macro", float(m.get("f1_macro", 0.0)))

            # Log feature importances
            for feat_name, feat_val in imp_sorted.items():
                mlflow.log_metric(f"feat_imp_{feat_name}", float(feat_val))

            # Log model and JSON artifacts
            if FEAT_PATH.exists():
                mlflow.log_artifact(str(FEAT_PATH), artifact_path="metadata")
            if METRICS_PATH.exists():
                mlflow.log_artifact(str(METRICS_PATH), artifact_path="metadata")
            if MODEL_PATH.exists():
                mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_pickle")

            try:
                mlflow.lightgbm.log_model(model, name="lightgbm_model")
            except Exception as model_err:
                logger.warning(f"mlflow.lightgbm.log_model failed ({model_err}), pickled model logged via log_artifact.")

            logger.success(f"MLflow Run logged successfully (Run ID: {run.info.run_id})")
    except Exception as exc:
        logger.warning(f"MLflow experiment logging skipped or failed: {exc}")

    return model, features, metrics


if __name__ == "__main__":
    run_training()
