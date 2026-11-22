"""
MarketMind — Stage 2D (cont.): SHAP Explainability
====================================================
Generates SHAP values for a trained LightGBM model.

Usage:
    from src.models.explainer import explain_prediction
    shap_vals = explain_prediction(feature_vector_dict)
    # returns sorted list of {"feature": str, "shap_value": float}
"""

import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH  = PROJECT_ROOT / "models" / "lightgbm_marketmind_v1.pkl"
FEAT_PATH   = PROJECT_ROOT / "models" / "feature_columns.json"


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run trainer.py first: python -m src.models.trainer"
        )
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["features"], payload["label_map"]


def predict_with_shap(feature_dict: dict) -> dict:
    """
    Predict class and generate SHAP values for a single observation.

    Args:
        feature_dict: {feature_name: value} for all model features.

    Returns:
        {
          "predicted_label": "POSITIVE"|"NEUTRAL"|"NEGATIVE",
          "probabilities":   {"POSITIVE": 0.72, "NEUTRAL": 0.20, "NEGATIVE": 0.08},
          "shap_values":     [{"feature": str, "shap_value": float}, ...],
          "confidence":      0.72
        }
    """
    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap")
        return {}

    model, features, label_map = _load_model()
    label_remap = {v: k for k, v in label_map.items()}

    # Build feature vector
    X = np.array([[feature_dict.get(f, 0.0) for f in features]], dtype=np.float32)

    # Predict
    proba  = model.predict_proba(X)[0]
    pred_i = int(np.argmax(proba))
    conf   = float(proba[pred_i])
    label  = label_remap.get(pred_i, "NEUTRAL")

    proba_dict = {label_remap[i]: round(float(p), 4) for i, p in enumerate(proba)}

    # SHAP — TreeExplainer
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)

    # For multiclass: shap_vals can be array of shape (n_samples, n_features, n_classes) or list
    if isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3:
            vals = shap_vals[0, :, pred_i]
        elif shap_vals.ndim == 2:
            vals = shap_vals[0]
        else:
            vals = shap_vals.flatten()
    elif isinstance(shap_vals, list):
        vals = shap_vals[pred_i][0]
    else:
        vals = np.array(shap_vals).flatten()

    shap_list = [
        {"feature": f, "shap_value": round(float(v), 5)}
        for f, v in zip(features, vals)
    ]
    shap_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "predicted_label": label,
        "probabilities"  : proba_dict,
        "shap_values"    : shap_list,
        "confidence"     : round(conf, 4),
    }


def predict_batch(df: pd.DataFrame, features: Optional[list] = None) -> pd.DataFrame:
    """Run predictions on a DataFrame with model features as columns."""
    model, feat_cols, label_map = _load_model()
    label_remap = {v: k for k, v in label_map.items()}

    if features is None:
        features = feat_cols

    df = df.copy()
    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    X = df[features].fillna(0).values.astype(np.float32)

    proba  = model.predict_proba(X)
    preds  = np.argmax(proba, axis=1)
    confs  = proba[np.arange(len(preds)), preds]

    df["predicted_label"] = [label_remap.get(p, "NEUTRAL") for p in preds]
    df["confidence"]      = confs.round(4)
    return df


if __name__ == "__main__":
    # Quick smoke test
    sample = {
        "sent_ceo": 0.72, "sent_cfo": 0.60, "sent_analyst": 0.30,
        "sent_ceo_delta": 0.08, "sent_cfo_delta": 0.05,
        "evasion_mean": 0.42, "evasion_max": 0.75, "evasive_count": 3,
        "revenue_qoq_pct": 12.5, "margin_qoq_pp": 1.2,
        "discrepancy_count": 1, "momentum_30d": 8.3, "volatility_30d": 32.1,
    }
    result = predict_with_shap(sample)
    print(json.dumps(result, indent=2))
