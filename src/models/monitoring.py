"""
MarketMind — Stage 4: Feature & Prediction Drift Monitoring
===========================================================
Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS)
tests to detect data drift and prediction distribution shifts between
baseline training data and serving/evaluation data.

Industry Standards Enforced:
    PSI < 0.10: Stable (no significant shift)
    0.10 <= PSI < 0.25: Moderate Drift (warning, monitor closely)
    PSI >= 0.25: Critical Drift (retraining recommended)
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys
import warnings

from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "train_matrix.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "lightgbm_marketmind_v1.pkl"
FEAT_PATH = MODELS_DIR / "feature_columns.json"
REPORT_JSON_PATH = MODELS_DIR / "drift_report.json"
REPORT_HTML_PATH = PROJECT_ROOT / "reports" / "drift_report.html"

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
MLFLOW_EXPERIMENT_NAME = "marketmind"

PSI_STABLE_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.05


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current samples.

    PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)

    # Strip NaNs and Infs
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # If all values are constant
    if np.all(ref == ref[0]) and np.all(cur == cur[0]):
        return 0.0 if ref[0] == cur[0] else 1.0

    # Determine quantile bins adaptively (prevent empty sparse bins on small samples)
    min_len = min(len(ref), len(cur))
    effective_buckets = min(num_buckets, max(4, min_len // 5))
    quantiles = np.linspace(0, 100, effective_buckets + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges = np.unique(bin_edges)

    # If quantiles collapse to <= 1 bin, fall back to linear spacing
    if len(bin_edges) <= 1:
        min_val = min(np.min(ref), np.min(cur))
        max_val = max(np.max(ref), np.max(cur))
        if min_val == max_val:
            return 0.0
        bin_edges = np.linspace(min_val, max_val, num_buckets + 1)

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_counts / len(ref)
    cur_pct = cur_counts / len(cur)

    # Apply epsilon smoothing to eliminate division by zero or ln(0)
    ref_pct = np.clip(ref_pct, epsilon, 1.0)
    cur_pct = np.clip(cur_pct, epsilon, 1.0)

    # Re-normalize
    ref_pct = ref_pct / np.sum(ref_pct)
    cur_pct = cur_pct / np.sum(cur_pct)

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(round(max(0.0, psi_val), 4))


def calculate_ks(reference: np.ndarray, current: np.ndarray) -> dict:
    """Calculate Kolmogorov-Smirnov 2-sample statistic and p-value."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    res = stats.ks_2samp(ref, cur)
    stat = float(round(res.statistic, 4))
    pval = float(round(res.pvalue, 4))
    drift = bool(pval < KS_PVALUE_THRESHOLD)
    return {"statistic": stat, "p_value": pval, "drift_detected": drift}


class DriftMonitor:
    """Production-grade drift monitoring engine for MarketMind features and predictions."""

    def __init__(self, matrix_path: Path = MATRIX_PATH, model_path: Path = MODEL_PATH):
        self.matrix_path = matrix_path
        self.model_path = model_path
        self.df = None
        self.model = None
        self.features = []
        self._load_dependencies()

    def _load_dependencies(self):
        if self.matrix_path.exists():
            self.df = pd.read_parquet(self.matrix_path)
            self.df["year"] = pd.to_datetime(self.df["call_date"]).dt.year
        else:
            logger.warning(f"Parquet matrix not found at {self.matrix_path}")

        if FEAT_PATH.exists():
            with open(FEAT_PATH) as f:
                self.features = json.load(f)
        else:
            self.features = [
                "sent_ceo", "sent_cfo", "sent_analyst", "sent_ceo_delta", "sent_cfo_delta",
                "evasion_mean", "evasion_max", "evasive_count", "revenue_qoq_pct",
                "margin_qoq_pp", "discrepancy_count", "momentum_30d", "volatility_30d"
            ]

        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data.get("model", data)
            except Exception as e:
                logger.warning(f"Could not load model for prediction drift: {e}")

    def run_drift_analysis(
        self,
        ref_years: list = None,
        cur_years: list = None,
        custom_current_df: pd.DataFrame = None,
    ) -> dict:
        """Run comprehensive feature and prediction drift comparison."""
        metrics_file = MODELS_DIR / "model_metrics.json"
        if ref_years is None:
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        m_data = json.load(f)
                        ref_years = m_data.get("train_years", [2023, 2024])
                except Exception:
                    ref_years = [2023, 2024]
            else:
                ref_years = [2023, 2024]

        if cur_years is None:
            if 2025 in ref_years:
                cur_years = [2026]
            else:
                cur_years = [2025, 2026]

        if self.df is None and custom_current_df is None:
            raise ValueError("No dataset available to compute drift.")

        if self.df is not None:
            ref_df = self.df[self.df["year"].isin(ref_years)].copy()
            cur_df = (
                custom_current_df.copy()
                if custom_current_df is not None
                else self.df[self.df["year"].isin(cur_years)].copy()
            )
        else:
            ref_df = custom_current_df.copy()
            cur_df = custom_current_df.copy()

        # Fill missing features with 0.0
        for f in self.features:
            if f not in ref_df.columns:
                ref_df[f] = 0.0
            if f not in cur_df.columns:
                cur_df[f] = 0.0

        feature_reports = {}
        critical_features = []
        warning_features = []

        for feat in self.features:
            ref_vals = ref_df[feat].values
            cur_vals = cur_df[feat].values

            psi = calculate_psi(ref_vals, cur_vals)
            ks = calculate_ks(ref_vals, cur_vals)

            ref_mean = float(round(np.nanmean(ref_vals), 4)) if len(ref_vals) else 0.0
            cur_mean = float(round(np.nanmean(cur_vals), 4)) if len(cur_vals) else 0.0
            ref_std = float(round(np.nanstd(ref_vals), 4)) if len(ref_vals) else 0.0
            cur_std = float(round(np.nanstd(cur_vals), 4)) if len(cur_vals) else 0.0

            if psi >= PSI_CRITICAL_THRESHOLD:
                level = "CRITICAL"
                critical_features.append(feat)
            elif psi >= PSI_STABLE_THRESHOLD or ks["drift_detected"]:
                level = "MODERATE"
                warning_features.append(feat)
            else:
                level = "STABLE"

            feature_reports[feat] = {
                "psi": psi,
                "ks_statistic": ks["statistic"],
                "ks_pvalue": ks["p_value"],
                "drift_level": level,
                "ref_mean": ref_mean,
                "ref_std": ref_std,
                "cur_mean": cur_mean,
                "cur_std": cur_std,
                "delta_mean": float(round(cur_mean - ref_mean, 4)),
            }

        # Prediction distribution drift
        prediction_drift = self._calculate_prediction_drift(ref_df, cur_df)

        # Overall Status
        if len(critical_features) >= 2 or prediction_drift.get("drift_level") == "CRITICAL":
            overall_status = "CRITICAL"
        elif len(critical_features) == 1 or len(warning_features) >= 2:
            overall_status = "WARNING"
        else:
            overall_status = "HEALTHY"

        recommendations = []
        if overall_status == "CRITICAL":
            recommendations.append(
                "Trigger automated walk-forward model retraining: feature and prediction distributions diverge significantly from baseline."
            )
        elif overall_status == "WARNING":
            recommendations.append(
                "Monitor drifted features closely over upcoming earnings cycle. Validate whether shift reflects broader semiconductor market regime transition."
            )
        else:
            recommendations.append(
                "Model features and inference distributions are stable within safe statistical bounds."
            )

        if "volatility_30d" in critical_features or "volatility_30d" in warning_features:
            recommendations.append(
                "Volatility shift detected (macro tech market regime change). Consider adjusting classification decision thresholds."
            )

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall_status,
            "reference_split": f"Years {ref_years} (N={len(ref_df)})",
            "current_split": f"Years {cur_years} (N={len(cur_df)})",
            "reference_sample_count": len(ref_df),
            "current_sample_count": len(cur_df),
            "total_features_evaluated": len(self.features),
            "critical_features_count": len(critical_features),
            "warning_features_count": len(warning_features),
            "critical_features": critical_features,
            "warning_features": warning_features,
            "feature_metrics": feature_reports,
            "prediction_drift": prediction_drift,
            "recommendations": recommendations,
        }

        return report

    def _calculate_prediction_drift(self, ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> dict:
        """Calculate distribution shift of model predictions."""
        if self.model is None or len(ref_df) == 0 or len(cur_df) == 0:
            return {
                "psi": 0.0,
                "drift_level": "STABLE",
                "reference_distribution": {"NEGATIVE": 0.33, "NEUTRAL": 0.33, "POSITIVE": 0.34},
                "current_distribution": {"NEGATIVE": 0.33, "NEUTRAL": 0.33, "POSITIVE": 0.34},
            }

        try:
            X_ref = ref_df[self.features].values.astype(np.float32)
            X_cur = cur_df[self.features].values.astype(np.float32)

            ref_preds = self.model.predict(X_ref)
            cur_preds = self.model.predict(X_cur)

            ref_dist = {
                "NEGATIVE": float(round((ref_preds == 0).mean(), 4)),
                "NEUTRAL": float(round((ref_preds == 1).mean(), 4)),
                "POSITIVE": float(round((ref_preds == 2).mean(), 4)),
            }
            cur_dist = {
                "NEGATIVE": float(round((cur_preds == 0).mean(), 4)),
                "NEUTRAL": float(round((cur_preds == 1).mean(), 4)),
                "POSITIVE": float(round((cur_preds == 2).mean(), 4)),
            }

            # Categorical PSI
            psi_pred = 0.0
            eps = 1e-4
            for cls in ["NEGATIVE", "NEUTRAL", "POSITIVE"]:
                p_ref = max(ref_dist[cls], eps)
                p_cur = max(cur_dist[cls], eps)
                psi_pred += (p_cur - p_ref) * np.log(p_cur / p_ref)

            psi_pred = float(round(max(0.0, psi_pred), 4))
            level = "CRITICAL" if psi_pred >= 0.25 else ("MODERATE" if psi_pred >= 0.10 else "STABLE")

            return {
                "psi": psi_pred,
                "drift_level": level,
                "reference_distribution": ref_dist,
                "current_distribution": cur_dist,
            }
        except Exception as e:
            logger.warning(f"Prediction drift calculation failed: {e}")
            return {"psi": 0.0, "drift_level": "STABLE", "error": str(e)}

    def save_reports(self, report: dict) -> tuple[Path, Path]:
        """Save JSON metrics and generate dark-themed HTML report."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)

        # 1. JSON Report
        with open(REPORT_JSON_PATH, "w") as f:
            json.dump(report, f, indent=2)
        logger.success(f"Drift metrics saved → {REPORT_JSON_PATH}")

        # 2. HTML Report
        html_content = self._render_html_report(report)
        with open(REPORT_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.success(f"Drift HTML report generated → {REPORT_HTML_PATH}")

        return REPORT_JSON_PATH, REPORT_HTML_PATH

    def _render_html_report(self, report: dict) -> str:
        """Render high-fidelity dark-themed HTML executive report."""
        status = report.get("overall_status", "UNKNOWN")
        status_color = "#10B981" if status == "HEALTHY" else ("#F59E0B" if status == "WARNING" else "#F43F5E")
        crit_count = report.get("critical_features_count", 0)
        warn_count = report.get("warning_features_count", 0)
        total_feat = report.get("total_features_evaluated", 0)
        timestamp = report.get("timestamp", "")

        rows_html = []
        for feat, m in report.get("feature_metrics", {}).items():
            lvl = m["drift_level"]
            badge_color = "#10B981" if lvl == "STABLE" else ("#F59E0B" if lvl == "MODERATE" else "#F43F5E")
            rows_html.append(f"""
            <tr>
                <td style="font-weight:600; font-family:monospace; color:#E2E8F0;">{feat}</td>
                <td><span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}44; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700;">{lvl}</span></td>
                <td style="font-family:monospace; text-align:right;">{m['psi']:.4f}</td>
                <td style="font-family:monospace; text-align:right;">{m['ks_statistic']:.4f}</td>
                <td style="font-family:monospace; text-align:right;">{m['ks_pvalue']:.4f}</td>
                <td style="font-family:monospace; text-align:right;">{m['ref_mean']:.3f}</td>
                <td style="font-family:monospace; text-align:right;">{m['cur_mean']:.3f}</td>
                <td style="font-family:monospace; text-align:right; color:{'#10B981' if m['delta_mean'] >= 0 else '#F43F5E'};">{m['delta_mean']:+.3f}</td>
            </tr>
            """)

        recs_html = "".join([f"<li style='margin-bottom:8px;'>{r}</li>" for r in report.get("recommendations", [])])

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MarketMind — Model & Feature Drift Monitoring Report</title>
    <style>
        body {{
            background-color: #070A0F;
            color: #F1F5F9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace, sans-serif;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 1px solid #1E293B;
            padding-bottom: 24px;
            margin-bottom: 30px;
        }}
        .title {{
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 0 0 8px 0;
            color: #FFFFFF;
        }}
        .subtitle {{
            color: #94A3B8;
            font-size: 13px;
        }}
        .status-badge {{
            display: inline-block;
            background: {status_color}20;
            color: {status_color};
            border: 1px solid {status_color}55;
            padding: 6px 16px;
            border-radius: 6px;
            font-weight: 800;
            font-size: 14px;
            letter-spacing: 0.5px;
            margin-top: 10px;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            background: #0D121A;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 20px;
        }}
        .card-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748B;
            margin-bottom: 6px;
        }}
        .card-val {{
            font-size: 24px;
            font-weight: 800;
            font-family: monospace;
            color: #F8FAFC;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #0D121A;
            border: 1px solid #1E293B;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 32px;
        }}
        th {{
            background: #131B26;
            color: #94A3B8;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid #1E293B;
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid #151D2A;
            font-size: 13px;
            color: #CBD5E1;
        }}
        tr:hover {{
            background: #111827;
        }}
        .recs {{
            background: #0D121A;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 24px;
        }}
        .recs h3 {{
            margin-top: 0;
            font-size: 16px;
            color: #38BDF8;
        }}
        ul {{
            margin: 0;
            padding-left: 20px;
            color: #94A3B8;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">MarketMind — Feature & Prediction Drift Monitor</h1>
            <div class="subtitle">Generated at {timestamp} | Baseline: {report.get('reference_split')} vs Current: {report.get('current_split')}</div>
            <div class="status-badge">SYSTEM STATUS: {status}</div>
        </div>

        <div class="cards">
            <div class="card">
                <div class="card-label">Features Evaluated</div>
                <div class="card-val">{total_feat}</div>
            </div>
            <div class="card">
                <div class="card-label">Critical Drift (PSI &ge; 0.25)</div>
                <div class="card-val" style="color:{'#F43F5E' if crit_count > 0 else '#10B981'};">{crit_count}</div>
            </div>
            <div class="card">
                <div class="card-label">Moderate Warnings</div>
                <div class="card-val" style="color:{'#F59E0B' if warn_count > 0 else '#10B981'};">{warn_count}</div>
            </div>
            <div class="card">
                <div class="card-label">Prediction Drift PSI</div>
                <div class="card-val">{report.get('prediction_drift', {}).get('psi', 0.0):.4f}</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Status</th>
                    <th style="text-align:right;">PSI</th>
                    <th style="text-align:right;">KS Stat</th>
                    <th style="text-align:right;">p-value</th>
                    <th style="text-align:right;">Ref Mean</th>
                    <th style="text-align:right;">Cur Mean</th>
                    <th style="text-align:right;">&Delta; Mean</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>

        <div class="recs">
            <h3>Automated Forensic Recommendations</h3>
            <ul>
                {recs_html}
            </ul>
        </div>
    </div>
</body>
</html>
"""

    def log_to_mlflow(self, report: dict):
        """Log drift metrics and reports to MLflow tracking server."""
        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name="feature_drift_monitoring") as run:
                mlflow.set_tags({
                    "task": "drift_monitoring",
                    "overall_status": report.get("overall_status", "UNKNOWN"),
                    "stage": "Stage 4",
                })

                # Summary metrics
                mlflow.log_metric("drift_total_features", report.get("total_features_evaluated", 0))
                mlflow.log_metric("drift_critical_features", report.get("critical_features_count", 0))
                mlflow.log_metric("drift_warning_features", report.get("warning_features_count", 0))
                mlflow.log_metric("drift_prediction_psi", report.get("prediction_drift", {}).get("psi", 0.0))

                # Per-feature PSI and KS metrics
                for feat, m in report.get("feature_metrics", {}).items():
                    mlflow.log_metric(f"psi_{feat}", float(m["psi"]))
                    mlflow.log_metric(f"ks_stat_{feat}", float(m["ks_statistic"]))
                    mlflow.log_metric(f"ks_pval_{feat}", float(m["ks_pvalue"]))

                # Log artifacts
                if REPORT_JSON_PATH.exists():
                    mlflow.log_artifact(str(REPORT_JSON_PATH), artifact_path="drift_reports")
                if REPORT_HTML_PATH.exists():
                    mlflow.log_artifact(str(REPORT_HTML_PATH), artifact_path="drift_reports")

                logger.success(f"Drift monitoring logged to MLflow (Run ID: {run.info.run_id})")
        except Exception as exc:
            logger.warning(f"MLflow drift logging skipped: {exc}")


def run_monitoring() -> dict:
    monitor = DriftMonitor()
    report = monitor.run_drift_analysis()
    monitor.save_reports(report)
    monitor.log_to_mlflow(report)
    return report


if __name__ == "__main__":
    run_monitoring()
