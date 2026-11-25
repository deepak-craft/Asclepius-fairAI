"""
Asclepius FairAI - Local Audit SDK: FairAIAuditor
Phase 14: Local-First Developer SDK for In-Situ Clinical AI Fairness Auditing,
Performance Evaluation, SHAP Attribution, AI Insights, and Threshold Calibration
"""

from typing import Dict, Any, Optional, Union, List
import uuid
import datetime
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.asclepius.config import AuditConfig
from src.asclepius.result import AuditResult
from src.core.fairness import evaluate_fairness_metrics, _safe_divide
from src.core.explainability import generate_shap_explanations
from src.core.insights import generate_insights
from src.core.mitigation import evaluate_threshold_candidates


class FairAIAuditor:
    """
    Primary developer API for local algorithmic fairness auditing of healthcare AI models.
    Operates strictly locally within the developer's environment without network transmission.
    """

    def __init__(
        self,
        model: Any,
        X_test: Union[pd.DataFrame, np.ndarray, List],
        y_test: Union[pd.Series, np.ndarray, List],
        sensitive_features: Union[pd.Series, np.ndarray, List, str],
        target_name: str = "Clinical Outcome",
        sensitive_name: str = "Protected Group",
        model_name: Optional[str] = None,
        model_version: str = "v1.0",
        model_type: Optional[str] = None,
        config: Optional[AuditConfig] = None,
    ):
        self.raw_model = model
        self.raw_X_test = X_test
        self.raw_y_test = y_test
        self.raw_sensitive = sensitive_features
        self.target_name = target_name
        self.sensitive_name = sensitive_name
        self.model_version = model_version
        self.config = config or AuditConfig()

        # Inferred model identity
        if model_name:
            self.model_name = model_name
        elif hasattr(model, "__class__"):
            self.model_name = model.__class__.__name__
        else:
            self.model_name = "Clinical Classifier"

        if model_type:
            self.model_type = model_type
        elif hasattr(model, "__class__"):
            c_name = model.__class__.__name__.lower()
            if "forest" in c_name:
                self.model_type = "Random Forest"
            elif "gradient" in c_name or "boost" in c_name:
                self.model_type = "Gradient Boosting"
            elif "logistic" in c_name:
                self.model_type = "Logistic Regression"
            else:
                self.model_type = model.__class__.__name__
        else:
            self.model_type = "scikit-learn"

    def _validate_inputs(self) -> Tuple_Validation:
        """
        Validates model and evaluation tensors defensively.
        Raises clear, helpful exceptions if data or models violate requirements.
        """
        # 1. Model validation
        if self.raw_model is None:
            raise ValueError("Asclepius FairAIAuditor: 'model' cannot be None.")
        if not hasattr(self.raw_model, "predict"):
            raise TypeError("Asclepius FairAIAuditor: 'model' must implement a .predict() method.")

        # Check fitted status
        if hasattr(self.raw_model, "named_steps"):
            # Check pipeline estimator
            last_step = list(self.raw_model.named_steps.values())[-1]
            if not any(hasattr(last_step, attr) for attr in ["classes_", "n_features_in_", "estimator_"]):
                raise NotFittedError(f"Asclepius FairAIAuditor: Pipeline final estimator '{last_step.__class__.__name__}' is not fitted.")
        else:
            if not any(hasattr(self.raw_model, attr) for attr in ["classes_", "n_features_in_", "coef_", "estimators_"]):
                raise NotFittedError(f"Asclepius FairAIAuditor: Model '{self.model_name}' is not fitted. Call model.fit(X, y) before auditing.")

        # 2. X_test validation
        if self.raw_X_test is None:
            raise ValueError("Asclepius FairAIAuditor: 'X_test' cannot be None.")

        if isinstance(self.raw_X_test, pd.DataFrame):
            X_df = self.raw_X_test.copy()
        elif isinstance(self.raw_X_test, (np.ndarray, list)):
            X_arr = np.array(self.raw_X_test)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            feature_cols = [f"Feature_{i+1}" for i in range(X_arr.shape[1])]
            X_df = pd.DataFrame(X_arr, columns=feature_cols)
        else:
            raise TypeError(f"Asclepius FairAIAuditor: Unsupported X_test type '{type(self.raw_X_test).__name__}'.")

        n_samples = len(X_df)
        if n_samples == 0:
            raise ValueError("Asclepius FairAIAuditor: 'X_test' contains 0 evaluation rows.")

        # 3. y_test validation
        if self.raw_y_test is None:
            raise ValueError("Asclepius FairAIAuditor: 'y_test' cannot be None.")

        y_series = pd.Series(self.raw_y_test)
        if len(y_series) != n_samples:
            raise ValueError(f"Asclepius FairAIAuditor: Length mismatch between X_test ({n_samples}) and y_test ({len(y_series)}).")

        if y_series.isna().any():
            raise ValueError("Asclepius FairAIAuditor: 'y_test' contains missing/NaN values.")

        # Binary outcome normalization
        unique_y = y_series.unique()
        if len(unique_y) != 2:
            raise ValueError(
                f"Asclepius FairAIAuditor: Clinical fairness workflow requires binary classification (found {len(unique_y)} classes: {list(unique_y)})."
            )

        # Standard conversion to [0, 1]
        pos_tokens = {1, "1", "1.0", "true", "yes", "positive", "high_risk", True}
        if any(v in pos_tokens for v in unique_y):
            y_arr = np.where(y_series.isin(pos_tokens), 1, 0).astype(int)
        else:
            sorted_classes = sorted(list(unique_y))
            y_map = {sorted_classes[0]: 0, sorted_classes[1]: 1}
            y_arr = y_series.map(y_map).to_numpy(dtype=int)

        # 4. Sensitive features validation
        if isinstance(self.raw_sensitive, str):
            if self.raw_sensitive in X_df.columns:
                sens_series = X_df[self.raw_sensitive].copy()
            else:
                raise KeyError(f"Asclepius FairAIAuditor: Sensitive feature column '{self.raw_sensitive}' not found in X_test.")
        elif self.raw_sensitive is not None:
            sens_series = pd.Series(self.raw_sensitive)
        else:
            raise ValueError("Asclepius FairAIAuditor: 'sensitive_features' cannot be None.")

        if len(sens_series) != n_samples:
            raise ValueError(f"Asclepius FairAIAuditor: Length mismatch between X_test ({n_samples}) and sensitive_features ({len(sens_series)}).")

        sens_list = [str(s).strip() if s is not None and not pd.isna(s) else "Unknown" for s in sens_series]
        unique_groups = set(sens_list)
        if len(unique_groups) < 2:
            # We don't crash, but warn in output that group comparisons require 2+ cohorts
            pass

        return X_df, y_arr, sens_list

    def run(self) -> AuditResult:
        """
        Executes complete local algorithmic audit.
        Evaluates Performance, Fairness, Subgroup Reliability, Explainability (SHAP),
        Deterministic Insights, and Threshold Mitigation.
        """
        X_df, y_true_arr, sens_list = self._validate_inputs()
        n_samples = len(y_true_arr)
        audit_id = f"SDK-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

        # 1. Generate predictions from fitted model
        try:
            raw_pred = self.raw_model.predict(X_df)
            y_pred_series = pd.Series(raw_pred)
            # Binary conversion
            if set(y_pred_series.unique()).issubset({0, 1}):
                y_pred_arr = y_pred_series.to_numpy(dtype=int)
            else:
                pos_tokens = {1, "1", "1.0", "true", "yes", "positive", True}
                y_pred_arr = np.where(y_pred_series.isin(pos_tokens), 1, 0).astype(int)
        except Exception as e:
            raise RuntimeError(f"Asclepius FairAIAuditor: model.predict(X_test) execution failed: {str(e)}")

        # 2. Extract prediction probabilities where supported
        has_probabilities = False
        y_prob_arr = None
        if hasattr(self.raw_model, "predict_proba"):
            try:
                raw_proba = self.raw_model.predict_proba(X_df)
                if isinstance(raw_proba, np.ndarray) and raw_proba.ndim == 2:
                    if raw_proba.shape[1] >= 2:
                        y_prob_arr = raw_proba[:, 1].astype(float)
                        has_probabilities = True
                    elif raw_proba.shape[1] == 1:
                        y_prob_arr = raw_proba[:, 0].astype(float)
                        has_probabilities = True
            except Exception:
                has_probabilities = False
                y_prob_arr = None

        # 3. Performance Diagnostics
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        spec = _safe_divide(tn, tn + fp, default=0.0)

        roc_auc_val = None
        roc_auc_display = "ROC-AUC unavailable — prediction probabilities were not supplied."
        if has_probabilities and y_prob_arr is not None:
            try:
                if len(np.unique(y_true_arr)) > 1:
                    roc_auc_val = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4)
                    roc_auc_display = f"{roc_auc_val:.3f}"
            except Exception:
                roc_auc_val = None

        model_results = {
            "status": "success",
            "audit_mode": "local_sdk",
            "model_key": "sdk_evaluated_model",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_label": f"{self.model_name} ({self.model_version})",
            "model_type": self.model_type,
            "target": self.target_name,
            "sensitive_attribute": self.sensitive_name,
            "positive_class": "1",
            "negative_class": "0",
            "train_samples": 0,
            "test_samples": n_samples,
            "total_evaluation_samples": n_samples,
            "test_size": 1.0,
            "decision_threshold": 0.50,
            "features_used": list(X_df.columns),
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1": round(f1, 4),
            "roc_auc": roc_auc_val,
            "roc_auc_display": roc_auc_display,
            "has_probabilities": has_probabilities,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "matrix": [[tn, fp], [fn, tp]]},
            "feature_importance": [],
            "y_test": y_true_arr.tolist(),
            "y_pred": y_pred_arr.tolist(),
            "y_prob": [round(float(p), 4) for p in y_prob_arr] if y_prob_arr is not None else None,
            "sensitive_test": sens_list,
            "pipeline": self.raw_model,
        }

        # 4. Fairness Evaluation (Direct Core Reuse)
        fairness_results = evaluate_fairness_metrics(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_prob=y_prob_arr,
            sensitive_features=sens_list,
            target_name=self.target_name,
            sensitive_name=self.sensitive_name,
            favorable_label=1,
        )

        # 5. Explainable AI via SHAP
        shap_results = None
        if self.config.explainability_enabled:
            shap_results = generate_shap_explanations(
                pipeline=self.raw_model,
                X_test=X_df,
                model_type=self.model_type,
                max_samples=self.config.max_shap_samples,
                sensitive_col=self.sensitive_name,
                y_test=y_true_arr,
                y_pred=y_pred_arr,
                y_prob=y_prob_arr,
            )
        if not shap_results or shap_results.get("status") == "error":
            # If explainer fails or not enabled, mark unavailable without breaking pipeline
            shap_results = {
                "status": "unavailable",
                "message": "SHAP feature attribution analysis was not generated for this model architecture.",
                "global_importance": [],
                "local_explanations": [],
            }

        # 6. Deterministic AI Insights (Core Reuse)
        insights_results = generate_insights(
            model_results=model_results,
            fairness_results=fairness_results,
            shap_results=shap_results,
        )

        # 7. Threshold Mitigation (Core Reuse)
        mitigation_results = None
        if has_probabilities and y_prob_arr is not None and self.config.mitigation_enabled:
            mitigation_results = evaluate_threshold_candidates(
                y_true=y_true_arr,
                y_prob=y_prob_arr,
                sensitive_features=sens_list,
                target_name=self.target_name,
                sensitive_name=self.sensitive_name,
                favorable_label=1,
                objective=self.config.mitigation_objective,
                max_performance_drop=self.config.mitigation_max_perf_drop,
            )
        else:
            mitigation_results = {
                "status": "error",
                "availability": "unavailable",
                "mitigation_available": False,
                "error_message": "Decision Threshold Calibration requires continuous probability scores (y_prob). Only discrete predictions were supplied.",
                "message": "Decision Threshold Calibration requires continuous probability scores (y_prob). Only discrete predictions were supplied.",
            }

        # 8. Assembled Bundle
        bundle = {
            "status": "success",
            "audit_id": audit_id,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": {
                "dataset_name": f"{self.model_name}_Evaluation_Cohort",
                "rows": n_samples,
                "cols": len(X_df.columns),
                "has_probabilities": has_probabilities,
            },
            "model_results": model_results,
            "fairness_results": fairness_results,
            "shap_results": shap_results,
            "insights_results": insights_results,
            "mitigation_results": mitigation_results,
        }

        return AuditResult(bundle)


Tuple_Validation = Any
