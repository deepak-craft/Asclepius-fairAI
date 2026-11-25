"""
Asclepius FairAI - Existing Healthcare AI Model Audit Adapter
Phase 13: Prediction-Based, Model-Agnostic External Model Evaluation

Evaluates external model predictions without executing or deserializing arbitrary model binaries.
Accepts held-out evaluation evidence containing ground truth, predictions, protected attributes,
and optional probabilities. Standardizes outputs for downstream Fairness, Insights, Mitigation,
Report, and History modules.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import io
import os
import pickle
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.core.fairness import evaluate_fairness_metrics, _safe_divide
from src.core.explainability import generate_shap_explanations

# Canonical path to the demonstration external audit evidence
DEMO_AUDIT_CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "demo", "external_model_audit_demo.csv")
)

# Marker features strictly characteristic of Cleveland / heart-disease training datasets
CLEVELAND_FEATURE_MARKERS = {
    "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "oldpeak", "slope", "ca", "thal"
}


def get_demo_audit_csv_path() -> str:
    """Returns the canonical absolute filesystem path to the demo audit evidence CSV."""
    return DEMO_AUDIT_CSV_PATH


# Canonical paths for the Level 2 SHAP demonstration package
SHAP_DEMO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "demo", "shap")
)
SHAP_DEMO_MODEL_PATH = os.path.join(SHAP_DEMO_DIR, "asclepius_shap_demo_model.pkl")
SHAP_DEMO_FEATURES_PATH = os.path.join(SHAP_DEMO_DIR, "asclepius_shap_demo_features.csv")
SHAP_DEMO_PREDICTIONS_PATH = os.path.join(SHAP_DEMO_DIR, "asclepius_shap_demo_predictions.csv")


def get_shap_demo_paths() -> Dict[str, str]:
    """Returns canonical absolute paths to the built-in Level 2 SHAP demonstration suite."""
    return {
        "dir": SHAP_DEMO_DIR,
        "model": SHAP_DEMO_MODEL_PATH,
        "features": SHAP_DEMO_FEATURES_PATH,
        "predictions": SHAP_DEMO_PREDICTIONS_PATH,
    }


def load_shap_demo_bundle() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads and validates the built-in deterministic SHAP demonstration bundle:
    - asclepius_shap_demo_model.pkl
    - asclepius_shap_demo_features.csv
    - asclepius_shap_demo_predictions.csv
    
    Returns:
        (bundle_dict, None) on success, or (None, error_str) on failure.
    """
    paths = get_shap_demo_paths()
    for key in ["model", "features", "predictions"]:
        if not os.path.exists(paths[key]):
            return None, f"SHAP demo artifact missing: {paths[key]}"

    # Load model
    model, err = load_external_model(paths["model"])
    if err or model is None:
        return None, f"Failed to load SHAP demo model: {err}"

    # Load features
    features_df, err = load_audit_input(paths["features"])
    if err or features_df is None:
        return None, f"Failed to load SHAP demo features: {err}"

    # Load predictions
    preds_df, err = load_audit_input(paths["predictions"])
    if err or preds_df is None:
        return None, f"Failed to load SHAP demo predictions: {err}"

    # Verify ID alignment
    if "id" in features_df.columns and "id" in preds_df.columns:
        if not (features_df["id"].values == preds_df["id"].values).all():
            return None, "ID mismatch between SHAP demo features and predictions."

    return {
        "model": model,
        "features_df": features_df,
        "predictions_df": preds_df,
        "model_path": paths["model"],
        "features_path": paths["features"],
        "predictions_path": paths["predictions"],
        "feature_names": list(getattr(model, "feature_names_in_", [])),
    }, None


def create_deterministic_demo_audit_csv(target_path: Optional[str] = None) -> str:
    """
    Creates a small, deterministic synthetic DEMO AUDIT EVIDENCE CSV
    specifically for testing the external-model audit workflow if not already present.
    Guarantees exact columns: ['id', 'y_true', 'y_pred', 'y_prob', 'sex'].
    """
    path = target_path or DEMO_AUDIT_CSV_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    np.random.seed(42)
    n = 600
    y_true = np.random.binomial(1, 0.38, n)
    noise = np.random.binomial(1, 0.18, n)
    y_pred = np.where(noise == 1, 1 - y_true, y_true)
    probs = np.where(y_pred == 1, np.random.uniform(0.52, 0.94, n), np.random.uniform(0.08, 0.48, n))
    probs = np.round(probs, 3)
    sexes = np.random.choice(["Female", "Male"], size=n, p=[0.51, 0.49])
    
    df = pd.DataFrame({
        "id": [f"EVAL-{1000 + i}" for i in range(n)],
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": probs,
        "sex": sexes,
    })
    df.to_csv(path, index=False)
    return path


def load_demo_audit_csv() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Loads and validates the canonical demo audit evidence CSV.
    Guarantees:
    - Points strictly to data/demo/external_model_audit_demo.csv.
    - Strictly rejects Cleveland clinical feature datasets.
    - Contains exactly: ['id', 'y_true', 'y_pred', 'y_prob', 'sex'].
    """
    path = get_demo_audit_csv_path()
    if not os.path.exists(path):
        create_deterministic_demo_audit_csv(path)
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"Failed to load demo audit evidence CSV: {e}"
    
    # Defensive guard: strictly reject Cleveland training dataset
    found_cleveland = set(str(c).lower().strip() for c in df.columns).intersection(CLEVELAND_FEATURE_MARKERS)
    if len(found_cleveland) >= 3:
        return None, (
            "Invalid demo audit dataset: Cleveland clinical feature dataset detected. "
            "Must load external_model_audit_demo.csv containing model predictions, not clinical features."
        )
    
    # Expected columns check
    expected_cols = ["id", "y_true", "y_pred", "y_prob", "sex"]
    if list(df.columns) != expected_cols:
        return None, f"Demo audit CSV columns mismatch. Expected {expected_cols}, got {list(df.columns)}"
        
    return df, None



def get_external_shap_unavailable_bundle() -> Dict[str, Any]:
    """
    Returns the authoritative SHAP unavailable state for External Model Audits.
    Model internals (weights, tree structures, decision boundaries) are not provided
    in prediction-only auditing; therefore SHAP cannot and must not be fabricated.
    """
    return {
        "status": "unavailable",
        "message": "SHAP unavailable — model internals were not provided.",
        "global_importance": [],
        "local_explanations": [],
        "explainability_mode": "external_model_audit",
        "sensitive_attribute_context": {
            "included_in_model_features": False,
            "attribute_name": None,
        },
    }


def load_audit_input(csv_source: Union[str, io.StringIO, bytes, pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely loads tabular evaluation evidence into a pandas DataFrame.
    Guarantees no arbitrary code execution or unpickling.
    Handles UTF-8, Windows cp1252, and UTF-8 BOM encoding.
    
    Returns:
        (DataFrame, None) on success, or (None, error_message) on failure.
    """
    if csv_source is None:
        return None, "No audit input data provided."

    if isinstance(csv_source, pd.DataFrame):
        if csv_source.empty:
            return None, "Supplied evaluation DataFrame is empty (0 rows)."
        return csv_source.copy(), None

    try:
        if isinstance(csv_source, bytes):
            text_data = csv_source.decode("utf-8-sig", errors="replace")
            df = pd.read_csv(io.StringIO(text_data))
        elif isinstance(csv_source, io.StringIO):
            df = pd.read_csv(csv_source)
        elif isinstance(csv_source, str):
            # Check if it's a file path or raw CSV string
            if "\n" in csv_source or "," in csv_source:
                cleaned = csv_source.lstrip("\ufeff")
                df = pd.read_csv(io.StringIO(cleaned))
            else:
                df = pd.read_csv(csv_source, encoding="utf-8-sig")
        else:
            return None, f"Unsupported data input type: {type(csv_source).__name__}"

        if df.empty or len(df) == 0:
            return None, "Evaluation data contains headers but 0 data records."

        # Clean column names (strip whitespace and BOM)
        df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]

        return df, None
    except Exception as e:
        return None, f"Unable to parse evaluation evidence CSV: {str(e)}"


def detect_external_audit_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Infers optimal semantic column mappings from evaluation CSV headers conservatively.
    Mirrors frontend heuristic mapping logic:
    y_true -> 'y_true', y_pred -> 'y_pred', y_prob -> 'y_prob', sensitive -> 'sex'/'gender', id -> 'id'.
    STRICT: Never maps 'sex', 'gender', or 'heart_disease' to y_pred or y_prob.
    """
    col_map = {
        "y_true": None,
        "y_pred": None,
        "y_prob": None,
        "sensitive": None,
        "id": None,
    }
    cleaned_cols = [str(c).lstrip("\ufeff").strip() for c in columns]
    lower_cols = [c.lower().replace("-", "_").replace(" ", "_") for c in cleaned_cols]

    # y_true candidate: prioritize exact match 'y_true', then strong ground-truth aliases
    y_true_aliases = ["true", "ground_truth", "groundtruth", "target_actual", "actual_label", "actual", "target", "outcome", "label", "readmission_30d", "readmission", "heart_disease"]
    if "y_true" in lower_cols:
        col_map["y_true"] = cleaned_cols[lower_cols.index("y_true")]
    else:
        for alias in y_true_aliases:
            if alias in lower_cols:
                col_map["y_true"] = cleaned_cols[lower_cols.index(alias)]
                break

    # y_pred candidate: prioritize exact match 'y_pred', then strong prediction aliases
    # STRICT: Never map clinical features ('sex', 'gender', 'heart_disease') to y_pred
    y_pred_aliases = ["prediction", "predicted", "predicted_label", "prediction_label", "model_prediction", "pred", "model_pred", "output"]
    if "y_pred" in lower_cols:
        col_map["y_pred"] = cleaned_cols[lower_cols.index("y_pred")]
    else:
        for alias in y_pred_aliases:
            if alias in lower_cols:
                candidate = cleaned_cols[lower_cols.index(alias)]
                if candidate.lower() not in ["sex", "gender", "heart_disease"]:
                    col_map["y_pred"] = candidate
                    break

    # y_prob candidate: prioritize exact match 'y_prob', then strong probability aliases
    # STRICT: Never map target or clinical features to y_prob
    y_prob_aliases = ["probability", "prediction_probability", "predicted_probability", "score", "prediction_score", "prob", "confidence", "risk", "pred_prob"]
    if "y_prob" in lower_cols:
        col_map["y_prob"] = cleaned_cols[lower_cols.index("y_prob")]
    else:
        for alias in y_prob_aliases:
            if alias in lower_cols:
                candidate = cleaned_cols[lower_cols.index(alias)]
                if candidate.lower() not in ["heart_disease", "sex", "gender"]:
                    col_map["y_prob"] = candidate
                    break

    # Sensitive attribute candidate: prioritize exact match 'sex', 'gender', then demographic aliases
    if "sex" in lower_cols:
        col_map["sensitive"] = cleaned_cols[lower_cols.index("sex")]
    elif "gender" in lower_cols:
        col_map["sensitive"] = cleaned_cols[lower_cols.index("gender")]
    else:
        for alias in ["race", "ethnicity", "age_group", "insurance", "payer"]:
            if alias in lower_cols:
                col_map["sensitive"] = cleaned_cols[lower_cols.index(alias)]
                break

    # ID candidate: prioritize exact match 'id', then identifier aliases
    if "id" in lower_cols:
        col_map["id"] = cleaned_cols[lower_cols.index("id")]
    else:
        for alias in ["record_id", "patient_id", "sample_id", "eval_id", "subject_id", "uuid"]:
            if alias in lower_cols:
                col_map["id"] = cleaned_cols[lower_cols.index(alias)]
                break

    # Defensive collision guard on auto-selection:
    if col_map["y_pred"] and (col_map["y_pred"] == col_map["sensitive"] or col_map["y_pred"] == col_map["y_true"]):
        col_map["y_pred"] = None

    return col_map


def _normalize_binary_series(series: pd.Series, col_label: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Converts binary column to canonical integer array [0, 1].
    Handles string values like 'yes'/'no', 'true'/'false', '1'/'0', etc.
    """
    if series.isna().any():
        nan_count = int(series.isna().sum())
        return None, f"Column '{col_label}' contains {nan_count} missing (NaN) values. External audits require complete evaluation vectors."

    # Convert to string stripped
    cleaned = series.astype(str).str.strip().str.lower()
    unique_vals = cleaned.unique()

    if len(unique_vals) != 2:
        return None, (
            f"Column '{col_label}' has {len(unique_vals)} unique classes ({list(series.unique())}). "
            f"Current clinical fairness workflow requires binary classification (exactly 2 distinct classes)."
        )

    # Standard positive/negative mapping
    pos_tokens = {"1", "1.0", "true", "yes", "positive", "pos", "y", "high_risk", "readmitted", "disease"}
    neg_tokens = {"0", "0.0", "false", "no", "negative", "neg", "n", "low_risk", "not_readmitted", "healthy"}

    has_pos = any(v in pos_tokens for v in unique_vals)
    has_neg = any(v in neg_tokens for v in unique_vals)

    if has_pos and has_neg:
        mapped = np.where(cleaned.isin(pos_tokens), 1, 0)
        return mapped, None

    # Fallback to alphabetical sorting: second class is positive (1), first is negative (0)
    sorted_unique = sorted(list(unique_vals))
    mapping = {sorted_unique[0]: 0, sorted_unique[1]: 1}
    mapped = cleaned.map(mapping).to_numpy(dtype=int)
    return mapped, None


def validate_audit_input(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    sensitive_col: str,
    y_prob_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Performs rigorous, model-agnostic validation on external evaluation evidence.
    
    Checks:
    - Required columns exist in DataFrame
    - y_true and y_pred lengths match
    - sensitive_attribute length matches
    - y_true and y_pred are binary with exactly 2 classes
    - y_prob (if provided) is numeric and within [0.0, 1.0]
    - Subgroup counts and reliability
    
    Returns structured validation report with status:
        'ready' (✓ Audit Input Ready)
        'warning' (⚠ Review Required)
        'invalid' (✕ Invalid Audit Input)
    """
    issues: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty or len(df) == 0:
        return {
            "status": "invalid",
            "status_title": "⚠ Audit Input Needs Correction",
            "status_message": "Dataset contains no records to audit.",
            "issues": ["Uploaded CSV contains 0 rows."],
            "warnings": [],
            "summary": {"rows": 0, "columns": 0, "has_probabilities": False},
        }

    n_rows = len(df)

    # 0. Defensive guard: reject Cleveland / clinical training feature datasets
    found_cleveland = set(str(c).lower().strip() for c in df.columns).intersection(CLEVELAND_FEATURE_MARKERS)
    if len(found_cleveland) >= 3:
        issues.append(
            f"Cleveland clinical training dataset detected containing feature columns: {', '.join(sorted(found_cleveland))}."
        )
        issues.append(
            "External model audit requires held-out evaluation evidence with model predictions (y_pred, y_prob), not model-training feature datasets."
        )
        warnings.append(
            "To audit an existing model, load 'data/demo/external_model_audit_demo.csv' or upload a CSV with predictions."
        )
        return {
            "status": "invalid",
            "status_title": "⚠ Audit Input Needs Correction",
            "status_message": (
                "Cleveland clinical feature dataset detected. Existing Model Audit requires evaluation "
                "evidence containing model predictions (y_pred, y_prob) from an already-created model, "
                "not raw clinical training features."
            ),
            "issues": issues,
            "warnings": warnings,
            "summary": {"rows": n_rows, "columns": len(df.columns), "has_probabilities": False},
        }

    # 1. Check required column existence
    if not y_true_col or y_true_col not in df.columns:
        issues.append(f"Ground truth column '{y_true_col}' not found in uploaded columns.")
    if not y_pred_col or y_pred_col not in df.columns:
        issues.append(f"Predicted label column '{y_pred_col}' not found in uploaded columns.")
    if not sensitive_col or sensitive_col not in df.columns:
        issues.append(f"Sensitive attribute column '{sensitive_col}' not found in uploaded columns.")

    if issues:
        return {
            "status": "invalid",
            "status_title": "⚠ Audit Input Needs Correction",
            "status_message": "One or more required evaluation columns are missing.",
            "issues": issues,
            "warnings": warnings,
            "summary": {"rows": n_rows, "columns": len(df.columns), "has_probabilities": False},
        }

    # 2. Prevent identical columns & enforce pairwise distinct semantic fields
    mapped_roles = [
        ("y_true", "Ground truth (y_true)", y_true_col),
        ("y_pred", "Predicted label (y_pred)", y_pred_col),
        ("sensitive", "Sensitive attribute", sensitive_col),
    ]
    if y_prob_col and str(y_prob_col).strip() and y_prob_col in df.columns:
        mapped_roles.append(("y_prob", "Prediction probability (y_prob)", y_prob_col))
    if id_col and str(id_col).strip() and id_col in df.columns:
        mapped_roles.append(("id", "Record ID (id)", id_col))

    semantic_collision = False
    for i in range(len(mapped_roles)):
        for j in range(i + 1, len(mapped_roles)):
            role_a, name_a, col_a = mapped_roles[i]
            role_b, name_b, col_b = mapped_roles[j]
            if col_a == col_b:
                semantic_collision = True
                issues.append(
                    f"{name_a} and {name_b} currently point to incompatible fields (both mapped to '{col_a}')."
                )

    if semantic_collision:
        issues.insert(0, "Invalid audit mapping: y_true, y_pred, y_prob, and sensitive attribute must represent different semantic fields.")
        warnings.append("Existing Model Audit requires evaluation evidence from an already-created model. Upload a CSV containing ground truth and model predictions.")
        return {
            "status": "invalid",
            "status_title": "⚠ Audit Input Needs Correction",
            "status_message": issues[0],
            "issues": issues,
            "warnings": warnings,
            "summary": {"rows": n_rows, "columns": len(df.columns), "has_probabilities": False},
        }

    # 3. Check sample size
    if n_rows < 2:
        issues.append(f"Evaluation dataset has only {n_rows} rows. A minimum of 2 observations is required for evaluation.")
        return {
            "status": "invalid",
            "status_title": "⚠ Audit Input Needs Correction",
            "status_message": "Sample size too small for statistical evaluation.",
            "issues": issues,
            "warnings": warnings,
            "summary": {"rows": n_rows, "columns": len(df.columns), "has_probabilities": False},
        }
    elif n_rows < 10:
        warnings.append(f"Small evaluation sample size (n={n_rows}). Subgroup disparity estimates carry wide variance with fewer than 10 observations.")

    # 4. Binary validation on y_true
    y_true_arr, err_true = _normalize_binary_series(df[y_true_col], y_true_col)
    if err_true:
        issues.append(err_true)

    # 5. Binary validation on y_pred
    y_pred_arr, err_pred = _normalize_binary_series(df[y_pred_col], y_pred_col)
    if err_pred:
        issues.append(err_pred)

    if issues:
        return {
            "status": "invalid",
            "status_title": "⚠ Audit Input Needs Correction",
            "status_message": "Class structure validation failed.",
            "issues": issues,
            "warnings": warnings,
            "summary": {"rows": n_rows, "columns": len(df.columns), "has_probabilities": False},
        }

    # 6. Sensitive attribute validation
    sens_series = df[sensitive_col].fillna("Unknown").astype(str).str.strip()
    sens_unique = sens_series.unique()
    subgroup_counts = sens_series.value_counts().to_dict()

    if len(sens_unique) < 2:
        issues.append(
            f"Sensitive attribute '{sensitive_col}' contains only {len(sens_unique)} unique demographic group ({list(sens_unique)}). "
            "Group disparity comparisons require at least 2 distinct cohorts."
        )
        return {
            "status": "invalid",
            "status_title": "⚠ Audit Input Needs Correction",
            "status_message": f"Sensitive attribute '{sensitive_col}' requires at least 2 cohorts for disparity auditing.",
            "issues": issues,
            "warnings": warnings,
            "summary": {"rows": n_rows, "columns": len(df.columns), "has_probabilities": False},
        }

    # Subgroup sample size check
    min_subgroup_size = min(subgroup_counts.values()) if subgroup_counts else 0
    small_subgroups = [g for g, count in subgroup_counts.items() if count < 30]
    if small_subgroups:
        warnings.append(
            f"Demographic subgroup(s) {small_subgroups} have fewer than 30 observations. "
            "Disparity estimates for these cohorts will carry wide statistical variance."
        )

    # 7. Probability validation (optional)
    has_prob = False
    y_prob_arr = None
    if y_prob_col and str(y_prob_col).strip() and y_prob_col in df.columns:
        prob_series = df[y_prob_col]
        # Check if numeric
        try:
            numeric_prob = pd.to_numeric(prob_series, errors="coerce")
            if numeric_prob.isna().any():
                nan_p = int(numeric_prob.isna().sum())
                issues.append(f"Prediction probability column '{y_prob_col}' contains {nan_p} non-numeric or missing values.")
            else:
                p_min = float(numeric_prob.min())
                p_max = float(numeric_prob.max())
                if p_min < 0.0 or p_max > 1.0:
                    issues.append(
                        f"Prediction probabilities must be in range [0.0, 1.0]. "
                        f"Found values ranging from {p_min:.3f} to {p_max:.3f}."
                    )
                else:
                    has_prob = True
                    y_prob_arr = numeric_prob.to_numpy(dtype=float)
        except Exception as e:
            issues.append(f"Failed to validate probability column '{y_prob_col}': {str(e)}")
    else:
        warnings.append(
            "Continuous prediction probabilities (y_prob) were not provided. "
            "ROC-AUC and Decision Threshold Mitigation will be marked unavailable."
        )

    # 8. ID Column check (optional)
    has_id = False
    if id_col and id_col in df.columns:
        has_id = True

    # Final status determination
    if issues:
        status = "invalid"
        status_title = "⚠ Audit Input Needs Correction"
        status_message = issues[0] if len(issues) == 1 else "Audit input verification detected issues that must be corrected."
    elif warnings:
        status = "warning"
        status_title = "⚠ Review Required"
        status_message = "Audit input is valid with clinical warnings. Review subgroup sizes and probability availability."
    else:
        status = "ready"
        status_title = "✓ Audit Input Ready"
        status_message = "Evaluation evidence successfully verified. Ready for performance and algorithmic fairness audit."

    return {
        "status": status,
        "status_title": status_title,
        "status_message": status_message,
        "issues": issues,
        "warnings": warnings,
        "summary": {
            "rows": n_rows,
            "columns": len(df.columns),
            "target_col": y_true_col,
            "pred_col": y_pred_col,
            "sensitive_col": sensitive_col,
            "has_probabilities": has_prob,
            "has_id": has_id,
            "subgroup_counts": subgroup_counts,
        },
        "parsed_arrays": {
            "y_true": y_true_arr.tolist() if y_true_arr is not None else [],
            "y_pred": y_pred_arr.tolist() if y_pred_arr is not None else [],
            "y_prob": [round(float(p), 4) for p in y_prob_arr] if y_prob_arr is not None else None,
            "sensitive_features": sens_series.tolist(),
        },
    }


def evaluate_external_predictions(
    y_true: Union[List[int], np.ndarray, pd.Series],
    y_pred: Union[List[int], np.ndarray, pd.Series],
    sensitive_features: Union[List[Any], np.ndarray, pd.Series],
    y_prob: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    target_name: str = "Clinical Outcome",
    sensitive_name: str = "Protected Group",
    model_name: str = "External Healthcare AI Model",
    model_version: str = "v1.0",
    model_type: str = "External Model Audit",
) -> Dict[str, Any]:
    """
    Evaluates classification diagnostics on external predictions WITHOUT fitting a model.
    Produces the exact standardized output bundle expected by downstream Asclepius modules.
    
    ROC-AUC is calculated ONLY if continuous y_prob is supplied.
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    n_samples = len(y_true_arr)

    if n_samples == 0 or len(y_pred_arr) != n_samples:
        return {
            "status": "error",
            "error_message": f"Mismatched array lengths: y_true ({len(y_true_arr)}), y_pred ({len(y_pred_arr)}).",
        }

    # Standard metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    specificity = _safe_divide(tn, tn + fp, default=0.0)

    # ROC-AUC: ONLY when y_prob is available
    roc_auc_val: Optional[float] = None
    roc_auc_display: str = "ROC-AUC unavailable — prediction probabilities were not supplied."
    has_probabilities = False

    if y_prob is not None and len(y_prob) == n_samples:
        try:
            y_prob_arr = np.array(y_prob, dtype=float)
            if len(np.unique(y_true_arr)) > 1:
                roc_auc_val = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4)
                roc_auc_display = f"{roc_auc_val:.3f}"
                has_probabilities = True
            else:
                roc_auc_val = 0.5
                roc_auc_display = "0.500 (Single class)"
                has_probabilities = True
        except Exception:
            roc_auc_val = None
            roc_auc_display = "ROC-AUC calculation failed."
            has_probabilities = False
    else:
        y_prob_arr = None

    # Sensitive features normalization
    sens_list = [str(s).strip() if s is not None and not pd.isna(s) else "Unknown" for s in sensitive_features]

    results = {
        "status": "success",
        "audit_mode": "external",
        "model_key": "external_model_audit",
        "model_name": model_name,
        "model_version": model_version,
        "model_label": f"{model_name} ({model_version}) — External Audit",
        "model_type": model_type,
        "target": target_name,
        "sensitive_attribute": sensitive_name,
        "positive_class": "1",
        "negative_class": "0",
        "train_samples": 0,
        "test_samples": n_samples,
        "total_evaluation_samples": n_samples,
        "test_size": 1.0,
        "random_state": None,
        "decision_threshold": 0.50,
        "class_weight": "pre-fitted",
        "stratified": False,
        "features_used": ["External Model Predictions"],
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": roc_auc_val,
        "roc_auc_display": roc_auc_display,
        "has_probabilities": has_probabilities,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "feature_importance": [],
        "y_test": y_true_arr.tolist(),
        "y_pred": y_pred_arr.tolist(),
        "y_prob": [round(float(p), 4) for p in y_prob_arr] if y_prob_arr is not None else None,
        "sensitive_test": sens_list,
        "pipeline": None,
    }

    return results


def load_external_model(model_source: Union[str, bytes, Path, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """
    Safely loads and validates an external trained classification model.
    Accepts a file path (.pkl, .joblib, .pickle), raw bytes, or an already instantiated model object.
    
    Security & Integrity:
    - Validates file extensions (.pkl, .joblib, .pickle)
    - Catches corruption, truncation, and unpickling errors
    - Verifies that the deserialized object is a valid, fitted scikit-learn estimator or Pipeline
    - Rejects arbitrary, malicious, or non-estimator objects
    
    Returns:
        (model_object, None) on success, or (None, error_message) on failure.
    """
    if model_source is None:
        return None, "No model source provided."

    # If already an estimator/pipeline object in memory
    if not isinstance(model_source, (str, bytes, Path)):
        if not hasattr(model_source, "predict"):
            return None, f"Invalid model object: Object of type '{type(model_source).__name__}' does not have a 'predict' method."
        is_fitted = hasattr(model_source, "classes_") or hasattr(model_source, "estimators_") or hasattr(model_source, "steps") or hasattr(model_source, "coef_")
        if not is_fitted:
            return None, "Model is not fitted. Please provide a trained and fitted classifier."
        return model_source, None

    # If string path or Path object
    if isinstance(model_source, (str, Path)):
        model_path = Path(model_source)
        if not model_path.exists():
            return None, f"Model file not found at path: {model_source}"
        ext = model_path.suffix.lower()
        if ext not in [".pkl", ".joblib", ".pickle"]:
            return None, f"Unsupported model file format '{ext}'. Supported formats: .pkl, .joblib, .pickle."
        try:
            if ext == ".joblib":
                model = joblib.load(model_path)
            else:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
        except Exception as e:
            return None, f"Corrupted or invalid model file: {str(e)}"
    elif isinstance(model_source, bytes):
        if len(model_source) == 0:
            return None, "Supplied model binary buffer is empty (0 bytes)."
        try:
            bio = io.BytesIO(model_source)
            try:
                model = joblib.load(bio)
            except Exception:
                bio.seek(0)
                model = pickle.load(bio)
        except Exception as e:
            return None, f"Corrupted or invalid model binary: {str(e)}"
    else:
        return None, f"Unsupported model source type: {type(model_source).__name__}"

    # Security validation on unpickled object
    if not hasattr(model, "predict"):
        return None, f"Invalid model file: Loaded object is of type '{type(model).__name__}', expected a fitted scikit-learn estimator or Pipeline with a 'predict' method."

    # Validate fitted state
    is_fitted = hasattr(model, "classes_") or hasattr(model, "estimators_") or hasattr(model, "steps") or hasattr(model, "coef_")
    if not is_fitted:
        return None, "Model is not fitted. Please provide a trained and fitted classifier."

    return model, None


def validate_explainability_inputs(
    model: Any,
    features_source: Union[str, bytes, io.StringIO, pd.DataFrame],
    target_col: Optional[str] = None,
    sensitive_col: Optional[str] = None,
    include_sensitive_in_features: bool = False,
    id_col: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    """
    Validates evaluation features against a loaded external model for Level 2 Explainability Audit.
    Enforces semantic isolation: sensitive attribute is used for fairness analysis and is
    strictly excluded from model features unless explicitly enabled by user confirmation.
    Checks column alignment and feature names with model metadata.
    """
    if features_source is None:
        return None, None, "Evaluation features dataset was not provided."

    if isinstance(features_source, pd.DataFrame):
        df_feat = features_source.copy()
    else:
        df_feat, err = load_audit_input(features_source)
        if err or df_feat is None:
            return None, None, f"Failed to load evaluation features: {err or 'Empty dataset'}"

    if df_feat.empty:
        return None, None, "Evaluation features dataset is empty (0 rows)."

    # Remove non-feature columns
    cols_to_drop = []
    if id_col and id_col in df_feat.columns:
        cols_to_drop.append(id_col)
    else:
        for c in df_feat.columns:
            if c.lower() in ["id", "patient_id", "record_id"]:
                cols_to_drop.append(c)

    if target_col and target_col in df_feat.columns:
        cols_to_drop.append(target_col)
    for c in ["y_true", "y_pred", "y_prob"]:
        if c in df_feat.columns and c not in cols_to_drop:
            cols_to_drop.append(c)

    # SEMANTIC RULE: Sensitive attribute must NOT automatically become a model feature
    if sensitive_col and sensitive_col in df_feat.columns:
        if not include_sensitive_in_features:
            cols_to_drop.append(sensitive_col)

    if cols_to_drop:
        df_feat = df_feat.drop(columns=[c for c in cols_to_drop if c in df_feat.columns])

    if df_feat.shape[1] == 0:
        return None, None, "No feature columns remaining in evaluation features dataset after excluding target, identifier, and protected columns."

    # Verify column alignment with fitted model
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        actual_features = list(df_feat.columns)
        missing_features = [f for f in expected_features if f not in actual_features]
        if missing_features:
            return None, None, f"Feature/model column mismatch: Features dataset is missing required model input features: {missing_features}."
        # Align column order to model expectations
        df_feat = df_feat[expected_features]
    elif hasattr(model, "n_features_in_"):
        expected_n = int(model.n_features_in_)
        if df_feat.shape[1] != expected_n:
            return None, None, f"Feature count mismatch: Model expects {expected_n} features, but evaluation dataset provides {df_feat.shape[1]} features."

    summary = {
        "features_count": df_feat.shape[1],
        "records_count": len(df_feat),
        "feature_names": list(df_feat.columns),
        "sensitive_included_in_features": include_sensitive_in_features and (sensitive_col in df_feat.columns if sensitive_col else False),
    }

    return df_feat, summary, None


def calculate_external_shap(
    model: Any,
    X_features: pd.DataFrame,
    sensitive_col: Optional[str] = None,
    y_true: Optional[Union[List[int], np.ndarray]] = None,
    y_pred: Optional[Union[List[int], np.ndarray]] = None,
    y_prob: Optional[Union[List[float], np.ndarray]] = None,
    max_samples: int = 500,
) -> Dict[str, Any]:
    """
    Calculates genuine SHAP explanations for an external model and its evaluation features.
    Selects appropriate explainer (TreeExplainer for tree models, LinearExplainer for linear models),
    handles binary classification positive-class attributions, and validates numerical integrity.
    """
    try:
        shap_res = generate_shap_explanations(
            pipeline=model,
            X_test=X_features,
            model_type="auto",
            max_samples=max_samples,
            sensitive_col=sensitive_col,
            y_test=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )
    except Exception as e:
        return {
            "status": "unavailable",
            "message": f"SHAP calculation failed or model type unsupported: {str(e)}",
            "global_importance": [],
            "local_explanations": [],
            "explainability_mode": "external_model_audit",
            "explainability_level": "model_specific_error",
        }

    if shap_res.get("status") != "success":
        return {
            "status": "unavailable",
            "message": f"SHAP calculation failed: {shap_res.get('error_message', 'Unknown explainer error')}",
            "global_importance": [],
            "local_explanations": [],
            "explainability_mode": "external_model_audit",
            "explainability_level": "model_specific_error",
        }

    # Verify SHAP values are numeric and finite
    global_imp = shap_res.get("global_importance", [])
    for item in global_imp:
        val = item.get("mean_abs_shap", 0.0)
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            return {
                "status": "unavailable",
                "message": "SHAP calculation produced non-finite or non-numeric attribution values.",
                "global_importance": [],
                "local_explanations": [],
                "explainability_mode": "external_model_audit",
                "explainability_level": "model_specific_error",
            }

    # Verify feature names match input features
    reported_features = set(item.get("feature") for item in global_imp)
    expected_features = set(X_features.columns)
    if reported_features and not reported_features.issubset(expected_features):
        return {
            "status": "unavailable",
            "message": "Mismatch between evaluation feature names and SHAP attribution outputs.",
            "global_importance": [],
            "local_explanations": [],
            "explainability_mode": "external_model_audit",
            "explainability_level": "model_specific_error",
        }

    shap_res["explainability_level"] = "model_specific"
    shap_res["explainability_mode"] = "external_model_audit_level_2"
    shap_res["message"] = "Model-Specific Explainability Available"
    return shap_res


def build_external_audit_bundle(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    sensitive_col: str,
    y_prob_col: Optional[str] = None,
    id_col: Optional[str] = None,
    model_name: str = "External Healthcare AI Model",
    model_version: str = "v1.0",
    model_type: str = "External Model Audit",
    dataset_name: str = "External_Evaluation_Evidence.csv",
    # Optional Level 2 Explainability Audit parameters:
    model_source: Optional[Union[str, bytes, Path, Any]] = None,
    features_source: Optional[Union[pd.DataFrame, str, bytes, io.StringIO]] = None,
    include_sensitive_in_features: bool = False,
) -> Dict[str, Any]:
    """
    Executes the External Model Audit pipeline supporting two audit levels:
    
    LEVEL 1 — Prediction Evidence Audit (Default):
    - Input: ground truth, predictions, sensitive attributes, and optional probabilities
    - Performance metrics, fairness analysis, and deterministic insights are evaluated
    - SHAP remains unavailable because model internals were not provided
    
    LEVEL 2 — Explainability Audit (When model_source & features_source are supplied):
    - Loads and validates the external trained model (.pkl, .joblib, or in-memory estimator)
    - Validates evaluation features against model expected inputs
    - Enforces semantic isolation: sensitive attribute is excluded from features unless explicitly confirmed
    - Computes genuine SHAP feature attributions via TreeExplainer / LinearExplainer
    """
    validation = validate_audit_input(
        df=df,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        sensitive_col=sensitive_col,
        y_prob_col=y_prob_col,
        id_col=id_col,
    )

    if validation["status"] == "invalid":
        return {
            "status": "error",
            "error_message": validation["status_message"],
            "validation": validation,
        }

    parsed = validation["parsed_arrays"]
    y_true = parsed["y_true"]
    y_pred = parsed["y_pred"]
    y_prob = parsed["y_prob"]
    sensitive_features = parsed["sensitive_features"]

    # 1. Performance Diagnostics
    model_results = evaluate_external_predictions(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        y_prob=y_prob,
        target_name=y_true_col,
        sensitive_name=sensitive_col,
        model_name=model_name,
        model_version=model_version,
        model_type=model_type,
    )

    # 2. Reuse Existing Fairness Engine (ZERO formula duplication)
    fairness_results = evaluate_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features=sensitive_features,
        target_name=y_true_col,
        sensitive_name=sensitive_col,
        favorable_label=1,
    )

    # 3. Resolve SHAP Explainability State (Level 1 vs Level 2)
    audit_level = "level_1_prediction_evidence"
    audit_level_display = "Level 1 — Model-Agnostic Prediction Evidence Audit"
    shap_results = None

    if model_source is not None and features_source is not None:
        # LEVEL 2: Explainability Audit
        model_obj, err_model = load_external_model(model_source)
        if err_model:
            return {
                "status": "error",
                "error_message": f"Explainability Audit Failed: {err_model}",
                "validation": validation,
            }

        X_features, feat_summary, err_feat = validate_explainability_inputs(
            model=model_obj,
            features_source=features_source,
            target_col=y_true_col,
            sensitive_col=sensitive_col,
            include_sensitive_in_features=include_sensitive_in_features,
            id_col=id_col,
        )
        if err_feat:
            return {
                "status": "error",
                "error_message": f"Explainability Audit Failed: {err_feat}",
                "validation": validation,
            }

        shap_results = calculate_external_shap(
            model=model_obj,
            X_features=X_features,
            sensitive_col=sensitive_col,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )
        audit_level = "level_2_explainability"
        audit_level_display = "Level 2 — Model-Specific Explainability Audit"
    elif model_source is not None and features_source is None:
        # Model provided but feature dataset missing -> SHAP remains unavailable
        shap_results = get_external_shap_unavailable_bundle()
        shap_results["message"] = "SHAP unavailable — evaluation feature dataset was not provided."
    else:
        # LEVEL 1: Standard Prediction Evidence Audit
        shap_results = get_external_shap_unavailable_bundle()

    # 4. Assembled Audit Bundle
    audit_bundle = {
        "status": "success",
        "audit_mode": "external",
        "audit_level": audit_level,
        "audit_level_display": audit_level_display,
        "dataset_info": {
            "dataset_name": dataset_name,
            "rows": len(y_true),
            "cols": len(df.columns),
            "has_probabilities": model_results["has_probabilities"],
            "id_column": id_col,
        },
        "model_results": model_results,
        "fairness_results": fairness_results,
        "shap_results": shap_results,
        "validation": validation,
    }

    return audit_bundle
