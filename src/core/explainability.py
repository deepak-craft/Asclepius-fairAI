"""
Asclepius FairAI - Explainable AI (XAI) Module using SHAP
Phase 5: Global Feature Importance, Local Prediction Explanations, Sensitive Attribute Transparency, and Clinical Safety Disclaimers
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import shap


def _extract_feature_names(preprocessor, X_sample: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Extracts clean, human-readable feature names from the fitted scikit-learn ColumnTransformer.
    Strips internal sklearn pipeline prefixes ('num__', 'cat__', 'remainder__').
    """
    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            raw_names = preprocessor.get_feature_names_out()
            clean_names = [
                str(name)
                .replace("num__", "")
                .replace("cat__", "")
                .replace("remainder__", "")
                for name in raw_names
            ]
            return clean_names
    except Exception:
        pass

    # Fallback inspection of transformers
    feature_names = []
    try:
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
                try:
                    cat_encoder = trans.named_steps["onehot"]
                    cat_names = cat_encoder.get_feature_names_out(cols)
                    feature_names.extend([str(c) for c in cat_names])
                except Exception:
                    feature_names.extend([str(c) for c in cols])
            elif isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                feature_names.extend([str(c) for c in cols])
    except Exception:
        pass

    if not feature_names and X_sample is not None:
        feature_names = [str(c) for c in X_sample.columns]

    return feature_names


def generate_shap_explanations(
    pipeline: Any,
    X_test: pd.DataFrame,
    model_type: str = "random_forest",
    max_samples: int = 500,
    sensitive_col: Optional[str] = None,
    y_test: Optional[Union[List[int], np.ndarray]] = None,
    y_pred: Optional[Union[List[int], np.ndarray]] = None,
    y_prob: Optional[Union[List[float], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Generates comprehensive SHAP explanations (Global feature importance & Local patient-level attributions).
    
    Supports:
    - Random Forest (TreeExplainer)
    - Gradient Boosting (TreeExplainer)
    - Logistic Regression (LinearExplainer)

    Parameters:
    -----------
    pipeline: Fitted sklearn Pipeline with 'preprocessor' and 'classifier'
    X_test: Unprocessed raw test feature DataFrame
    model_type: Model identifier ('random_forest', 'gradient_boosting', 'logistic_regression')
    max_samples: Maximum test samples to explain (performance cap, default 500)
    sensitive_col: Name of the protected/sensitive attribute
    y_test: Ground truth binary targets for test set
    y_pred: Predicted binary labels
    y_prob: Predicted positive class probabilities

    Returns:
    --------
    Dict containing structured global rankings, local explanations, metadata, and clinical disclaimers.
    """
    if pipeline is None:
        return {
            "status": "error",
            "error_message": "No fitted model pipeline provided.",
        }

    if X_test is None or len(X_test) == 0:
        return {
            "status": "error",
            "error_message": "Test features dataframe X_test is empty.",
        }

    # 1. Cap test set to max_samples for fast performance
    total_test_samples = len(X_test)
    if total_test_samples > max_samples:
        X_eval = X_test.iloc[:max_samples].copy()
    else:
        X_eval = X_test.copy()

    n_samples_explained = len(X_eval)

    # 2. Extract components from Pipeline or standalone Estimator
    if hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get("preprocessor")
        classifier = pipeline.named_steps.get("classifier", pipeline)
    else:
        preprocessor = None
        classifier = pipeline

    # 3. Transform evaluation data if preprocessor is present
    if preprocessor is not None:
        try:
            X_trans = preprocessor.transform(X_eval)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Failed to transform test features through pipeline preprocessor: {str(e)}",
            }
        feature_names = _extract_feature_names(preprocessor, X_eval)
    else:
        if isinstance(X_eval, pd.DataFrame):
            X_trans = X_eval.to_numpy()
            feature_names = list(X_eval.columns)
        else:
            X_trans = np.array(X_eval)
            feature_names = [f"Feature_{i+1}" for i in range(X_trans.shape[1])]

    if len(feature_names) != X_trans.shape[1]:
        feature_names = [f"Feature_{i+1}" for i in range(X_trans.shape[1])]

    # 4. Resolve model type identifier
    model_norm = str(model_type).lower().replace("-", "_").replace(" ", "_") if model_type else "auto"
    if model_norm in ["auto", "none", ""]:
        cls_name = type(classifier).__name__.lower()
        if "forest" in cls_name:
            model_norm = "random_forest"
        elif "gradient" in cls_name or "boost" in cls_name:
            model_norm = "gradient_boosting"
        elif "logistic" in cls_name:
            model_norm = "logistic_regression"

    shap_values_raw = None
    base_value = 0.0

    try:
        if model_norm in ["random_forest", "gradient_boosting", "tree"]:
            explainer = shap.TreeExplainer(classifier)
            shap_values_raw = explainer.shap_values(X_trans)
            expected_val = explainer.expected_value
        elif model_norm in ["logistic_regression", "linear"]:
            bg_size = min(len(X_trans), 100)
            background = X_trans[:bg_size]
            explainer = shap.LinearExplainer(classifier, background)
            shap_values_raw = explainer.shap_values(X_trans)
            expected_val = explainer.expected_value
        else:
            # Fallback to general Explainer
            bg_size = min(len(X_trans), 50)
            background = X_trans[:bg_size]
            explainer = shap.Explainer(classifier, background)
            shap_obj = explainer(X_trans)
            shap_values_raw = shap_obj.values
            expected_val = getattr(shap_obj, "base_values", 0.0)

        # Handle base value formatting (extract positive class baseline if multi-class / list)
        if isinstance(expected_val, (list, tuple, np.ndarray)):
            if len(expected_val) >= 2:
                base_value = float(expected_val[1])
            elif len(expected_val) == 1:
                base_value = float(expected_val[0])
            else:
                base_value = 0.0
        elif isinstance(expected_val, (float, int, np.floating, np.integer)):
            base_value = float(expected_val)
        else:
            base_value = 0.0

        # Handle SHAP values shape (binary classification positive class attribution)
        if isinstance(shap_values_raw, list):
            if len(shap_values_raw) >= 2:
                shap_matrix = np.array(shap_values_raw[1])
            elif len(shap_values_raw) == 1:
                shap_matrix = np.array(shap_values_raw[0])
            else:
                shap_matrix = np.array(shap_values_raw)
        elif isinstance(shap_values_raw, np.ndarray):
            if shap_values_raw.ndim == 3 and shap_values_raw.shape[2] >= 2:
                shap_matrix = shap_values_raw[:, :, 1]
            elif shap_values_raw.ndim == 3 and shap_values_raw.shape[2] == 1:
                shap_matrix = shap_values_raw[:, :, 0]
            else:
                shap_matrix = shap_values_raw
        else:
            shap_matrix = np.array(shap_values_raw)

    except Exception as explainer_err:
        return {
            "status": "error",
            "error_message": f"SHAP calculation failed: {str(explainer_err)}",
        }

    # Verify dimensions
    if shap_matrix.ndim != 2 or shap_matrix.shape[1] != len(feature_names):
        return {
            "status": "error",
            "error_message": f"Dimension mismatch in SHAP matrix ({shap_matrix.shape}) and features ({len(feature_names)}).",
        }

    # 6. Global Feature Importance (Mean Absolute SHAP)
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    mean_direction_shap = np.mean(shap_matrix, axis=0)
    max_global_val = float(np.max(mean_abs_shap)) if np.max(mean_abs_shap) > 0 else 1.0

    global_importance_list = []
    for idx, f_name in enumerate(feature_names):
        val = float(mean_abs_shap[idx])
        dir_val = float(mean_direction_shap[idx])
        global_importance_list.append({
            "feature": f_name,
            "mean_abs_shap": round(val, 4),
            "relative_pct": round((val / max_global_val) * 100, 1),
            "direction": "Increases Risk" if dir_val >= 0 else "Decreases Risk",
            "direction_val": round(dir_val, 4),
            "rank": 0,
        })

    # Sort descending by mean absolute SHAP
    global_importance_list.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    for rank_idx, item in enumerate(global_importance_list):
        item["rank"] = rank_idx + 1

    # 7. Local Sample Explanations
    # Prepare individual sample explanation breakdowns
    local_explanations = []
    y_test_list = list(y_test[:n_samples_explained]) if y_test is not None else None
    y_pred_list = list(y_pred[:n_samples_explained]) if y_pred is not None else None
    y_prob_list = list(y_prob[:n_samples_explained]) if y_prob is not None else None

    for sample_idx in range(n_samples_explained):
        sample_shap_row = shap_matrix[sample_idx]
        raw_row = X_eval.iloc[sample_idx].to_dict()

        feature_contributions = []
        for f_idx, f_name in enumerate(feature_names):
            contrib = float(sample_shap_row[f_idx])
            feature_contributions.append({
                "feature": f_name,
                "shap_value": round(contrib, 4),
                "abs_shap": round(abs(contrib), 4),
                "impact": "Increases Prediction" if contrib > 0 else ("Decreases Prediction" if contrib < 0 else "Neutral"),
                "sign": "+" if contrib > 0 else ("-" if contrib < 0 else "0"),
            })

        # Sort by absolute SHAP impact
        feature_contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

        top_positive = [f for f in feature_contributions if f["shap_value"] > 0][:5]
        top_negative = [f for f in feature_contributions if f["shap_value"] < 0][:5]

        # Baseline offset calculation
        shap_sum = float(np.sum(sample_shap_row))
        calculated_score = base_value + shap_sum

        local_explanations.append({
            "sample_index": sample_idx,
            "patient_features": {str(k): (round(v, 2) if isinstance(v, float) else str(v)) for k, v in raw_row.items()},
            "actual_outcome": int(y_test_list[sample_idx]) if y_test_list is not None else None,
            "predicted_label": int(y_pred_list[sample_idx]) if y_pred_list is not None else None,
            "predicted_probability": float(y_prob_list[sample_idx]) if y_prob_list is not None else None,
            "base_value": round(base_value, 4),
            "shap_sum": round(shap_sum, 4),
            "calculated_score": round(calculated_score, 4),
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "all_contributions": feature_contributions,
        })

    # 8. Check Sensitive Attribute Usage in Feature Space
    sensitive_used = False
    sensitive_features_matched = []
    if sensitive_col:
        sens_clean = str(sensitive_col).lower()
        for f in feature_names:
            f_lower = f.lower()
            if sens_clean == f_lower or f_lower.startswith(f"{sens_clean}_") or f_lower.endswith(f"_{sens_clean}"):
                sensitive_used = True
                sensitive_features_matched.append(f)

    # 9. Structured Response Bundle
    return {
        "status": "success",
        "model_type": model_norm,
        "explainer_type": explainer.__class__.__name__,
        "total_test_samples": total_test_samples,
        "samples_explained": n_samples_explained,
        "max_samples_cap": max_samples,
        "base_value": round(base_value, 4),
        "feature_names": feature_names,
        "global_importance": global_importance_list,
        "local_explanations": local_explanations,
        "sensitive_attribute_context": {
            "sensitive_attribute_name": sensitive_col or "None",
            "included_in_model_features": sensitive_used,
            "matched_feature_columns": sensitive_features_matched,
            "transparency_note": (
                f"The sensitive attribute '{sensitive_col}' is directly present in the model's feature set ({', '.join(sensitive_features_matched)})."
                if sensitive_used
                else f"The sensitive attribute '{sensitive_col}' was excluded from model training features to prevent direct demographic decisioning."
            ),
        },
        "clinical_safety": {
            "causality_disclaimer": "SHAP values quantify statistical feature influence within this specific machine learning model. They do not establish biological or clinical causality.",
            "fairness_vs_explainability": "High or low SHAP importance for a demographic attribute does not prove or disprove algorithmic bias. Fairness must be audited through validated group-level disparity metrics (DPD, DIR, Equalized Odds, Underdiagnosis FNR).",
        },
    }

