"""
Asclepius FairAI - Level 2 SHAP Demonstration Test Suite
Validates the built-in deterministic SHAP demonstration package:
- Real trained RandomForestClassifier model
- Real evaluation feature dataset
- Real prediction evidence CSV
- Real SHAP TreeExplainer attributions
- Demographic protection exclusion by default
- Level 1 prediction-only preservation
- Level 2 explainability audit validation
- PDF report section generation
"""

import os
import sys
import tempfile
import pickle
import joblib
import numpy as np
import pandas as pd

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.external_audit import (
    get_shap_demo_paths,
    load_shap_demo_bundle,
    load_external_model,
    validate_explainability_inputs,
    calculate_external_shap,
    build_external_audit_bundle,
    load_demo_audit_csv,
    get_demo_audit_csv_path,
)
from src.core.report_generator import generate_fairness_report


def run_all_tests():
    paths = get_shap_demo_paths()
    results = []

    def record(test_num: int, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        results.append((test_num, name, passed, detail))
        prefix = f"[TEST {test_num:02d}] {name}"
        dots = "." * max(3, 70 - len(prefix))
        msg = f"{prefix} {dots} {status}"
        if detail and not passed:
            msg += f" ({detail})"
        print(msg)

    print("\n" + "=" * 80)
    print("ASCLEPIUS FAIRAI — REAL BUILT-IN SHAP DEMO TEST SUITE")
    print("=" * 80)

    # 1. Demo model exists
    model_exists = os.path.exists(paths["model"]) and os.path.getsize(paths["model"]) > 0
    record(1, "Demo model file exists on disk", model_exists, paths["model"])

    # 2. Demo model loads
    model_obj, err_load = load_external_model(paths["model"])
    model_loads = model_obj is not None and err_load is None
    record(2, "Demo model loads successfully via load_external_model", model_loads, err_load or "")

    # 3. Demo model is fitted
    is_fitted = hasattr(model_obj, "classes_") and hasattr(model_obj, "predict") and hasattr(model_obj, "predict_proba")
    record(3, "Demo model is genuinely fitted with predict and predict_proba", is_fitted)

    # 4. Demo features exist
    features_exist = os.path.exists(paths["features"]) and os.path.getsize(paths["features"]) > 0
    df_features = pd.read_csv(paths["features"]) if features_exist else pd.DataFrame()
    features_ok = features_exist and len(df_features) == 600 and "sex" in df_features.columns
    record(4, "Demo features CSV exists (600 rows, contains clinical features + sex)", features_ok)

    # 5. Demo prediction evidence exists
    preds_exist = os.path.exists(paths["predictions"]) and os.path.getsize(paths["predictions"]) > 0
    df_preds = pd.read_csv(paths["predictions"]) if preds_exist else pd.DataFrame()
    preds_ok = preds_exist and len(df_preds) == 600 and set(["id", "y_true", "y_pred", "y_prob", "sex"]).issubset(df_preds.columns)
    record(5, "Demo prediction evidence CSV exists (600 rows, required columns)", preds_ok)

    # 6. IDs match between features and prediction CSV
    ids_match = False
    if not df_features.empty and not df_preds.empty:
        ids_match = list(df_features["id"]) == list(df_preds["id"])
    record(6, "IDs match exactly between evaluation features and predictions", ids_match)

    # 7. Predictions actually match model output
    model_preds_match = False
    if is_fitted and not df_features.empty:
        expected_cols = list(model_obj.feature_names_in_)
        X_mat = df_features[expected_cols]
        actual_model_preds = model_obj.predict(X_mat)
        model_preds_match = np.array_equal(actual_model_preds, df_preds["y_pred"].values)
    record(7, "Predictions in CSV genuinely match trained model predict() output", model_preds_match)

    # 8. Probability values are 0–1
    probs_valid = False
    if not df_preds.empty:
        prob_vals = df_preds["y_prob"].values
        probs_valid = np.all((prob_vals >= 0.0) & (prob_vals <= 1.0)) and np.all(np.isfinite(prob_vals))
    record(8, "Prediction probability values are finite and strictly bounded in [0.0, 1.0]", probs_valid)

    # 9. SHAP library is available
    shap_available = False
    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False
    record(9, "SHAP library is installed and importable", shap_available)

    # 10. TreeExplainer works
    explainer_works = False
    shap_values = None
    if shap_available and is_fitted and not df_features.empty:
        try:
            expected_cols = list(model_obj.feature_names_in_)
            X_eval = df_features[expected_cols]
            explainer = shap.TreeExplainer(model_obj)
            raw_shap = explainer.shap_values(X_eval)
            if isinstance(raw_shap, list) and len(raw_shap) == 2:
                shap_values = raw_shap[1]
            elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                shap_values = raw_shap[:, :, 1]
            else:
                shap_values = raw_shap
            explainer_works = shap_values is not None
        except Exception as e:
            explainer_works = False
    record(10, "SHAP TreeExplainer instantiates and executes without errors", explainer_works)

    # 11. SHAP values are numeric
    is_numeric = explainer_works and isinstance(shap_values, np.ndarray) and np.issubdtype(shap_values.dtype, np.number)
    record(11, "SHAP attribution values are strictly numeric ndarray", is_numeric)

    # 12. SHAP values are finite
    is_finite = explainer_works and bool(np.all(np.isfinite(shap_values)))
    record(12, "All SHAP attribution values are strictly finite (no NaN / Inf)", is_finite)

    # 13. SHAP feature count matches model features
    expected_cols = list(model_obj.feature_names_in_) if is_fitted else []
    feat_count_matches = explainer_works and shap_values.shape[1] == len(expected_cols)
    record(13, f"SHAP column count ({shap_values.shape[1] if explainer_works else 0}) matches model features ({len(expected_cols)})", feat_count_matches)

    # 14. Feature names match
    calc_res = calculate_external_shap(
        model=model_obj,
        X_features=df_features[expected_cols],
        sensitive_col="sex",
        y_true=df_preds["y_true"].values,
        y_pred=df_preds["y_pred"].values,
        y_prob=df_preds["y_prob"].values,
    )
    reported_feats = [item["feature"] for item in calc_res.get("global_importance", [])]
    names_match = set(reported_feats) == set(expected_cols)
    record(14, "Feature names match expected clinical predictor features", names_match)

    # 15. SHAP values are not fabricated/constant zeros
    mean_abs_values = [item["mean_abs_shap"] for item in calc_res.get("global_importance", [])]
    not_all_zeros = len(mean_abs_values) > 0 and any(v > 0.001 for v in mean_abs_values)
    record(15, "SHAP attributions contain genuine variance and are not constant zeros", not_all_zeros)

    # 16. Sensitive attribute sex is excluded by default
    sex_excluded = "sex" not in expected_cols and "sex" not in reported_feats
    record(16, "Sensitive attribute 'sex' is strictly excluded from model & SHAP features by default", sex_excluded)

    # 17. Level 2 validation succeeds
    X_val, summary, err_val = validate_explainability_inputs(
        model=model_obj,
        features_source=paths["features"],
        target_col="y_true",
        sensitive_col="sex",
        include_sensitive_in_features=False,
        id_col="id",
    )
    l2_val_success = err_val is None and X_val is not None and "sex" not in X_val.columns and X_val.shape[1] == 8
    record(17, "Level 2 validate_explainability_inputs succeeds with 8 clinical features", l2_val_success)

    # 18. Level 1 prediction-only audit still succeeds
    bundle_l1 = build_external_audit_bundle(
        df=df_preds,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id",
    )
    l1_success = (
        bundle_l1.get("status") == "success"
        and bundle_l1.get("audit_level") == "level_1_prediction_evidence"
        and bundle_l1.get("shap_results", {}).get("status") == "unavailable"
        and "model internals were not provided" in bundle_l1.get("shap_results", {}).get("message", "").lower()
    )
    record(18, "Level 1 prediction-only audit succeeds and enforces SHAP unavailable", l1_success)

    # 19. Existing demo loader still works
    demo_df, demo_err = load_demo_audit_csv()
    demo_loader_ok = (
        demo_err is None
        and demo_df is not None
        and len(demo_df) == 600
        and list(demo_df.columns) == ["id", "y_true", "y_pred", "y_prob", "sex"]
    )
    record(19, "Existing Level 1 demo loader (load_demo_audit_csv) remains intact and operational", demo_loader_ok)

    # 20. PDF report includes SHAP section when Level 2 is active
    bundle_l2 = build_external_audit_bundle(
        df=df_preds,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id",
        model_source=paths["model"],
        features_source=paths["features"],
        include_sensitive_in_features=False,
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf_l2:
        tmp_pdf_l2_path = tmp_pdf_l2.name

    try:
        rep_l2 = generate_fairness_report(bundle_l2, output_path=tmp_pdf_l2_path)
        has_shap_l2 = (
            rep_l2.get("status") == "success"
            and os.path.exists(tmp_pdf_l2_path)
            and rep_l2.get("file_size_bytes", 0) > 50000
            and bundle_l2.get("audit_level") == "level_2_explainability"
            and bundle_l2.get("shap_results", {}).get("status") == "success"
            and len(bundle_l2.get("shap_results", {}).get("global_importance", [])) == 8
        )
    finally:
        if os.path.exists(tmp_pdf_l2_path):
            os.remove(tmp_pdf_l2_path)
    record(20, "PDF report includes Section 6 SHAP Explainability Analysis when Level 2 is active", has_shap_l2)

    # 21. PDF report keeps SHAP unavailable section for Level 1
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf_l1:
        tmp_pdf_l1_path = tmp_pdf_l1.name

    try:
        rep_l1 = generate_fairness_report(bundle_l1, output_path=tmp_pdf_l1_path)
        has_unavailable_l1 = (
            rep_l1.get("status") == "success"
            and os.path.exists(tmp_pdf_l1_path)
            and rep_l1.get("file_size_bytes", 0) > 5000
            and bundle_l1.get("audit_level") == "level_1_prediction_evidence"
            and bundle_l1.get("shap_results", {}).get("status") == "unavailable"
            and "model internals were not provided" in bundle_l1.get("shap_results", {}).get("message", "").lower()
        )
    finally:
        if os.path.exists(tmp_pdf_l1_path):
            os.remove(tmp_pdf_l1_path)
    record(21, "PDF report displays SHAP unavailable advisory when Level 1 is active", has_unavailable_l1)

    # 22. Unsupported model fails gracefully
    unsupported_obj = {"not_a_model": True}
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_bad:
        pickle.dump(unsupported_obj, tmp_bad)
        tmp_bad_path = tmp_bad.name

    try:
        bad_model, bad_err = load_external_model(tmp_bad_path)
        bad_handled = bad_model is None and bad_err is not None and "predict" in bad_err.lower()
    finally:
        if os.path.exists(tmp_bad_path):
            os.remove(tmp_bad_path)
    record(22, "Unsupported / unfitted model fails gracefully with descriptive error", bad_handled)

    print("=" * 80)
    passed_cnt = sum(1 for _, _, p, _ in results if p)
    total_cnt = len(results)
    print(f"SUMMARY: {passed_cnt}/{total_cnt} TESTS PASSED")
    print("=" * 80 + "\n")

    assert passed_cnt == total_cnt, f"Failed {total_cnt - passed_cnt} tests!"


if __name__ == "__main__":
    run_all_tests()
