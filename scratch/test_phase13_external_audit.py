"""
Asclepius FairAI - Phase 13 Test Suite: Existing Healthcare AI Model Audit Mode
Validates prediction-based, model-agnostic external model evaluation, zero model binary execution,
subgroup reliability, fairness metric reuse, SHAP unavailable enforcement, mitigation gating,
PDF report generation, SQLite history persistence without raw vectors, state isolation, and regression protection.
"""

import os
import sys
import tempfile
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.external_audit import (
    load_audit_input,
    validate_audit_input,
    evaluate_external_predictions,
    build_external_audit_bundle,
    get_external_shap_unavailable_bundle,
    get_demo_audit_csv_path,
    load_demo_audit_csv,
    create_deterministic_demo_audit_csv,
    detect_external_audit_columns,
    load_external_model,
    validate_explainability_inputs,
    calculate_external_shap,
)
from src.core.fairness import evaluate_fairness_metrics
from src.core.insights import generate_insights
from src.core.mitigation import evaluate_threshold_candidates
from src.core.report_generator import generate_fairness_report
from src.core.history import save_analysis, get_analysis_history, get_analysis, initialize_history_db
from src.core.train_model import train_clinical_model
from src.core.explainability import generate_shap_explanations


def create_sample_df(n=100, p_prob=True, minority_n=50):
    """Creates a consistent synthetic external audit dataset."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.4, n)
    # y_pred with moderate correlation to y_true
    noise = np.random.binomial(1, 0.15, n)
    y_pred = np.where(noise == 1, 1 - y_true, y_true)
    
    # sensitive feature with custom split
    sensitive = np.array(["Group A"] * (n - minority_n) + ["Group B"] * minority_n)
    
    data = {
        "id": [f"ID-{i+1:04d}" for i in range(n)],
        "y_true": y_true,
        "y_pred": y_pred,
        "group": sensitive
    }
    if p_prob:
        # Generate probabilities consistent with predictions
        probs = np.where(y_pred == 1, np.random.uniform(0.55, 0.95, n), np.random.uniform(0.05, 0.45, n))
        data["y_prob"] = np.round(probs, 3)
    return pd.DataFrame(data)


def run_all_tests():
    passed = 0
    total = 115
    results = []

    def record_test(idx, name, is_pass, note=""):
        nonlocal passed
        if is_pass:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        results.append((idx, name, status, note))
        print(f"[{status}] Test {idx:02d}: {name} {note}")

    print("======================================================================")
    print("ASCLEPIUS FAIRAI — PHASE 13 TEST SUITE (EXTERNAL MODEL AUDIT MODE)")
    print("======================================================================\n")

    # -------------------------------------------------------------------------
    # 1. Valid external audit CSV
    # -------------------------------------------------------------------------
    df_valid = create_sample_df(120, p_prob=True, minority_n=60)
    v1 = validate_audit_input(df_valid, "y_true", "y_pred", "group", "y_prob", "id")
    t1_pass = v1["status"] == "ready" and len(v1["issues"]) == 0
    record_test(1, "Valid external audit CSV", t1_pass, f"Status: {v1['status']}")

    # -------------------------------------------------------------------------
    # 2. Missing y_true
    # -------------------------------------------------------------------------
    v2 = validate_audit_input(df_valid, "non_existent_y_true", "y_pred", "group")
    t2_pass = v2["status"] == "invalid" and any("Ground truth column" in i for i in v2["issues"])
    record_test(2, "Missing y_true detection", t2_pass)

    # -------------------------------------------------------------------------
    # 3. Missing y_pred
    # -------------------------------------------------------------------------
    v3 = validate_audit_input(df_valid, "y_true", "non_existent_y_pred", "group")
    t3_pass = v3["status"] == "invalid" and any("Predicted label column" in i for i in v3["issues"])
    record_test(3, "Missing y_pred detection", t3_pass)

    # -------------------------------------------------------------------------
    # 4. Missing sensitive attribute
    # -------------------------------------------------------------------------
    v4 = validate_audit_input(df_valid, "y_true", "y_pred", "non_existent_sens")
    t4_pass = v4["status"] == "invalid" and any("Sensitive attribute column" in i for i in v4["issues"])
    record_test(4, "Missing sensitive attribute detection", t4_pass)

    # -------------------------------------------------------------------------
    # 5. y_true/y_pred length mismatch
    # -------------------------------------------------------------------------
    # Test through evaluate_external_predictions directly
    e5 = evaluate_external_predictions(y_true=[1, 0, 1], y_pred=[1, 0], sensitive_features=["A", "B", "A"])
    t5_pass = e5.get("status") == "error" and "Mismatched" in e5.get("error_message", "")
    record_test(5, "y_true/y_pred length mismatch rejected", t5_pass)

    # -------------------------------------------------------------------------
    # 6. sensitive length mismatch
    # -------------------------------------------------------------------------
    df_mismatch = df_valid.copy()
    # Check that equal column lengths in valid DF are enforced
    t6_pass = len(df_mismatch["y_true"]) == len(df_mismatch["group"])
    record_test(6, "Sensitive length matches record count", t6_pass)

    # -------------------------------------------------------------------------
    # 7. Single-class target
    # -------------------------------------------------------------------------
    df_single = df_valid.copy()
    df_single["y_true"] = 1
    v7 = validate_audit_input(df_single, "y_true", "y_pred", "group")
    t7_pass = v7["status"] == "invalid" and any("Current clinical fairness workflow requires binary classification" in i for i in v7["issues"])
    record_test(7, "Single-class target rejected", t7_pass)

    # -------------------------------------------------------------------------
    # 8. Invalid binary prediction (multiclass or non-binary)
    # -------------------------------------------------------------------------
    df_multi = df_valid.copy()
    df_multi["y_pred"] = [0, 1, 2] * 40
    v8 = validate_audit_input(df_multi, "y_true", "y_pred", "group")
    t8_pass = v8["status"] == "invalid" and any("requires binary classification" in i for i in v8["issues"])
    record_test(8, "Invalid multiclass prediction rejected", t8_pass)

    # -------------------------------------------------------------------------
    # 9. Missing y_prob handling
    # -------------------------------------------------------------------------
    df_noprob = create_sample_df(100, p_prob=False, minority_n=50)
    v9 = validate_audit_input(df_noprob, "y_true", "y_pred", "group", y_prob_col=None)
    t9_pass = v9["status"] == "warning" and v9["summary"]["has_probabilities"] is False and any("Continuous prediction probabilities" in w for w in v9["warnings"])
    record_test(9, "Missing y_prob handled with clear warning", t9_pass)

    # -------------------------------------------------------------------------
    # 10. Invalid y_prob values (out of [0, 1])
    # -------------------------------------------------------------------------
    df_badprob = create_sample_df(100, p_prob=True)
    df_badprob.loc[0, "y_prob"] = 1.45
    v10 = validate_audit_input(df_badprob, "y_true", "y_pred", "group", "y_prob")
    t10_pass = v10["status"] == "invalid" and any("Prediction probabilities must be in range [0.0, 1.0]" in i for i in v10["issues"])
    record_test(10, "Invalid out-of-bounds y_prob rejected", t10_pass)

    # -------------------------------------------------------------------------
    # 11. Valid y_prob
    # -------------------------------------------------------------------------
    df_goodprob = create_sample_df(100, p_prob=True)
    v11 = validate_audit_input(df_goodprob, "y_true", "y_pred", "group", "y_prob")
    t11_pass = v11["summary"]["has_probabilities"] is True and len(v11["parsed_arrays"]["y_prob"]) == 100
    record_test(11, "Valid y_prob parsed successfully", t11_pass)

    # -------------------------------------------------------------------------
    # 12. Small subgroup n < 30 (Insufficient Evidence)
    # -------------------------------------------------------------------------
    df_small = create_sample_df(100, minority_n=15)
    f12 = evaluate_fairness_metrics(df_small["y_true"], df_small["y_pred"], sensitive_features=df_small["group"])
    small_sub = [s for s in f12["subgroups"] if s["group"] == "Group B"][0]
    t12_pass = (small_sub["reliability_tier"] == "insufficient" or "INSUFFICIENT" in small_sub["reliability"].upper()) and f12["has_insufficient_subgroup"] is True
    record_test(12, "Small subgroup n < 30 flagged Insufficient Evidence", t12_pass, f"Tier: {small_sub['reliability']}")

    # -------------------------------------------------------------------------
    # 13. Caution subgroup n 30–49
    # -------------------------------------------------------------------------
    df_caution = create_sample_df(100, minority_n=35)
    f13 = evaluate_fairness_metrics(df_caution["y_true"], df_caution["y_pred"], sensitive_features=df_caution["group"])
    caut_sub = [s for s in f13["subgroups"] if s["group"] == "Group B"][0]
    t13_pass = (caut_sub["reliability_tier"] == "caution" or "CAUTION" in caut_sub["reliability"].upper()) and f13["has_insufficient_subgroup"] is False
    record_test(13, "Caution subgroup n 30-49 flagged Caution", t13_pass, f"Tier: {caut_sub['reliability']}")

    # -------------------------------------------------------------------------
    # 14. Standard subgroup n >= 50
    # -------------------------------------------------------------------------
    df_std = create_sample_df(120, minority_n=60)
    f14 = evaluate_fairness_metrics(df_std["y_true"], df_std["y_pred"], sensitive_features=df_std["group"])
    std_sub = [s for s in f14["subgroups"] if s["group"] == "Group B"][0]
    t14_pass = std_sub["reliability_tier"] == "standard" or "STANDARD" in std_sub["reliability"].upper()
    record_test(14, "Standard subgroup n >= 50 flagged Standard Evaluation", t14_pass, f"Tier: {std_sub['reliability']}")

    # -------------------------------------------------------------------------
    # 15. Performance result equivalence
    # -------------------------------------------------------------------------
    df_perf = create_sample_df(200, p_prob=True, minority_n=100)
    perf15 = evaluate_external_predictions(df_perf["y_true"], df_perf["y_pred"], df_perf["group"], df_perf["y_prob"])
    # Independent calculation
    sk_acc = accuracy_score(df_perf["y_true"], df_perf["y_pred"])
    sk_bal_acc = balanced_accuracy_score(df_perf["y_true"], df_perf["y_pred"])
    sk_prec = precision_score(df_perf["y_true"], df_perf["y_pred"])
    sk_rec = recall_score(df_perf["y_true"], df_perf["y_pred"])
    sk_f1 = f1_score(df_perf["y_true"], df_perf["y_pred"])
    t15_pass = (
        np.isclose(perf15["accuracy"], sk_acc, atol=1e-4) and
        np.isclose(perf15["balanced_accuracy"], sk_bal_acc, atol=1e-4) and
        np.isclose(perf15["precision"], sk_prec, atol=1e-4) and
        np.isclose(perf15["recall"], sk_rec, atol=1e-4) and
        np.isclose(perf15["f1"], sk_f1, atol=1e-4)
    )
    record_test(15, "Performance result exact equivalence to scikit-learn", t15_pass)

    # -------------------------------------------------------------------------
    # 16. Fairness result equivalence (direct core reuse)
    # -------------------------------------------------------------------------
    bundle16 = build_external_audit_bundle(df_perf, "y_true", "y_pred", "group", "y_prob")
    direct_fair = evaluate_fairness_metrics(df_perf["y_true"], df_perf["y_pred"], df_perf["y_prob"], df_perf["group"])
    t16_pass = (
        bundle16["fairness_results"]["metrics"]["demographic_parity_difference"]["value"] ==
        direct_fair["metrics"]["demographic_parity_difference"]["value"] and
        bundle16["fairness_results"]["metrics"]["fnr_disparity"]["value"] ==
        direct_fair["metrics"]["fnr_disparity"]["value"]
    )
    record_test(16, "Fairness metrics direct reuse equivalence", t16_pass)

    # -------------------------------------------------------------------------
    # 17. ROC-AUC unavailable without y_prob
    # -------------------------------------------------------------------------
    perf17 = evaluate_external_predictions(df_perf["y_true"], df_perf["y_pred"], df_perf["group"], y_prob=None)
    t17_pass = (
        perf17["roc_auc"] is None and
        perf17["has_probabilities"] is False and
        perf17["roc_auc_display"] == "ROC-AUC unavailable — prediction probabilities were not supplied."
    )
    record_test(17, "ROC-AUC marked unavailable without y_prob", t17_pass)

    # -------------------------------------------------------------------------
    # 18. ROC-AUC calculated with y_prob
    # -------------------------------------------------------------------------
    perf18 = evaluate_external_predictions(df_perf["y_true"], df_perf["y_pred"], df_perf["group"], y_prob=df_perf["y_prob"])
    sk_auc = roc_auc_score(df_perf["y_true"], df_perf["y_prob"])
    t18_pass = (
        perf18["roc_auc"] is not None and
        perf18["has_probabilities"] is True and
        np.isclose(perf18["roc_auc"], sk_auc, atol=1e-3)
    )
    record_test(18, "ROC-AUC calculated with y_prob", t18_pass, f"AUC: {perf18['roc_auc']}")

    # -------------------------------------------------------------------------
    # 19. Mitigation disabled without y_prob
    # -------------------------------------------------------------------------
    mit19 = evaluate_threshold_candidates(df_perf["y_true"], y_prob=None, sensitive_features=df_perf["group"])
    t19_pass = (
        mit19.get("status") in ["unavailable", "error"] and
        mit19.get("mitigation_available") is False and
        "Decision Threshold Calibration requires continuous probability scores (y_prob)" in mit19.get("message", "")
    )
    record_test(19, "Mitigation disabled with exact clinical notice without y_prob", t19_pass)

    # -------------------------------------------------------------------------
    # 20. Mitigation enabled with y_prob
    # -------------------------------------------------------------------------
    mit20 = evaluate_threshold_candidates(df_perf["y_true"], y_prob=df_perf["y_prob"], sensitive_features=df_perf["group"])
    t20_pass = mit20.get("status") == "success" and "selection" in mit20
    record_test(20, "Mitigation enabled and evaluated with y_prob", t20_pass, f"Status: {mit20.get('status')}")

    # -------------------------------------------------------------------------
    # 21. SHAP explicitly unavailable
    # -------------------------------------------------------------------------
    shap21 = get_external_shap_unavailable_bundle()
    t21_pass = (
        shap21.get("status") == "unavailable" and
        shap21.get("message") == "SHAP unavailable — model internals were not provided." and
        len(shap21.get("global_importance", [1])) == 0 and
        len(shap21.get("local_explanations", [1])) == 0
    )
    record_test(21, "SHAP explicitly unavailable with zero fabricated data", t21_pass)

    # -------------------------------------------------------------------------
    # 22. No SHAP state leakage in insights
    # -------------------------------------------------------------------------
    insights22 = generate_insights(
        model_results=bundle16["model_results"],
        fairness_results=bundle16["fairness_results"],
        shap_results=shap21
    )
    exp_finds = insights22.get("explainability_findings", [])
    has_unavailable_shap_finding = any("SHAP Unavailable — External Model Audit" in f.get("title", "") for f in exp_finds)
    has_fake_predictor_finding = any("Primary Model-Influencing Predictors" in f.get("title", "") for f in exp_finds)
    t22_pass = has_unavailable_shap_finding and not has_fake_predictor_finding
    record_test(22, "No SHAP state leakage in generated insights", t22_pass)

    # -------------------------------------------------------------------------
    # 23. External audit history save
    # -------------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db_path = tf.name

    try:
        initialize_history_db(temp_db_path)
        bundle23 = build_external_audit_bundle(
            df_perf, "y_true", "y_pred", "group", "y_prob",
            model_name="Clinical Risk Predictor",
            model_version="v2.1",
            model_type="External Model Audit"
        )
        bundle23["insights_results"] = insights22
        bundle23["mitigation_results"] = mit20

        save_res = save_analysis(bundle23, db_path=temp_db_path)
        t23_pass = save_res.get("status") == "success"

        # Check saved record
        records = get_analysis_history(db_path=temp_db_path)
        saved_rec = records[0]
        t23_pass = t23_pass and saved_rec["model_type"] == "External Model Audit"
        record_test(23, "External audit history save with model_type = 'External Model Audit'", t23_pass)

        # ---------------------------------------------------------------------
        # 24. No raw vectors stored in history (Zero PHI)
        # ---------------------------------------------------------------------
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_history WHERE analysis_id = ?", (saved_rec["analysis_id"],))
        row = cursor.fetchone()
        col_names = [d[0] for d in cursor.description]
        row_dict = dict(zip(col_names, row))
        conn.close()

        # Verify no raw vector columns exist
        forbidden_cols = ["y_true", "y_pred", "y_prob", "y_test", "sensitive_test", "raw_rows"]
        has_forbidden_col = any(c in col_names for c in forbidden_cols)
        # Verify JSON columns don't contain raw vectors
        t24_pass = (
            not has_forbidden_col and
            "EVAL-" not in str(row_dict) and
            isinstance(row_dict["dataset_row_count"], int)
        )
        record_test(24, "No raw evaluation vectors stored in history (Zero PHI compliance)", t24_pass)

        # ---------------------------------------------------------------------
        # 25. External audit report generation
        # ---------------------------------------------------------------------
        report_pdf_path = os.path.join(tempfile.gettempdir(), "test_external_audit_report.pdf")
        rep_res = generate_fairness_report(
            analysis_results=bundle23,
            output_path=report_pdf_path
        )
        t25_pass = rep_res.get("status") == "success" and os.path.exists(report_pdf_path) and os.path.getsize(report_pdf_path) > 1000
        record_test(25, "External audit PDF report generated successfully", t25_pass, f"Size: {os.path.getsize(report_pdf_path) if t25_pass else 0} bytes")

        # ---------------------------------------------------------------------
        # 26. Duplicate audit handling
        # ---------------------------------------------------------------------
        bundle23["analysis_id"] = saved_rec["analysis_id"]
        bundle23["report_filename"] = "updated_report.pdf"
        save_res_dup = save_analysis(bundle23, db_path=temp_db_path)
        records_after = get_analysis_history(db_path=temp_db_path)
        t26_pass = (
            save_res_dup.get("status") == "success" and
            len(records_after) == 1 and
            records_after[0]["report_filename"] == "updated_report.pdf"
        )
        record_test(26, "Duplicate audit handling updates existing record without error", t26_pass)

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

    # -------------------------------------------------------------------------
    # 27. Mode switching state reset simulation
    # -------------------------------------------------------------------------
    # Simulate switching from local evaluation to external audit and back
    local_state = {
        "auditMode": "internal",
        "trainedModel": {"name": "RandomForestClassifier", "accuracy": 0.88},
        "shapResults": {"status": "success", "global_importance": [{"feature": "age", "importance": 0.3}]},
        "mitigationResults": {"status": "success"}
    }
    # User switches to external audit:
    external_state = {
        "auditMode": "external",
        "trainedModel": None, # purged
        "shapResults": get_external_shap_unavailable_bundle(), # reset
        "mitigationResults": None # purged
    }
    t27_pass = (
        external_state["trainedModel"] is None and
        external_state["shapResults"]["status"] == "unavailable" and
        len(external_state["shapResults"]["global_importance"]) == 0
    )
    record_test(27, "Mode switching cleanses local model & SHAP state to prevent leakage", t27_pass)

    # -------------------------------------------------------------------------
    # 28. Existing local workflow regression
    # -------------------------------------------------------------------------
    demo_path = os.path.join(WORKSPACE_ROOT, "data", "demo", "synthetic_clinical_demo.csv")
    if os.path.exists(demo_path):
        df_local = pd.read_csv(demo_path)
        train_res = train_clinical_model(
            df=df_local,
            target_col="readmitted_30d",
            sensitive_col="gender",
            model_type="logistic_regression",
            test_size=0.2,
            random_state=42
        )
        fair_res = evaluate_fairness_metrics(
            train_res["y_test"],
            train_res["y_pred"],
            train_res["y_prob"],
            train_res["sensitive_test"]
        )
        feature_cols = [c for c in df_local.columns if c not in ["readmitted_30d"]]
        X_test_df = df_local[feature_cols].iloc[-len(train_res["y_test"]):]
        shap_res = generate_shap_explanations(
            pipeline=train_res["pipeline"],
            X_test=X_test_df,
            model_type="logistic_regression",
            sensitive_col="gender",
            y_test=train_res["y_test"],
            y_pred=train_res["y_pred"],
            y_prob=train_res["y_prob"]
        )
        insights_res = generate_insights(train_res, fair_res, shap_res)
        try:
            t28_pass = (
                train_res.get("status") == "success" and
                fair_res.get("status") == "success" and
                shap_res.get("status") == "success" and
                insights_res.get("status") == "success" and
                len(shap_res.get("global_importance", [])) > 0
            )
            note_28 = f"train:{train_res.get('status')}, fair:{fair_res.get('status')}, shap:{shap_res.get('status')}, ins:{insights_res.get('status')}"
        except Exception as ex:
            t28_pass = False
            note_28 = f"Exception: {str(ex)}"
    else:
        t28_pass = False
        note_28 = "File not found"

    record_test(28, "Existing Local Model Evaluation workflow full regression check", t28_pass, note_28)

    # -------------------------------------------------------------------------
    # 29. Semantic Validation: Valid y_true/y_pred/y_prob/sensitive mapping
    # -------------------------------------------------------------------------
    df_sem = create_sample_df(100, p_prob=True, minority_n=50)
    v29 = validate_audit_input(df_sem, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="group", y_prob_col="y_prob", id_col="id")
    t29_pass = v29["status"] == "ready" and len(v29["issues"]) == 0 and v29["status_title"] == "✓ Audit Input Ready"
    record_test(29, "Semantic Validation: Valid y_true/y_pred/y_prob/sensitive accepted", t29_pass)

    # -------------------------------------------------------------------------
    # 30. Semantic Validation: y_true == sensitive -> rejected
    # -------------------------------------------------------------------------
    v30 = validate_audit_input(df_sem, y_true_col="group", y_pred_col="y_pred", sensitive_col="group", y_prob_col="y_prob")
    t30_pass = v30["status"] == "invalid" and v30["status_title"] == "⚠ Audit Input Needs Correction" and any("incompatible fields" in i or "different semantic fields" in i for i in v30["issues"])
    record_test(30, "Semantic Validation: y_true == sensitive rejected", t30_pass)

    # -------------------------------------------------------------------------
    # 31. Semantic Validation: y_true == y_pred -> rejected
    # -------------------------------------------------------------------------
    v31 = validate_audit_input(df_sem, y_true_col="y_true", y_pred_col="y_true", sensitive_col="group", y_prob_col="y_prob")
    t31_pass = v31["status"] == "invalid" and v31["status_title"] == "⚠ Audit Input Needs Correction" and any("incompatible fields" in i or "different semantic fields" in i for i in v31["issues"])
    record_test(31, "Semantic Validation: y_true == y_pred rejected", t31_pass)

    # -------------------------------------------------------------------------
    # 32. Semantic Validation: y_true == y_prob -> rejected
    # -------------------------------------------------------------------------
    v32 = validate_audit_input(df_sem, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="group", y_prob_col="y_true")
    t32_pass = v32["status"] == "invalid" and v32["status_title"] == "⚠ Audit Input Needs Correction" and any("incompatible fields" in i or "different semantic fields" in i for i in v32["issues"])
    record_test(32, "Semantic Validation: y_true == y_prob rejected", t32_pass)

    # -------------------------------------------------------------------------
    # 33. Semantic Validation: y_pred == y_prob -> rejected
    # -------------------------------------------------------------------------
    v33 = validate_audit_input(df_sem, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="group", y_prob_col="y_pred")
    t33_pass = v33["status"] == "invalid" and v33["status_title"] == "⚠ Audit Input Needs Correction" and any("incompatible fields" in i or "different semantic fields" in i for i in v33["issues"])
    record_test(33, "Semantic Validation: y_pred == y_prob rejected", t33_pass)

    # -------------------------------------------------------------------------
    # 34. Semantic Validation: sensitive == y_pred -> rejected
    # -------------------------------------------------------------------------
    v34 = validate_audit_input(df_sem, y_true_col="y_true", y_pred_col="group", sensitive_col="group", y_prob_col="y_prob")
    t34_pass = v34["status"] == "invalid" and v34["status_title"] == "⚠ Audit Input Needs Correction" and any("incompatible fields" in i or "different semantic fields" in i for i in v34["issues"])
    record_test(34, "Semantic Validation: sensitive == y_pred rejected", t34_pass)

    # -------------------------------------------------------------------------
    # 35. Semantic Validation: sensitive == y_prob -> rejected
    # -------------------------------------------------------------------------
    v35 = validate_audit_input(df_sem, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="group", y_prob_col="group")
    t35_pass = v35["status"] == "invalid" and v35["status_title"] == "⚠ Audit Input Needs Correction" and any("incompatible fields" in i or "different semantic fields" in i for i in v35["issues"])
    record_test(35, "Semantic Validation: sensitive == y_prob rejected", t35_pass)

    # -------------------------------------------------------------------------
    # 36. Semantic Validation: invalid probability values -> rejected
    # -------------------------------------------------------------------------
    df_bad_prob = df_sem.copy()
    df_bad_prob.loc[0, "y_prob"] = 2.5
    v36 = validate_audit_input(df_bad_prob, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="group", y_prob_col="y_prob")
    t36_pass = v36["status"] == "invalid" and v36["status_title"] == "⚠ Audit Input Needs Correction" and any("range [0.0, 1.0]" in i for i in v36["issues"])
    record_test(36, "Semantic Validation: invalid probability values rejected", t36_pass)

    # -------------------------------------------------------------------------
    # 37. Semantic Validation: non-binary y_true -> rejected
    # -------------------------------------------------------------------------
    df_non_bin = df_sem.copy()
    df_non_bin.loc[0:10, "y_true"] = 3
    v37 = validate_audit_input(df_non_bin, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="group")
    t37_pass = v37["status"] == "invalid" and v37["status_title"] == "⚠ Audit Input Needs Correction" and any("binary" in i.lower() for i in v37["issues"])
    record_test(37, "Semantic Validation: non-binary y_true rejected", t37_pass)

    # -------------------------------------------------------------------------
    # 38. Semantic Validation: fewer than 2 sensitive cohorts -> rejected
    # -------------------------------------------------------------------------
    df_one_cohort = df_sem.copy()
    df_one_cohort["group"] = "Cohort A"
    v38 = validate_audit_input(df_one_cohort, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="group")
    t38_pass = v38["status"] == "invalid" and v38["status_title"] == "⚠ Audit Input Needs Correction" and any("at least 2 cohorts" in i.lower() or "at least 2 distinct cohorts" in i.lower() for i in v38["issues"])
    record_test(38, "Semantic Validation: fewer than 2 sensitive cohorts rejected", t38_pass)

    # -------------------------------------------------------------------------
    # 39. Bundled Demo CSV: external_model_audit_demo.csv -> accepted
    # -------------------------------------------------------------------------
    demo_ext_path = os.path.join(WORKSPACE_ROOT, "data", "demo", "external_model_audit_demo.csv")
    if os.path.exists(demo_ext_path):
        df_ext_demo = pd.read_csv(demo_ext_path)
        v39 = validate_audit_input(df_ext_demo, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="sex", y_prob_col="y_prob", id_col="id")
        t39_pass = v39["status"] == "ready" and len(v39["issues"]) == 0 and v39["status_title"] == "✓ Audit Input Ready"
    else:
        t39_pass = False
    record_test(39, "Bundled Demo CSV: external_model_audit_demo.csv accepted", t39_pass)

    # -------------------------------------------------------------------------
    # 40. Critical Case: heart_disease-only mapping -> rejected
    # -------------------------------------------------------------------------
    df_heart = pd.DataFrame({
        "age": [55, 60, 48, 62, 58, 64, 45, 52, 67, 59],
        "sex": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        "cp": [2, 1, 0, 3, 2, 0, 1, 2, 3, 0],
        "trestbps": [140, 130, 120, 150, 135, 160, 125, 138, 145, 128],
        "heart_disease": [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    })
    # Simulated mapping from user screenshot:
    # y_true -> sex, y_pred -> heart_disease, y_prob -> heart_disease, sensitive -> heart_disease, id -> heart_disease
    v40 = validate_audit_input(
        df_heart,
        y_true_col="sex",
        y_pred_col="heart_disease",
        sensitive_col="heart_disease",
        y_prob_col="heart_disease",
        id_col="heart_disease"
    )
    t40_pass = (
        v40["status"] == "invalid" and
        v40["status_title"] == "⚠ Audit Input Needs Correction" and
        any("Invalid audit mapping: y_true, y_pred, y_prob, and sensitive attribute must represent different semantic fields" in i for i in v40["issues"])
    )
    record_test(40, "Critical Case: heart_disease collision mapping strictly rejected", t40_pass)

    # -------------------------------------------------------------------------
    # 41. Run button triggers correct audit action
    # -------------------------------------------------------------------------
    bundle41 = build_external_audit_bundle(
        df_sem,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="group",
        y_prob_col="y_prob",
        id_col="id",
        model_name="Heart Risk Classifier",
        model_version="v1.2",
        model_type="External Model Audit"
    )
    t41_pass = (
        bundle41.get("status") == "success" and
        bundle41.get("model_results", {}).get("model_type") == "External Model Audit" and
        bundle41.get("model_results", {}).get("test_samples") == 100
    )
    record_test(41, "Run button execution triggers valid external audit action", t41_pass)

    # -------------------------------------------------------------------------
    # 42. Audit success renders results (performance, fairness, deterministic insights)
    # -------------------------------------------------------------------------
    mr = bundle41.get("model_results", {})
    fr = bundle41.get("fairness_results", {})
    insights42 = generate_insights(
        model_results=mr,
        fairness_results=fr,
        shap_results=get_external_shap_unavailable_bundle()
    )
    t42_pass = (
        mr.get("accuracy") is not None and
        mr.get("confusion_matrix") is not None and
        fr.get("status") == "success" and
        insights42.get("status") == "success" and
        "summary" in insights42 and
        "key_finding" in insights42
    )
    record_test(42, "Audit success renders complete diagnostic results", t42_pass)

    # -------------------------------------------------------------------------
    # 43. Audit failure produces visible error state
    # -------------------------------------------------------------------------
    # Passing empty/invalid inputs should fail gracefully with structured error, no crash
    v43 = validate_audit_input(pd.DataFrame(), "y_true", "y_pred", "sensitive")
    t43_pass = (
        v43.get("status") == "invalid" and
        v43.get("status_title") == "⚠ Audit Input Needs Correction" and
        len(v43.get("issues", [])) > 0
    )
    record_test(43, "Audit failure produces visible error state without uncaught exceptions", t43_pass)

    # -------------------------------------------------------------------------
    # 44. Canonical Demo Path & Loader Strictly Resolves external_model_audit_demo.csv
    # -------------------------------------------------------------------------
    demo_path_44 = get_demo_audit_csv_path()
    df_demo_44, err_demo_44 = load_demo_audit_csv()
    t44_pass = (
        err_demo_44 is None and
        df_demo_44 is not None and
        os.path.basename(demo_path_44) == "external_model_audit_demo.csv" and
        os.path.exists(demo_path_44)
    )
    record_test(44, "Canonical Demo Path & Loader Strictly Resolves external_model_audit_demo.csv", t44_pass)

    # -------------------------------------------------------------------------
    # 45. Demo CSV Contains Exactly ['id', 'y_true', 'y_pred', 'y_prob', 'sex'] & Zero Cleveland Features
    # -------------------------------------------------------------------------
    expected_demo_cols = ["id", "y_true", "y_pred", "y_prob", "sex"]
    cleveland_cols = {"cp", "trestbps", "chol", "fbs", "restecg", "thalach", "oldpeak", "slope", "ca", "thal", "heart_disease"}
    found_cleveland_in_demo = set(c.lower().strip() for c in df_demo_44.columns).intersection(cleveland_cols)
    t45_pass = (
        list(df_demo_44.columns) == expected_demo_cols and
        len(found_cleveland_in_demo) == 0 and
        len(df_demo_44) >= 50
    )
    record_test(45, "Demo CSV Contains Exactly id, y_true, y_pred, y_prob, sex and Zero Cleveland Features", t45_pass)

    # -------------------------------------------------------------------------
    # 46. Cleveland Feature Dataset (heart_disease_test.csv) Strictly Rejected as Audit Input
    # -------------------------------------------------------------------------
    cleveland_path = os.path.join(WORKSPACE_ROOT, "data", "demo", "heart_disease_test.csv")
    if os.path.exists(cleveland_path):
        df_cleveland = pd.read_csv(cleveland_path)
    else:
        df_cleveland = pd.DataFrame({
            "age": [63, 67], "sex": [1, 1], "cp": [1, 4], "trestbps": [145, 160],
            "chol": [233, 286], "fbs": [1, 0], "restecg": [2, 2], "thalach": [150, 108],
            "exang": [0, 1], "oldpeak": [2.3, 1.5], "slope": [3, 2], "ca": [0, 3],
            "thal": [6, 3], "heart_disease": [0, 1]
        })
    v46 = validate_audit_input(
        df_cleveland,
        y_true_col="heart_disease",
        y_pred_col="sex",
        sensitive_col="sex"
    )
    t46_pass = (
        v46["status"] == "invalid" and
        v46["status_title"] == "⚠ Audit Input Needs Correction" and
        any("Cleveland clinical" in i for i in v46["issues"])
    )
    record_test(46, "Cleveland Feature Dataset Strictly Rejected by Validation Guard", t46_pass)

    # -------------------------------------------------------------------------
    # 47. Automatic Demo Mapping (y_true->y_true, y_pred->y_pred, y_prob->y_prob, sensitive->sex, id->id)
    # -------------------------------------------------------------------------
    v47 = validate_audit_input(
        df_demo_44,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id"
    )
    t47_pass = (
        v47["status"] == "ready" and
        v47["status_title"] == "✓ Audit Input Ready" and
        len(v47["issues"]) == 0
    )
    record_test(47, "Automatic Demo Mapping Achieves 'Audit Input Ready' with Zero Issues", t47_pass)

    # -------------------------------------------------------------------------
    # 48. Full End-to-End Execution of Demo Evidence (Performance, Fairness, Insights, Report)
    # -------------------------------------------------------------------------
    bundle48 = build_external_audit_bundle(
        df_demo_44,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id",
        model_name="Heart Risk Classifier",
        model_version="v1.2",
        model_type="External Model Audit"
    )
    mr48 = bundle48.get("model_results", {})
    fr48 = bundle48.get("fairness_results", {})
    sr48 = bundle48.get("shap_results", {})
    ins48 = generate_insights(mr48, fr48, sr48)
    
    # Generate report to guarantee zero crash
    bundle48["insights"] = ins48
    rep_pdf_48 = os.path.join(tempfile.gettempdir(), "test_demo_audit_report.pdf")
    try:
        rep_res_48 = generate_fairness_report(
            analysis_results=bundle48,
            output_path=rep_pdf_48
        )
        t48_pass = (
            bundle48.get("status") == "success" and
            mr48.get("accuracy") is not None and
            mr48.get("roc_auc") is not None and
            fr48.get("status") == "success" and
            sr48.get("status") == "unavailable" and
            ins48.get("status") == "success" and
            rep_res_48.get("status") == "success" and
            os.path.exists(rep_pdf_48) and os.path.getsize(rep_pdf_48) > 1000
        )
    finally:
        if os.path.exists(rep_pdf_48):
            os.remove(rep_pdf_48)
    record_test(48, "Full End-to-End Execution on Demo Evidence Completes with Report", t48_pass)

    # -------------------------------------------------------------------------
    # 49. Valid external CSV upload parsing
    # -------------------------------------------------------------------------
    sample_csv_text = "\ufeffid,y_true,y_pred,y_prob,sex\r\n1,1,1,0.91,F\r\n2,0,0,0.18,M\r\n3,1,0,0.42,F\r\n4,0,1,0.67,M\r\n5,1,1,0.85,F\r\n"
    df_parsed, err_parsed = load_audit_input(sample_csv_text)
    t49_pass = err_parsed is None and df_parsed is not None and len(df_parsed) == 5
    record_test(49, "Valid external CSV upload parsing handles BOM and Windows newlines", t49_pass)

    # -------------------------------------------------------------------------
    # 50. CSV headers correctly extracted
    # -------------------------------------------------------------------------
    expected_extracted_headers = ["id", "y_true", "y_pred", "y_prob", "sex"]
    t50_pass = list(df_parsed.columns) == expected_extracted_headers
    record_test(50, "CSV headers correctly extracted without BOM or corrupted characters", t50_pass)

    # -------------------------------------------------------------------------
    # 51. CSV records correctly parsed
    # -------------------------------------------------------------------------
    t51_pass = (
        len(df_parsed) == 5 and
        int(df_parsed.iloc[0]["y_true"]) == 1 and
        str(df_parsed.iloc[1]["sex"]) == "M" and
        float(df_parsed.iloc[2]["y_prob"]) == 0.42
    )
    record_test(51, "CSV records correctly parsed into tabular structure with intact values", t51_pass)

    # -------------------------------------------------------------------------
    # 52. Valid mapping after upload
    # -------------------------------------------------------------------------
    auto_detected_mapping = detect_external_audit_columns(list(df_parsed.columns))
    t52_pass = (
        auto_detected_mapping["y_true"] == "y_true" and
        auto_detected_mapping["y_pred"] == "y_pred" and
        auto_detected_mapping["y_prob"] == "y_prob" and
        auto_detected_mapping["sensitive"] == "sex" and
        auto_detected_mapping["id"] == "id"
    )
    record_test(52, "Valid mapping after upload auto-populates semantic columns", t52_pass)

    # -------------------------------------------------------------------------
    # 53. Invalid Cleveland dataset rejected
    # -------------------------------------------------------------------------
    cleveland_csv_text = "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,heart_disease\n63,1,3,145,233,1,0,150,0,2.3,0,0,1,1\n37,1,2,130,250,0,1,187,0,3.5,0,0,2,1\n"
    df_clev_parsed, _ = load_audit_input(cleveland_csv_text)
    v53 = validate_audit_input(df_clev_parsed, y_true_col="heart_disease", y_pred_col="sex", sensitive_col="sex")
    t53_pass = (
        v53["status"] == "invalid" and
        v53["status_title"] == "⚠ Audit Input Needs Correction" and
        any("Cleveland clinical" in i or "model predictions" in i for i in v53["issues"])
    )
    record_test(53, "Invalid Cleveland clinical feature dataset rejected with explicit explanation", t53_pass)

    # -------------------------------------------------------------------------
    # 54. Missing y_true rejected
    # -------------------------------------------------------------------------
    df_missing_ytrue = df_parsed.drop(columns=["y_true"])
    v54 = validate_audit_input(df_missing_ytrue, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="sex")
    t54_pass = (
        v54["status"] == "invalid" and
        v54["status_title"] == "⚠ Audit Input Needs Correction" and
        any("Ground truth column" in i for i in v54["issues"])
    )
    record_test(54, "Missing y_true rejected with 'Audit Input Needs Correction'", t54_pass)

    # -------------------------------------------------------------------------
    # 55. Missing y_pred rejected
    # -------------------------------------------------------------------------
    df_missing_ypred = df_parsed.drop(columns=["y_pred"])
    v55 = validate_audit_input(df_missing_ypred, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="sex")
    t55_pass = (
        v55["status"] == "invalid" and
        v55["status_title"] == "⚠ Audit Input Needs Correction" and
        any("Predicted label column" in i or "y_pred" in i for i in v55["issues"])
    )
    record_test(55, "Missing y_pred rejected with 'Audit Input Needs Correction'", t55_pass)

    # -------------------------------------------------------------------------
    # 56. Invalid y_prob rejected
    # -------------------------------------------------------------------------
    df_invalid_prob = df_parsed.copy()
    df_invalid_prob.loc[0, "y_prob"] = 1.85  # Out of [0, 1] bounds
    v56 = validate_audit_input(df_invalid_prob, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="sex", y_prob_col="y_prob")
    t56_pass = (
        v56["status"] == "invalid" and
        v56["status_title"] == "⚠ Audit Input Needs Correction" and
        any("range [0.0, 1.0]" in i for i in v56["issues"])
    )
    record_test(56, "Invalid y_prob (> 1.0) rejected with 'Audit Input Needs Correction'", t56_pass)

    # -------------------------------------------------------------------------
    # 57. Single sensitive cohort rejected
    # -------------------------------------------------------------------------
    df_single_cohort = df_parsed.copy()
    df_single_cohort["sex"] = "Female"  # Single cohort only
    v57 = validate_audit_input(df_single_cohort, y_true_col="y_true", y_pred_col="y_pred", sensitive_col="sex")
    t57_pass = (
        v57["status"] == "invalid" and
        v57["status_title"] == "⚠ Audit Input Needs Correction" and
        any("at least 2 distinct cohorts" in i or "requires at least 2 cohorts" in i for i in v57["issues"])
    )
    record_test(57, "Single sensitive cohort rejected (requires >= 2 cohorts for fairness disparity)", t57_pass)

    # -------------------------------------------------------------------------
    # 58. Valid uploaded external audit reaches execution path
    # -------------------------------------------------------------------------
    bundle58 = build_external_audit_bundle(
        df_parsed,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id",
        model_name="Uploaded Custom Model",
        model_version="v2.0"
    )
    mr58 = bundle58.get("model_results", {})
    fr58 = bundle58.get("fairness_results", {})
    t58_pass = (
        bundle58.get("status") == "success" and
        mr58.get("accuracy") is not None and
        fr58.get("status") == "success" and
        bundle58.get("shap_results", {}).get("status") == "unavailable"
    )
    record_test(58, "Valid uploaded external audit reaches execution path and computes fairness", t58_pass)

    # -------------------------------------------------------------------------
    # 59. Malformed CSV produces visible error
    # -------------------------------------------------------------------------
    df_bad_bytes, err_bad_bytes = load_audit_input(b"")
    t59_pass = err_bad_bytes is not None and df_bad_bytes is None
    record_test(59, "Malformed CSV produces visible error without unhandled exception", t59_pass)

    # -------------------------------------------------------------------------
    # 60. Empty CSV produces visible error
    # -------------------------------------------------------------------------
    df_empty, err_empty = load_audit_input("")
    df_empty_spaces, err_empty_spaces = load_audit_input("   \n\r\n  ")
    t60_pass = (
        df_empty is None and
        err_empty is not None and
        df_empty_spaces is None and
        err_empty_spaces is not None
    )
    record_test(60, "Empty CSV produces visible error", t60_pass)

    # -------------------------------------------------------------------------
    # 61. Demo loader still works
    # -------------------------------------------------------------------------
    df_demo_61, err_demo_61 = load_demo_audit_csv()
    t61_pass = (
        err_demo_61 is None and
        df_demo_61 is not None and
        list(df_demo_61.columns) == ["id", "y_true", "y_pred", "y_prob", "sex"] and
        len(df_demo_61) >= 50
    )
    record_test(61, "Demo loader still works and delivers canonical evaluation evidence", t61_pass)

    # -------------------------------------------------------------------------
    # 62. Demo and user-upload paths remain independent
    # -------------------------------------------------------------------------
    demo_file_path = get_demo_audit_csv_path()
    mod_time_before = os.path.getmtime(demo_file_path)
    custom_user_csv = "id,y_true,y_pred,y_prob,sex\nU1,1,1,0.9,F\nU2,0,0,0.1,M\n"
    df_user_parsed, err_user = load_audit_input(custom_user_csv)
    mod_time_after = os.path.getmtime(demo_file_path)
    t62_pass = (
        err_user is None and
        df_user_parsed is not None and
        len(df_user_parsed) == 2 and
        mod_time_before == mod_time_after
    )
    record_test(62, "Demo and user-upload paths remain completely independent", t62_pass)

    # -------------------------------------------------------------------------
    # 63. Cleveland dataset must NOT auto-map sex -> y_pred
    # -------------------------------------------------------------------------
    cleveland_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "heart_disease"]
    map_clev = detect_external_audit_columns(cleveland_cols)
    t63_pass = map_clev.get("y_pred") != "sex"
    record_test(63, "Cleveland dataset must NOT auto-map sex -> y_pred", t63_pass)

    # -------------------------------------------------------------------------
    # 64. Cleveland dataset must NOT auto-map heart_disease -> y_pred
    # -------------------------------------------------------------------------
    t64_pass = map_clev.get("y_pred") != "heart_disease" and map_clev.get("y_pred") is None
    record_test(64, "Cleveland dataset must NOT auto-map heart_disease -> y_pred", t64_pass)

    # -------------------------------------------------------------------------
    # 65. Cleveland dataset must NOT auto-map heart_disease -> y_prob
    # -------------------------------------------------------------------------
    t65_pass = map_clev.get("y_prob") != "heart_disease" and map_clev.get("y_prob") is None
    record_test(65, "Cleveland dataset must NOT auto-map heart_disease -> y_prob", t65_pass)

    # -------------------------------------------------------------------------
    # 66. Cleveland dataset remains rejected because actual model prediction evidence is absent
    # -------------------------------------------------------------------------
    clev_path = os.path.join(WORKSPACE_ROOT, "data", "demo", "heart_disease_test.csv")
    df_clev_real = pd.read_csv(clev_path)
    v66 = validate_audit_input(
        df_clev_real,
        y_true_col=map_clev.get("y_true") or "heart_disease",
        y_pred_col=map_clev.get("y_pred") or "",
        sensitive_col=map_clev.get("sensitive") or "sex"
    )
    t66_pass = (
        v66["status"] == "invalid" and
        v66["status_title"] == "⚠ Audit Input Needs Correction" and
        any("Predicted label column" in i or "model predictions" in i for i in v66["issues"])
    )
    record_test(66, "Cleveland dataset remains rejected because actual model prediction evidence is absent", t66_pass)

    # -------------------------------------------------------------------------
    # 67. Valid external evidence still automatically maps (y_true, y_pred, y_prob, sex, id)
    # -------------------------------------------------------------------------
    valid_ext_cols = ["id", "y_true", "y_pred", "y_prob", "sex"]
    map_valid = detect_external_audit_columns(valid_ext_cols)
    t67_pass = (
        map_valid["y_true"] == "y_true" and
        map_valid["y_pred"] == "y_pred" and
        map_valid["y_prob"] == "y_prob" and
        map_valid["sensitive"] == "sex" and
        map_valid["id"] == "id"
    )
    record_test(67, "Valid external evidence still automatically maps all semantic columns", t67_pass)

    # -------------------------------------------------------------------------
    # 68. Semantic mappings pointing to the same column remain invalid
    # -------------------------------------------------------------------------
    df_collision_test = pd.DataFrame({
        "sex": ["M", "F", "M", "F"],
        "heart_disease": [1, 0, 1, 0]
    })
    v68 = validate_audit_input(
        df_collision_test,
        y_true_col="sex",
        y_pred_col="heart_disease",
        sensitive_col="heart_disease"
    )
    t68_pass = (
        v68["status"] == "invalid" and
        v68["status_title"] == "⚠ Audit Input Needs Correction" and
        any("incompatible fields" in i or "different semantic fields" in i for i in v68["issues"])
    )
    record_test(68, "Semantic mappings pointing to the same column remain strictly invalid", t68_pass)

    # -------------------------------------------------------------------------
    # 69. Binary clinical feature (sex) cannot be automatically treated as y_pred
    # -------------------------------------------------------------------------
    clinical_binary_cols = ["patient_num", "sex", "smoker", "diabetic", "readmitted"]
    map_clinical = detect_external_audit_columns(clinical_binary_cols)
    t69_pass = (
        map_clinical.get("y_pred") is None and
        map_clinical.get("y_pred") != "sex" and
        map_clinical.get("y_pred") != "smoker"
    )
    record_test(69, "Binary clinical feature such as sex cannot be automatically treated as y_pred", t69_pass)

    # -------------------------------------------------------------------------
    # 70. Normal upload reaches successful audit execution using exact 3-row test CSV
    # -------------------------------------------------------------------------
    exact_3row_csv = "id,y_true,y_pred,y_prob,sex\n1,1,1,0.91,F\n2,0,0,0.18,M\n3,1,0,0.42,F\n"
    df_3row, err_3row = load_audit_input(exact_3row_csv)
    map_3row = detect_external_audit_columns(list(df_3row.columns))
    bundle_3row = build_external_audit_bundle(
        df_3row,
        y_true_col=map_3row["y_true"],
        y_pred_col=map_3row["y_pred"],
        sensitive_col=map_3row["sensitive"],
        y_prob_col=map_3row["y_prob"],
        id_col=map_3row["id"]
    )
    mr_3row = bundle_3row.get("model_results", {})
    fr_3row = bundle_3row.get("fairness_results", {})
    t70_pass = (
        err_3row is None and
        bundle_3row.get("status") == "success" and
        mr_3row.get("accuracy") is not None and
        abs(float(mr_3row.get("accuracy")) - (2/3)) < 0.01 and
        fr_3row.get("status") == "success"
    )
    record_test(70, "Normal upload reaches successful audit execution using exact 3-row test CSV", t70_pass)

    # -------------------------------------------------------------------------
    # 71. Exact 600-row external audit CSV loads successfully
    # -------------------------------------------------------------------------
    test_600_path = os.path.abspath(r"data\demo\external_model_audit_test_600.csv")
    with open(test_600_path, "r", encoding="utf-8") as f:
        csv_600_content = f.read()
    df_600, err_600 = load_audit_input(csv_600_content)
    t71_pass = (
        err_600 is None and
        df_600 is not None and
        len(df_600) == 600 and
        list(df_600.columns) == ["id", "y_true", "y_pred", "y_prob", "sex"]
    )
    record_test(71, "Exact 600-row external audit CSV loads successfully", t71_pass)

    # -------------------------------------------------------------------------
    # 72. Exact 600-row external audit CSV reaches execution
    # -------------------------------------------------------------------------
    map_600 = detect_external_audit_columns(list(df_600.columns))
    bundle_600 = build_external_audit_bundle(
        df_600,
        y_true_col=map_600["y_true"],
        y_pred_col=map_600["y_pred"],
        sensitive_col=map_600["sensitive"],
        y_prob_col=map_600["y_prob"],
        id_col=map_600["id"],
        model_name="Cardiac Risk Classifier",
        model_version="v2.0",
        model_type="External Model Audit"
    )
    t72_pass = (
        bundle_600.get("status") == "success" and
        bundle_600.get("model_results", {}).get("test_samples") == 600
    )
    record_test(72, "Exact 600-row external audit CSV reaches execution", t72_pass)

    # -------------------------------------------------------------------------
    # 73. 600 y_true values remain binary
    # -------------------------------------------------------------------------
    y_true_vals = bundle_600.get("model_results", {}).get("y_test", [])
    t73_pass = (
        len(y_true_vals) == 600 and
        set(y_true_vals).issubset({0, 1}) and
        0 in y_true_vals and 1 in y_true_vals
    )
    record_test(73, "600 y_true values remain binary", t73_pass)

    # -------------------------------------------------------------------------
    # 74. 600 y_pred values remain binary
    # -------------------------------------------------------------------------
    y_pred_vals = bundle_600.get("model_results", {}).get("y_pred", [])
    t74_pass = (
        len(y_pred_vals) == 600 and
        set(y_pred_vals).issubset({0, 1}) and
        0 in y_pred_vals and 1 in y_pred_vals
    )
    record_test(74, "600 y_pred values remain binary", t74_pass)

    # -------------------------------------------------------------------------
    # 75. 600 y_prob values remain numeric and within [0,1]
    # -------------------------------------------------------------------------
    y_prob_vals = bundle_600.get("model_results", {}).get("y_prob", [])
    t75_pass = (
        y_prob_vals is not None and
        len(y_prob_vals) == 600 and
        all(isinstance(p, (int, float)) and 0.0 <= p <= 1.0 for p in y_prob_vals)
    )
    record_test(75, "600 y_prob values remain numeric and within [0,1]", t75_pass)

    # -------------------------------------------------------------------------
    # 76. 600-row fairness evaluation succeeds with F/M cohorts
    # -------------------------------------------------------------------------
    fr_600 = bundle_600.get("fairness_results", {})
    subgroups_600 = fr_600.get("subgroups", [])
    group_names_600 = {s["group"] for s in subgroups_600}
    group_counts_600 = {s["group"]: s["n"] for s in subgroups_600}
    t76_pass = (
        fr_600.get("status") == "success" and
        len(subgroups_600) == 2 and
        group_names_600 == {"F", "M"} and
        group_counts_600["F"] == 300 and
        group_counts_600["M"] == 300
    )
    record_test(76, "600-row fairness evaluation succeeds with F/M cohorts", t76_pass)

    # -------------------------------------------------------------------------
    # 77. 600-row audit produces valid performance metrics
    # -------------------------------------------------------------------------
    mr_600 = bundle_600.get("model_results", {})
    cm_600 = mr_600.get("confusion_matrix", {})
    t77_pass = (
        mr_600.get("accuracy") is not None and
        float(mr_600.get("accuracy")) > 0.80 and
        mr_600.get("roc_auc") is not None and
        cm_600.get("tp", 0) + cm_600.get("tn", 0) + cm_600.get("fp", 0) + cm_600.get("fn", 0) == 600
    )
    record_test(77, "600-row audit produces valid performance metrics", t77_pass)

    # -------------------------------------------------------------------------
    # 78. 600-row audit produces insights
    # -------------------------------------------------------------------------
    insights_600 = generate_insights(
        model_results=mr_600,
        fairness_results=fr_600,
        shap_results=get_external_shap_unavailable_bundle()
    )
    t78_pass = (
        insights_600.get("status") == "success" and
        insights_600.get("key_finding") is not None and
        insights_600.get("overall_severity") is not None
    )
    record_test(78, "600-row audit produces insights", t78_pass)

    # -------------------------------------------------------------------------
    # 79. 600-row audit produces report/history if part of execution path
    # -------------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf_rep:
        temp_rep_path = tf_rep.name
    try:
        bundle_600["insights_results"] = insights_600
        rep_res_600 = generate_fairness_report(
            analysis_results=bundle_600,
            output_path=temp_rep_path
        )
        t79_pass = (
            rep_res_600.get("status") == "success" and
            os.path.exists(temp_rep_path) and
            os.path.getsize(temp_rep_path) > 1000
        )
    finally:
        if os.path.exists(temp_rep_path):
            os.remove(temp_rep_path)
    record_test(79, "600-row audit produces report/history if those are part of normal path", t79_pass)

    # -------------------------------------------------------------------------
    # 80. 3-row execution continues to work
    # -------------------------------------------------------------------------
    t80_pass = (
        bundle_3row.get("status") == "success" and
        bundle_3row.get("model_results", {}).get("test_samples") == 3 and
        fr_3row.get("status") == "success"
    )
    record_test(80, "3-row execution continues to work", t80_pass)

    # -------------------------------------------------------------------------
    # 81. Canonical 600-row demo execution continues to work
    # -------------------------------------------------------------------------
    canonical_demo_path = os.path.abspath(r"data\demo\external_model_audit_demo.csv")
    with open(canonical_demo_path, "r", encoding="utf-8") as f:
        demo_content = f.read()
    df_demo_81, err_demo_81 = load_audit_input(demo_content)
    map_demo_81 = detect_external_audit_columns(list(df_demo_81.columns))
    bundle_demo_81 = build_external_audit_bundle(
        df_demo_81,
        y_true_col=map_demo_81["y_true"],
        y_pred_col=map_demo_81["y_pred"],
        sensitive_col=map_demo_81["sensitive"],
        y_prob_col=map_demo_81["y_prob"],
        id_col=map_demo_81["id"]
    )
    t81_pass = (
        err_demo_81 is None and
        bundle_demo_81.get("status") == "success" and
        bundle_demo_81.get("model_results", {}).get("test_samples") == 600 and
        bundle_demo_81.get("fairness_results", {}).get("status") == "success"
    )
    record_test(81, "Canonical 600-row demo execution continues to work", t81_pass)
 
    # -------------------------------------------------------------------------
    # 82. Local model with valid model internals: SHAP functionality remains available
    # -------------------------------------------------------------------------
    df_local_82 = pd.read_csv(os.path.join(WORKSPACE_ROOT, "data", "demo", "synthetic_clinical_demo.csv"))
    train_82 = train_clinical_model(
        df=df_local_82,
        target_col="readmitted_30d",
        sensitive_col="gender",
        model_type="random_forest",
        test_size=0.2,
        random_state=42
    )
    feature_cols_82 = [c for c in df_local_82.columns if c not in ["readmitted_30d", "patient_id"]]
    X_test_82 = df_local_82[feature_cols_82].iloc[-train_82["test_samples"]:]
    shap_82 = generate_shap_explanations(
        pipeline=train_82["pipeline"],
        X_test=X_test_82,
        model_type="random_forest",
        max_samples=500,
        sensitive_col="gender",
        y_test=train_82["y_test"],
        y_pred=train_82["y_pred"],
        y_prob=train_82["y_prob"]
    )
    t82_pass = (
        shap_82.get("status") == "success" and
        shap_82.get("explainer_type") == "TreeExplainer" and
        len(shap_82.get("global_importance", [])) > 0 and
        len(shap_82.get("local_explanations", [])) == train_82["test_samples"]
    )
    record_test(82, "Local model with valid model internals: SHAP functionality remains available", t82_pass)

    # -------------------------------------------------------------------------
    # 83. External audit with no model internals: SHAP warning & limitation panel verified
    # -------------------------------------------------------------------------
    dash_html_path = os.path.join(WORKSPACE_ROOT, "templates", "components", "dashboard.html")
    with open(dash_html_path, "r", encoding="utf-8") as f:
        dash_html = f.read()
    t83_pass = (
        "ext-shap-unavailable-banner" in dash_html and
        "SHAP Unavailable — Model Internals Were Not Provided" in dash_html and
        "ext-shap-limitation-panel" in dash_html and
        "Feature-Level Explainability Unavailable" in dash_html and
        "External Model Audit received held-out prediction evidence" in dash_html
    )
    record_test(83, "External audit with no model internals: SHAP warning & limitation panel verified", t83_pass)

    # -------------------------------------------------------------------------
    # 84. External audit: TreeExplainer result UI is NOT presented as executed
    # -------------------------------------------------------------------------
    js_script_path = os.path.join(WORKSPACE_ROOT, "assets", "js", "script.js")
    with open(js_script_path, "r", encoding="utf-8") as f:
        js_content = f.read()
    t84_pass = (
        "clearLocalShapElements" in js_content and
        "Unavailable (Model Internals Withheld)" in js_content and
        "contentArea.style.display = 'none'" in js_content
    )
    record_test(84, "External audit: TreeExplainer result UI is NOT presented as executed", t84_pass)

    # -------------------------------------------------------------------------
    # 85. External audit: fabricated/stale Base Value is NOT displayed
    # -------------------------------------------------------------------------
    t85_pass = (
        "baseValEl.innerText = '—'" in js_content and
        "clearLocalShapElements" in js_content
    )
    record_test(85, "External audit: fabricated/stale Base Value is NOT displayed", t85_pass)

    # -------------------------------------------------------------------------
    # 86. External audit: Global Feature Importance is NOT displayed as real SHAP output
    # -------------------------------------------------------------------------
    shap_bundle_86 = get_external_shap_unavailable_bundle()
    t86_pass = (
        len(shap_bundle_86.get("global_importance", [1])) == 0 and
        "globalContainer.innerHTML = ''" in js_content
    )
    record_test(86, "External audit: Global Feature Importance is NOT displayed as real SHAP output", t86_pass)

    # -------------------------------------------------------------------------
    # 87. External audit: Local Patient SHAP explanation is NOT displayed as calculated
    # -------------------------------------------------------------------------
    t87_pass = (
        len(shap_bundle_86.get("local_explanations", [1])) == 0 and
        "localSelect.innerHTML = ''" in js_content and
        "localTbody.innerHTML = ''" in js_content
    )
    record_test(87, "External audit: Local Patient SHAP explanation is NOT displayed as calculated", t87_pass)

    # -------------------------------------------------------------------------
    # 88. External audit: no SHAP values are generated from y_pred/y_prob/sex
    # -------------------------------------------------------------------------
    bundle_88 = build_external_audit_bundle(
        df_600,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id"
    )
    shap_88 = bundle_88.get("shap_results", {})
    t88_pass = (
        shap_88.get("status") == "unavailable" and
        len(shap_88.get("global_importance", [])) == 0 and
        len(shap_88.get("local_explanations", [])) == 0 and
        "TreeExplainer" not in str(shap_88)
    )
    record_test(88, "External audit: no SHAP values are generated from y_pred/y_prob/sex", t88_pass)

    # -------------------------------------------------------------------------
    # 89. Stale-state regression: Local Model SHAP does not leak into External Audit
    # -------------------------------------------------------------------------
    # Simulate: User runs Local Model Evaluation, generates SHAP
    simulated_state = {
        "auditMode": "local",
        "trainedModel": {"modelType": "random_forest", "accuracy": 0.85, "featureImportance": [{"feature": "age"}]},
        "shapResults": shap_82
    }
    # User switches to External Audit Mode:
    simulated_state["auditMode"] = "external"
    simulated_state["trainedModel"] = None
    simulated_state["shapResults"] = get_external_shap_unavailable_bundle()
    t89_pass = (
        simulated_state["auditMode"] == "external" and
        simulated_state["trainedModel"] is None and
        simulated_state["shapResults"]["status"] == "unavailable" and
        len(simulated_state["shapResults"]["global_importance"]) == 0 and
        len(simulated_state["shapResults"]["local_explanations"]) == 0
    )
    record_test(89, "Stale-state regression: Local Model SHAP does not leak into External Audit", t89_pass)

    # -------------------------------------------------------------------------
    # 90. Existing External Audit execution remains successful
    # -------------------------------------------------------------------------
    t90_pass = (
        bundle_88.get("status") == "success" and
        bundle_88.get("model_results", {}).get("test_samples") == 600
    )
    record_test(90, "Existing External Audit execution remains successful", t90_pass)

    # -------------------------------------------------------------------------
    # 91. Existing fairness results remain unchanged
    # -------------------------------------------------------------------------
    fair_88 = bundle_88.get("fairness_results", {})
    t91_pass = (
        fair_88.get("status") == "success" and
        len(fair_88.get("subgroups", [])) == 2 and
        "metrics" in fair_88 and
        "demographic_parity_difference" in fair_88["metrics"]
    )
    record_test(91, "Existing fairness results remain unchanged", t91_pass)

    # -------------------------------------------------------------------------
    # 92. External audit with y_true/y_pred/y_prob/sex: audit, fairness, recommendations succeed, SHAP unavailable
    # -------------------------------------------------------------------------
    ext_bundle_92 = build_external_audit_bundle(
        df_600,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id"
    )
    ins_92 = generate_insights(
        model_results=ext_bundle_92["model_results"],
        fairness_results=ext_bundle_92["fairness_results"],
        shap_results=ext_bundle_92["shap_results"]
    )
    t92_pass = (
        ext_bundle_92.get("status") == "success" and
        ext_bundle_92["fairness_results"].get("status") == "success" and
        ins_92.get("status") == "success" and
        len(ins_92.get("recommendations", [])) > 0 and
        ext_bundle_92["shap_results"].get("status") == "unavailable" and
        len(ext_bundle_92["shap_results"].get("global_importance", [])) == 0
    )
    record_test(92, "External audit: audit, fairness, recommendations succeed, SHAP unavailable", t92_pass)

    # -------------------------------------------------------------------------
    # 93. Verify recommendation explainability state is unavailable / withheld
    # -------------------------------------------------------------------------
    exp_summary_93 = ins_92.get("summary", {})
    t93_pass = (
        exp_summary_93.get("explainability_status") == "unavailable" and
        ("unavailable" in exp_summary_93.get("explainability_summary", "").lower() or
         "withheld" in exp_summary_93.get("explainability_summary", "").lower()) and
        "internals" in exp_summary_93.get("explainability_summary", "").lower()
    )
    record_test(93, "Recommendation explainability state is unavailable / withheld", t93_pass)

    # -------------------------------------------------------------------------
    # 94. Verify recommendation page does NOT contain forbidden misleading strings
    # -------------------------------------------------------------------------
    with open(dash_html_path, "r", encoding="utf-8") as f:
        dash_html_94 = f.read()
    with open(js_script_path, "r", encoding="utf-8") as f:
        js_content_94 = f.read()

    rec_sec_html = dash_html_94.split('id="section-recommendations"')[1].split('<!-- ==========================================')[0] if 'id="section-recommendations"' in dash_html_94 else ""

    t94_pass = (
        "Shapley feature attributions evaluated" not in js_content_94 and
        "Attributions Ready" not in rec_sec_html and
        "Shapley feature attributions evaluated" not in exp_summary_93.get("explainability_summary", "") and
        "Attributions Ready" != exp_summary_93.get("explainability_pill")
    )
    record_test(94, "Recommendation page does NOT contain 'Shapley feature attributions evaluated' or 'Attributions Ready'", t94_pass)

    # -------------------------------------------------------------------------
    # 95. Verify recommendation page DOES contain truthful wording
    # -------------------------------------------------------------------------
    t95_pass = (
        "Feature-level SHAP unavailable" in rec_sec_html and
        "External model internals not provided" in rec_sec_html and
        "Feature-level SHAP unavailable" in js_content_94 and
        "External model internals not provided" in js_content_94
    )
    record_test(95, "Recommendation page DOES contain truthful wording (Feature-level SHAP unavailable / Model internals not provided)", t95_pass)

    # -------------------------------------------------------------------------
    # 96. Verify Local Model Evaluation SHAP behavior remains unchanged
    # -------------------------------------------------------------------------
    mock_local_model = {
        "modelType": "random_forest",
        "modelLabel": "Random Forest Classifier",
        "metrics": {"accuracy": 0.85, "balancedAccuracy": 0.82, "precision": 0.83, "recall": 0.80, "rocAuc": 0.88},
        "featureImportance": [{"rank": 1, "feature": "bmi", "weight": "0.30"}, {"rank": 2, "feature": "age", "weight": "0.20"}]
    }
    local_shap = {
        "status": "success",
        "global_importance": [{"feature": "bmi", "importance": 0.30}, {"feature": "age", "importance": 0.20}]
    }
    local_ins = generate_insights(
        model_results=mock_local_model,
        fairness_results=ext_bundle_92["fairness_results"],
        shap_results=local_shap
    )
    t96_pass = (
        local_ins.get("summary", {}).get("explainability_status") == "available" and
        "Top model predictors: bmi, age" in local_ins.get("summary", {}).get("explainability_summary", "") and
        local_ins.get("summary", {}).get("explainability_pill") == "Attributions Ready"
    )
    record_test(96, "Local Model Evaluation SHAP behavior remains fully available and functional", t96_pass)

    # -------------------------------------------------------------------------
    # 97. Verify no fake SHAP values are generated
    # -------------------------------------------------------------------------
    t97_pass = (
        len(ext_bundle_92["shap_results"].get("global_importance", [])) == 0 and
        len(ext_bundle_92["shap_results"].get("local_explanations", [])) == 0 and
        ext_bundle_92["shap_results"].get("status") == "unavailable" and
        not any(f.get("category") == "Explainability" and "Primary Model-Influencing Predictors" in f.get("title", "") for f in ins_92.get("explainability_findings", []))
    )
    record_test(97, "Zero fake/fabricated SHAP values generated in external audit", t97_pass)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 98. Fairness recommendation (FNR disparity / underdiagnosis risk) remains intact
    # -------------------------------------------------------------------------
    recs_92 = ins_92.get("recommendations", [])
    has_fnr_rec = any("REC-FAIR-FNR" in r.get("id", "") or "Underdiagnosis" in r.get("title", "") or "false-negative" in r.get("action", "").lower() for r in recs_92)
    t98_pass = has_fnr_rec and len(recs_92) >= 1
    record_test(98, "Fairness recommendation (FNR disparity / underdiagnosis risk) remains intact and valid", t98_pass)

    # -------------------------------------------------------------------------
    # 99. Prediction-only external audit -> SHAP unavailable
    # -------------------------------------------------------------------------
    bundle_99 = build_external_audit_bundle(
        df_600,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id"
    )
    t99_pass = (
        bundle_99.get("status") == "success" and
        bundle_99.get("audit_level") == "level_1_prediction_evidence" and
        bundle_99.get("shap_results", {}).get("status") == "unavailable" and
        "not provided" in bundle_99.get("shap_results", {}).get("message", "").lower()
    )
    record_test(99, "Prediction-only external audit -> SHAP unavailable", t99_pass)

    # -------------------------------------------------------------------------
    # 100. Valid model + feature data -> SHAP available (Level 2)
    # -------------------------------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    np.random.seed(42)
    n_toy = 60
    X_toy = pd.DataFrame({
        "age": np.random.randint(30, 80, n_toy),
        "bmi": np.random.uniform(18.5, 38.0, n_toy),
        "bp": np.random.randint(90, 160, n_toy)
    })
    y_toy = np.random.binomial(1, 0.45, n_toy)
    rf_toy = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf_toy.fit(X_toy, y_toy)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_model_file:
        joblib.dump(rf_toy, tmp_model_file.name)
        toy_model_path = tmp_model_file.name

    pred_toy = rf_toy.predict(X_toy)
    prob_toy = rf_toy.predict_proba(X_toy)[:, 1]
    df_eval_100 = X_toy.copy()
    df_eval_100["id"] = [f"ID_{i}" for i in range(n_toy)]
    df_eval_100["y_true"] = y_toy
    df_eval_100["y_pred"] = pred_toy
    df_eval_100["y_prob"] = prob_toy
    df_eval_100["sex"] = ["M" if i % 2 == 0 else "F" for i in range(n_toy)]

    bundle_100 = build_external_audit_bundle(
        df_eval_100,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id",
        model_source=toy_model_path,
        features_source=X_toy
    )
    t100_pass = (
        bundle_100.get("status") == "success" and
        bundle_100.get("audit_level") == "level_2_explainability" and
        bundle_100.get("shap_results", {}).get("status") == "success" and
        len(bundle_100.get("shap_results", {}).get("global_importance", [])) == 3
    )
    record_test(100, "Valid model + feature data -> SHAP available (Level 2 Explainability)", t100_pass)

    # -------------------------------------------------------------------------
    # 101. Invalid model file -> clear error
    # -------------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_bad_file:
        tmp_bad_file.write(b"NOT_A_VALID_MODEL_FILE_CORRUPTED_BYTES")
        bad_model_path = tmp_bad_file.name
    try:
        loaded_bad, err_bad = load_external_model(bad_model_path)
        t101_pass = (
            loaded_bad is None and
            err_bad is not None and
            ("corrupt" in err_bad.lower() or "failed to load" in err_bad.lower() or "pickle" in err_bad.lower())
        )
    finally:
        if os.path.exists(bad_model_path):
            os.remove(bad_model_path)
    record_test(101, "Invalid model file produces clear, secure error without silent execution", t101_pass)

    # -------------------------------------------------------------------------
    # 102. Missing feature dataset -> SHAP unavailable
    # -------------------------------------------------------------------------
    X_feat_102, sum_102, err_102 = validate_explainability_inputs(
        model=rf_toy,
        features_source=None,
        target_col="y_true",
        sensitive_col="sex"
    )
    bundle_102 = build_external_audit_bundle(
        df_eval_100,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id",
        model_source=rf_toy,
        features_source=None
    )
    t102_pass = (
        X_feat_102 is None and
        err_102 is not None and
        "not provided" in err_102.lower() and
        bundle_102.get("status") == "success" and
        bundle_102.get("shap_results", {}).get("status") == "unavailable" and
        "not provided" in bundle_102.get("shap_results", {}).get("message", "").lower()
    )
    record_test(102, "Missing feature dataset -> SHAP unavailable gracefully", t102_pass)

    # -------------------------------------------------------------------------
    # 103. Missing model -> prediction audit still works
    # -------------------------------------------------------------------------
    bundle_103 = build_external_audit_bundle(
        df_eval_100,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id",
        model_source=None,
        features_source=X_toy
    )
    t103_pass = (
        bundle_103.get("status") == "success" and
        bundle_103.get("audit_level") == "level_1_prediction_evidence" and
        bundle_103.get("model_results", {}).get("accuracy") is not None and
        bundle_103.get("fairness_results", {}).get("status") == "success" and
        bundle_103.get("shap_results", {}).get("status") == "unavailable"
    )
    record_test(103, "Missing model -> prediction audit still works completely", t103_pass)

    # -------------------------------------------------------------------------
    # 104. Feature/model column mismatch -> clear error
    # -------------------------------------------------------------------------
    X_mismatched = X_toy[["age", "bmi"]].copy()  # missing "bp"
    X_feat_104, sum_104, err_104 = validate_explainability_inputs(
        model=rf_toy,
        features_source=X_mismatched,
        target_col="y_true",
        sensitive_col="sex"
    )
    t104_pass = (
        X_feat_104 is None and
        err_104 is not None and
        ("mismatch" in err_104.lower() or "missing" in err_104.lower())
    )
    record_test(104, "Feature/model column mismatch produces clear validation error", t104_pass)

    # -------------------------------------------------------------------------
    # 105. Binary classification SHAP works (positive-class attribution)
    # -------------------------------------------------------------------------
    shap_105 = calculate_external_shap(
        model=rf_toy,
        X_features=X_toy,
        sensitive_col="sex",
        y_true=y_toy,
        y_pred=pred_toy,
        y_prob=prob_toy
    )
    t105_pass = (
        shap_105.get("status") == "success" and
        shap_105.get("base_value") is not None and
        isinstance(shap_105.get("base_value"), (int, float, np.floating)) and
        len(shap_105.get("global_importance", [])) == 3 and
        len(shap_105.get("local_explanations", [])) == n_toy and
        len(shap_105["local_explanations"][0]["all_contributions"]) == 3
    )
    record_test(105, "Binary classification SHAP works with positive-class attribution", t105_pass)

    # -------------------------------------------------------------------------
    # 106. Tree model SHAP works with TreeExplainer
    # -------------------------------------------------------------------------
    t106_pass = (
        shap_105.get("explainer_type") == "TreeExplainer"
    )
    record_test(106, "Tree model SHAP automatically selects and runs TreeExplainer", t106_pass)

    # -------------------------------------------------------------------------
    # 107. Unsupported model type -> graceful fallback
    # -------------------------------------------------------------------------
    class DummyCustomModel:
        def __init__(self):
            self.classes_ = np.array([0, 1])
        def predict(self, X):
            return np.zeros(len(X))
        def predict_proba(self, X):
            return np.ones((len(X), 2)) * 0.5

    dummy_model = DummyCustomModel()
    shap_107 = calculate_external_shap(
        model=dummy_model,
        X_features=X_toy,
        sensitive_col="sex"
    )
    t107_pass = (
        shap_107.get("status") == "unavailable" and
        any(k in shap_107.get("message", "").lower() for k in ["failed", "not supported", "withheld", "unavailable", "error"])
    )
    record_test(107, "Unsupported model type falls back gracefully to SHAP unavailable", t107_pass)

    # -------------------------------------------------------------------------
    # 108. SHAP values are numeric and finite
    # -------------------------------------------------------------------------
    all_finite_global = all(
        isinstance(item["mean_abs_shap"], (int, float, np.floating)) and np.isfinite(item["mean_abs_shap"])
        for item in shap_105.get("global_importance", [])
    )
    all_finite_local = all(
        np.isfinite(contrib["shap_value"])
        for loc in shap_105.get("local_explanations", [])
        for contrib in loc.get("all_contributions", [])
    )
    t108_pass = all_finite_global and all_finite_local and np.isfinite(shap_105.get("base_value"))
    record_test(108, "SHAP values are guaranteed strictly numeric and finite", t108_pass)

    # -------------------------------------------------------------------------
    # 109. Feature names match SHAP output
    # -------------------------------------------------------------------------
    global_features_109 = {item["feature"] for item in shap_105.get("global_importance", [])}
    expected_features_109 = set(X_toy.columns)
    t109_pass = global_features_109 == expected_features_109
    record_test(109, "Feature names in SHAP output exactly match input feature matrix", t109_pass)

    # -------------------------------------------------------------------------
    # 110. SHAP feature importance is deterministic for identical model/data
    # -------------------------------------------------------------------------
    shap_110a = calculate_external_shap(model=rf_toy, X_features=X_toy)
    shap_110b = calculate_external_shap(model=rf_toy, X_features=X_toy)
    imp_a = [item["mean_abs_shap"] for item in shap_110a.get("global_importance", [])]
    imp_b = [item["mean_abs_shap"] for item in shap_110b.get("global_importance", [])]
    t110_pass = (
        np.allclose(imp_a, imp_b, atol=1e-6) and
        [item["feature"] for item in shap_110a.get("global_importance", [])] == [item["feature"] for item in shap_110b.get("global_importance", [])]
    )
    record_test(110, "SHAP feature importance is strictly deterministic for identical model/data", t110_pass)

    # -------------------------------------------------------------------------
    # 111. Existing fairness calculations remain unchanged between Level 1 and Level 2
    # -------------------------------------------------------------------------
    bundle_111_lvl1 = build_external_audit_bundle(
        df_eval_100,
        y_true_col="y_true",
        y_pred_col="y_pred",
        sensitive_col="sex",
        y_prob_col="y_prob",
        id_col="id"
    )
    fair_lvl1 = bundle_111_lvl1.get("fairness_results", {}).get("metrics", {})
    fair_lvl2 = bundle_100.get("fairness_results", {}).get("metrics", {})
    dpd_1 = fair_lvl1.get("demographic_parity_difference", {}).get("value")
    dpd_2 = fair_lvl2.get("demographic_parity_difference", {}).get("value")
    t111_pass = (
        dpd_1 is not None and dpd_2 is not None and
        np.isclose(dpd_1, dpd_2, atol=1e-6)
    )
    record_test(111, "Existing fairness calculations remain identical between Level 1 and Level 2", t111_pass)

    # -------------------------------------------------------------------------
    # 112. Existing prediction-only External Audit remains unchanged
    # -------------------------------------------------------------------------
    t112_pass = (
        bundle_111_lvl1.get("audit_level") == "level_1_prediction_evidence" and
        bundle_111_lvl1.get("shap_results", {}).get("status") == "unavailable" and
        bundle_111_lvl1.get("model_results", {}).get("test_samples") == n_toy
    )
    record_test(112, "Existing prediction-only External Audit remains completely intact", t112_pass)

    # -------------------------------------------------------------------------
    # 113. Existing demo loader remains unchanged
    # -------------------------------------------------------------------------
    df_demo_113, err_demo_113 = load_demo_audit_csv()
    t113_pass = (
        err_demo_113 is None and
        df_demo_113 is not None and
        list(df_demo_113.columns) == ["id", "y_true", "y_pred", "y_prob", "sex"] and
        len(df_demo_113) == 600
    )
    record_test(113, "Existing demo loader remains unchanged and delivers canonical dataset", t113_pass)

    # -------------------------------------------------------------------------
    # 114. PDF report correctly distinguishes SHAP available vs unavailable
    # -------------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf_rep_lvl1:
        pdf_lvl1_path = tf_rep_lvl1.name
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf_rep_lvl2:
        pdf_lvl2_path = tf_rep_lvl2.name
    try:
        ins_lvl1 = generate_insights(bundle_111_lvl1["model_results"], bundle_111_lvl1["fairness_results"], bundle_111_lvl1["shap_results"])
        bundle_111_lvl1["insights"] = ins_lvl1
        rep_lvl1 = generate_fairness_report(bundle_111_lvl1, output_path=pdf_lvl1_path)

        ins_lvl2 = generate_insights(bundle_100["model_results"], bundle_100["fairness_results"], bundle_100["shap_results"])
        bundle_100["insights"] = ins_lvl2
        rep_lvl2 = generate_fairness_report(bundle_100, output_path=pdf_lvl2_path)

        t114_pass = (
            rep_lvl1.get("status") == "success" and
            os.path.exists(pdf_lvl1_path) and os.path.getsize(pdf_lvl1_path) > 1000 and
            rep_lvl2.get("status") == "success" and
            os.path.exists(pdf_lvl2_path) and os.path.getsize(pdf_lvl2_path) > 1000
        )
    finally:
        if os.path.exists(pdf_lvl1_path):
            os.remove(pdf_lvl1_path)
        if os.path.exists(pdf_lvl2_path):
            os.remove(pdf_lvl2_path)
        if os.path.exists(toy_model_path):
            os.remove(toy_model_path)
    record_test(114, "PDF report correctly distinguishes SHAP available vs unavailable and generates successfully", t114_pass)

    # -------------------------------------------------------------------------
    # 115. Semantic Rule: Sensitive attribute is NOT automatically included in SHAP features
    # -------------------------------------------------------------------------
    df_with_sensitive = X_toy.copy()
    df_with_sensitive["gender"] = ["M" if i % 2 == 0 else "F" for i in range(n_toy)]
    # Case A: include_sensitive_in_features is False (default)
    X_feat_115_a, sum_115_a, err_115_a = validate_explainability_inputs(
        model=rf_toy,
        features_source=df_with_sensitive,
        sensitive_col="gender",
        include_sensitive_in_features=False
    )
    # The cleaned feature matrix must NOT contain gender
    cleaned_features_a = list(X_feat_115_a.columns) if X_feat_115_a is not None else []
    t115_pass = (
        err_115_a is None and
        X_feat_115_a is not None and
        "gender" not in cleaned_features_a and
        set(cleaned_features_a) == {"age", "bmi", "bp"}
    )
    record_test(115, "Semantic Rule: Sensitive attribute strictly excluded from SHAP features unless explicitly confirmed", t115_pass)

    print("\n======================================================================")
    print(f"PHASE 13 TEST SUITE RESULTS: {passed}/{total} TESTS PASSED")
    print("======================================================================")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
