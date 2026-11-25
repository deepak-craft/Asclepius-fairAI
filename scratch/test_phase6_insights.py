import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.train_model import train_clinical_model
from core.fairness import evaluate_fairness_metrics
from core.explainability import generate_shap_explanations
from core.insights import generate_insights, SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_HIGH_ATTENTION, SEVERITY_INSUFFICIENT


def run_unit_tests():
    print("=== Testing Phase 6: AI Insights & Recommendations (14 Test Scenarios) ===")
    
    base_model_res = {
        "status": "success",
        "model_name": "Random Forest",
        "model_label": "Random Forest Classifier",
        "accuracy": 0.90,
        "balanced_accuracy": 0.88,
        "precision": 0.89,
        "recall": 0.87,
        "specificity": 0.89,
        "f1": 0.88,
        "roc_auc": 0.93,
        "target": "heart_disease_outcome",
        "sensitive_attribute": "gender",
    }
    
    base_fairness_res = {
        "status": "success",
        "metrics": {
            "dpd": {"value": 0.02, "pass": True},
            "dir": {"value": 0.95, "pass": True},
            "eopd": {"value": 0.03, "pass": True},
            "eod": {"value": 0.03, "pass": True},
            "fnr_disparity": {"value": 0.02, "pass": True},
        },
        "has_insufficient_subgroup": False,
        "subgroups": [
            {"group": "Female", "tpr": 0.88, "fnr": 0.12, "n": 60, "selection_rate": 0.45},
            {"group": "Male", "tpr": 0.86, "fnr": 0.14, "n": 60, "selection_rate": 0.47},
        ],
        "reliability": [
            {"group": "Female", "n": 60, "tier": "standard"},
            {"group": "Male", "n": 60, "tier": "standard"},
        ],
        "overall_status": "FAIR_BALANCED",
        "sensitive_name": "gender",
    }
    
    base_shap_res = {
        "status": "success",
        "global_importance": [
            {"feature": "Glucose", "mean_abs_shap": 0.45, "rank": 1},
            {"feature": "Age", "mean_abs_shap": 0.38, "rank": 2},
            {"feature": "Blood_Pressure", "mean_abs_shap": 0.30, "rank": 3},
        ],
        "sensitive_attribute_context": {
            "included_in_model_features": False,
            "sensitive_attribute_name": "gender"
        }
    }

    # Scenario 1: All metrics pass
    print("\n--- Scenario 1: All fairness metrics pass ---")
    ins1 = generate_insights(base_model_res, base_fairness_res, base_shap_res)
    assert ins1["status"] == "success"
    assert ins1["overall_severity"] == SEVERITY_INFO
    assert "Fairness Metrics Within Project Thresholds" in [f["title"] for f in ins1["fairness_findings"]]
    print("[OK] Scenario 1 passed.")

    # Scenario 2: DPD exceeds threshold
    print("\n--- Scenario 2: DPD exceeds threshold ---")
    f_dpd = dict(base_fairness_res)
    f_dpd["metrics"] = dict(base_fairness_res["metrics"])
    f_dpd["metrics"]["dpd"] = {"value": 0.12, "pass": False}
    ins2 = generate_insights(base_model_res, f_dpd, base_shap_res)
    assert ins2["overall_severity"] == SEVERITY_WARNING
    assert any("Demographic Parity Selection Gap" in f["title"] for f in ins2["fairness_findings"])
    assert any(r["id"] == "REC-FAIR-DPD" for r in ins2["recommendations"])
    print("[OK] Scenario 2 passed.")

    # Scenario 3: DIR below 0.80
    print("\n--- Scenario 3: DIR below 0.80 ---")
    f_dir = dict(base_fairness_res)
    f_dir["metrics"] = dict(base_fairness_res["metrics"])
    f_dir["metrics"]["dir"] = {"value": 0.65, "pass": False}
    ins3 = generate_insights(base_model_res, f_dir, base_shap_res)
    assert ins3["overall_severity"] == SEVERITY_WARNING
    assert any("Disparate Impact Ratio" in f["title"] for f in ins3["fairness_findings"])
    assert any(r["id"] == "REC-FAIR-DIR" for r in ins3["recommendations"])
    print("[OK] Scenario 3 passed.")

    # Scenario 4: EOpD exceeds threshold
    print("\n--- Scenario 4: EOpD exceeds threshold ---")
    f_eopd = dict(base_fairness_res)
    f_eopd["metrics"] = dict(base_fairness_res["metrics"])
    f_eopd["metrics"]["eopd"] = {"value": 0.09, "pass": False}
    ins4 = generate_insights(base_model_res, f_eopd, base_shap_res)
    assert any("Equal Opportunity" in f["title"] for f in ins4["fairness_findings"])
    print("[OK] Scenario 4 passed.")

    # Scenario 5: EOD exceeds threshold
    print("\n--- Scenario 5: EOD exceeds threshold ---")
    f_eod = dict(base_fairness_res)
    f_eod["metrics"] = dict(base_fairness_res["metrics"])
    f_eod["metrics"]["eod"] = {"value": 0.08, "pass": False}
    ins5 = generate_insights(base_model_res, f_eod, base_shap_res)
    assert any("Equalized Odds Disparity" in f["title"] for f in ins5["fairness_findings"])
    print("[OK] Scenario 5 passed.")

    # Scenario 6: FNR disparity exceeds threshold (> 0.10 -> HIGH ATTENTION)
    print("\n--- Scenario 6: FNR disparity exceeds threshold ---")
    f_fnr = dict(base_fairness_res)
    f_fnr["metrics"] = dict(base_fairness_res["metrics"])
    f_fnr["metrics"]["fnr_disparity"] = {"value": 0.14, "pass": False}
    ins6 = generate_insights(base_model_res, f_fnr, base_shap_res)
    assert ins6["overall_severity"] == SEVERITY_HIGH_ATTENTION
    assert any("False-Negative Rate Disparity" in f["title"] for f in ins6["fairness_findings"])
    assert any(r["id"] == "REC-FAIR-FNR" for r in ins6["recommendations"])
    print("[OK] Scenario 6 passed.")

    # Scenario 7: n < 30
    print("\n--- Scenario 7: Subgroup sample size n < 30 ---")
    f_small = dict(base_fairness_res)
    f_small["has_insufficient_subgroup"] = True
    f_small["reliability"] = [{"group": "Female", "n": 12, "tier": "insufficient"}]
    ins7 = generate_insights(base_model_res, f_small, base_shap_res)
    assert ins7["overall_severity"] == SEVERITY_INSUFFICIENT
    assert any(r["id"] == "REC-DATA-RELIABILITY" for r in ins7["recommendations"])
    print("[OK] Scenario 7 passed.")

    # Scenario 8: Subgroup sensitivity spread
    print("\n--- Scenario 8: Subgroup sensitivity spread ---")
    f_spread = dict(base_fairness_res)
    f_spread["subgroups"] = [
        {"group": "Female", "tpr": 0.92, "fnr": 0.08, "n": 50},
        {"group": "Male", "tpr": 0.72, "fnr": 0.28, "n": 50},
    ]
    ins8 = generate_insights(base_model_res, f_spread, base_shap_res)
    assert any("Subgroup Sensitivity Variation" in f["title"] for f in ins8["subgroup_findings"])
    print("[OK] Scenario 8 passed.")

    # Scenario 9: Strong model performance
    print("\n--- Scenario 9: Strong model performance ---")
    m_strong = dict(base_model_res)
    m_strong["accuracy"] = 0.95
    m_strong["balanced_accuracy"] = 0.94
    ins9 = generate_insights(m_strong, base_fairness_res, base_shap_res)
    assert any("Robust Overall Classification" in f["title"] for f in ins9["performance_findings"])
    print("[OK] Scenario 9 passed.")

    # Scenario 10: Weak recall (Precision-Recall Gap)
    print("\n--- Scenario 10: Weak recall (Precision-Recall Asymmetry) ---")
    m_asym = dict(base_model_res)
    m_asym["precision"] = 0.92
    m_asym["recall"] = 0.65
    ins10 = generate_insights(m_asym, base_fairness_res, base_shap_res)
    assert any("Precision-Recall Asymmetry" in f["title"] for f in ins10["performance_findings"])
    assert any(r["id"] == "REC-PERF-THRESHOLD" for r in ins10["recommendations"])
    print("[OK] Scenario 10 passed.")

    # Scenario 11: SHAP dominant feature with sensitive attribute included
    print("\n--- Scenario 11: SHAP dominant feature & sensitive attribute inclusion ---")
    s_feat = dict(base_shap_res)
    s_feat["sensitive_attribute_context"] = {"included_in_model_features": True, "sensitive_attribute_name": "gender"}
    ins11 = generate_insights(base_model_res, base_fairness_res, s_feat)
    assert any("Direct Protected Attribute Inclusion" in f["title"] for f in ins11["explainability_findings"])
    assert any(r["id"] == "REC-XAI-SENSITIVE-FEATURE" for r in ins11["recommendations"])
    print("[OK] Scenario 11 passed.")

    # Scenario 12: Missing / None SHAP results
    print("\n--- Scenario 12: Missing SHAP results ---")
    ins12 = generate_insights(base_model_res, base_fairness_res, None)
    assert ins12["status"] == "success"
    assert len(ins12["explainability_findings"]) > 0
    print("[OK] Scenario 12 passed.")

    # Scenario 13: Multiple simultaneous findings
    print("\n--- Scenario 13: Multiple simultaneous findings ---")
    f_multi = dict(base_fairness_res)
    f_multi["metrics"] = {
        "dpd": {"value": 0.15, "pass": False},
        "dir": {"value": 0.60, "pass": False},
        "eopd": {"value": 0.12, "pass": False},
        "eod": {"value": 0.12, "pass": False},
        "fnr_disparity": {"value": 0.16, "pass": False},
    }
    m_multi = dict(base_model_res)
    m_multi["precision"] = 0.90
    m_multi["recall"] = 0.60
    ins13 = generate_insights(m_multi, f_multi, s_feat)
    assert ins13["overall_severity"] == SEVERITY_HIGH_ATTENTION
    assert len(ins13["recommendations"]) >= 3
    print(f"[OK] Scenario 13 passed with {len(ins13['recommendations'])} active recommendations.")

    # Scenario 14: End-to-End Real Cohort Pipeline
    print("\n--- Scenario 14: End-to-End Real Cohort Pipeline ---")
    csv_data = """patient_id,age,gender,race,bmi,systolic_bp,cholesterol,glucose,readmission_30d,heart_disease_outcome
P1001,58,Female,White,28.4,138,240,110,0,1
P1002,64,Male,Black,31.2,145,260,145,1,1
P1003,45,Female,Hispanic,24.1,120,180,95,0,0
P1004,72,Male,White,29.8,155,275,160,1,1
P1005,51,Female,Asian,22.7,118,190,92,0,0
P1006,66,Male,Black,33.5,150,290,175,1,1
P1007,39,Female,White,23.0,115,175,88,0,0
P1008,55,Male,Hispanic,27.9,130,215,105,0,0
P1009,61,Female,Black,30.4,142,250,138,0,1
P1010,48,Male,Asian,25.6,125,200,98,0,0
P1011,70,Female,White,28.1,148,265,150,1,1
P1012,53,Male,White,26.4,132,225,115,0,0
P1013,67,Female,Black,32.0,152,280,165,1,1
P1014,42,Male,Hispanic,24.8,122,185,90,0,0
P1015,59,Female,Asian,23.5,128,210,102,0,0
P1016,63,Male,Black,29.9,140,245,130,0,1
P1017,75,Female,White,30.8,160,295,180,1,1
P1018,49,Male,White,25.2,126,195,96,0,0
P1019,56,Female,Hispanic,27.3,134,230,112,0,0
P1020,68,Male,Asian,26.1,144,255,140,1,1
P1021,52,Female,White,24.5,124,198,94,0,0
P1022,60,Male,Black,34.1,154,285,170,1,1
P1023,44,Female,Asian,21.9,116,170,89,0,0
P1024,71,Male,White,28.7,146,260,148,1,1
P1025,57,Female,Black,31.5,139,248,125,0,1
P1026,46,Male,Hispanic,26.8,128,205,100,0,0
P1027,65,Female,White,29.2,142,252,135,0,1
P1028,50,Male,Asian,23.8,120,188,91,0,0
P1029,73,Female,Black,32.8,158,290,178,1,1
P1030,54,Male,White,27.5,136,235,118,0,0
P1031,62,Female,Hispanic,28.9,140,242,122,0,1
P1032,41,Male,White,23.2,118,178,87,0,0
P1033,69,Female,Asian,24.6,138,228,114,0,0
P1034,58,Male,Black,30.6,148,270,155,1,1
P1035,47,Female,White,25.0,122,192,93,0,0
P1036,66,Male,Hispanic,29.4,144,258,142,1,1
P1037,74,Female,Black,33.1,162,300,185,1,1
P1038,52,Male,Asian,24.0,126,202,99,0,0
P1039,59,Female,White,27.8,135,238,120,0,0
P1040,64,Male,White,28.3,142,250,132,0,1
P1041,43,Female,Black,26.5,125,196,95,0,0
P1042,70,Male,Hispanic,31.0,150,282,168,1,1
P1043,55,Female,Asian,22.4,119,182,90,0,0
P1044,61,Male,Black,32.4,152,278,162,1,1
P1045,48,Female,White,24.9,121,190,92,0,0
P1046,67,Male,White,29.1,145,264,150,1,1
P1047,53,Female,Hispanic,28.0,130,220,108,0,0
P1048,76,Male,Black,34.5,165,310,192,1,1
P1049,40,Female,Asian,21.5,114,168,85,0,0
P1050,58,Male,White,26.9,134,232,116,0,0
P1051,63,Female,Black,30.1,146,262,144,0,1
P1052,47,Male,Hispanic,25.7,124,198,97,0,0
P1053,72,Female,White,29.5,152,276,158,1,1
P1054,51,Male,Asian,23.3,122,186,93,0,0
P1055,68,Female,Black,33.0,156,288,172,1,1
P1056,56,Male,White,27.2,132,228,110,0,0
P1057,45,Female,Hispanic,25.4,120,194,94,0,0
P1058,65,Male,Black,31.8,149,274,156,1,1
P1059,71,Female,White,28.6,147,260,146,1,1
P1060,54,Male,Asian,24.2,126,204,101,0,0"""

    import io
    df = pd.read_csv(io.StringIO(csv_data))
    
    t_res = train_clinical_model(
        df=df,
        target_col="heart_disease_outcome",
        sensitive_col="gender",
        model_type="random_forest",
        test_size=0.2,
        random_state=42
    )
    f_res = evaluate_fairness_metrics(
        y_true=t_res["y_test"],
        y_pred=t_res["y_pred"],
        y_prob=t_res["y_prob"],
        sensitive_features=t_res["sensitive_test"],
        target_name="heart_disease_outcome",
        sensitive_name="gender"
    )
    drop_cols = ["heart_disease_outcome", "patient_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_test_df = df[feature_cols].iloc[-t_res["test_samples"]:]
    
    s_res = generate_shap_explanations(
        pipeline=t_res["pipeline"],
        X_test=X_test_df,
        model_type="random_forest",
        max_samples=500,
        sensitive_col="gender",
        y_test=t_res["y_test"],
        y_pred=t_res["y_pred"],
        y_prob=t_res["y_prob"]
    )
    
    ins_e2e = generate_insights(t_res, f_res, s_res)
    assert ins_e2e["status"] == "success"
    assert "summary" in ins_e2e
    assert "key_finding" in ins_e2e
    assert "recommendations" in ins_e2e
    print(f"[OK] Scenario 14 passed.")
    print(f"     Overall Severity: {ins_e2e['overall_severity']}")
    print(f"     Primary Finding: {ins_e2e['summary']['primary_finding']}")
    print(f"     Recommended Action: {ins_e2e['summary']['recommended_next_action']}")

    print("\n=== [ALL 14 PHASE 6 TEST SCENARIOS PASSED 100%!] ===")

if __name__ == "__main__":
    run_unit_tests()

