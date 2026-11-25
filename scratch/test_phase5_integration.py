import os
import sys
import io
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.train_model import train_clinical_model
from core.fairness import evaluate_fairness_metrics
from core.explainability import generate_shap_explanations

def run_integration_tests():
    print("=== Phase 5 End-to-End Integration Test ===")
    
    # 1. 60-patient realistic dataset
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

    df = pd.read_csv(io.StringIO(csv_data))
    
    # 2. Test Pipeline Execution
    for m_type in ["random_forest", "gradient_boosting", "logistic_regression"]:
        print(f"\n[Test Pipeline] Model: {m_type}")
        
        # Step 3: Train
        t_res = train_clinical_model(
            df=df,
            target_col="heart_disease_outcome",
            sensitive_col="gender",
            model_type=m_type,
            test_size=0.2,
            random_state=42
        )
        assert t_res["status"] == "success"
        print(f"  -> Model Accuracy: {t_res['accuracy']*100:.1f}%, Test size: {t_res['test_samples']}")
        
        # Step 4: Fairness
        f_res = evaluate_fairness_metrics(
            y_true=t_res["y_test"],
            y_pred=t_res["y_pred"],
            y_prob=t_res["y_prob"],
            sensitive_features=t_res["sensitive_test"],
            target_name="heart_disease_outcome",
            sensitive_name="gender"
        )
        assert f_res["status"] == "success"
        print(f"  -> Fairness Verdict: {f_res['overall_status']} ({f_res['verdict_title']})")
        
        # Step 5: Explainability (SHAP)
        drop_cols = ["heart_disease_outcome", "patient_id"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X_test_df = df[feature_cols].iloc[-t_res["test_samples"]:]
        
        shap_res = generate_shap_explanations(
            pipeline=t_res["pipeline"],
            X_test=X_test_df,
            model_type=m_type,
            max_samples=500,
            sensitive_col="gender",
            y_test=t_res["y_test"],
            y_pred=t_res["y_pred"],
            y_prob=t_res["y_prob"]
        )
        assert shap_res["status"] == "success"
        assert len(shap_res["global_importance"]) > 0
        assert len(shap_res["local_explanations"]) == t_res["test_samples"]
        assert "causality_disclaimer" in shap_res["clinical_safety"]
        assert "fairness_vs_explainability" in shap_res["clinical_safety"]
        
        print(f"  -> SHAP Explainer: {shap_res['explainer_type']}")
        print(f"  -> Global top feature: {shap_res['global_importance'][0]['feature']} (mean |SHAP|={shap_res['global_importance'][0]['mean_abs_shap']})")
        print(f"  -> Local patient #1 base={shap_res['local_explanations'][0]['base_value']}, prob={shap_res['local_explanations'][0]['predicted_probability']}")
        print(f"  -> Sensitive attribute used in model: {shap_res['sensitive_attribute_context']['included_in_model_features']}")

    # 3. Verify HTML & JS assets exist and contain Section 5 IDs
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dash_html_path = os.path.join(base_dir, "templates", "components", "dashboard.html")
    with open(dash_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    assert "section-explain" in html_content, "Missing section-explain in dashboard.html"
    assert "shap-global-bars-container" in html_content, "Missing shap-global-bars-container in dashboard.html"
    assert "local-patient-select" in html_content, "Missing local-patient-select in dashboard.html"
    assert "local-shap-detail-table" in html_content, "Missing local-shap-detail-table in dashboard.html"
    print("\n[OK] dashboard.html contains all Phase 5 Section 5 elements.")
    
    js_path = os.path.join(base_dir, "assets", "js", "script.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()
        
    assert "runExplainabilityAnalysis" in js_content, "Missing runExplainabilityAnalysis in script.js"
    assert "renderExplainabilityResults" in js_content, "Missing renderExplainabilityResults in script.js"
    assert "onSelectPatientExplanation" in js_content, "Missing onSelectPatientExplanation in script.js"
    print("[OK] script.js contains all Phase 5 interactive functions.")

    css_path = os.path.join(base_dir, "assets", "css", "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
        
    assert "shap-global-card" in css_content, "Missing shap-global-card in style.css"
    assert "local-explainer-card" in css_content, "Missing local-explainer-card in style.css"
    assert "contrib-val-badge" in css_content, "Missing contrib-val-badge in style.css"
    print("[OK] style.css contains all Phase 5 Explainability styles.")

    print("\n=== [ALL PHASE 5 INTEGRATION TESTS PASSED 100%] ===")

if __name__ == "__main__":
    run_integration_tests()
