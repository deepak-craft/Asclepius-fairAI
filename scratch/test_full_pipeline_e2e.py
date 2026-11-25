"""
Asclepius FairAI - Full End-to-End Pipeline Integration Test (Phases 1 through 9)
Validates complete clinical AI pipeline:
Data Ingestion -> Model Training -> Fairness Audit -> SHAP Explainability -> AI Insights -> Mitigation Optimization -> PDF Report -> SQLite History
"""

import os
import sys
import numpy as np
import pandas as pd
import datetime

# Add workspace to path
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.core.train_model import train_clinical_model
from src.core.fairness import evaluate_fairness_metrics
from src.core.explainability import generate_shap_explanations
from src.core.insights import generate_insights
from src.core.mitigation import evaluate_threshold_candidates
from src.core.report_generator import generate_fairness_report
from src.core.history import initialize_history_db, save_analysis, get_analysis, get_history_stats


def run_full_pipeline_e2e():
    print("=" * 80)
    print("ASCLEPIUS FAIRAI - FULL 9-PHASE PIPELINE INTEGRATION TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Generate Synthetic Clinical ICU Cohort Dataset
    # -------------------------------------------------------------------------
    print("\n[Step 1] Creating Synthetic Clinical ICU Dataset...")
    np.random.seed(42)
    n = 600
    genders = np.random.choice(["Female", "Male"], size=n, p=[0.52, 0.48])
    ages = np.random.randint(25, 88, size=n)
    los = np.random.exponential(scale=3.5, size=n) + 1.0
    sofa = np.random.poisson(lam=2.5, size=n)
    charlson = np.random.poisson(lam=1.5, size=n)

    # Outcome with realistic demographic disparity (higher baseline readmission in Male cohort)
    logits = -3.5 + 0.03 * ages + 0.15 * los + 0.25 * sofa + 0.20 * charlson + (genders == "Male") * 0.75
    probs = 1 / (1 + np.exp(-logits))
    readmission_30d = np.random.binomial(1, probs)

    df = pd.DataFrame({
        "patient_id": [f"ICU-{10000+i}" for i in range(n)],
        "gender": genders,
        "age": ages,
        "icu_los_days": np.round(los, 1),
        "sofa_score": sofa,
        "charlson_comorbidity": charlson,
        "readmission_30d": readmission_30d,
    })

    print(f" -> Generated DataFrame shape: {df.shape}")
    print(f" -> Readmission target prevalence: {df['readmission_30d'].mean()*100:.1f}%")

    # -------------------------------------------------------------------------
    # STEP 2: Phase 3 - Train Clinical Model (Random Forest)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Executing Phase 3: Train Clinical Model (Random Forest)...")
    model_bundle = train_clinical_model(
        df=df,
        target_col="readmission_30d",
        sensitive_col="gender",
        model_type="random_forest",
        test_size=0.20,
        random_state=42,
    )
    assert model_bundle["status"] == "success", f"Training failed: {model_bundle.get('error_message')}"
    print(f" -> Accuracy: {model_bundle['accuracy']*100:.1f}% | ROC-AUC: {model_bundle['roc_auc']:.3f} | F1: {model_bundle['f1']:.3f}")

    # -------------------------------------------------------------------------
    # STEP 3: Phase 4 - Algorithmic Fairness & Bias Evaluation
    # -------------------------------------------------------------------------
    print("\n[Step 3] Executing Phase 4: Fairness & Algorithmic Bias Evaluation...")
    fairness_bundle = evaluate_fairness_metrics(
        y_true=model_bundle["y_test"],
        y_pred=model_bundle["y_pred"],
        y_prob=model_bundle["y_prob"],
        sensitive_features=model_bundle["sensitive_test"],
        target_name="30-Day ICU Readmission",
        sensitive_name="Gender Cohort",
    )
    assert fairness_bundle["status"] == "success"
    print(f" -> Demographic Parity Diff: {fairness_bundle['metrics']['demographic_parity_difference']['value']:.3f} ({fairness_bundle['metrics']['demographic_parity_difference']['status']})")
    print(f" -> Disparate Impact Ratio:  {fairness_bundle['metrics']['disparate_impact_ratio']['value']:.2f} ({fairness_bundle['metrics']['disparate_impact_ratio']['status']})")
    print(f" -> Verdict: {fairness_bundle['verdict_title']}")

    # -------------------------------------------------------------------------
    # STEP 4: Phase 5 - SHAP Explainability Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 4] Executing Phase 5: SHAP Explainability Analysis...")
    shap_bundle = generate_shap_explanations(
        pipeline=model_bundle["pipeline"],
        X_test=model_bundle["X_test"],
        model_type="random_forest",
        sensitive_col="gender",
    )
    assert shap_bundle["status"] == "success"
    top_feature = shap_bundle["global_importance"][0]
    print(f" -> Top SHAP feature: {top_feature['feature']} ({top_feature['mean_abs_shap']:.4f})")

    # -------------------------------------------------------------------------
    # STEP 5: Phase 6 - Deterministic AI Insights & Recommendations
    # -------------------------------------------------------------------------
    print("\n[Step 5] Executing Phase 6: Deterministic AI Insights & Recommendations...")
    insights_bundle = generate_insights(
        model_results=model_bundle,
        fairness_results=fairness_bundle,
        shap_results=shap_bundle,
    )
    assert insights_bundle["status"] == "success"
    print(f" -> Key Finding: {insights_bundle['key_finding']['title']} [{insights_bundle['key_finding']['severity']}]")
    recs = insights_bundle.get("recommendations", [])
    print(f" -> Generated {len(recs)} Actionable Recommendations.")

    # -------------------------------------------------------------------------
    # STEP 6: Phase 9 - Fairness Mitigation & Decision Threshold Optimization
    # -------------------------------------------------------------------------
    print("\n[Step 6] Executing Phase 9: Fairness Mitigation & Decision Boundary Optimization...")
    mitigation_bundle = evaluate_threshold_candidates(
        y_true=model_bundle["y_test"],
        y_prob=model_bundle["y_prob"],
        sensitive_features=model_bundle["sensitive_test"],
        target_name="30-Day ICU Readmission",
        sensitive_name="Gender Cohort",
        threshold_min=0.20,
        threshold_max=0.80,
        step_size=0.05,
        max_performance_drop=0.05,
        objective="balance_fairness_performance",
    )
    assert mitigation_bundle["status"] == "success"
    sel_cand = mitigation_bundle["selection"]["selected_candidate"]
    print(f" -> Evaluated {len(mitigation_bundle['candidates'])} Candidate Thresholds.")
    print(f" -> Selected Candidate: Threshold = {sel_cand['threshold']:.2f}")
    print(f" -> Mitigation Verdict: {mitigation_bundle['mitigation_verdict']}")
    print(f" -> Performance Delta: {sel_cand['deltas']['balanced_accuracy_pp']:+.1f} pp | FNR Disparity Delta: {sel_cand['deltas']['fnr_disparity']:+.3f}")

    # -------------------------------------------------------------------------
    # STEP 7: Phase 7 - Professional PDF Report Generation (with Mitigation)
    # -------------------------------------------------------------------------
    print("\n[Step 7] Executing Phase 7: Professional PDF Report Generation (with Mitigation)...")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_output_path = os.path.join(workspace_dir, "outputs", "reports", f"asclepius_fairai_mitigated_report_{timestamp_str}.pdf")
    
    analysis_payload = {
        "dataset_info": {"dataset_name": "Synthetic_ICU_Readmissions.csv", "rows": n, "cols": df.shape[1]},
        "model_results": model_bundle,
        "fairness_results": fairness_bundle,
        "shap_results": shap_bundle,
        "insights_results": insights_bundle,
        "mitigation_results": mitigation_bundle,
    }

    report_bundle = generate_fairness_report(analysis_payload, output_path=report_output_path)
    assert report_bundle["status"] == "success"
    assert os.path.exists(report_output_path)
    file_size = os.path.getsize(report_output_path)
    print(f" -> PDF Report successfully compiled: {report_bundle['filename']}")
    print(f" -> File Size: {file_size:,} bytes")

    # -------------------------------------------------------------------------
    # STEP 8: Phase 8 - SQLite Persistent History & Mitigation Linkage
    # -------------------------------------------------------------------------
    print("\n[Step 8] Executing Phase 8: Analysis History & Persistent Results (SQLite)...")
    db_path = initialize_history_db()
    analysis_payload["report_filename"] = report_bundle["filename"]

    save_res = save_analysis(analysis_payload, db_path=db_path)
    assert save_res["status"] == "success"
    analysis_id = save_res["analysis_id"]
    print(f" -> Persisted into SQLite with ID: {analysis_id}")

    # Verify retrieval
    retrieved = get_analysis(analysis_id, db_path=db_path)
    assert retrieved is not None
    assert retrieved["mitigation"] is not None
    assert retrieved["mitigation"]["strategy"] == "Decision Threshold Adjustment"
    print(f" -> Retrieved Record: {retrieved['analysis_id']} | Status: {retrieved['fairness_status']} | Mitigation Verdict: {retrieved['mitigation']['mitigation_verdict']}")

    stats = get_history_stats(db_path=db_path)
    print(f" -> Real DB Stats: Total Analyses = {stats['total_analyses']}, Latest = {stats['latest_analysis']['analysis_id']}")

    print("\n" + "=" * 80)
    print("SUCCESS: ALL 9 PHASES EXECUTED & INTEGRATED FLAWLESSLY IN FULL PIPELINE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_full_pipeline_e2e()
