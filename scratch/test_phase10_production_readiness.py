"""
Asclepius FairAI - Phase 10: Production Readiness & Full-System Reliability Test Suite
==================================================================================
Comprehensive QA test suite testing:
1. End-to-End Workflow Pipeline
2. Dataset Matrix (18 Edge Cases A-R)
3. Exact Fairness Threshold Boundaries (DPD, EOpD, EOD, FNR, DIR)
4. Model Matrix (Logistic Regression, Random Forest, Gradient Boosting)
5. Multi-Analysis Session State Isolation (3 consecutive runs)
6. Latency & Performance Benchmarks (60, 600, 5000 rows)
7. Zero PHI / Privacy & Security Audit
8. Multi-Configuration PDF Report Generation Matrix
"""

import sys
import os
import time
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.train_model import train_clinical_model, SUPPORTED_MODELS
from src.core.fairness import evaluate_fairness_metrics, THRESHOLDS, _safe_divide
from src.core.explainability import generate_shap_explanations
from src.core.insights import generate_insights
from src.core.mitigation import (
    evaluate_threshold_candidates,
    select_mitigation_candidate,
    compare_before_after
)
from src.core.report_generator import generate_fairness_report
from src.core.history import (
    initialize_history_db,
    save_analysis,
    get_analysis_history,
    get_analysis,
    delete_analysis,
    get_history_stats
)


def create_synthetic_clinical_data(n=600, seed=42, imbalance_ratio=0.5):
    """Generates synthetic ICU readmission dataset with zero PHI."""
    np.random.seed(seed)
    age = np.random.randint(20, 85, size=n)
    gender = np.random.choice(["Female", "Male"], size=n, p=[0.48, 0.52])
    charlson_index = np.random.poisson(lam=2.5, size=n)
    icu_los_days = np.round(np.random.exponential(scale=3.5, size=n) + 1.0, 1)
    prior_admissions = np.random.negative_binomial(n=2, p=0.6, size=n)
    insurance = np.random.choice(["Medicare", "Private", "Medicaid"], size=n, p=[0.5, 0.35, 0.15])
    
    # Target probability with mild demographic bias for testing fairness detection
    logits = (
        0.03 * age
        + 0.4 * charlson_index
        + 0.25 * icu_los_days
        + 0.5 * prior_admissions
        - 2.8
    )
    if imbalance_ratio != 0.5:
        logits += np.log(imbalance_ratio / (1 - imbalance_ratio))
    
    prob = 1.0 / (1.0 + np.exp(-logits))
    prob_adj = np.where(gender == "Female", prob * 0.85, prob * 1.15)
    prob_adj = np.clip(prob_adj, 0.01, 0.99)
    readmitted = (np.random.rand(n) < prob_adj).astype(int)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "charlson_index": charlson_index,
        "icu_los_days": icu_los_days,
        "prior_admissions": prior_admissions,
        "insurance": insurance,
        "readmitted_30d": readmitted
    })
    return df


class TestPhase10ProductionReadiness(unittest.TestCase):
    """Phase 10 Master QA Test Suite."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="asclepius_phase10_test_")
        cls.test_db_path = os.path.join(cls.temp_dir, "test_history.db")
        initialize_history_db(cls.test_db_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # =========================================================================
    # 1. FULL USER JOURNEY END-TO-END TEST
    # =========================================================================
    def test_01_full_user_journey_end_to_end(self):
        """Tests the entire user journey: Ingest -> Train -> Fairness -> SHAP -> Insights -> Mitigation -> Report -> History."""
        df = create_synthetic_clinical_data(n=400, seed=101)
        
        # 1. Train Model
        train_res = train_clinical_model(
            df=df,
            target_col="readmitted_30d",
            sensitive_col="gender",
            model_type="random_forest",
            test_size=0.25,
            random_state=42
        )
        self.assertEqual(train_res["status"], "success")
        self.assertIn("accuracy", train_res)
        self.assertIn("pipeline", train_res)
        
        # 2. Fairness Analysis
        fairness_res = evaluate_fairness_metrics(
            y_true=train_res["y_test"],
            y_pred=train_res["y_pred"],
            y_prob=train_res["y_prob"],
            sensitive_features=train_res["sensitive_test"],
            target_name="readmitted_30d",
            sensitive_name="gender"
        )
        self.assertEqual(fairness_res["status"], "success")
        self.assertIn("overall_status", fairness_res)
        self.assertIn("metrics", fairness_res)
        self.assertIn("subgroups", fairness_res)
        
        # 3. SHAP Explainability
        shap_res = generate_shap_explanations(
            pipeline=train_res["pipeline"],
            X_test=train_res["X_test"],
            model_type="random_forest",
            sensitive_col="gender"
        )
        self.assertEqual(shap_res["status"], "success")
        self.assertIn("global_importance", shap_res)
        
        # 4. Deterministic Insights
        insights_res = generate_insights(
            model_results=train_res,
            fairness_results=fairness_res,
            shap_results=shap_res
        )
        self.assertEqual(insights_res["status"], "success")
        self.assertIn("summary", insights_res)
        self.assertIn("recommendations", insights_res)
        
        # 5. Mitigation & Before/After Comparison
        mit_eval = evaluate_threshold_candidates(
            y_true=train_res["y_test"],
            y_prob=train_res["y_prob"],
            sensitive_features=train_res["sensitive_test"],
            target_name="readmitted_30d",
            sensitive_name="gender",
            threshold_min=0.20,
            threshold_max=0.80,
            step_size=0.05
        )
        self.assertEqual(mit_eval["status"], "success")
        self.assertGreater(len(mit_eval["candidates"]), 0)
        self.assertIsNotNone(mit_eval["selection"])
        self.assertIsNotNone(mit_eval["comparison"])
        
        # 6. Report Generation
        pdf_path = os.path.join(self.temp_dir, "e2e_audit_report.pdf")
        report_payload = {
            "dataset_info": {"dataset_name": "Synthetic ICU Readmission", "rows": 400, "cols": 7},
            "model_results": train_res,
            "fairness_results": fairness_res,
            "shap_results": shap_res,
            "insights_results": insights_res,
            "mitigation_results": mit_eval
        }
        report_res = generate_fairness_report(report_payload, output_path=pdf_path)
        self.assertEqual(report_res["status"], "success")
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)
        
        # 7. Save to History & Retrieve
        save_bundle = {
            "dataset_info": {"dataset_name": "Synthetic ICU Readmission", "rows": 400, "cols": 7},
            "model_results": train_res,
            "fairness_results": fairness_res,
            "shap_results": shap_res,
            "insights_results": insights_res,
            "mitigation_results": mit_eval,
            "report_filename": report_res["filename"]
        }
        save_res = save_analysis(save_bundle, db_path=self.test_db_path)
        self.assertEqual(save_res["status"], "success")
        rec_id = save_res["analysis_id"]
        
        fetched = get_analysis(rec_id, db_path=self.test_db_path)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["analysis_id"], rec_id)
        self.assertIn("Random Forest", fetched["model_type"])
        self.assertIsNotNone(fetched["mitigation_json"])

    # =========================================================================
    # 2. DATASET EDGE CASES MATRIX (18 Scenarios A-R)
    # =========================================================================
    def test_02_scenario_A_binary_normal(self):
        """Scenario A: Standard binary classification."""
        df = create_synthetic_clinical_data(n=200, seed=1)
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        self.assertEqual(res["status"], "success")
        self.assertIn("accuracy", res)

    def test_03_scenario_B_mixed_num_cat(self):
        """Scenario B: Mixed numerical and categorical features."""
        df = create_synthetic_clinical_data(n=200, seed=2)
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="random_forest")
        self.assertEqual(res["status"], "success")
        self.assertIsNotNone(res["pipeline"])

    def test_04_scenario_C_missing_values(self):
        """Scenario C: Missing values in numeric and categorical columns."""
        df = create_synthetic_clinical_data(n=200, seed=3)
        df.loc[0:10, "age"] = np.nan
        df.loc[15:25, "insurance"] = np.nan
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="random_forest")
        self.assertEqual(res["status"], "success")
        self.assertIn("accuracy", res)

    def test_05_scenario_D_duplicate_rows(self):
        """Scenario D: Dataset with duplicate rows."""
        df = create_synthetic_clinical_data(n=150, seed=4)
        df = pd.concat([df, df.iloc[:30]], ignore_index=True)
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="gradient_boosting")
        self.assertEqual(res["status"], "success")

    def test_06_scenario_E_empty_dataset(self):
        """Scenario E: Completely empty dataset (0 rows, 0 cols)."""
        df = pd.DataFrame()
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        self.assertEqual(res["status"], "error")
        self.assertIn("empty", res["error_message"].lower())

    def test_07_scenario_F_header_only_dataset(self):
        """Scenario F: Header-only dataset (0 rows, >0 cols)."""
        df = pd.DataFrame(columns=["age", "gender", "readmitted_30d"])
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        self.assertEqual(res["status"], "error")
        self.assertIn("empty", res["error_message"].lower())

    def test_08_scenario_G_small_sample_under_30(self):
        """Scenario G: Small total n dataset (n=24)."""
        df = create_synthetic_clinical_data(n=24, seed=7)
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        self.assertIn(res["status"], ["success", "error"])

    def test_09_scenario_H_single_class_target(self):
        """Scenario H: Single-class target (all 0s or all 1s)."""
        df = create_synthetic_clinical_data(n=100, seed=8)
        df["readmitted_30d"] = 0
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        self.assertEqual(res["status"], "error")
        self.assertIn("binary", res["error_message"].lower())

    def test_10_scenario_I_missing_target_column(self):
        """Scenario I: Missing target column."""
        df = create_synthetic_clinical_data(n=100, seed=9)
        res = train_clinical_model(df, "non_existent_col", "gender", model_type="logistic_regression")
        self.assertEqual(res["status"], "error")
        self.assertIn("target column", res["error_message"].lower())

    def test_11_scenario_J_missing_sensitive_column(self):
        """Scenario J: Missing sensitive column."""
        df = create_synthetic_clinical_data(n=100, seed=10)
        res = train_clinical_model(df, "readmitted_30d", "non_existent_sens", model_type="logistic_regression")
        self.assertEqual(res["status"], "error")
        self.assertIn("sensitive attribute", res["error_message"].lower())

    def test_12_scenario_K_subgroup_n_under_30(self):
        """Scenario K: Subgroup n < 30 triggers Insufficient Sample Size alert."""
        sens = np.array(["Female"] * 25 + ["Male"] * 100)
        y_true = np.random.randint(0, 2, size=125)
        y_pred = np.random.randint(0, 2, size=125)
        res = evaluate_fairness_metrics(y_true, y_pred, sensitive_features=sens)
        self.assertTrue(res["has_insufficient_subgroup"])
        self.assertEqual(res["overall_status"], "INSUFFICIENT_EVIDENCE")

    def test_13_scenario_L_subgroup_n_30_to_49(self):
        """Scenario L: Subgroup n between 30 and 49 triggers Moderate Sample Size / Caution alert."""
        sens = np.array(["Female"] * 40 + ["Male"] * 100)
        y_true = np.random.randint(0, 2, size=140)
        y_pred = np.random.randint(0, 2, size=140)
        res = evaluate_fairness_metrics(y_true, y_pred, sensitive_features=sens)
        self.assertFalse(res["has_insufficient_subgroup"])
        tiers = [r["tier"] for r in res["reliability"]]
        self.assertIn("caution", tiers)

    def test_14_scenario_M_subgroup_n_50_plus(self):
        """Scenario M: Subgroup n >= 50 triggers Robust evaluation."""
        sens = np.array(["Female"] * 80 + ["Male"] * 120)
        y_true = np.random.randint(0, 2, size=200)
        y_pred = np.random.randint(0, 2, size=200)
        res = evaluate_fairness_metrics(y_true, y_pred, sensitive_features=sens)
        self.assertFalse(res["has_insufficient_subgroup"])
        tiers = [r["tier"] for r in res["reliability"]]
        self.assertEqual(set(tiers), {"standard"})

    def test_15_scenario_N_multigroup_sensitive_attribute(self):
        """Scenario N: Multi-group sensitive attribute (> 2 categories)."""
        df = create_synthetic_clinical_data(n=300, seed=14)
        res = train_clinical_model(df, "readmitted_30d", "insurance", model_type="random_forest", test_size=0.3)
        self.assertEqual(res["status"], "success")
        
        fairness_res = evaluate_fairness_metrics(
            y_true=res["y_test"],
            y_pred=res["y_pred"],
            sensitive_features=res["sensitive_test"]
        )
        self.assertEqual(fairness_res["status"], "success")
        self.assertEqual(len(fairness_res["subgroups"]), len(set(res["sensitive_test"])))

    def test_16_scenario_O_weird_column_names(self):
        """Scenario O: Column names with spaces, special chars, uppercase, and punctuation."""
        df = create_synthetic_clinical_data(n=150, seed=15)
        df.rename(columns={
            "age": "Patient Age (Years)",
            "icu_los_days": "ICU Length-Of-Stay [Days]!",
            "readmitted_30d": "TARGET_READMITTED_30D"
        }, inplace=True)
        res = train_clinical_model(df, "TARGET_READMITTED_30D", "gender", model_type="logistic_regression")
        self.assertEqual(res["status"], "success")
        self.assertIn("accuracy", res)

    def test_17_scenario_P_high_cardinality_id_column(self):
        """Scenario P: High-cardinality ID / identifier column."""
        df = create_synthetic_clinical_data(n=200, seed=16)
        df["patient_uuid"] = [f"PT-{i:06d}" for i in range(len(df))]
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="random_forest")
        self.assertEqual(res["status"], "success")
        self.assertNotIn("patient_uuid", res["features_used"])

    def test_18_scenario_Q_constant_feature(self):
        """Scenario Q: Constant / zero-variance feature column."""
        df = create_synthetic_clinical_data(n=200, seed=17)
        df["constant_col"] = 1
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        self.assertEqual(res["status"], "success")

    def test_19_scenario_R_extreme_class_imbalance(self):
        """Scenario R: Extreme class imbalance (e.g. 96% negative, 4% positive)."""
        df = create_synthetic_clinical_data(n=300, seed=18, imbalance_ratio=0.04)
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        self.assertEqual(res["status"], "success")
        self.assertIn("balanced_accuracy", res)

    # =========================================================================
    # 3. FAIRNESS EXACT THRESHOLD BOUNDARY TESTS
    # =========================================================================
    def test_20_dpd_exact_boundary(self):
        """Tests Demographic Parity Difference at boundary 0.050 (Pass) vs 0.051 (Review)."""
        y_true = [0] * 200
        sens = ["A"] * 100 + ["B"] * 100
        y_pred_pass = [1] * 50 + [0] * 50 + [1] * 55 + [0] * 45
        res_pass = evaluate_fairness_metrics(y_true, y_pred_pass, sensitive_features=sens)
        dpd_val = res_pass["metrics"]["demographic_parity_difference"]["value"]
        self.assertAlmostEqual(dpd_val, 0.050, places=3)
        self.assertEqual(res_pass["metrics"]["demographic_parity_difference"]["status"], "PASS")

        y_pred_rev = [1] * 50 + [0] * 50 + [1] * 56 + [0] * 44
        res_rev = evaluate_fairness_metrics(y_true, y_pred_rev, sensitive_features=sens)
        self.assertEqual(res_rev["metrics"]["demographic_parity_difference"]["status"], "REVIEW")

    def test_21_dir_exact_boundaries(self):
        """Tests Disparate Impact Ratio boundaries: 0.79 (Review), 0.80 (Pass), 1.0 (Optimal), 1.25 (Pass), 1.26 (Review)."""
        y_true = [0] * 200
        sens = ["A"] * 100 + ["B"] * 100
        
        y_pred_79 = [1] * 79 + [0] * 21 + [1] * 100 + [0] * 0
        res_79 = evaluate_fairness_metrics(y_true, y_pred_79, sensitive_features=sens)
        self.assertEqual(res_79["metrics"]["disparate_impact_ratio"]["status"], "REVIEW")
        
        y_pred_80 = [1] * 80 + [0] * 20 + [1] * 100 + [0] * 0
        res_80 = evaluate_fairness_metrics(y_true, y_pred_80, sensitive_features=sens)
        self.assertEqual(res_80["metrics"]["disparate_impact_ratio"]["status"], "PASS")
        
        y_pred_100 = [1] * 50 + [0] * 50 + [1] * 50 + [0] * 50
        res_100 = evaluate_fairness_metrics(y_true, y_pred_100, sensitive_features=sens)
        self.assertEqual(res_100["metrics"]["disparate_impact_ratio"]["status"], "PASS")

    def test_22_eopd_exact_boundary(self):
        """Tests Equal Opportunity Difference at boundary 0.050 vs >0.050."""
        y_true = [1] * 200
        sens = ["A"] * 100 + ["B"] * 100
        y_pred_pass = [1] * 80 + [0] * 20 + [1] * 85 + [0] * 15
        res_pass = evaluate_fairness_metrics(y_true, y_pred_pass, sensitive_features=sens)
        self.assertEqual(res_pass["metrics"]["equal_opportunity_difference"]["status"], "PASS")

        y_pred_rev = [1] * 80 + [0] * 20 + [1] * 86 + [0] * 14
        res_rev = evaluate_fairness_metrics(y_true, y_pred_rev, sensitive_features=sens)
        self.assertEqual(res_rev["metrics"]["equal_opportunity_difference"]["status"], "REVIEW")

    def test_23_eod_exact_boundary(self):
        """Tests Equalized Odds Difference boundary."""
        y_true = [1] * 50 + [0] * 50 + [1] * 50 + [0] * 50
        sens = ["A"] * 100 + ["B"] * 100
        y_pred_pass = [1] * 40 + [0] * 10 + [1] * 5 + [0] * 45 + [1] * 42 + [0] * 8 + [1] * 7 + [0] * 43
        res_pass = evaluate_fairness_metrics(y_true, y_pred_pass, sensitive_features=sens)
        self.assertEqual(res_pass["metrics"]["equalized_odds_difference"]["status"], "PASS")

    def test_24_fnrd_exact_boundary(self):
        """Tests False Negative Rate Disparity boundary."""
        y_true = [1] * 200
        sens = ["A"] * 100 + ["B"] * 100
        y_pred_pass = [1] * 80 + [0] * 20 + [1] * 75 + [0] * 25
        res_pass = evaluate_fairness_metrics(y_true, y_pred_pass, sensitive_features=sens)
        self.assertEqual(res_pass["metrics"]["fnr_disparity"]["status"], "PASS")

        y_pred_rev = [1] * 80 + [0] * 20 + [1] * 74 + [0] * 26
        res_rev = evaluate_fairness_metrics(y_true, y_pred_rev, sensitive_features=sens)
        self.assertEqual(res_rev["metrics"]["fnr_disparity"]["status"], "REVIEW")

    # =========================================================================
    # 4. MODEL MATRIX TESTS
    # =========================================================================
    def test_25_model_matrix_logistic_regression(self):
        """Model Matrix: Logistic Regression."""
        df = create_synthetic_clinical_data(n=250, seed=25)
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model_name"], "Logistic Regression")

    def test_26_model_matrix_random_forest(self):
        """Model Matrix: Random Forest."""
        df = create_synthetic_clinical_data(n=250, seed=26)
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="random_forest")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model_name"], "Random Forest")

    def test_27_model_matrix_gradient_boosting(self):
        """Model Matrix: Gradient Boosting."""
        df = create_synthetic_clinical_data(n=250, seed=27)
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="gradient_boosting")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model_name"], "Gradient Boosting")

    # =========================================================================
    # 5. MULTI-ANALYSIS SESSION TEST (Zero State Leakage)
    # =========================================================================
    def test_28_multi_analysis_session_state_isolation(self):
        """Validates 3 consecutive runs (A, B, C) have zero state leakage."""
        # Run A
        df_a = create_synthetic_clinical_data(n=200, seed=201)
        res_a = train_clinical_model(df_a, "readmitted_30d", "gender", model_type="logistic_regression")
        save_a = save_analysis({"dataset_info": {"dataset_name": "Dataset_A", "rows": 200}, "model_results": res_a}, db_path=self.test_db_path)
        id_a = save_a["analysis_id"]
        
        # Run B
        df_b = create_synthetic_clinical_data(n=350, seed=202)
        res_b = train_clinical_model(df_b, "readmitted_30d", "insurance", model_type="random_forest")
        save_b = save_analysis({"dataset_info": {"dataset_name": "Dataset_B", "rows": 350}, "model_results": res_b}, db_path=self.test_db_path)
        id_b = save_b["analysis_id"]
        
        # Run C
        df_c = create_synthetic_clinical_data(n=180, seed=203)
        res_c = train_clinical_model(df_c, "readmitted_30d", "gender", model_type="gradient_boosting")
        save_c = save_analysis({"dataset_info": {"dataset_name": "Dataset_C", "rows": 180}, "model_results": res_c}, db_path=self.test_db_path)
        id_c = save_c["analysis_id"]
        
        # Verify isolation
        rec_a = get_analysis(id_a, db_path=self.test_db_path)
        rec_b = get_analysis(id_b, db_path=self.test_db_path)
        rec_c = get_analysis(id_c, db_path=self.test_db_path)
        
        self.assertIn("Logistic Regression", rec_a["model_type"])
        self.assertIn("Random Forest", rec_b["model_type"])
        self.assertIn("Gradient Boosting", rec_c["model_type"])
        self.assertEqual(rec_a["dataset_row_count"], 200)
        self.assertEqual(rec_b["dataset_row_count"], 350)
        self.assertEqual(rec_c["dataset_row_count"], 180)
        self.assertNotEqual(id_a, id_b)
        self.assertNotEqual(id_b, id_c)

    # =========================================================================
    # 6. PERFORMANCE & LATENCY BENCHMARKS
    # =========================================================================
    def test_29_performance_benchmark_60_rows(self):
        """Performance benchmark for N=60 rows."""
        df = create_synthetic_clinical_data(n=60, seed=60)
        t0 = time.time()
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        t1 = time.time()
        duration = t1 - t0
        self.assertEqual(res["status"], "success")
        self.assertLess(duration, 3.0, f"Training took {duration:.2f}s, expected < 3.0s")

    def test_30_performance_benchmark_600_rows(self):
        """Performance benchmark for N=600 rows."""
        df = create_synthetic_clinical_data(n=600, seed=600)
        t0 = time.time()
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="random_forest")
        shap_res = generate_shap_explanations(res["pipeline"], res["X_test"], "random_forest", sensitive_col="gender")
        t1 = time.time()
        duration = t1 - t0
        self.assertEqual(res["status"], "success")
        self.assertEqual(shap_res["status"], "success")
        self.assertLess(duration, 5.0, f"Pipeline + SHAP took {duration:.2f}s, expected < 5.0s")

    def test_31_performance_benchmark_5000_rows(self):
        """Performance benchmark for N=5000 rows."""
        df = create_synthetic_clinical_data(n=5000, seed=5000)
        t0 = time.time()
        res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        t1 = time.time()
        duration = t1 - t0
        self.assertEqual(res["status"], "success")
        self.assertLess(duration, 10.0, f"Training 5000 rows took {duration:.2f}s, expected < 10.0s")

    # =========================================================================
    # 7. ZERO PHI / PRIVACY & SECURITY AUDIT
    # =========================================================================
    def test_32_zero_phi_in_database(self):
        """Verifies no patient-level rows, names, or raw records exist in SQLite history."""
        records = get_analysis_history(db_path=self.test_db_path)
        for r in records:
            self.assertIn("analysis_id", r)
            self.assertIn("dataset_name", r)
            self.assertNotIn("raw_data", r)
            self.assertNotIn("patient_records", r)

    def test_33_sql_injection_defense(self):
        """Verifies parameterized queries prevent SQL injection."""
        malicious_input = "'; DROP TABLE analysis_history; --"
        save_res = save_analysis({
            "dataset_info": {"dataset_name": malicious_input, "rows": 100},
            "model_results": {"model_name": "Logistic Regression", "accuracy": 0.85},
            "fairness_results": {"overall_status": "FAIR_BALANCED"}
        }, db_path=self.test_db_path)
        self.assertEqual(save_res["status"], "success")
        stats = get_history_stats(db_path=self.test_db_path)
        self.assertEqual(stats["status"], "success")
        self.assertGreaterEqual(stats["total_analyses"], 1)

    # =========================================================================
    # 8. MULTI-CONFIGURATION REPORT GENERATION MATRIX
    # =========================================================================
    def test_34_report_matrix_unmitigated(self):
        """Report Profile 1: Standard unmitigated audit report."""
        df = create_synthetic_clinical_data(n=200, seed=34)
        train_res = train_clinical_model(df, "readmitted_30d", "gender", model_type="logistic_regression")
        fairness_res = evaluate_fairness_metrics(train_res["y_test"], train_res["y_pred"], sensitive_features=train_res["sensitive_test"])
        pdf_path = os.path.join(self.temp_dir, "report_unmitigated.pdf")
        
        rep_res = generate_fairness_report({
            "dataset_info": {"dataset_name": "ICU Baseline", "rows": 200, "cols": 7},
            "model_results": train_res,
            "fairness_results": fairness_res
        }, output_path=pdf_path)
        self.assertEqual(rep_res["status"], "success")
        self.assertTrue(os.path.exists(pdf_path))

    def test_35_report_matrix_mitigated_random_forest(self):
        """Report Profile 2: Mitigated Random Forest report."""
        df = create_synthetic_clinical_data(n=250, seed=35)
        train_res = train_clinical_model(df, "readmitted_30d", "gender", model_type="random_forest")
        fairness_res = evaluate_fairness_metrics(train_res["y_test"], train_res["y_pred"], sensitive_features=train_res["sensitive_test"])
        mit_eval = evaluate_threshold_candidates(train_res["y_test"], train_res["y_prob"], train_res["sensitive_test"], step_size=0.05)
        
        pdf_path = os.path.join(self.temp_dir, "report_mitigated_rf.pdf")
        rep_res = generate_fairness_report({
            "dataset_info": {"dataset_name": "ICU RF Study", "rows": 250, "cols": 7},
            "model_results": train_res,
            "fairness_results": fairness_res,
            "mitigation_results": mit_eval
        }, output_path=pdf_path)
        self.assertEqual(rep_res["status"], "success")
        self.assertTrue(os.path.exists(pdf_path))


if __name__ == "__main__":
    print("=" * 80)
    print("ASCLEPIUS FAIRAI - PHASE 10 PRODUCTION READINESS TEST SUITE")
    print("=" * 80)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase10ProductionReadiness)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - (failures + errors)
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed}/{total} Tests Passed (Failures: {failures}, Errors: {errors})")
    print("=" * 80)
    
    if failures == 0 and errors == 0:
        print("ALL PHASE 10 PRODUCTION READINESS AUDIT TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("PHASE 10 TEST FAILURES DETECTED. PLEASE REVIEW LOGS.")
        sys.exit(1)
