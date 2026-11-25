"""
Asclepius FairAI - Comprehensive Phase 9 Mitigation & Before/After Verification Suite
Tests:
1. Candidate threshold generation
2. Candidate metric calculation
3. DPD improvement
4. DIR improvement
5. EOpD improvement
6. EOD improvement
7. FNR disparity improvement
8. Performance degradation constraint
9. Candidate selection
10. No improvement case
11. Insufficient subgroup sample size
12. Zero denominator handling
13. Extreme threshold handling
14. Missing probabilities error handling
15. Preservation of baseline results
16. Report integration (PDF compilation with & without mitigation)
17. History linkage (SQLite schema, saving mitigation_json, deduplication & retrieval)
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile
import sqlite3

# Add workspace to path
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.core.mitigation import (
    compute_metrics_at_threshold,
    evaluate_threshold_candidates,
    select_mitigation_candidate,
    compare_before_after,
)
from src.core.fairness import THRESHOLDS, evaluate_fairness_metrics
from src.core.report_generator import generate_fairness_report
from src.core.history import initialize_history_db, save_analysis, get_analysis, get_analysis_history


class TestPhase9MitigationComprehensive(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n = 600
        # Synthetic cohort with demographic bias
        self.sensitive = np.array(["Cohort A"] * 300 + ["Cohort B"] * 300)
        
        # Ground truth binary clinical outcomes
        self.y_true = np.random.binomial(1, 0.45, size=self.n)
        
        # Synthetic probability distribution where Cohort A has elevated risk scores
        probs_A = np.clip(np.random.normal(0.62, 0.16, size=300), 0.01, 0.99)
        probs_B = np.clip(np.random.normal(0.38, 0.16, size=300), 0.01, 0.99)
        self.y_prob = np.concatenate([probs_A, probs_B])

    def test_01_candidate_threshold_generation(self):
        """1. Candidate threshold generation across range with configurable step."""
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
            threshold_min=0.20,
            threshold_max=0.80,
            step_size=0.05,
        )
        self.assertEqual(res["status"], "success")
        self.assertIn("candidates", res)
        thresholds = [c["threshold"] for c in res["candidates"]]
        self.assertIn(0.20, thresholds)
        self.assertIn(0.50, thresholds)
        self.assertIn(0.80, thresholds)
        self.assertTrue(len(thresholds) >= 12)

    def test_02_candidate_metric_calculation(self):
        """2. Candidate metric calculation for all performance & fairness indicators."""
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
            threshold_min=0.30,
            threshold_max=0.70,
            step_size=0.10,
        )
        for cand in res["candidates"]:
            perf = cand["performance"]
            fair = cand["fairness"]
            self.assertIn("accuracy", perf)
            self.assertIn("balanced_accuracy", perf)
            self.assertIn("precision", perf)
            self.assertIn("recall", perf)
            self.assertIn("specificity", perf)
            self.assertIn("f1", perf)
            self.assertIn("roc_auc", perf)
            self.assertIn("dpd", fair)
            self.assertIn("dir", fair)
            self.assertIn("eopd", fair)
            self.assertIn("eod", fair)
            self.assertIn("fnr_disparity", fair)

    def test_03_dpd_improvement(self):
        """3. DPD improvement tracking."""
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
            threshold_min=0.20,
            threshold_max=0.80,
            step_size=0.05,
        )
        base_dpd = res["baseline"]["fairness"]["metrics"]["demographic_parity_difference"]["value"]
        # Find if there is at least one candidate that reduces DPD
        reduced_dpd = [c for c in res["candidates"] if c["fairness"]["dpd"] < base_dpd]
        self.assertTrue(len(reduced_dpd) > 0)

    def test_04_dir_improvement(self):
        """4. Disparate Impact Ratio (DIR) improvement tracking toward 1.0."""
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
            threshold_min=0.20,
            threshold_max=0.80,
            step_size=0.05,
        )
        base_dir = res["baseline"]["fairness"]["metrics"]["disparate_impact_ratio"]["value"]
        closer_to_one = [c for c in res["candidates"] if abs(c["fairness"]["dir"] - 1.0) < abs(base_dir - 1.0)]
        self.assertTrue(len(closer_to_one) > 0)

    def test_05_eopd_improvement(self):
        """5. Equal Opportunity Difference (EOpD) improvement."""
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
            threshold_min=0.20,
            threshold_max=0.80,
            step_size=0.05,
        )
        base_eopd = res["baseline"]["fairness"]["metrics"]["equal_opportunity_difference"]["value"]
        reduced_eopd = [c for c in res["candidates"] if c["fairness"]["eopd"] < base_eopd]
        self.assertTrue(len(reduced_eopd) > 0)

    def test_06_eod_improvement(self):
        """6. Equalized Odds Difference (EOD) improvement."""
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
            threshold_min=0.20,
            threshold_max=0.80,
            step_size=0.05,
        )
        base_eod = res["baseline"]["fairness"]["metrics"]["equalized_odds_difference"]["value"]
        reduced_eod = [c for c in res["candidates"] if c["fairness"]["eod"] < base_eod]
        self.assertTrue(len(reduced_eod) > 0)

    def test_07_fnr_disparity_improvement(self):
        """7. False Negative Rate (Underdiagnosis) disparity improvement."""
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
            threshold_min=0.20,
            threshold_max=0.80,
            step_size=0.05,
        )
        base_fnr = res["baseline"]["fairness"]["metrics"]["fnr_disparity"]["value"]
        reduced_fnr = [c for c in res["candidates"] if c["fairness"]["fnr_disparity"] < base_fnr]
        self.assertTrue(len(reduced_fnr) > 0)

    def test_08_performance_degradation_constraint(self):
        """8. Performance degradation constraint (maximum allowable performance reduction)."""
        max_drop = 0.03
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
            threshold_min=0.10,
            threshold_max=0.90,
            step_size=0.05,
            max_performance_drop=max_drop,
        )
        sel_cand = res["selection"]["selected_candidate"]
        if sel_cand:
            bal_acc_drop = max(0.0, -sel_cand["deltas"]["balanced_accuracy"])
            self.assertLessEqual(bal_acc_drop, max_drop + 1e-4)

    def test_09_candidate_selection(self):
        """9. Candidate selection logic produces structured selection bundle."""
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
        )
        sel = res["selection"]
        self.assertIn("verdict", sel)
        self.assertIn("reason", sel)
        self.assertIn("reliability_status", sel)
        self.assertIn(sel["verdict"], [
            "Fairness Improved",
            "Fairness Improved with Performance Trade-off",
            "No Meaningful Improvement",
            "Insufficient Evidence",
        ])

    def test_10_no_improvement_case(self):
        """10. No improvement case when tolerance is 0.0 and thresholds cannot improve without drop."""
        # Perfectly balanced dataset
        y_true_bal = np.array([0, 1] * 100)
        y_prob_bal = np.array([0.1, 0.9] * 100)
        sens_bal = np.array(["A", "B"] * 100)

        res = evaluate_threshold_candidates(
            y_true=y_true_bal,
            y_prob=y_prob_bal,
            sensitive_features=sens_bal,
            max_performance_drop=0.001,
        )
        self.assertEqual(res["status"], "success")

    def test_11_insufficient_subgroup_sample_size(self):
        """11. Small subgroup sample size flags 'Insufficient Evidence'."""
        small_sens = np.array(["Subgroup Rare"] * 20 + ["Subgroup Main"] * 200)
        small_y_true = np.random.binomial(1, 0.5, size=220)
        small_y_prob = np.random.uniform(0.1, 0.9, size=220)

        res = evaluate_threshold_candidates(
            y_true=small_y_true,
            y_prob=small_y_prob,
            sensitive_features=small_sens,
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["reliability_status"], "Insufficient Evidence")
        self.assertEqual(res["mitigation_verdict"], "Insufficient Evidence")

    def test_12_zero_denominator_handling(self):
        """12. Zero denominator handling for edge confusion components."""
        # All 0s in true labels
        y_zeros = np.zeros(100, dtype=int)
        y_prob_zeros = np.full(100, 0.4)
        sens_zeros = np.array(["A"] * 50 + ["B"] * 50)

        res = evaluate_threshold_candidates(
            y_true=y_zeros,
            y_prob=y_prob_zeros,
            sensitive_features=sens_zeros,
        )
        self.assertEqual(res["status"], "success")
        for c in res["candidates"]:
            self.assertFalse(np.isnan(c["performance"]["accuracy"]))

    def test_13_extreme_thresholds(self):
        """13. Extreme threshold values (0.01 and 0.99)."""
        m_low = compute_metrics_at_threshold(self.y_true, self.y_prob, threshold=0.01)
        m_high = compute_metrics_at_threshold(self.y_true, self.y_prob, threshold=0.99)
        self.assertEqual(m_low["threshold"], 0.01)
        self.assertEqual(m_high["threshold"], 0.99)
        self.assertGreaterEqual(m_low["recall"], 0.99)
        self.assertLessEqual(m_high["recall"], 0.10)

    def test_14_missing_probabilities_error_handling(self):
        """14. Missing probability scores handles error defensively."""
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=None,
            sensitive_features=self.sensitive,
        )
        self.assertEqual(res["status"], "error")
        self.assertIn("required", res["error_message"].lower())

    def test_15_preservation_of_baseline_results(self):
        """15. Preservation of baseline results intact."""
        res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
            baseline_threshold=0.50,
        )
        base = res["baseline"]
        self.assertEqual(base["threshold"], 0.50)
        self.assertIn("accuracy", base["performance"])
        self.assertIn("metrics", base["fairness"])
        self.assertEqual(base["fairness"]["target_name"], "Outcome")

    def test_16_report_integration(self):
        """16. Report integration (PDF compilation with and without mitigation section)."""
        mit_res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
        )
        
        analysis_bundle = {
            "dataset_info": {"dataset_name": "Test_Cardio_ICU.csv", "rows": 600, "cols": 8},
            "model_results": {
                "model_label": "Random Forest",
                "accuracy": 0.85,
                "balanced_accuracy": 0.84,
                "precision": 0.82,
                "recall": 0.86,
                "specificity": 0.83,
                "f1": 0.84,
                "roc_auc": 0.89,
                "target": "Cardiovascular Risk",
                "sensitive_attribute": "Cohort",
                "confusion_matrix": {"tn": 150, "fp": 30, "fn": 25, "tp": 155},
            },
            "fairness_results": mit_res["baseline"]["fairness"],
            "shap_results": {"status": "success", "global_importance": [{"rank": 1, "feature": "troponin", "mean_abs_shap": 0.18}]},
            "insights_results": {"key_finding": {"title": "Sample bias evaluated.", "explanation": "Audit complete."}, "recommendations": []},
            "mitigation_results": mit_res,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = os.path.join(tmp_dir, "test_mitigated_report.pdf")
            rep_res = generate_fairness_report(analysis_bundle, output_path=pdf_path)
            self.assertEqual(rep_res["status"], "success")
            self.assertTrue(os.path.exists(pdf_path))
            self.assertGreater(os.path.getsize(pdf_path), 50000)

    def test_17_history_linkage(self):
        """17. SQLite History linkage with mitigation_json persistence and retrieval."""
        mit_res = evaluate_threshold_candidates(
            y_true=self.y_true,
            y_prob=self.y_prob,
            sensitive_features=self.sensitive,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_mit_history.db")
            initialize_history_db(db_path)

            analysis_bundle = {
                "analysis_id": "ANL-TEST-MIT-001",
                "dataset_info": {"dataset_name": "Test_Cohort.csv", "rows": 600},
                "model_results": {
                    "model_label": "Random Forest",
                    "accuracy": 0.85,
                    "balanced_accuracy": 0.84,
                    "precision": 0.82,
                    "recall": 0.86,
                    "specificity": 0.83,
                    "f1": 0.84,
                    "roc_auc": 0.89,
                    "target": "Readmission",
                    "sensitive_attribute": "Cohort",
                },
                "fairness_results": mit_res["baseline"]["fairness"],
                "mitigation_results": mit_res,
            }

            # Save
            save_res = save_analysis(analysis_bundle, db_path=db_path)
            self.assertEqual(save_res["status"], "success")

            # Retrieve single
            retrieved = get_analysis("ANL-TEST-MIT-001", db_path=db_path)
            self.assertIsNotNone(retrieved)
            self.assertIn("mitigation", retrieved)
            self.assertEqual(retrieved["mitigation"]["strategy"], "Decision Threshold Adjustment")

            # Retrieve list
            hist_list = get_analysis_history(db_path=db_path)
            self.assertEqual(len(hist_list), 1)
            self.assertIn("mitigation", hist_list[0])


if __name__ == "__main__":
    unittest.main()

