# Asclepius FairAI -- Comprehensive Testing Documentation

Asclepius FairAI implements a rigorous, multi-tier testing framework validating every component from unit functions to full end-to-end integration and security audits.

---

## Test Execution Summary

| Phase | Test Suite | Scope & Coverage | Test Count | Result |
| :---: | :--- | :--- | :---: | :---: |
| **Phase 3** | `scratch/test_ml_pipeline.py` | Preprocessing, ColumnTransformer, Stratified Splits, 3 Models | 12 | **12 / 12 (100%)** |
| **Phase 4** | `scratch/test_phase4_fairness.py` | 5 Core Fairness Metrics, Subgroups, Reliability Tiers | 15 | **15 / 15 (100%)** |
| **Phase 5** | `scratch/test_phase5_shap.py` | TreeExplainer, LinearExplainer, Global/Local Features | 10 | **10 / 10 (100%)** |
| **Phase 6** | `scratch/test_phase6_insights.py` | Rule Engine, Severity Tiers, Deterministic Outputs | 14 | **14 / 14 (100%)** |
| **Phase 7** | `scratch/test_phase7_reports.py` | ReportLab PDF, NumberedCanvas, Type-1 Font Safety | 8 | **8 / 8 (100%)** |
| **Phase 8** | `scratch/test_phase8_history.py` | SQLite Persistence, Deduplication, Summary Stats | 12 | **12 / 12 (100%)** |
| **Phase 9** | `scratch/test_phase9_mitigation.py` | Threshold Optimization, Trade-off Guards, Baseline Preservation | 17 | **17 / 17 (100%)** |
| **Phase 10** | `scratch/test_phase10_production_readiness.py` | 18 Dataset Edge Cases, Boundaries, Multi-Session, Benchmarks | 35 | **35 / 35 (100%)** |
| **Full E2E**| `scratch/test_full_pipeline_e2e.py` | 10-Phase End-to-End Workflow Execution | 8 | **8 / 8 (100%)** |

---

## Key Test Matrices in Phase 10

### 1. Dataset Matrix (18 Edge Cases A-R)
- **Scenario A**: Standard binary classification.
- **Scenario B**: Mixed numerical & categorical features.
- **Scenario C**: Missing values in numerical and categorical features (Imputation validation).
- **Scenario D**: Duplicate rows handling.
- **Scenario E**: Completely empty dataset handling (Graceful exception).
- **Scenario F**: Header-only dataset handling (Graceful exception).
- **Scenario G**: Small total sample (n = 24).
- **Scenario H**: Single-class target rejection (Binary classification enforcement).
- **Scenario I**: Missing target column validation.
- **Scenario J**: Missing sensitive attribute validation.
- **Scenario K**: Subgroup n < 30 (*Insufficient Sample Size* trigger).
- **Scenario L**: Subgroup n in [30, 49] (*Caution* trigger).
- **Scenario M**: Subgroup n >= 50 (*Robust Evaluation* trigger).
- **Scenario N**: Multi-group categorical sensitive attribute (> 2 groups).
- **Scenario O**: Unusual column names (spaces, special characters, punctuation).
- **Scenario P**: High-cardinality patient identifier column auto-drop.
- **Scenario Q**: Constant / zero-variance feature column handling.
- **Scenario R**: Extreme class imbalance (96% negative / 4% positive).

### 2. Fairness Exact Boundary Verification
- **DPD**: 0.050 (PASS) vs 0.051 (REVIEW).
- **DIR**: 0.79 (REVIEW), 0.80 (PASS), 1.00 (PASS/Optimal), 1.25 (PASS), 1.26 (REVIEW).
- **EOpD**: 0.050 (PASS) vs 0.051 (REVIEW).
- **EOD**: 0.050 (PASS) vs 0.051 (REVIEW).
- **FNR Disparity**: 0.050 (PASS) vs 0.051 (REVIEW).

### 3. Performance Benchmarks
- **N = 60 rows**: 0.023s execution latency (< 3.0s SLA budget).
- **N = 600 rows**: 0.210s execution latency with SHAP (< 5.0s SLA budget).
- **N = 5,000 rows**: 0.485s execution latency (< 10.0s SLA budget).

### 4. Privacy & Security Checks
- **Zero PHI**: Verified that `analysis_history` table stores only aggregated metrics and summary metadata.
- **SQL Injection**: Verified parameterized query bindings against malicious SQL payloads.
