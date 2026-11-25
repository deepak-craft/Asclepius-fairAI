# Asclepius FairAI -- 5-to-10 Minute Demonstration Script

This demonstration script provides an end-to-end presentation flow for demonstrating **Asclepius FairAI** to clinical leaders, data science teams, compliance officers, and academic peers.

---

## Demo Overview
- **Duration**: 5-10 Minutes
- **Target Audience**: Clinical Data Scientists, Bioethicists, Healthcare AI Researchers, Technical Auditors
- **Demonstration Dataset**: `data/demo/synthetic_clinical_demo.csv` (100% Synthetic ICU Readmission Cohort, Zero PHI)

---

## Step-by-Step Presentation Flow

### 1. Introduction & Clinical Problem Statement (1 Minute)
- **What to Explain**:
  - Machine learning models deployed in healthcare risk perpetuating historical demographic disparities across race, gender, age, or insurance tiers.
  - Standard diagnostic evaluations (e.g. overall ROC-AUC, accuracy) mask subgroup-level harms such as disparate false-negative rates (underdiagnosis).
- **Positioning**:
  - *"Asclepius FairAI is an AI-powered healthcare fairness and explainability platform that evaluates machine-learning models for predictive performance and demographic disparities, explains model behavior using SHAP, tests candidate fairness mitigation strategies, and produces structured audit-style reports."*
- **Safety Disclaimer**:
  - Emphasize that Asclepius FairAI is a research/audit decision-support system, not an autonomous medical diagnostic device or clinical decision-maker.

---

### 2. Landing Page & Clinical Onboarding (30 Seconds)
- **What to Click**:
  - Open `http://localhost:8501`. Show the Landing Page Hero section.
- **What to Explain**:
  - The clean, evidence-based user experience built with clinical-grade aesthetics (Emerald Green `#16A34A`, Soft Mint, and Slate).
  - Highlights the three core architectural pillars: ML Training, Fairness Auditing, and Explainable AI.
- **What to Click**: Click the **"Get Started"** or **"Launch Audit Dashboard"** CTA button.
- **What Audience Notices**: Instant smooth view transition into the interactive 9-section analysis dashboard.

---

### 3. Dataset Upload & Validation (1 Minute)
- **What to Click**:
  - In **Section 1 (Dataset Ingestion)**, click **"Upload CSV File"** and select `data/demo/synthetic_clinical_demo.csv`.
- **What to Explain**:
  - Automated column type inference (numerical vs categorical).
  - Identification and automated exclusion of high-cardinality patient identifiers.
  - Zero-storage policy: raw patient data is analyzed in-memory and never permanently stored in databases.
- **What Audience Notices**: Dataset statistics summary (600 rows, 8 features), target distribution preview, and feature table.

---

### 4. Target & Sensitive Attribute Configuration (45 Seconds)
- **What to Click**:
  - Set **Target Variable**: `readmitted_30d` (30-day ICU readmission).
  - Set **Sensitive Attribute**: `gender` (or `insurance_type`).
  - Set **Algorithm**: Select `Random Forest` (or `Logistic Regression` / `Gradient Boosting`).
  - Select Test Split (25%) and random seed (42).
- **What to Explain**:
  - Scikit-Learn `ColumnTransformer` handles missing values (median/mode imputation), standard scaling, and one-hot encoding without data leakage.
- **What to Click**: Click **"Train Clinical Model & Run Audit"**.

---

### 5. Model Performance Evaluation (1 Minute)
- **What to Explain**:
  - Standard clinical metrics: Accuracy, Balanced Accuracy, Sensitivity (Recall), Specificity, F1-Score, and ROC-AUC.
  - Confusion matrix breakdown highlighting true negatives, false positives, false negatives, and true positives.
- **What Audience Notices**: Interactive metric cards and confusion matrix heatmap.

---

### 6. Algorithmic Fairness & Subgroup Disparities (1.5 Minutes)
- **What to Click**: Scroll to **Section 4 (Fairness & Subgroup Disparities)**.
- **What to Explain**:
  - Project screening indicators evaluated against explicit thresholds:
    1. **Demographic Parity Difference (DPD <= 0.05)**: Selection rate balance.
    2. **Disparate Impact Ratio (DIR 0.80 - 1.25)**: The 4/5ths Rule.
    3. **Equal Opportunity Difference (EOpD <= 0.05)**: Sensitivity gap across cohorts.
    4. **Equalized Odds Difference (EOD <= 0.05)**: Combined TPR/FPR gap.
    5. **False Negative Rate Disparity (FNR <= 0.05)**: Clinical underdiagnosis protection.
  - **Subgroup Reliability Tiering**:
    - `n < 30`: *Insufficient Evidence* (alerts against over-interpreting noisy metrics).
    - `30 <= n < 50`: *Caution*.
    - `n >= 50`: *Robust Evaluation*.
- **What Audience Notices**: Clear status badges (`PASS` / `REVIEW`), subgroup breakdown table, and reason-backed overall verdict.

---

### 7. Explainable AI (SHAP) (1 Minute)
- **What to Click**: Scroll to **Section 5 (Explainable AI - SHAP)**.
- **What to Explain**:
  - Uses SHAP (`TreeExplainer` for tree models, `LinearExplainer` for linear models) to calculate exact Shapley attributions.
  - Differentiates global feature importance (e.g., ICU length of stay, Charlson comorbidity index) from protected demographic attributes.
- **What Audience Notices**: Global feature ranking bar chart and local case waterfall explanation.

---

### 8. Deterministic AI Insights & Recommendations (45 Seconds)
- **What to Click**: Scroll to **Section 6 (AI Insights & Governance Recommendations)**.
- **What to Explain**:
  - Rule-based, deterministic engine synthesizing model metrics, fairness screening, and SHAP drivers into structured findings.
  - Categorized severity tiers (`INFO`, `WARNING`, `HIGH ATTENTION`) with actionable governance steps.

---

### 9. Fairness Mitigation & Before/After Comparison (1.5 Minutes)
- **What to Click**: Scroll to **Section 9 (Mitigation & Decision Boundary Optimization)**.
- **What to Explain**:
  - Explains the post-processing *Decision Threshold Adjustment* technique.
  - **Baseline Preservation**: The original trained model and threshold (theta = 0.50) are strictly preserved.
  - **Constraint Guard**: Disqualifies any candidate threshold causing Balanced Accuracy degradation > 5 percentage points.
  - Demonstrates why ROC-AUC remains unchanged (continuous rank-ordering is invariant to classification threshold changes).
- **What to Click**: Select an optimal candidate threshold (e.g., theta = 0.65).
- **What Audience Notices**: Side-by-side Before/After comparison table displaying exact percentage point deltas and fairness gains.

---

### 10. Audit PDF Report & SQLite History (1 Minute)
- **What to Click**:
  - Click **"Generate Audit PDF Report"** and open the compiled document.
  - Click **"Save Analysis to History"** and navigate to **Section 8 (Analysis History)**.
- **What to Explain**:
  - PDF contains a complete executive record, embedded charts, mitigation tables, and academic/clinical disclaimers.
  - SQLite history database stores only aggregate metadata, with Zero PHI.
- **What Audience Notices**: Multi-page report with "Page X of Y" footers and search/filter capabilities in the audit history table.

---

### 11. Conclusion & Q&A (30 Seconds)
- **Summary Statement**: Asclepius FairAI provides transparent, reproducible, and privacy-preserving algorithmic fairness auditing to support ethical clinical AI research.
