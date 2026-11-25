"""
Asclepius FairAI - Basic Local Model Audit Example
Phase 14: Developer Integration Example

Demonstrates:
1. Loading / generating synthetic clinical cohort data
2. Training a scikit-learn clinical classifier (Random Forest)
3. Partitioning test features and sensitive demographic attributes
4. Executing an in-situ fairness, performance, SHAP, and mitigation audit via FairAIAuditor
5. Printing the formatted console summary
6. Exporting a publication-ready PDF audit report
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ensure workspace root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from asclepius import FairAIAuditor


def run_basic_audit():
    print("======================================================================")
    print("ASCLÉPIUS FAIRAI — LOCAL DEVELOPER AUDIT WORKFLOW")
    print("======================================================================\n")

    # 1. Generate realistic synthetic clinical cohort data
    print("[1/5] Generating synthetic clinical cohort data...")
    np.random.seed(42)
    n = 600
    age = np.random.randint(30, 85, size=n)
    sex = np.random.choice(["Female", "Male"], size=n, p=[0.52, 0.48])
    bmi = np.random.normal(27.5, 5.0, size=n)
    systolic_bp = np.random.normal(132, 18, size=n)
    cholesterol = np.random.normal(215, 40, size=n)
    prior_heart_event = np.random.binomial(1, 0.25, size=n)

    # Clinical risk equation with baseline demographic disparity
    logits = (
        -4.5
        + 0.04 * age
        + 0.03 * systolic_bp
        + 0.015 * cholesterol
        + 1.2 * prior_heart_event
        + (sex == "Male") * 0.70
    )
    risk_prob = 1 / (1 + np.exp(-logits))
    heart_disease = np.random.binomial(1, risk_prob)

    df = pd.DataFrame({
        "age": age,
        "sex_male": (sex == "Male").astype(int),
        "bmi": np.round(bmi, 1),
        "systolic_bp": np.round(systolic_bp, 1),
        "cholesterol": np.round(cholesterol, 1),
        "prior_heart_event": prior_heart_event,
        "demographic_sex": sex,
        "heart_disease": heart_disease,
    })

    # 2. Train a local model in the developer's environment
    print("[2/5] Training local Random Forest classifier...")
    feature_cols = ["age", "sex_male", "bmi", "systolic_bp", "cholesterol", "prior_heart_event"]
    X = df[feature_cols]
    y = df["heart_disease"]
    sensitive = df["demographic_sex"]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 3. Instantiate local FairAIAuditor
    print("[3/5] Instantiating Asclepius FairAIAuditor (Local-First Engine)...")
    auditor = FairAIAuditor(
        model=model,
        X_test=X_test,
        y_test=y_test,
        sensitive_features=sens_test,
        target_name="Heart Disease Outcome",
        sensitive_name="Sex (Demographic)",
        model_name="Coronary Risk Predictor",
        model_version="v1.2",
    )

    # 4. Run in-situ audit
    print("[4/5] Executing Performance, Fairness, SHAP, and Mitigation audit...\n")
    result = auditor.run()

    # 5. Print summary
    print(result.summary())
    print("\n----------------------------------------------------------------------")

    # 6. Generate report
    report_out_dir = os.path.join(ROOT_DIR, "outputs", "reports")
    os.makedirs(report_out_dir, exist_ok=True)
    report_path = os.path.join(report_out_dir, "developer_audit_example_report.pdf")
    generated_path = result.generate_report(report_path)
    print(f"\n[5/5] Audit Report successfully exported to:\n -> {generated_path}")

    # Optional: Save summarized record to SQLite
    history_res = result.save_history()
    print(f" -> Persisted summarized audit to local history database (Zero PHI): {history_res.get('status')}")


if __name__ == "__main__":
    run_basic_audit()
