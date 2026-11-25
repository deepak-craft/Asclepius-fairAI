"""
Asclepius FairAI - Deterministic Level 2 SHAP Demo Artifacts Generator
Generates:
1. data/demo/shap/asclepius_shap_demo_model.pkl
2. data/demo/shap/asclepius_shap_demo_features.csv
3. data/demo/shap/asclepius_shap_demo_predictions.csv
4. data/demo/shap/README.md

Features:
age, bmi, blood_pressure, cholesterol, glucose, heart_rate, smoking, activity_level
Sensitive attribute: sex (strictly excluded from model features)
Observations: 600
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHAP_DEMO_DIR = os.path.join(WORKSPACE_ROOT, "data", "demo", "shap")

MODEL_PATH = os.path.join(SHAP_DEMO_DIR, "asclepius_shap_demo_model.pkl")
FEATURES_PATH = os.path.join(SHAP_DEMO_DIR, "asclepius_shap_demo_features.csv")
PREDICTIONS_PATH = os.path.join(SHAP_DEMO_DIR, "asclepius_shap_demo_predictions.csv")
README_PATH = os.path.join(SHAP_DEMO_DIR, "README.md")

MODEL_FEATURES = [
    "age",
    "bmi",
    "blood_pressure",
    "cholesterol",
    "glucose",
    "heart_rate",
    "smoking",
    "activity_level",
]
PROTECTED_ATTRIBUTE = "sex"


def generate_shap_demo_package(n_samples: int = 600, seed: int = 42):
    os.makedirs(SHAP_DEMO_DIR, exist_ok=True)
    rng = np.random.RandomState(seed)

    # 1. Generate physiological & lifestyle synthetic clinical features
    age = rng.randint(28, 84, size=n_samples)
    bmi = np.round(rng.normal(loc=27.6, scale=5.1, size=n_samples).clip(18.5, 42.0), 1)
    blood_pressure = rng.normal(loc=128, scale=17, size=n_samples).clip(95, 185).astype(int)
    cholesterol = rng.normal(loc=212, scale=35, size=n_samples).clip(135, 310).astype(int)
    glucose = rng.normal(loc=112, scale=31, size=n_samples).clip(70, 230).astype(int)
    heart_rate = rng.normal(loc=75, scale=11, size=n_samples).clip(52, 115).astype(int)
    smoking = rng.binomial(1, 0.28, size=n_samples)  # 0 = non-smoker, 1 = smoker
    activity_level = rng.choice([0, 1, 2], size=n_samples, p=[0.35, 0.45, 0.20])  # 0=low, 1=mod, 2=high

    # Protected demographic attribute: balanced Female/Male cohorts
    sexes = rng.choice(["Female", "Male"], size=n_samples, p=[0.50, 0.50])

    # 2. Synthetic ground-truth binary outcome (readmission / cardiovascular adverse event)
    logit = (
        -0.4
        + 0.038 * (age - 50)
        + 0.068 * (bmi - 25)
        + 0.016 * (blood_pressure - 120)
        + 0.012 * (cholesterol - 200)
        + 0.022 * (glucose - 100)
        + 0.014 * (heart_rate - 75)
        + 0.62 * smoking
        - 0.42 * activity_level
    )
    # Subtle disparity for fairness analysis
    logit += np.where(sexes == "Female", 0.18, -0.18)

    prob_true = 1.0 / (1.0 + np.exp(-logit))
    y_true = (rng.uniform(0, 1, size=n_samples) < prob_true).astype(int)

    ids = [f"SHAP-PT-{i+1:04d}" for i in range(n_samples)]

    df_full = pd.DataFrame({
        "id": ids,
        "age": age,
        "bmi": bmi,
        "blood_pressure": blood_pressure,
        "cholesterol": cholesterol,
        "glucose": glucose,
        "heart_rate": heart_rate,
        "smoking": smoking,
        "activity_level": activity_level,
        "sex": sexes,
        "y_true": y_true,
    })

    # 3. Train RandomForestClassifier strictly on the 8 model features
    X_train = df_full[MODEL_FEATURES]
    y_train = df_full["y_true"]

    rf = RandomForestClassifier(
        n_estimators=45,
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=seed,
    )
    rf.fit(X_train, y_train)

    # 4. Generate model predictions
    y_pred = rf.predict(X_train)
    y_prob = rf.predict_proba(X_train)[:, 1]

    # 5. Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 6. Save evaluation features CSV (id, features..., sex)
    features_cols = ["id"] + MODEL_FEATURES + [PROTECTED_ATTRIBUTE]
    df_features = df_full[features_cols]
    df_features.to_csv(FEATURES_PATH, index=False)

    # 7. Save predictions CSV (id, y_true, y_pred, y_prob, sex)
    df_preds = pd.DataFrame({
        "id": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": np.round(y_prob, 4),
        "sex": sexes,
    })
    df_preds.to_csv(PREDICTIONS_PATH, index=False)

    # 8. Save README.md
    readme_content = f"""# Asclepius FairAI — Level 2 SHAP Explainability Demo Package

This directory provides a deterministic, synthetic healthcare-style demonstration package for verifying **Level 2 External Model Audit (Real SHAP Explainability)**.

> [!NOTE]
> All records in this package are synthetically generated for algorithmic fairness and explainability testing. They contain no real protected health information (PHI).

---

## Package Contents

1. **`asclepius_shap_demo_model.pkl`**
   - Trained scikit-learn `RandomForestClassifier(n_estimators=45, max_depth=5, random_state={seed})`.
   - Trained strictly on **{len(MODEL_FEATURES)} clinical features**:
     `{', '.join(MODEL_FEATURES)}`
   - **Semantic Protection**: Protected attribute `{PROTECTED_ATTRIBUTE}` is strictly excluded from model feature inputs.

2. **`asclepius_shap_demo_features.csv`**
   - Exactly **{n_samples} synthetic patient evaluation records**.
   - Schema: `id, {', '.join(MODEL_FEATURES)}, {PROTECTED_ATTRIBUTE}`.
   - Ground truth (`y_true`) is intentionally excluded from this file.

3. **`asclepius_shap_demo_predictions.csv`**
   - Exactly **{n_samples} matching prediction records** generated by `asclepius_shap_demo_model.pkl`.
   - Schema: `id, y_true, y_pred, y_prob, {PROTECTED_ATTRIBUTE}`.
   - IDs match `asclepius_shap_demo_features.csv` 1-to-1.

---

## Verification & Usage Instructions

### In the Asclepius FairAI UI:
1. Navigate to **Existing AI Model Audit**.
2. **Prediction Evidence**: Select `asclepius_shap_demo_predictions.csv`.
3. **Section 3: Explainability Evidence (Level 2 Audit)**:
   - Choose Model: Select `asclepius_shap_demo_model.pkl`.
   - Choose Features: Select `asclepius_shap_demo_features.csv`.
   - Leave *Demographic Protection* enabled (default).
   - Click **Validate Explainability Inputs**.
4. Verify status badge changes to **"Model-Specific Explainability Available"**.
5. Click **Run Model Audit & Fairness Evaluation**.
6. Navigate to **Explainable AI** to inspect genuine `TreeExplainer` feature attributions, base value $E[f(x)]$, and the top contributing features chart.
"""

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"Generated Level 2 SHAP Demo Package in {SHAP_DEMO_DIR}:")
    print(f" - Model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")
    print(f" - Features CSV: {FEATURES_PATH} ({len(df_features)} rows)")
    print(f" - Predictions CSV: {PREDICTIONS_PATH} ({len(df_preds)} rows)")
    print(f" - Documentation: {README_PATH}")
    return df_full, rf


if __name__ == "__main__":
    generate_shap_demo_package()
