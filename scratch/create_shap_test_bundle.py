"""
Asclepius FairAI - Level 2 Explainability Audit Test Bundle Generator
Generates a complete, deterministic, synthetic clinical test dataset and trained
RandomForestClassifier for Level 2 External Model Audit (Real SHAP).

Artifacts generated:
1. data/demo/shap_test_predictions.csv (id, y_true, y_pred, y_prob, sex)
2. data/demo/shap_test_features.csv (id, age, bmi, blood_pressure, cholesterol, glucose, heart_rate, sex)
3. data/demo/shap_test_model.pkl (trained scikit-learn RandomForestClassifier)

Semantic Isolation Guard:
- Protected attribute 'sex' is strictly excluded from model feature inputs.
- The model is trained ONLY on: [age, bmi, blood_pressure, cholesterol, glucose, heart_rate].
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_DIR = os.path.join(WORKSPACE_ROOT, "data", "demo")

MODEL_PATH = os.path.join(DEMO_DIR, "shap_test_model.pkl")
FEATURES_PATH = os.path.join(DEMO_DIR, "shap_test_features.csv")
PREDICTIONS_PATH = os.path.join(DEMO_DIR, "shap_test_predictions.csv")

MODEL_FEATURES = ["age", "bmi", "blood_pressure", "cholesterol", "glucose", "heart_rate"]
PROTECTED_ATTRIBUTE = "sex"


def generate_synthetic_clinical_data(n_samples: int = 600, seed: int = 42):
    """
    Generates realistic synthetic clinical-style records.
    NOTE: This is synthetic data intended for algorithmic auditing verification;
    it does not contain real protected health information (PHI).
    """
    rng = np.random.RandomState(seed)

    # 1. Feature distributions (clinical physiological metrics)
    age = rng.randint(28, 85, size=n_samples)
    bmi = np.round(rng.normal(loc=27.8, scale=5.2, size=n_samples).clip(18.5, 42.0), 1)
    blood_pressure = rng.normal(loc=128, scale=18, size=n_samples).clip(95, 185).astype(int)
    cholesterol = rng.normal(loc=210, scale=36, size=n_samples).clip(135, 310).astype(int)
    glucose = rng.normal(loc=112, scale=32, size=n_samples).clip(70, 230).astype(int)
    heart_rate = rng.normal(loc=76, scale=12, size=n_samples).clip(52, 115).astype(int)

    # Protected demographic attribute: balanced F/M cohorts
    sexes = np.array(["F" if i % 2 == 0 else "M" for i in range(n_samples)])
    rng.shuffle(sexes)

    # 2. Clinically correlated risk score for synthetic ground-truth target (30-day readmission / high risk)
    # Using clinical risk factors: age, bmi, BP, glucose, cholesterol, HR
    z = (
        -4.2
        + 0.035 * (age - 50)
        + 0.065 * (bmi - 25)
        + 0.018 * (blood_pressure - 120)
        + 0.012 * (cholesterol - 200)
        + 0.024 * (glucose - 100)
        + 0.015 * (heart_rate - 75)
    )
    # Demographic base prevalence difference (subtle disparity for fairness audit)
    z += np.where(sexes == "F", 0.15, -0.15)

    prob_true = 1.0 / (1.0 + np.exp(-z))
    y_true = (rng.uniform(0, 1, size=n_samples) < prob_true).astype(int)

    # IDs
    ids = [f"PT-{i+1:04d}" for i in range(n_samples)]

    df = pd.DataFrame({
        "id": ids,
        "age": age,
        "bmi": bmi,
        "blood_pressure": blood_pressure,
        "cholesterol": cholesterol,
        "glucose": glucose,
        "heart_rate": heart_rate,
        "sex": sexes,
        "y_true": y_true,
    })
    return df


def build_and_save_bundle():
    os.makedirs(DEMO_DIR, exist_ok=True)
    print("======================================================================")
    print("ASCLEPIUS FAIRAI — CREATING LEVEL 2 SHAP TEST BUNDLE")
    print("======================================================================\n")

    # 1. Generate synthetic dataset
    print("[1/4] Generating 600 synthetic patient records...")
    df = generate_synthetic_clinical_data(n_samples=600, seed=42)
    print(f" -> Generated {len(df)} records across cohorts: {dict(df['sex'].value_counts())}")

    # 2. Train deterministic RandomForestClassifier ONLY on the 6 clinical features
    print("\n[2/4] Training deterministic RandomForestClassifier on 6 clinical features...")
    print(f" -> Model Features: {MODEL_FEATURES}")
    print(f" -> Protected attribute ('{PROTECTED_ATTRIBUTE}') is strictly EXCLUDED from model features.")

    X_train = df[MODEL_FEATURES]
    y_train = df["y_true"]

    rf = RandomForestClassifier(
        n_estimators=40,
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # 3. Generate predictions using the model
    y_pred = rf.predict(X_train)
    y_prob = rf.predict_proba(X_train)[:, 1]

    # 4. Save Model
    print("\n[3/4] Saving model to data/demo/shap_test_model.pkl...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f" -> Model saved: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")

    # 5. Save Features CSV (id, age, bmi, blood_pressure, cholesterol, glucose, heart_rate, sex)
    features_cols = ["id"] + MODEL_FEATURES + [PROTECTED_ATTRIBUTE]
    df_features = df[features_cols]
    df_features.to_csv(FEATURES_PATH, index=False)
    print(f" -> Features CSV saved: {FEATURES_PATH} ({len(df_features)} rows, {list(df_features.columns)})")

    # 6. Save Predictions CSV (id, y_true, y_pred, y_prob, sex)
    df_preds = pd.DataFrame({
        "id": df["id"],
        "y_true": df["y_true"],
        "y_pred": y_pred,
        "y_prob": np.round(y_prob, 4),
        "sex": df["sex"],
    })
    df_preds.to_csv(PREDICTIONS_PATH, index=False)
    print(f" -> Predictions CSV saved: {PREDICTIONS_PATH} ({len(df_preds)} rows, {list(df_preds.columns)})")

    # 7. Verification checks
    print("\n[4/4] Running rigorous verification checks on generated bundle...")
    verify_bundle()


def verify_bundle():
    print("----------------------------------------------------------------------")
    print("VERIFICATION SUITE")
    print("----------------------------------------------------------------------")

    # 1. Model loading
    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)
    print("1. Model load check: PASS (Type: {})".format(type(loaded_model).__name__))
    assert isinstance(loaded_model, RandomForestClassifier)

    # 2. Model feature_names_in_
    model_features = list(getattr(loaded_model, "feature_names_in_", []))
    print(f"2. Model features check: PASS ({model_features})")
    assert model_features == MODEL_FEATURES

    # 3. Row count check
    df_features = pd.read_csv(FEATURES_PATH)
    df_preds = pd.read_csv(PREDICTIONS_PATH)
    print(f"3. Record count check: PASS (Features: {len(df_features)} rows, Predictions: {len(df_preds)} rows)")
    assert len(df_features) == 600 and len(df_preds) == 600
    assert list(df_features["id"]) == list(df_preds["id"])

    # 4. Binary check
    y_true_unique = set(df_preds["y_true"].unique())
    y_pred_unique = set(df_preds["y_pred"].unique())
    print(f"4. Binary outcome check: PASS (y_true classes: {y_true_unique}, y_pred classes: {y_pred_unique})")
    assert y_true_unique.issubset({0, 1}) and y_pred_unique.issubset({0, 1})

    # 5. Probability bounds check
    prob_min = df_preds["y_prob"].min()
    prob_max = df_preds["y_prob"].max()
    print(f"5. Probability range check: PASS ([{prob_min:.4f}, {prob_max:.4f}])")
    assert 0.0 <= prob_min and prob_max <= 1.0

    # 6. Sensitive cohorts check
    cohorts = dict(df_preds["sex"].value_counts())
    print(f"6. Sensitive cohorts check: PASS ({cohorts})")
    assert "F" in cohorts and "M" in cohorts and cohorts["F"] > 0 and cohorts["M"] > 0

    # 7. SHAP TreeExplainer calculation
    X_explain = df_features[MODEL_FEATURES]
    explainer = shap.TreeExplainer(loaded_model)
    raw_shap_values = explainer.shap_values(X_explain)

    # Scikit-learn binary tree explainer yields either (600, 6) or list of [(600, 6), (600, 6)]
    if isinstance(raw_shap_values, list):
        shap_vals_pos = raw_shap_values[1]  # positive class
    elif raw_shap_values.ndim == 3:
        shap_vals_pos = raw_shap_values[:, :, 1]
    else:
        shap_vals_pos = raw_shap_values

    is_finite = np.all(np.isfinite(shap_vals_pos))
    print(f"7. SHAP TreeExplainer calculation check: PASS (Shape: {shap_vals_pos.shape}, All Finite: {is_finite})")
    assert is_finite and shap_vals_pos.shape == (600, 6)

    # 8. SHAP feature names alignment
    mean_abs_shap = np.mean(np.abs(shap_vals_pos), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1]
    ranked_features = [MODEL_FEATURES[i] for i in top_indices]
    print(f"8. SHAP feature names check: PASS ({ranked_features})")
    assert set(ranked_features) == set(MODEL_FEATURES)

    # 9. Semantic rule: Sex must NOT be present in model or SHAP features
    print(f"9. Semantic isolation check: PASS ('sex' present in model features? {'sex' in model_features})")
    assert "sex" not in model_features
    assert "sex" not in ranked_features

    # 10. Summary table
    print("\n======================================================================")
    print("SHAP GLOBAL FEATURE IMPORTANCE (MEAN |SHAP| ON 600 SAMPLES):")
    print("----------------------------------------------------------------------")
    for rank, idx in enumerate(top_indices, 1):
        print(f" {rank}. {MODEL_FEATURES[idx]:<16} : {mean_abs_shap[idx]:.4f}")
    print("======================================================================\n")
    print("ALL 10 VERIFICATION CHECKS PASSED FLAWLESSLY!")


if __name__ == "__main__":
    build_and_save_bundle()
