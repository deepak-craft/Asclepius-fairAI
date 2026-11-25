"""
Asclepius FairAI - Clinical Machine Learning Model Training & Performance Evaluation Module
Phase 3: Clinical Binary Classification, ColumnTransformer Preprocessing, Metrics & Feature Importance
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


SUPPORTED_MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "label": "Logistic Regression — Clinical Baseline",
        "class": LogisticRegression,
    },
    "random_forest": {
        "name": "Random Forest",
        "label": "Random Forest — Ensemble",
        "class": RandomForestClassifier,
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "label": "Gradient Boosting — Ensemble",
        "class": GradientBoostingClassifier,
    },
}


def _identify_id_columns(df: pd.DataFrame) -> List[str]:
    """Identifies obvious ID / identifier columns to avoid passing them as predictive features."""
    id_cols = []
    for col in df.columns:
        col_str = str(col).lower()
        if col_str in ["id", "patient_id", "mrn", "patientid", "record_id", "subject_id", "uuid"]:
            id_cols.append(col)
        elif "id" in col_str and df[col].nunique() == len(df):
            id_cols.append(col)
    return id_cols


def train_clinical_model(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    model_type: str = "logistic_regression",
    test_size: float = 0.2,
    random_state: int = 42,
    class_weight: Optional[str] = None,
    include_sensitive_in_features: bool = True,
    decision_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Trains a clinical binary classification model and calculates comprehensive performance metrics,
    confusion matrix, and explainability-ready feature importances. Preserves the sensitive attribute
    partition for subsequent group-level fairness auditing.
    """
    # 1. Validation of inputs
    if df is None or df.empty or len(df) == 0:
        return {
            "status": "error",
            "error_message": "Dataset is empty. Please upload or load a valid healthcare dataset.",
        }

    if target_col not in df.columns:
        return {
            "status": "error",
            "error_message": f"Target column '{target_col}' not found in dataset columns.",
        }

    if sensitive_col not in df.columns:
        return {
            "status": "error",
            "error_message": f"Sensitive attribute column '{sensitive_col}' not found in dataset columns.",
        }

    if target_col == sensitive_col:
        return {
            "status": "error",
            "error_message": "Target variable and Sensitive attribute cannot be the same column.",
        }

    # Clean working copy
    clean_df = df.copy()

    # Drop rows where target is NaN
    clean_df = clean_df.dropna(subset=[target_col])
    if len(clean_df) < 10:
        return {
            "status": "error",
            "error_message": f"Dataset has only {len(clean_df)} valid rows after dropping missing targets. At least 10 observations required.",
        }

    # 2. Binary classification validation
    unique_targets = clean_df[target_col].unique()
    if len(unique_targets) != 2:
        return {
            "status": "error",
            "error_message": f"Target '{target_col}' has {len(unique_targets)} unique values ({unique_targets.tolist()}). Binary classification requires exactly 2 distinct classes (e.g. 0/1, Yes/No, Negative/Positive).",
        }

    # Convert binary target to 0 and 1
    classes_sorted = sorted(list(unique_targets), key=lambda x: str(x))
    pos_class = classes_sorted[1]
    neg_class = classes_sorted[0]
    
    # Mapping
    y_raw = clean_df[target_col]
    if pd.api.types.is_numeric_dtype(y_raw) and set(unique_targets).issubset({0, 1}):
        y = y_raw.astype(int).values
    else:
        y = (y_raw == pos_class).astype(int).values

    # 3. Extract Sensitive attribute series
    sensitive_series = clean_df[sensitive_col].copy()

    # 4. Separate model features (X)
    id_cols = _identify_id_columns(clean_df)
    drop_cols = [target_col] + id_cols
    if not include_sensitive_in_features and sensitive_col not in drop_cols:
        drop_cols.append(sensitive_col)

    feature_cols = [c for c in clean_df.columns if c not in drop_cols]
    if len(feature_cols) == 0:
        return {
            "status": "error",
            "error_message": "No predictor features remaining after dropping target and identifier columns.",
        }

    X = clean_df[feature_cols].copy()

    # Identify numerical and categorical column types
    numerical_cols = []
    categorical_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    # 5. Preprocessing Pipelines
    transformers = []
    if numerical_cols:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # 6. Model selection and initialization
    model_key = model_type.lower().replace("-", "_").replace(" ", "_")
    if model_key not in SUPPORTED_MODELS:
        model_key = "logistic_regression"

    selected_model_meta = SUPPORTED_MODELS[model_key]
    model_class = selected_model_meta["class"]

    # Parameter setup
    effective_class_weight = "balanced" if class_weight == "balanced" else None

    if model_key == "logistic_regression":
        classifier = model_class(
            class_weight=effective_class_weight,
            random_state=random_state,
            max_iter=1000,
            solver="lbfgs",
        )
    elif model_key == "random_forest":
        classifier = model_class(
            class_weight=effective_class_weight,
            random_state=random_state,
            n_estimators=100,
            max_depth=6,
        )
    elif model_key == "gradient_boosting":
        classifier = model_class(
            random_state=random_state,
            n_estimators=100,
            max_depth=3,
        )
    else:
        classifier = LogisticRegression(random_state=random_state, max_iter=1000)

    # 7. Train/Test Split (Stratified if allowed)
    if not (0.05 <= test_size <= 0.5):
        test_size = 0.2

    can_stratify = True
    class_counts = pd.Series(y).value_counts()
    if (class_counts < 2).any() or (class_counts * test_size < 1).any():
        can_stratify = False

    try:
        stratify_param = y if can_stratify else None
        (
            X_train,
            X_test,
            y_train,
            y_test,
            sensitive_train,
            sensitive_test,
        ) = train_test_split(
            X,
            y,
            sensitive_series,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
        )
    except Exception:
        # Fallback without stratification
        (
            X_train,
            X_test,
            y_train,
            y_test,
            sensitive_train,
            sensitive_test,
        ) = train_test_split(
            X,
            y,
            sensitive_series,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    # 8. Pipeline Construction & Fit (Fit ONLY on training data to prevent leakage)
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    try:
        full_pipeline.fit(X_train, y_train)
    except Exception as fit_err:
        return {
            "status": "error",
            "error_message": f"Model training failed: {str(fit_err)}",
        }

    # 9. Predictions and Probabilities
    # Probability estimation
    if hasattr(full_pipeline, "predict_proba"):
        y_prob = full_pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(full_pipeline, "decision_function"):
        decision = full_pipeline.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-decision))
    else:
        y_prob = full_pipeline.predict(X_test).astype(float)

    # Classification at specified decision threshold (default: 0.50)
    y_pred = (y_prob >= float(decision_threshold)).astype(int)

    # 10. Performance Metrics Calculation
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    
    # Safe metrics with zero division handling
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC calculation
    try:
        if len(np.unique(y_test)) > 1:
            auc = float(roc_auc_score(y_test, y_prob))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    # 11. Feature Importance Extraction
    feature_importance_list = []
    try:
        # Extract transformed feature names
        fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
        transformed_feature_names = []
        
        if numerical_cols:
            transformed_feature_names.extend(numerical_cols)
        if categorical_cols:
            cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_encoded_names = cat_encoder.get_feature_names_out(categorical_cols)
            transformed_feature_names.extend(list(cat_encoded_names))

        fitted_clf = full_pipeline.named_steps["classifier"]
        
        raw_importances = None
        if hasattr(fitted_clf, "feature_importances_"):
            raw_importances = fitted_clf.feature_importances_
        elif hasattr(fitted_clf, "coef_"):
            raw_importances = fitted_clf.coef_[0]

        if raw_importances is not None and len(raw_importances) == len(transformed_feature_names):
            max_val = np.max(np.abs(raw_importances)) if np.max(np.abs(raw_importances)) > 0 else 1.0
            
            for feat_name, val in zip(transformed_feature_names, raw_importances):
                clean_name = feat_name.replace("cat__", "").replace("num__", "")
                feature_importance_list.append({
                    "feature": clean_name,
                    "importance": float(round(val, 4)),
                    "abs_importance": float(round(abs(val), 4)),
                    "relative_pct": float(round((abs(val) / max_val) * 100, 1)),
                    "direction": "Positive" if val >= 0 else "Negative",
                })
            
            # Sort by absolute importance descending
            feature_importance_list.sort(key=lambda x: x["abs_importance"], reverse=True)
    except Exception:
        feature_importance_list = []

    # 12. Build Structured Output Bundle
    results = {
        "status": "success",
        "model_key": model_key,
        "model_name": selected_model_meta["name"],
        "model_label": selected_model_meta["label"],
        "target": target_col,
        "sensitive_attribute": sensitive_col,
        "positive_class": str(pos_class),
        "negative_class": str(neg_class),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "decision_threshold": round(float(decision_threshold), 4),
        "class_weight": class_weight or "none",
        "stratified": can_stratify,
        "features_used": feature_cols,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "feature_importance": feature_importance_list[:12],
        # Data artifacts needed for Phase 4: Fairness Analysis & Phase 5: SHAP
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": [round(float(p), 4) for p in y_prob],
        "sensitive_test": sensitive_test.astype(str).tolist(),
        "X_test": X_test,
        "X_train": X_train,
        "pipeline": full_pipeline,
    }

    return results


def train_model(data):
    """Legacy compatibility bridge."""
    return {"status": "ready"}

