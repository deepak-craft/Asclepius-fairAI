"""
Dataset Validation & Preprocessing Module for Asclepius FairAI
Handles file parsing, data quality checks, candidate attribute suggestions, and summary statistics.
"""

from typing import Dict, Any, List, Optional
import io
import pandas as pd
import numpy as np


SENSITIVE_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "age", "age_group", "insurance",
    "payer", "zip", "zipcode", "postal", "income", "socioeconomic", "language",
    "marital", "disability", "nationality", "religion"
]

TARGET_KEYWORDS = [
    "target", "outcome", "label", "diagnosis", "readmission", "mortality",
    "disease", "condition", "status", "result", "recurrence", "diabetes",
    "heart_disease", "cancer", "stroke", "hypertension", "y", "class"
]


def validate_dataset_from_csv(csv_content: str) -> Dict[str, Any]:
    """
    Parses and validates a CSV dataset from a string or file-like stream.
    Returns structured validation metrics, column info, preview, and status.
    """
    if not csv_content or not csv_content.strip():
        return {
            "status": "invalid",
            "status_title": "Dataset Invalid",
            "status_message": "The uploaded dataset is empty and cannot be analyzed.",
            "issues": ["Uploaded CSV file is completely empty (0 bytes)."],
            "summary": {"rows": 0, "columns": 0, "missing_values": 0, "missing_percentage": 0.0, "duplicate_rows": 0, "duplicate_percentage": 0.0},
            "columns_info": [],
            "target_candidates": [],
            "sensitive_candidates": [],
            "sample_size_eval": {"n": 0, "tier": "insufficient", "message": "No data rows found."},
            "preview_rows": [],
            "preview_columns": [],
        }

    try:
        df = pd.read_csv(io.StringIO(csv_content))
    except Exception as e:
        return {
            "status": "invalid",
            "status_title": "Dataset Invalid",
            "status_message": f"File could not be parsed as a valid CSV table: {str(e)}",
            "issues": [f"CSV Parsing Error: {str(e)}"],
            "summary": {"rows": 0, "columns": 0, "missing_values": 0, "missing_percentage": 0.0, "duplicate_rows": 0, "duplicate_percentage": 0.0},
            "columns_info": [],
            "target_candidates": [],
            "sensitive_candidates": [],
            "sample_size_eval": {"n": 0, "tier": "insufficient", "message": "Failed to parse data."},
            "preview_rows": [],
            "preview_columns": [],
        }

    return validate_dataframe(df)


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates a pandas DataFrame and returns a comprehensive health, preview, and candidate assessment.
    """
    issues: List[str] = []
    warnings: List[str] = []

    # Check for empty dataframe
    if df is None or df.empty or len(df) == 0:
        return {
            "status": "invalid",
            "status_title": "Dataset Invalid",
            "status_message": "The dataset contains column headers but no data rows (0 rows).",
            "issues": ["Dataset contains 0 rows."],
            "summary": {"rows": 0, "columns": len(df.columns) if df is not None else 0, "missing_values": 0, "missing_percentage": 0.0, "duplicate_rows": 0, "duplicate_percentage": 0.0},
            "columns_info": [],
            "target_candidates": [],
            "sensitive_candidates": [],
            "sample_size_eval": {"n": 0, "tier": "insufficient", "message": "No sample rows found."},
            "preview_rows": [],
            "preview_columns": list(df.columns) if df is not None else [],
        }

    n_rows = len(df)
    n_cols = len(df.columns)

    if n_cols == 0:
        return {
            "status": "invalid",
            "status_title": "Dataset Invalid",
            "status_message": "The dataset contains no usable columns.",
            "issues": ["Dataset has 0 columns."],
            "summary": {"rows": n_rows, "columns": 0, "missing_values": 0, "missing_percentage": 0.0, "duplicate_rows": 0, "duplicate_percentage": 0.0},
            "columns_info": [],
            "target_candidates": [],
            "sensitive_candidates": [],
            "sample_size_eval": {"n": n_rows, "tier": "insufficient", "message": "No columns found."},
            "preview_rows": [],
            "preview_columns": [],
        }

    # Missing values calculation
    total_cells = n_rows * n_cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

    if total_missing > 0:
        warnings.append(f"Dataset contains {total_missing} missing values ({missing_pct}% of all data cells).")

    # Duplicate rows calculation
    try:
        duplicate_rows = int(df.duplicated().sum())
    except Exception:
        duplicate_rows = 0
    duplicate_pct = round((duplicate_rows / n_rows) * 100, 2) if n_rows > 0 else 0.0

    if duplicate_rows > 0:
        warnings.append(f"Dataset contains {duplicate_rows} duplicate rows ({duplicate_pct}% of total records).")

    # Sample size tier evaluation (preparing architecture for later fairness reliability rules)
    if n_rows < 30:
        sample_tier = "insufficient"
        sample_message = f"Sample size (n={n_rows}) is below the standard minimum (n<30). Subgroup evaluations may be statistically unreliable."
        warnings.append(sample_message)
    elif 30 <= n_rows < 50:
        sample_tier = "caution"
        sample_message = f"Sample size (n={n_rows}) is in caution range (30-49). Interpret subgroup fairness metrics cautiously."
    else:
        sample_tier = "standard"
        sample_message = f"Sample size (n={n_rows}) meets standard evaluation thresholds (n>=50)."

    # Column level analysis
    columns_info: List[Dict[str, Any]] = []
    target_candidates: List[Dict[str, Any]] = []
    sensitive_candidates: List[Dict[str, Any]] = []

    for col in df.columns:
        col_name = str(col).strip()
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_col_pct = round((missing_count / n_rows) * 100, 1)
        unique_count = int(series.nunique(dropna=True))

        # Infer simplified data type
        dtype = str(series.dtype).lower()
        if "int" in dtype:
            inferred_type = "Numeric (Integer)"
        elif "float" in dtype:
            inferred_type = "Numeric (Float)"
        elif "bool" in dtype:
            inferred_type = "Boolean"
        elif "datetime" in dtype:
            inferred_type = "DateTime"
        else:
            inferred_type = "Categorical (Text)"

        sample_vals = [str(x) for x in series.dropna().unique()[:5]]

        col_info = {
            "name": col_name,
            "data_type": inferred_type,
            "missing_count": missing_count,
            "missing_pct": missing_col_pct,
            "unique_count": unique_count,
            "sample_values": sample_vals
        }
        columns_info.append(col_info)

        # Target candidate detection heuristics (Suggestion only)
        col_lower = col_name.lower().replace("_", " ").replace("-", " ")
        is_target_keyword = any(kw in col_lower for kw in TARGET_KEYWORDS)
        is_discrete = 2 <= unique_count <= 10
        is_binary = unique_count == 2

        if is_target_keyword or is_binary or (is_discrete and inferred_type in ["Categorical (Text)", "Numeric (Integer)", "Boolean"]):
            confidence = "High" if (is_target_keyword and is_discrete) else ("Medium" if (is_target_keyword or is_binary) else "Low")
            target_candidates.append({
                "column": col_name,
                "unique_values": unique_count,
                "data_type": inferred_type,
                "confidence": confidence,
                "reason": "Binary classification candidate" if is_binary else ("Matches clinical outcome naming" if is_target_keyword else "Discrete categories")
            })

        # Sensitive attribute candidate detection heuristics (Suggestion only)
        is_sensitive_keyword = any(kw in col_lower for kw in SENSITIVE_KEYWORDS)
        is_sensitive_discrete = 2 <= unique_count <= 25

        if is_sensitive_keyword or (is_sensitive_discrete and inferred_type in ["Categorical (Text)", "Numeric (Integer)"]):
            confidence = "High" if is_sensitive_keyword else "Medium"
            sensitive_candidates.append({
                "column": col_name,
                "unique_values": unique_count,
                "data_type": inferred_type,
                "confidence": confidence,
                "reason": "Demographic attribute candidate" if is_sensitive_keyword else "Categorical subgroup candidate"
            })

    # Overall status determination
    if len(df) < 2:
        status = "invalid"
        status_title = "Dataset Invalid"
        status_message = "The dataset has fewer than 2 rows and cannot be evaluated."
        issues.append("Dataset has fewer than 2 rows.")
    elif warnings:
        status = "warning"
        status_title = "Review Required"
        status_message = "Dataset successfully loaded with warnings. Review missing values or sample size below before proceeding."
    else:
        status = "ready"
        status_title = "Dataset Ready"
        status_message = "Dataset successfully loaded and verified. Ready for target and sensitive attribute configuration."

    # Preview rows (first 10 rows converted to string-safe dicts)
    preview_df = df.head(10).fillna("—")
    preview_rows = preview_df.to_dict(orient="records")
    preview_columns = list(df.columns)

    return {
        "status": status,
        "status_title": status_title,
        "status_message": status_message,
        "issues": issues,
        "warnings": warnings,
        "summary": {
            "rows": n_rows,
            "columns": n_cols,
            "missing_values": total_missing,
            "missing_percentage": missing_pct,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_pct,
        },
        "columns_info": columns_info,
        "target_candidates": target_candidates,
        "sensitive_candidates": sensitive_candidates,
        "sample_size_eval": {
            "n": n_rows,
            "tier": sample_tier,
            "message": sample_message
        },
        "preview_rows": preview_rows,
        "preview_columns": preview_columns,
    }


def preprocess_data(data):
    """Placeholder for data preprocessing transformations."""
    return data

