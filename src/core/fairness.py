"""
Asclepius FairAI - Healthcare Fairness Analysis & Algorithmic Bias Evaluation Module
Phase 4: Subgroup Analysis, Core Disparity Metrics, Reliability Tiering, and Transparent Verdicts
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd


# Project-level screening thresholds (Locked per requirements)
THRESHOLDS = {
    "dpd": 0.05,
    "dir_min": 0.80,
    "dir_max": 1.25,
    "eopd": 0.05,
    "eod": 0.05,
    "fnr_disparity": 0.05,
}


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely computes division, returning default if denominator is zero."""
    if denominator == 0 or np.isnan(denominator):
        return default
    return float(numerator / denominator)


def evaluate_fairness_metrics(
    y_true: Union[List[int], np.ndarray, pd.Series],
    y_pred: Union[List[int], np.ndarray, pd.Series],
    y_prob: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    sensitive_features: Optional[Union[List[Any], np.ndarray, pd.Series]] = None,
    target_name: str = "Outcome",
    sensitive_name: str = "Protected Group",
    favorable_label: int = 1,
) -> Dict[str, Any]:
    """
    Evaluates algorithmic fairness and subgroup performance disparities across protected demographic attributes.
    Returns a comprehensive structured fairness analysis object with subgroup metrics, disparity scores,
    reliability assessments, and reason-backed clinical screening verdicts.
    """
    # 1. Input normalization and validation
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n_samples = len(y_true_arr)

    if n_samples == 0 or len(y_pred_arr) != n_samples:
        return {
            "status": "error",
            "error_message": f"Mismatched or empty sample arrays (y_true: {len(y_true_arr)}, y_pred: {len(y_pred_arr)}).",
        }

    # Handle sensitive features
    if sensitive_features is None:
        sens_arr = np.array(["Overall"] * n_samples, dtype=object)
    else:
        # Explicit missing-value handling: Map NaN/None/empty to "Unknown"
        sens_list = []
        missing_sens_count = 0
        for val in sensitive_features:
            if val is None or pd.isna(val) or str(val).strip() in ["", "nan", "NaN", "None", "null"]:
                sens_list.append("Unknown")
                missing_sens_count += 1
            else:
                sens_list.append(str(val).strip())
        sens_arr = np.array(sens_list, dtype=object)

    # 2. Identify all observed demographic subgroups
    unique_groups = sorted(list(set(sens_arr)))
    if len(unique_groups) == 0:
        unique_groups = ["Overall"]

    subgroups_list: List[Dict[str, Any]] = []
    reliability_list: List[Dict[str, Any]] = []
    has_insufficient_subgroup = False

    # 3. Subgroup Metric Computations
    for group_name in unique_groups:
        group_mask = sens_arr == group_name
        n_group = int(np.sum(group_mask))

        if n_group == 0:
            continue

        y_true_g = y_true_arr[group_mask]
        y_pred_g = y_pred_arr[group_mask]

        # Favorable binary conversions
        pos_true = (y_true_g == favorable_label).astype(int)
        pos_pred = (y_pred_g == favorable_label).astype(int)

        # Confusion components
        tp = int(np.sum((pos_true == 1) & (pos_pred == 1)))
        fp = int(np.sum((pos_true == 0) & (pos_pred == 1)))
        tn = int(np.sum((pos_true == 0) & (pos_pred == 0)))
        fn = int(np.sum((pos_true == 1) & (pos_pred == 0)))

        # Rates
        pos_selection_rate = _safe_divide(tp + fp, n_group)
        tpr = _safe_divide(tp, tp + fn)              # Sensitivity / Recall
        specificity = _safe_divide(tn, tn + fp)      # True Negative Rate
        fpr = _safe_divide(fp, fp + tn)              # False Positive Rate
        fnr = _safe_divide(fn, fn + tp)              # False Negative Rate / Underdiagnosis
        accuracy = _safe_divide(tp + tn, n_group)

        # Subgroup sample size reliability rules
        if n_group < 30:
            reliability_status = "INSUFFICIENT SAMPLE SIZE"
            reliability_tier = "insufficient"
            reliability_desc = f"Subgroup size (n={n_group}) is below 30. Metrics have high variance and cannot support strong fairness conclusions."
            has_insufficient_subgroup = True
        elif 30 <= n_group < 50:
            reliability_status = "CAUTION"
            reliability_tier = "caution"
            reliability_desc = f"Subgroup size (n={n_group}) is in the caution range (30–49). Disparities should be interpreted cautiously."
        else:
            reliability_status = "STANDARD EVALUATION"
            reliability_tier = "standard"
            reliability_desc = f"Subgroup size (n={n_group}) meets standard evaluation thresholds (n ≥ 50)."

        reliability_info = {
            "group": group_name,
            "n": n_group,
            "tier": reliability_tier,
            "status": reliability_status,
            "description": reliability_desc,
        }
        reliability_list.append(reliability_info)

        subgroup_data = {
            "group": group_name,
            "n": n_group,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "positive_selection_rate": round(pos_selection_rate, 4),
            "selection_rate_pct": round(pos_selection_rate * 100, 1),
            "tpr": round(tpr, 4),
            "sensitivity_pct": round(tpr * 100, 1),
            "specificity": round(specificity, 4),
            "specificity_pct": round(specificity * 100, 1),
            "fpr": round(fpr, 4),
            "fpr_pct": round(fpr * 100, 1),
            "fnr": round(fnr, 4),
            "fnr_pct": round(fnr * 100, 1),
            "accuracy": round(accuracy, 4),
            "accuracy_pct": round(accuracy * 100, 1),
            "reliability": reliability_status,
            "reliability_tier": reliability_tier,
        }
        subgroups_list.append(subgroup_data)

    # 4. Core Disparity Metrics Calculation
    selection_rates = [g["positive_selection_rate"] for g in subgroups_list]
    tprs = [g["tpr"] for g in subgroups_list]
    fprs = [g["fpr"] for g in subgroups_list]
    fnrs = [g["fnr"] for g in subgroups_list]

    max_sel = max(selection_rates) if selection_rates else 0.0
    min_sel = min(selection_rates) if selection_rates else 0.0

    max_tpr = max(tprs) if tprs else 0.0
    min_tpr = min(tprs) if tprs else 0.0

    max_fpr = max(fprs) if fprs else 0.0
    min_fpr = min(fprs) if fprs else 0.0

    max_fnr = max(fnrs) if fnrs else 0.0
    min_fnr = min(fnrs) if fnrs else 0.0

    # A. Demographic Parity Difference (DPD)
    dpd = round(max_sel - min_sel, 6)

    # B. Disparate Impact Ratio (DIR) - Min to Max ratio
    dir_ratio = round(_safe_divide(min_sel, max_sel, default=1.0), 6) if max_sel > 0 else 1.0

    # C. Equal Opportunity Difference (EOpD)
    eopd = round(max_tpr - min_tpr, 6)

    # D. Equalized Odds Difference (EOD)
    tpr_gap = round(max_tpr - min_tpr, 6)
    fpr_gap = round(max_fpr - min_fpr, 6)
    eod = max(abs(tpr_gap), abs(fpr_gap))

    # E. False Negative Rate Disparity (Underdiagnosis Risk)
    fnr_disparity = round(max_fnr - min_fnr, 6)

    # 5. Metric Status Determinations (PASS / REVIEW)
    dpd_pass = round(dpd, 4) <= THRESHOLDS["dpd"]
    dir_pass = THRESHOLDS["dir_min"] <= round(dir_ratio, 4) <= THRESHOLDS["dir_max"]
    eopd_pass = round(eopd, 4) <= THRESHOLDS["eopd"]
    eod_pass = round(eod, 4) <= THRESHOLDS["eod"]
    fnr_pass = round(fnr_disparity, 4) <= THRESHOLDS["fnr_disparity"]

    metrics_status = {
        "demographic_parity_difference": {
            "value": round(dpd, 4),
            "threshold": f"≤ {THRESHOLDS['dpd']}",
            "status": "PASS" if dpd_pass else "REVIEW",
            "passed": bool(dpd_pass),
            "description": "Gap between maximum and minimum group positive selection rates.",
        },
        "disparate_impact_ratio": {
            "value": round(dir_ratio, 4),
            "threshold": f"{THRESHOLDS['dir_min']}–{THRESHOLDS['dir_max']}",
            "status": "PASS" if dir_pass else "REVIEW",
            "passed": bool(dir_pass),
            "description": "Ratio of lowest to highest group selection rate (80% Rule).",
        },
        "equal_opportunity_difference": {
            "value": round(eopd, 4),
            "threshold": f"≤ {THRESHOLDS['eopd']}",
            "status": "PASS" if eopd_pass else "REVIEW",
            "passed": bool(eopd_pass),
            "description": "Gap between maximum and minimum true-positive sensitivity rates.",
        },
        "equalized_odds_difference": {
            "value": round(eod, 4),
            "threshold": f"≤ {THRESHOLDS['eod']}",
            "status": "PASS" if eod_pass else "REVIEW",
            "passed": bool(eod_pass),
            "description": "Maximum of TPR disparity and FPR disparity.",
        },
        "fnr_disparity": {
            "value": round(fnr_disparity, 4),
            "threshold": f"≤ {THRESHOLDS['fnr_disparity']}",
            "status": "PASS" if fnr_pass else "REVIEW",
            "passed": bool(fnr_pass),
            "description": "Underdiagnosis / missed positive outcome rate disparity.",
        },
        "tpr_disparity": round(tpr_gap, 4),
        "fpr_disparity": round(fpr_gap, 4),
    }

    # 6. Overall Fairness Verdict Logic (Transparent & Reason-backed)
    reasons: List[str] = []
    failed_metrics_count = sum([not dpd_pass, not dir_pass, not eopd_pass, not eod_pass, not fnr_pass])

    # Reasons formulation
    if not dpd_pass:
        reasons.append(f"Demographic Parity Difference ({dpd:.3f}) exceeds the project screening threshold of {THRESHOLDS['dpd']}.")
    if not dir_pass:
        reasons.append(f"Disparate Impact Ratio ({dir_ratio:.2f}) falls outside the 80% screening range ({THRESHOLDS['dir_min']}–{THRESHOLDS['dir_max']}).")
    if not eopd_pass:
        reasons.append(f"Equal Opportunity Difference ({eopd:.3f}) exceeds the project threshold of {THRESHOLDS['eopd']}.")
    if not eod_pass:
        reasons.append(f"Equalized Odds Difference ({eod:.3f}) exceeds the project threshold of {THRESHOLDS['eod']}.")
    if not fnr_pass:
        reasons.append(f"Underdiagnosis disparity ({fnr_disparity:.3f}) indicates uneven false-negative rates across groups.")

    # Status assignment
    if has_insufficient_subgroup:
        overall_status = "INSUFFICIENT_EVIDENCE"
        verdict_title = "Insufficient Evidence"
        verdict_badge = "ⓘ INSUFFICIENT EVIDENCE"
        reasons.insert(0, "One or more demographic subgroups contain fewer than 30 observations; statistical conclusions require larger cohorts.")
    elif failed_metrics_count == 0:
        overall_status = "FAIR_BALANCED"
        verdict_title = "Fair / Balanced"
        verdict_badge = "✓ FAIR / BALANCED"
        reasons.append("All 5 screening fairness metrics satisfy project thresholds across evaluated demographic groups.")
    elif failed_metrics_count >= 2 or fnr_disparity > 0.10:
        overall_status = "SIGNIFICANT_DISPARITY"
        verdict_title = "Significant Disparity"
        verdict_badge = "⚠ SIGNIFICANT DISPARITY"
    else:
        overall_status = "MODERATE_DISPARITY"
        verdict_title = "Moderate Disparity"
        verdict_badge = "⚠ MODERATE DISPARITY"

    # 7. Final Structured Fairness Artifact Bundle
    results = {
        "status": "success",
        "target_name": target_name,
        "sensitive_name": sensitive_name,
        "favorable_label": favorable_label,
        "sample_size": n_samples,
        "subgroups": subgroups_list,
        "metrics": metrics_status,
        "thresholds": THRESHOLDS,
        "reliability": reliability_list,
        "has_insufficient_subgroup": has_insufficient_subgroup,
        "overall_status": overall_status,
        "verdict_title": verdict_title,
        "verdict_badge": verdict_badge,
        "reasons": reasons,
        "clinical_safety_note": (
            "Fairness screening thresholds are project criteria for bias detection and model evaluation, "
            "and do not represent universal regulatory or clinical standards. Metrics highlight areas for clinical audit."
        ),
    }

    return results


def evaluate_fairness(results):
    """Legacy bridge function."""
    if isinstance(results, dict) and "y_test" in results and "sensitive_test" in results:
        return evaluate_fairness_metrics(
            y_true=results["y_test"],
            y_pred=results["y_pred"],
            y_prob=results.get("y_prob"),
            sensitive_features=results["sensitive_test"],
            target_name=results.get("target", "Outcome"),
            sensitive_name=results.get("sensitive_attribute", "Protected Group"),
        )
    return {"status": "ready", "results": results}

