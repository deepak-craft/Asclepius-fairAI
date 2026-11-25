"""
Asclepius FairAI - Fairness Mitigation & Before/After Model Comparison Module
Phase 9: Decision Threshold Optimization, Trade-off Analytics, Before/After Auditing, and Safety Verification
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.core.fairness import evaluate_fairness_metrics, THRESHOLDS, _safe_divide


def compute_metrics_at_threshold(
    y_true: Union[List[int], np.ndarray, pd.Series],
    y_prob: Union[List[float], np.ndarray, pd.Series],
    threshold: float = 0.5,
    favorable_label: int = 1,
) -> Dict[str, Any]:
    """
    Computes classification performance metrics at a specific decision threshold.
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)

    # Apply decision threshold to probabilities
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    # Standard metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    specificity = _safe_divide(tn, tn + fp, default=0.0)

    # ROC-AUC (computed on underlying continuous probabilities; threshold-independent)
    try:
        if len(np.unique(y_true_arr)) > 1:
            auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 1),
        "balanced_accuracy": round(bal_acc, 4),
        "balanced_accuracy_pct": round(bal_acc * 100, 1),
        "precision": round(prec, 4),
        "precision_pct": round(prec * 100, 1),
        "recall": round(rec, 4),
        "recall_pct": round(rec * 100, 1),
        "specificity": round(specificity, 4),
        "specificity_pct": round(specificity * 100, 1),
        "f1": round(f1, 4),
        "f1_pct": round(f1 * 100, 1),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "y_pred": y_pred_arr.tolist(),
    }


def evaluate_threshold_candidates(
    y_true: Union[List[int], np.ndarray, pd.Series],
    y_prob: Union[List[float], np.ndarray, pd.Series],
    sensitive_features: Optional[Union[List[Any], np.ndarray, pd.Series]] = None,
    target_name: str = "Outcome",
    sensitive_name: str = "Protected Group",
    favorable_label: int = 1,
    threshold_min: float = 0.20,
    threshold_max: float = 0.80,
    step_size: float = 0.05,
    baseline_threshold: float = 0.50,
    max_performance_drop: float = 0.05,
    objective: str = "balance_fairness_performance",
    target_metric: str = "all",
) -> Dict[str, Any]:
    """
    Evaluates candidate decision thresholds across a range, calculating both performance
    and fairness metrics for each candidate. Preserves baseline evaluation intact.
    """
    # 1. Input Validation
    y_true_arr = np.array(y_true, dtype=int)
    if y_prob is None or len(y_prob) == 0:
        msg = "Decision Threshold Calibration requires continuous probability scores (y_prob). Only discrete predictions were supplied. Continuous probability scores are required."
        return {
            "status": "error",
            "availability": "unavailable",
            "mitigation_available": False,
            "error_message": msg,
            "message": msg,
        }

    y_prob_arr = np.array(y_prob, dtype=float)
    n_samples = len(y_true_arr)

    if n_samples == 0 or len(y_prob_arr) != n_samples:
        return {
            "status": "error",
            "error_message": f"Mismatched sample arrays (y_true: {len(y_true_arr)}, y_prob: {len(y_prob_arr)}).",
        }

    # Validate threshold bounds
    th_min = max(0.01, min(0.95, float(threshold_min)))
    th_max = max(th_min + 0.01, min(0.99, float(threshold_max)))
    step = max(0.01, min(0.20, float(step_size)))

    # 2. Compute Baseline (Standard Default Threshold = 0.50)
    baseline_perf = compute_metrics_at_threshold(
        y_true=y_true_arr,
        y_prob=y_prob_arr,
        threshold=baseline_threshold,
        favorable_label=favorable_label,
    )

    baseline_fairness = evaluate_fairness_metrics(
        y_true=y_true_arr,
        y_pred=baseline_perf["y_pred"],
        y_prob=y_prob_arr,
        sensitive_features=sensitive_features,
        target_name=target_name,
        sensitive_name=sensitive_name,
        favorable_label=favorable_label,
    )

    if baseline_fairness.get("status") == "error":
        return {
            "status": "error",
            "error_message": f"Baseline fairness calculation failed: {baseline_fairness.get('error_message')}",
        }

    has_insufficient_sample = baseline_fairness.get("has_insufficient_subgroup", False)

    # 3. Generate and Evaluate Threshold Candidates
    candidate_thresholds = np.arange(th_min, th_max + 1e-5, step)
    # Ensure baseline threshold is evaluated
    if not any(np.isclose(candidate_thresholds, baseline_threshold, atol=1e-3)):
        candidate_thresholds = np.sort(np.append(candidate_thresholds, baseline_threshold))

    candidates_list: List[Dict[str, Any]] = []

    base_bal_acc = baseline_perf["balanced_accuracy"]
    base_f1 = baseline_perf["f1"]
    base_dpd = baseline_fairness["metrics"]["demographic_parity_difference"]["value"]
    base_dir = baseline_fairness["metrics"]["disparate_impact_ratio"]["value"]
    base_eopd = baseline_fairness["metrics"]["equal_opportunity_difference"]["value"]
    base_eod = baseline_fairness["metrics"]["equalized_odds_difference"]["value"]
    base_fnr = baseline_fairness["metrics"]["fnr_disparity"]["value"]
    base_pass_count = sum(1 for m in baseline_fairness["metrics"].values() if isinstance(m, dict) and m.get("passed", False))

    for th in candidate_thresholds:
        th_val = round(float(th), 3)
        perf_cand = compute_metrics_at_threshold(
            y_true=y_true_arr,
            y_prob=y_prob_arr,
            threshold=th_val,
            favorable_label=favorable_label,
        )

        fair_cand = evaluate_fairness_metrics(
            y_true=y_true_arr,
            y_pred=perf_cand["y_pred"],
            y_prob=y_prob_arr,
            sensitive_features=sensitive_features,
            target_name=target_name,
            sensitive_name=sensitive_name,
            favorable_label=favorable_label,
        )

        if fair_cand.get("status") == "error":
            continue

        cand_metrics = fair_cand["metrics"]
        cand_dpd = cand_metrics["demographic_parity_difference"]["value"]
        cand_dir = cand_metrics["disparate_impact_ratio"]["value"]
        cand_eopd = cand_metrics["equal_opportunity_difference"]["value"]
        cand_eod = cand_metrics["equalized_odds_difference"]["value"]
        cand_fnr = cand_metrics["fnr_disparity"]["value"]
        cand_pass_count = sum(1 for m in cand_metrics.values() if isinstance(m, dict) and m.get("passed", False))

        # Performance Changes (relative to baseline)
        delta_bal_acc = round(perf_cand["balanced_accuracy"] - base_bal_acc, 4)
        delta_f1 = round(perf_cand["f1"] - base_f1, 4)
        bal_acc_drop = max(0.0, -delta_bal_acc)

        # Fairness Changes
        delta_dpd = round(cand_dpd - base_dpd, 4)
        delta_dir = round(cand_dir - base_dir, 4)
        delta_eopd = round(cand_eopd - base_eopd, 4)
        delta_eod = round(cand_eod - base_eod, 4)
        delta_fnr = round(cand_fnr - base_fnr, 4)

        # Disparity Improvements (negative is improvement for DPD/EOD/FNR/EOpD)
        dpd_improved = cand_dpd < base_dpd or (cand_dpd <= THRESHOLDS["dpd"] and not baseline_fairness["metrics"]["demographic_parity_difference"]["passed"])
        fnr_improved = cand_fnr < base_fnr or (cand_fnr <= THRESHOLDS["fnr_disparity"] and not baseline_fairness["metrics"]["fnr_disparity"]["passed"])
        dir_improved = abs(cand_dir - 1.0) < abs(base_dir - 1.0)
        eopd_improved = cand_eopd < base_eopd
        eod_improved = cand_eod < base_eod

        # Check performance drop constraint
        within_tolerance = bal_acc_drop <= max_performance_drop

        # Overall improvement flag
        overall_fairness_improved = (cand_pass_count > base_pass_count) or (
            cand_pass_count == base_pass_count and (delta_dpd < -0.01 or delta_fnr < -0.01 or delta_eopd < -0.01 or delta_eod < -0.01)
        )

        is_baseline = np.isclose(th_val, baseline_threshold, atol=1e-3)

        # Status Tag
        if is_baseline:
            suitability_tag = "Baseline Model"
        elif not within_tolerance:
            suitability_tag = "Exceeds Performance Drop Limit"
        elif overall_fairness_improved and bal_acc_drop <= 0.01:
            suitability_tag = "Strong Candidate (Minimal Trade-off)"
        elif overall_fairness_improved and within_tolerance:
            suitability_tag = "Viable Candidate (Acceptable Trade-off)"
        elif cand_pass_count == base_pass_count and within_tolerance:
            suitability_tag = "Neutral / Similar to Baseline"
        else:
            suitability_tag = "No Disparity Improvement"

        candidate_item = {
            "threshold": th_val,
            "is_baseline": bool(is_baseline),
            "performance": {
                "accuracy": perf_cand["accuracy"],
                "accuracy_pct": perf_cand["accuracy_pct"],
                "balanced_accuracy": perf_cand["balanced_accuracy"],
                "balanced_accuracy_pct": perf_cand["balanced_accuracy_pct"],
                "precision": perf_cand["precision"],
                "precision_pct": perf_cand["precision_pct"],
                "recall": perf_cand["recall"],
                "recall_pct": perf_cand["recall_pct"],
                "specificity": perf_cand["specificity"],
                "specificity_pct": perf_cand["specificity_pct"],
                "f1": perf_cand["f1"],
                "f1_pct": perf_cand["f1_pct"],
                "roc_auc": perf_cand["roc_auc"],
                "confusion_matrix": perf_cand["confusion_matrix"],
            },
            "fairness": {
                "dpd": cand_dpd,
                "dir": cand_dir,
                "eopd": cand_eopd,
                "eod": cand_eod,
                "fnr_disparity": cand_fnr,
                "passed_metrics_count": cand_pass_count,
                "overall_status": fair_cand["overall_status"],
                "verdict_summary": fair_cand.get("summary", ""),
                "metrics": cand_metrics,
                "subgroups": fair_cand.get("subgroups", []),
            },
            "deltas": {
                "balanced_accuracy": delta_bal_acc,
                "balanced_accuracy_pp": round(delta_bal_acc * 100, 1),
                "f1": delta_f1,
                "f1_pp": round(delta_f1 * 100, 1),
                "accuracy_pp": round((perf_cand["accuracy"] - baseline_perf["accuracy"]) * 100, 1),
                "dpd": delta_dpd,
                "dir": delta_dir,
                "eopd": delta_eopd,
                "eod": delta_eod,
                "fnr_disparity": delta_fnr,
            },
            "improvements": {
                "dpd_improved": bool(dpd_improved),
                "fnr_improved": bool(fnr_improved),
                "dir_improved": bool(dir_improved),
                "eopd_improved": bool(eopd_improved),
                "eod_improved": bool(eod_improved),
                "overall_fairness_improved": bool(overall_fairness_improved),
            },
            "within_tolerance": bool(within_tolerance),
            "suitability_tag": suitability_tag,
        }
        candidates_list.append(candidate_item)

    # 4. Automatically select best mitigation candidate
    selection_result = select_mitigation_candidate(
        candidates_list=candidates_list,
        baseline_perf=baseline_perf,
        baseline_fairness=baseline_fairness,
        max_performance_drop=max_performance_drop,
        objective=objective,
        target_metric=target_metric,
        has_insufficient_sample=has_insufficient_sample,
    )

    # 5. Build structured Before/After comparison if candidate is found
    selected_cand = selection_result.get("selected_candidate")
    comparison = None
    if selected_cand is not None:
        comparison = compare_before_after(
            baseline_perf=baseline_perf,
            baseline_fairness=baseline_fairness,
            candidate=selected_cand,
            sensitive_name=sensitive_name,
            target_name=target_name,
            baseline_threshold=baseline_threshold,
        )

    return {
        "status": "success",
        "strategy": "Decision Threshold Adjustment",
        "objective": objective,
        "target_metric": target_metric,
        "max_performance_drop": max_performance_drop,
        "max_performance_drop_pct": round(max_performance_drop * 100, 1),
        "threshold_range": {
            "min": th_min,
            "max": th_max,
            "step": step,
            "baseline": baseline_threshold,
        },
        "baseline": {
            "threshold": baseline_threshold,
            "performance": baseline_perf,
            "fairness": baseline_fairness,
            "has_insufficient_sample": has_insufficient_sample,
        },
        "candidates": candidates_list,
        "selection": selection_result,
        "comparison": comparison,
        "mitigation_verdict": selection_result["verdict"],
        "reliability_status": selection_result["reliability_status"],
        "governance_notice": (
            "Threshold mitigation is an experimental decision-boundary calibration. "
            "It modifies the operational classification cut-point on continuous model probabilities. "
            "Observed improvements require clinical peer-review and multi-cohort validation before deployment."
        ),
    }


def select_mitigation_candidate(
    candidates_list: List[Dict[str, Any]],
    baseline_perf: Dict[str, Any],
    baseline_fairness: Dict[str, Any],
    max_performance_drop: float = 0.05,
    objective: str = "balance_fairness_performance",
    target_metric: str = "all",
    has_insufficient_sample: bool = False,
) -> Dict[str, Any]:
    """
    Selects the optimal candidate decision threshold based on a documented, transparent objective rule.
    """
    if not candidates_list:
        return {
            "selected_candidate": None,
            "verdict": "Candidate Evaluation Failed",
            "reason": "No valid threshold candidates were evaluated.",
            "reliability_status": "Evaluation Error",
        }

    # Small subgroup sample check
    if has_insufficient_sample:
        # Still identify best mathematical candidate for exploration, but mark reliability as Insufficient Evidence
        reliability_status = "Insufficient Evidence"
        verdict_override = "Insufficient Evidence"
    else:
        reliability_status = "Standard Evaluation"
        verdict_override = None

    base_bal_acc = baseline_perf["balanced_accuracy"]
    base_f1 = baseline_perf["f1"]
    base_pass_count = sum(1 for m in baseline_fairness["metrics"].values() if isinstance(m, dict) and m.get("passed", False))

    # Eligible non-baseline candidates within performance degradation tolerance
    eligible = [
        c for c in candidates_list
        if not c["is_baseline"] and c["within_tolerance"]
    ]

    if not eligible:
        return {
            "selected_candidate": None,
            "verdict": verdict_override or "No Meaningful Improvement",
            "reason": f"No candidate thresholds met the maximum performance reduction constraint of {max_performance_drop*100:.1f} percentage points.",
            "reliability_status": reliability_status,
        }

    # Scoring Function for Candidate Selection
    def _score_candidate(cand: Dict[str, Any]) -> float:
        deltas = cand["deltas"]
        cand_fair = cand["fairness"]
        passed_cnt = cand_fair["passed_metrics_count"]

        # Component 1: Passing metrics gain (weight: 100 per additional passing metric)
        pass_gain = (passed_cnt - base_pass_count) * 100.0

        # Component 2: Disparity reduction on target metric
        if target_metric == "fnr_disparity":
            disp_gain = -deltas["fnr_disparity"] * 50.0
        elif target_metric == "dpd":
            disp_gain = -deltas["dpd"] * 50.0
        elif target_metric == "eopd":
            disp_gain = -deltas["eopd"] * 50.0
        elif target_metric == "eod":
            disp_gain = -deltas["eod"] * 50.0
        elif target_metric == "dir":
            disp_gain = (abs(baseline_fairness["metrics"]["disparate_impact_ratio"]["value"] - 1.0) - abs(cand_fair["dir"] - 1.0)) * 50.0
        else:
            # All metrics combined
            disp_gain = (
                -deltas["dpd"] * 20.0
                - deltas["fnr_disparity"] * 25.0
                - deltas["eopd"] * 15.0
                - deltas["eod"] * 15.0
                + (abs(baseline_fairness["metrics"]["disparate_impact_ratio"]["value"] - 1.0) - abs(cand_fair["dir"] - 1.0)) * 15.0
            )

        # Component 3: Performance penalty (depending on objective)
        bal_acc_drop = max(0.0, -deltas["balanced_accuracy"])
        if objective == "improve_fairness":
            perf_penalty = bal_acc_drop * 30.0  # Mild penalty within tolerance
        else:
            # balance_fairness_performance
            perf_penalty = bal_acc_drop * 80.0  # Moderate penalty for performance loss

        # Small preference for thresholds close to 0.5 to prevent extreme drift
        threshold_penalty = abs(cand["threshold"] - 0.5) * 2.0

        return float(pass_gain + disp_gain - perf_penalty - threshold_penalty)

    # Sort eligible by score descending
    scored_candidates = sorted(eligible, key=_score_candidate, reverse=True)
    best_cand = scored_candidates[0]
    best_score = _score_candidate(best_cand)

    # Determine whether best candidate provides genuine improvement
    cand_pass_count = best_cand["fairness"]["passed_metrics_count"]
    delta_fnr = best_cand["deltas"]["fnr_disparity"]
    delta_dpd = best_cand["deltas"]["dpd"]
    bal_acc_drop = max(0.0, -best_cand["deltas"]["balanced_accuracy"])

    has_meaningful_gain = (cand_pass_count > base_pass_count) or (delta_fnr <= -0.015 or delta_dpd <= -0.015)

    if not has_meaningful_gain:
        verdict = verdict_override or "No Meaningful Improvement"
        reason = "Evaluated candidates did not produce a significant reduction in algorithmic disparity within the specified performance tolerance."
    elif verdict_override == "Insufficient Evidence":
        verdict = "Insufficient Evidence"
        reason = (
            f"Candidate threshold {best_cand['threshold']} showed mathematical improvement in disparity metrics, "
            f"but subgroup sample sizes (n < 30) prevent reliable validation."
        )
    elif bal_acc_drop <= 0.01:
        verdict = "Fairness Improved"
        reason = (
            f"Candidate threshold {best_cand['threshold']} improved fairness metrics with minimal performance impact "
            f"({bal_acc_drop*100:.1f} pp drop in balanced accuracy)."
        )
    else:
        verdict = "Fairness Improved with Performance Trade-off"
        reason = (
            f"Candidate threshold {best_cand['threshold']} reduced disparity while balanced accuracy decreased by "
            f"{bal_acc_drop*100:.1f} percentage points (within {max_performance_drop*100:.1f} pp tolerance)."
        )

    return {
        "selected_candidate": best_cand,
        "verdict": verdict,
        "reason": reason,
        "reliability_status": reliability_status,
        "candidate_score": round(best_score, 2),
        "candidate_threshold": best_cand["threshold"],
    }


def compare_before_after(
    baseline_perf: Dict[str, Any],
    baseline_fairness: Dict[str, Any],
    candidate: Dict[str, Any],
    sensitive_name: str = "Protected Group",
    target_name: str = "Outcome",
    baseline_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Constructs a comprehensive, side-by-side Before/After comparison bundle.
    """
    base_th = round(float(baseline_threshold if baseline_threshold is not None else baseline_perf.get("threshold", 0.50)), 3)
    cand_perf = candidate["performance"]
    cand_fair = candidate["fairness"]
    cand_th = candidate["threshold"]
    deltas = candidate["deltas"]

    # 1. Performance Metrics Comparison Table
    performance_comparison = [
        {
            "metric": "Accuracy",
            "baseline": f"{baseline_perf['accuracy_pct']:.1f}%",
            "candidate": f"{cand_perf['accuracy_pct']:.1f}%",
            "change": f"{deltas['accuracy_pp']:+.1f} pp",
            "delta_raw": round(cand_perf["accuracy"] - baseline_perf["accuracy"], 4),
            "improved": cand_perf["accuracy"] >= baseline_perf["accuracy"],
        },
        {
            "metric": "Balanced Accuracy",
            "baseline": f"{baseline_perf['balanced_accuracy_pct']:.1f}%",
            "candidate": f"{cand_perf['balanced_accuracy_pct']:.1f}%",
            "change": f"{deltas['balanced_accuracy_pp']:+.1f} pp",
            "delta_raw": deltas["balanced_accuracy"],
            "improved": deltas["balanced_accuracy"] >= 0,
        },
        {
            "metric": "Sensitivity (Recall)",
            "baseline": f"{baseline_perf['recall_pct']:.1f}%",
            "candidate": f"{cand_perf['recall_pct']:.1f}%",
            "change": f"{(cand_perf['recall'] - baseline_perf['recall'])*100:+.1f} pp",
            "delta_raw": round(cand_perf["recall"] - baseline_perf["recall"], 4),
            "improved": cand_perf["recall"] >= baseline_perf["recall"],
        },
        {
            "metric": "Specificity (TNR)",
            "baseline": f"{baseline_perf['specificity_pct']:.1f}%",
            "candidate": f"{cand_perf['specificity_pct']:.1f}%",
            "change": f"{(cand_perf['specificity'] - baseline_perf['specificity'])*100:+.1f} pp",
            "delta_raw": round(cand_perf["specificity"] - baseline_perf["specificity"], 4),
            "improved": cand_perf["specificity"] >= baseline_perf["specificity"],
        },
        {
            "metric": "Precision (PPV)",
            "baseline": f"{baseline_perf['precision_pct']:.1f}%",
            "candidate": f"{cand_perf['precision_pct']:.1f}%",
            "change": f"{(cand_perf['precision'] - baseline_perf['precision'])*100:+.1f} pp",
            "delta_raw": round(cand_perf["precision"] - baseline_perf["precision"], 4),
            "improved": cand_perf["precision"] >= baseline_perf["precision"],
        },
        {
            "metric": "F1 Score",
            "baseline": f"{baseline_perf['f1_pct']:.1f}%",
            "candidate": f"{cand_perf['f1_pct']:.1f}%",
            "change": f"{deltas['f1_pp']:+.1f} pp",
            "delta_raw": deltas["f1"],
            "improved": deltas["f1"] >= 0,
        },
        {
            "metric": "ROC-AUC",
            "baseline": f"{baseline_perf['roc_auc']:.3f}",
            "candidate": f"{cand_perf['roc_auc']:.3f}",
            "change": "0.000 (Invariant)",
            "delta_raw": 0.0,
            "improved": True,
            "note": "ROC-AUC is calculated on continuous model probabilities and remains invariant under decision threshold shifts.",
        },
    ]

    # 2. Fairness Metrics Comparison Table
    base_metrics = baseline_fairness["metrics"]
    cand_metrics = cand_fair["metrics"]

    fairness_comparison = [
        {
            "metric": "Demographic Parity Difference (DPD)",
            "threshold": f"≤ {THRESHOLDS['dpd']}",
            "baseline": f"{base_metrics['demographic_parity_difference']['value']:.4f}",
            "candidate": f"{cand_fair['dpd']:.4f}",
            "baseline_status": base_metrics['demographic_parity_difference']['status'],
            "candidate_status": cand_metrics['demographic_parity_difference']['status'],
            "change": f"{deltas['dpd']:+.4f}",
            "delta_raw": deltas["dpd"],
            "improved": deltas["dpd"] < 0 or (cand_fair["dpd"] <= THRESHOLDS["dpd"] and base_metrics['demographic_parity_difference']['status'] == "REVIEW"),
        },
        {
            "metric": "Disparate Impact Ratio (DIR)",
            "threshold": f"{THRESHOLDS['dir_min']}–{THRESHOLDS['dir_max']}",
            "baseline": f"{base_metrics['disparate_impact_ratio']['value']:.4f}",
            "candidate": f"{cand_fair['dir']:.4f}",
            "baseline_status": base_metrics['disparate_impact_ratio']['status'],
            "candidate_status": cand_metrics['disparate_impact_ratio']['status'],
            "change": f"{deltas['dir']:+.4f}",
            "delta_raw": deltas["dir"],
            "improved": abs(cand_fair["dir"] - 1.0) < abs(base_metrics['disparate_impact_ratio']['value'] - 1.0),
        },
        {
            "metric": "Equal Opportunity Difference (EOpD)",
            "threshold": f"≤ {THRESHOLDS['eopd']}",
            "baseline": f"{base_metrics['equal_opportunity_difference']['value']:.4f}",
            "candidate": f"{cand_fair['eopd']:.4f}",
            "baseline_status": base_metrics['equal_opportunity_difference']['status'],
            "candidate_status": cand_metrics['equal_opportunity_difference']['status'],
            "change": f"{deltas['eopd']:+.4f}",
            "delta_raw": deltas["eopd"],
            "improved": deltas["eopd"] < 0,
        },
        {
            "metric": "Equalized Odds Difference (EOD)",
            "threshold": f"≤ {THRESHOLDS['eod']}",
            "baseline": f"{base_metrics['equalized_odds_difference']['value']:.4f}",
            "candidate": f"{cand_fair['eod']:.4f}",
            "baseline_status": base_metrics['equalized_odds_difference']['status'],
            "candidate_status": cand_metrics['equalized_odds_difference']['status'],
            "change": f"{deltas['eod']:+.4f}",
            "delta_raw": deltas["eod"],
            "improved": deltas["eod"] < 0,
        },
        {
            "metric": "False Negative Rate Disparity (FNR)",
            "threshold": f"≤ {THRESHOLDS['fnr_disparity']}",
            "baseline": f"{base_metrics['fnr_disparity']['value']:.4f}",
            "candidate": f"{cand_fair['fnr_disparity']:.4f}",
            "baseline_status": base_metrics['fnr_disparity']['status'],
            "candidate_status": cand_metrics['fnr_disparity']['status'],
            "change": f"{deltas['fnr_disparity']:+.4f}",
            "delta_raw": deltas["fnr_disparity"],
            "improved": deltas["fnr_disparity"] < 0,
        },
    ]

    # 3. Subgroup Demographic Comparison
    base_subgroups = {g["group"]: g for g in baseline_fairness.get("subgroups", [])}
    cand_subgroups = {g["group"]: g for g in cand_fair.get("subgroups", [])}

    subgroup_comparison = []
    for g_name, b_data in base_subgroups.items():
        c_data = cand_subgroups.get(g_name, {})
        n = b_data.get("n", 0)
        base_sel = b_data.get("selection_rate_pct", 0.0)
        cand_sel = c_data.get("selection_rate_pct", 0.0)
        base_sens = b_data.get("sensitivity_pct", 0.0)
        cand_sens = c_data.get("sensitivity_pct", 0.0)
        base_fnr = b_data.get("fnr_pct", 0.0)
        cand_fnr = c_data.get("fnr_pct", 0.0)

        subgroup_comparison.append({
            "group": g_name,
            "n": n,
            "reliability": b_data.get("reliability", "Standard"),
            "selection_rate": {
                "baseline": f"{base_sel:.1f}%",
                "candidate": f"{cand_sel:.1f}%",
                "change": f"{cand_sel - base_sel:+.1f} pp",
            },
            "sensitivity": {
                "baseline": f"{base_sens:.1f}%",
                "candidate": f"{cand_sens:.1f}%",
                "change": f"{cand_sens - base_sens:+.1f} pp",
            },
            "fnr": {
                "baseline": f"{base_fnr:.1f}%",
                "candidate": f"{cand_fnr:.1f}%",
                "change": f"{cand_fnr - base_fnr:+.1f} pp",
            },
        })

    # 4. Summary Text
    fnr_drop = deltas["fnr_disparity"]
    dpd_drop = deltas["dpd"]

    summary_statement = (
        f"Candidate decision threshold {cand_th:.2f} shifted the classification cut-point from baseline ({base_th:.2f}). "
        f"Under this candidate, FNR Disparity changed from {base_metrics['fnr_disparity']['value']:.3f} to {cand_fair['fnr_disparity']:.3f} ({fnr_drop:+.3f}), "
        f"and Demographic Parity Difference changed from {base_metrics['demographic_parity_difference']['value']:.3f} to {cand_fair['dpd']:.3f} ({dpd_drop:+.3f}), "
        f"with a balanced accuracy impact of {deltas['balanced_accuracy_pp']:+.1f} percentage points."
    )

    return {
        "candidate_threshold": cand_th,
        "baseline_threshold": base_th,
        "summary_statement": summary_statement,
        "performance_comparison": performance_comparison,
        "fairness_comparison": fairness_comparison,
        "subgroup_comparison": subgroup_comparison,
        "overall_status_baseline": baseline_fairness.get("overall_status", "UNKNOWN"),
        "overall_status_candidate": cand_fair.get("overall_status", "UNKNOWN"),
    }

