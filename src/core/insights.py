"""
Asclepius FairAI - AI Insights & Recommendations Engine
Phase 6: Deterministic, Evidence-Based Clinical AI Insights, Severity Tiering, and Actionable Governance Recommendations
"""

from typing import Dict, Any, List, Optional, Union


# Standard severity levels
SEVERITY_INFO = "INFO"
SEVERITY_WARNING = "WARNING"
SEVERITY_HIGH_ATTENTION = "HIGH ATTENTION"
SEVERITY_INSUFFICIENT = "INSUFFICIENT EVIDENCE"


def generate_insights(
    model_results: Optional[Dict[str, Any]] = None,
    fairness_results: Optional[Dict[str, Any]] = None,
    shap_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates a comprehensive, deterministic, and evidence-traceable insight bundle
    interpreting model performance, subgroup fairness disparities, and SHAP explainability.
    
    Parameters:
    -----------
    model_results: Output bundle from train_clinical_model
    fairness_results: Output bundle from evaluate_fairness_metrics
    shap_results: Output bundle from generate_shap_explanations
    
    Returns:
    --------
    Structured insight dictionary containing summary, key findings, category findings,
    severity assessments, and concrete recommendations.
    """
    if not model_results or not fairness_results:
        return {
            "status": "error",
            "error_message": "Both model_results and fairness_results are required to generate insights.",
        }

    # Extract model performance metrics safely
    acc = float(model_results.get("accuracy", 0.0))
    bal_acc = float(model_results.get("balanced_accuracy", 0.0))
    prec = float(model_results.get("precision", 0.0))
    rec = float(model_results.get("recall", 0.0))
    spec = float(model_results.get("specificity", 0.0))
    f1 = float(model_results.get("f1", 0.0))
    raw_auc = model_results.get("roc_auc")
    auc = float(raw_auc) if raw_auc is not None else None
    model_name = model_results.get("model_name", model_results.get("model_label", "Clinical Classifier"))
    target_col = model_results.get("target", "Outcome")
    sensitive_col = model_results.get("sensitive_attribute", fairness_results.get("sensitive_name", "Protected Group"))

    # Extract fairness metrics safely
    fairness_metrics = fairness_results.get("metrics", {})
    subgroups = fairness_results.get("subgroups", [])
    has_insufficient_sample = fairness_results.get("has_insufficient_subgroup", False)
    reliability_list = fairness_results.get("reliability", [])
    fairness_verdict_status = fairness_results.get("overall_status", "UNKNOWN")

    dpd_entry = fairness_metrics.get("demographic_parity_difference") or fairness_metrics.get("dpd") or {}
    dpd_val = float(dpd_entry.get("value", dpd_entry.get("val", 0.0)))
    dpd_pass = bool(dpd_entry.get("passed", dpd_entry.get("pass", True))) if dpd_entry else True

    dir_entry = fairness_metrics.get("disparate_impact_ratio") or fairness_metrics.get("dir") or {}
    dir_val = float(dir_entry.get("value", dir_entry.get("val", 1.0)))
    dir_pass = bool(dir_entry.get("passed", dir_entry.get("pass", True))) if dir_entry else True

    eopd_entry = fairness_metrics.get("equal_opportunity_difference") or fairness_metrics.get("eopd") or {}
    eopd_val = float(eopd_entry.get("value", eopd_entry.get("val", 0.0)))
    eopd_pass = bool(eopd_entry.get("passed", eopd_entry.get("pass", True))) if eopd_entry else True

    eod_entry = fairness_metrics.get("equalized_odds_difference") or fairness_metrics.get("eod") or {}
    eod_val = float(eod_entry.get("value", eod_entry.get("val", 0.0)))
    eod_pass = bool(eod_entry.get("passed", eod_entry.get("pass", True))) if eod_entry else True

    fnr_entry = fairness_metrics.get("fnr_disparity") or fairness_metrics.get("fnr") or {}
    fnr_val = float(fnr_entry.get("value", fnr_entry.get("val", 0.0)))
    fnr_pass = bool(fnr_entry.get("passed", fnr_entry.get("pass", True))) if fnr_entry else True

    # Extract SHAP metrics safely
    top_shap_features = []
    is_sensitive_in_features = False
    if shap_results and shap_results.get("status") == "success":
        top_shap_features = shap_results.get("global_importance", [])[:5]
        is_sensitive_in_features = shap_results.get("sensitive_attribute_context", {}).get("included_in_model_features", False)

    # Containers for findings and recommendations
    performance_findings = []
    fairness_findings = []
    subgroup_findings = []
    explainability_findings = []
    recommendations = []

    # =========================================================================
    # 1. MODEL PERFORMANCE INSIGHTS
    # =========================================================================
    if acc >= 0.85 and bal_acc >= 0.80:
        perf_summary_text = "Strong overall discrimination performance on the evaluation dataset."
        evidence_dict = {"accuracy": acc, "balanced_accuracy": bal_acc}
        if auc is not None:
            evidence_dict["roc_auc"] = auc
        performance_findings.append({
            "category": "Model Performance",
            "title": "Robust Overall Classification",
            "severity": SEVERITY_INFO,
            "explanation": f"Overall accuracy ({acc*100:.1f}%) and balanced accuracy ({bal_acc*100:.1f}%) reflect strong classification performance across evaluation classes.",
            "evidence": evidence_dict,
            "recommendation": "Maintain model baseline while monitoring calibration across operational subsets."
        })
    elif acc < 0.70 or bal_acc < 0.65:
        perf_summary_text = "Moderate or limited classification accuracy requiring clinical tuning."
        performance_findings.append({
            "category": "Model Performance",
            "title": "Sub-optimal Classification Baseline",
            "severity": SEVERITY_WARNING,
            "explanation": f"Model balanced accuracy ({bal_acc*100:.1f}%) indicates that predictions require calibration before high-stakes clinical deployment.",
            "evidence": {"accuracy": acc, "balanced_accuracy": bal_acc, "f1": f1},
            "recommendation": "Evaluate additional clinical features, hyperparameter tuning, or alternative ensemble architectures."
        })
    else:
        perf_summary_text = "Adequate classification performance across test observations."
        auc_desc = f" and ROC-AUC of {auc:.3f}." if auc is not None else "."
        evidence_dict = {"accuracy": acc, "f1": f1}
        if auc is not None:
            evidence_dict["roc_auc"] = auc
        performance_findings.append({
            "category": "Model Performance",
            "title": "Standard Baseline Performance",
            "severity": SEVERITY_INFO,
            "explanation": f"Model demonstrates an accuracy of {acc*100:.1f}%{auc_desc}",
            "evidence": evidence_dict,
            "recommendation": "Continue regular performance surveillance during prospective validation."
        })

    # Missed-positive cases / Recall vs Precision gap
    if (prec - rec) >= 0.15:
        performance_findings.append({
            "category": "Model Performance",
            "title": "Precision-Recall Asymmetry (Underdiagnosis Bias)",
            "severity": SEVERITY_WARNING,
            "explanation": f"Model recall ({rec*100:.1f}%) is substantially lower than precision ({prec*100:.1f}%), indicating that the model misses a larger proportion of true positive disease cases than it incorrectly flags.",
            "evidence": {"precision": prec, "recall": rec, "difference": round(prec - rec, 3)},
            "recommendation": "Review decision threshold calibration to prioritize clinical sensitivity and reduce missed positive cases."
        })
        recommendations.append({
            "id": "REC-PERF-THRESHOLD",
            "title": "Calibrate Decision Threshold for Sensitivity",
            "category": "Model Performance",
            "severity": SEVERITY_WARNING,
            "trigger": f"Recall ({rec*100:.1f}%) is {round((prec-rec)*100, 1)}% lower than precision ({prec*100:.1f}%).",
            "action": "Lower the classification probability threshold or apply class re-weighting to minimize false negatives in high-risk patient pathways.",
            "investigate": ["Operating threshold curve", "Cost of false negatives vs false positives", "Class imbalance ratio"]
        })

    # =========================================================================
    # 2. FAIRNESS DISPARITY INSIGHTS
    # =========================================================================
    primary_finding = None

    # A. FNR Disparity (Clinical Underdiagnosis) - Highest Priority
    if not fnr_pass:
        fnr_finding = {
            "category": "Fairness",
            "title": "False-Negative Rate Disparity (Underdiagnosis Risk)",
            "severity": SEVERITY_HIGH_ATTENTION if fnr_val > 0.10 else SEVERITY_WARNING,
            "explanation": f"Underdiagnosis disparity ({fnr_val:.3f}) exceeds the project screening threshold (0.05). The model misses positive disease cases at different rates across evaluated demographic groups.",
            "evidence": {"metric": "fnr_disparity", "value": fnr_val, "threshold": 0.05, "protected_group": sensitive_col},
            "recommendation": "Prioritize investigation of subgroup false-negative rates and calibrate group-specific diagnostic operating points."
        }
        fairness_findings.append(fnr_finding)
        if primary_finding is None or fnr_val > 0.10:
            primary_finding = fnr_finding

        recommendations.append({
            "id": "REC-FAIR-FNR",
            "title": "Investigate Subgroup Underdiagnosis and False-Negative Rates",
            "category": "Fairness & Clinical Safety",
            "severity": SEVERITY_HIGH_ATTENTION if fnr_val > 0.10 else SEVERITY_WARNING,
            "trigger": f"FNR disparity is {fnr_val:.3f} (exceeds threshold 0.05).",
            "action": "Audit false-negative error patterns by demographic group to understand why positive cases are under-detected in specific cohorts.",
            "investigate": [
                "Group-specific clinical presentation variations",
                "Subgroup label quality and ground-truth verification",
                "Operating threshold adjustments across demographic partitions"
            ]
        })

    # B. Demographic Parity Difference (DPD)
    if not dpd_pass:
        dpd_finding = {
            "category": "Fairness",
            "title": "Demographic Parity Selection Gap",
            "severity": SEVERITY_WARNING,
            "explanation": f"Demographic parity difference ({dpd_val:.3f}) exceeds the project screening threshold (0.05), showing a gap in overall positive selection rates across groups.",
            "evidence": {"metric": "demographic_parity_difference", "value": dpd_val, "threshold": 0.05},
            "recommendation": "Review group-level positive prediction rates and evaluate demographic prevalence differences."
        }
        fairness_findings.append(dpd_finding)
        if primary_finding is None:
            primary_finding = dpd_finding

        recommendations.append({
            "id": "REC-FAIR-DPD",
            "title": "Review Group Positive Selection Rates",
            "category": "Fairness",
            "severity": SEVERITY_WARNING,
            "trigger": f"Demographic Parity Difference is {dpd_val:.3f} (threshold ≤ 0.05).",
            "action": "Investigate underlying demographic prevalence and potential sampling selection biases in training data.",
            "investigate": ["Dataset recruitment demographics", "Feature distribution parity", "Target variable definition consistency"]
        })

    # C. Disparate Impact Ratio (DIR / 80% Rule)
    if not dir_pass:
        dir_finding = {
            "category": "Fairness",
            "title": "Disparate Impact Ratio Deviation (80% Rule)",
            "severity": SEVERITY_WARNING,
            "explanation": f"Disparate impact ratio ({dir_val:.2f}) falls outside the screening range (0.80–1.25), indicating relative selection rate divergence between comparison cohorts.",
            "evidence": {"metric": "disparate_impact_ratio", "value": dir_val, "threshold_range": "0.80 - 1.25"},
            "recommendation": "Review positive prediction rates for the lowest-selection group relative to the reference group."
        }
        fairness_findings.append(dir_finding)
        if primary_finding is None:
            primary_finding = dir_finding

        recommendations.append({
            "id": "REC-FAIR-DIR",
            "title": "Assess Relative Group Selection Proportions",
            "category": "Fairness",
            "severity": SEVERITY_WARNING,
            "trigger": f"Disparate Impact Ratio is {dir_val:.2f} (outside 0.80–1.25 range).",
            "action": "Evaluate whether downstream clinical referral or risk-tier allocations will disproportionately impact specific demographic groups.",
            "investigate": ["Historical clinical referral rates", "Baseline population incidence", "Pre-processing reweighing options"]
        })

    # D. Equal Opportunity Difference (EOpD)
    if not eopd_pass:
        fairness_findings.append({
            "category": "Fairness",
            "title": "Equal Opportunity / Sensitivity Disparity",
            "severity": SEVERITY_WARNING,
            "explanation": f"Equal opportunity difference ({eopd_val:.3f}) exceeds the 0.05 threshold. The model exhibits different True Positive Rates (Sensitivity) across demographic cohorts.",
            "evidence": {"metric": "equal_opportunity_difference", "value": eopd_val, "threshold": 0.05},
            "recommendation": "Investigate differences in true-positive sensitivity rates across groups."
        })

    # E. Equalized Odds Difference (EOD)
    if not eod_pass:
        fairness_findings.append({
            "category": "Fairness",
            "title": "Equalized Odds Disparity",
            "severity": SEVERITY_WARNING,
            "explanation": f"Equalized odds difference ({eod_val:.3f}) exceeds the 0.05 threshold, indicating compound differences in TPR and/or FPR across groups.",
            "evidence": {"metric": "equalized_odds_difference", "value": eod_val, "threshold": 0.05},
            "recommendation": "Review both false-positive and false-negative trade-offs simultaneously across demographic groups."
        })

    # F. Subgroup Reliability Notice
    if has_insufficient_sample:
        rel_finding = {
            "category": "Fairness Reliability",
            "title": "Sample Size Limitation in Subgroup Evaluation",
            "severity": SEVERITY_INSUFFICIENT,
            "explanation": "One or more demographic subgroups in the evaluation set contain fewer than 30 observations. Disparity metrics for small cohorts have wider variance and must be interpreted with caution.",
            "evidence": {"insufficient_subgroups": [r["group"] for r in reliability_list if r.get("tier") == "insufficient" or r.get("n", 0) < 30]},
            "recommendation": "Collect or evaluate additional representative data before drawing conclusive algorithmic fairness determinations."
        }
        fairness_findings.append(rel_finding)
        primary_finding = rel_finding

        recommendations.append({
            "id": "REC-DATA-RELIABILITY",
            "title": "Expand Evaluation Cohort for Under-Represented Groups",
            "category": "Data Governance",
            "severity": SEVERITY_INSUFFICIENT,
            "trigger": "Subgroups contain n < 30 samples.",
            "action": "Augment evaluation cohort with multi-site or representative demographic data to establish statistically sound subgroup performance baselines.",
            "investigate": ["Demographic representation targets", "Multi-institution data pooling", "Data collection equity"]
        })

    # If all metrics passed and sufficient samples
    if len(fairness_findings) == 0:
        fairness_findings.append({
            "category": "Fairness",
            "title": "Fairness Metrics Within Project Thresholds",
            "severity": SEVERITY_INFO,
            "explanation": "All 5 screening fairness metrics (DPD, DIR, EOpD, EOD, FNR disparity) satisfy project thresholds across evaluated demographic groups.",
            "evidence": {"status": "FAIR_BALANCED", "metrics_evaluated": 5},
            "recommendation": "Maintain ongoing algorithmic fairness surveillance in prospective clinical monitoring."
        })

    # =========================================================================
    # 3. SUBGROUP PERFORMANCE DIFFERENCES
    # =========================================================================
    if len(subgroups) >= 2:
        # Find highest and lowest TPR
        sorted_by_tpr = sorted(subgroups, key=lambda s: s.get("tpr", s.get("sensitivity", 0.0)), reverse=True)
        highest_tpr_grp = sorted_by_tpr[0]
        lowest_tpr_grp = sorted_by_tpr[-1]

        # Find highest FNR
        sorted_by_fnr = sorted(subgroups, key=lambda s: s.get("fnr", 0.0), reverse=True)
        highest_fnr_grp = sorted_by_fnr[0]

        subgroup_findings.append({
            "category": "Subgroup Performance",
            "title": "Subgroup Sensitivity Variation",
            "severity": SEVERITY_WARNING if (highest_tpr_grp.get("tpr", 0.0) - lowest_tpr_grp.get("tpr", 0.0)) > 0.05 else SEVERITY_INFO,
            "explanation": f"Sensitivity varies from {lowest_tpr_grp.get('group', 'Group A')} ({lowest_tpr_grp.get('tpr', 0.0)*100:.1f}%) to {highest_tpr_grp.get('group', 'Group B')} ({highest_tpr_grp.get('tpr', 0.0)*100:.1f}%).",
            "evidence": {
                "highest_sensitivity_group": highest_tpr_grp.get("group"),
                "highest_sensitivity_val": highest_tpr_grp.get("tpr", 0.0),
                "lowest_sensitivity_group": lowest_tpr_grp.get("group"),
                "lowest_sensitivity_val": lowest_tpr_grp.get("tpr", 0.0)
            },
            "recommendation": f"Monitor diagnostic performance closely for group '{lowest_tpr_grp.get('group', 'Unknown')}' to ensure equitable clinical sensitivity."
        })

        if highest_fnr_grp.get("fnr", 0.0) > 0.10:
            subgroup_findings.append({
                "category": "Subgroup Performance",
                "title": f"Elevated False-Negative Rate in {highest_fnr_grp.get('group')}",
                "severity": SEVERITY_WARNING,
                "explanation": f"Subgroup '{highest_fnr_grp.get('group')}' exhibits a false-negative rate of {highest_fnr_grp.get('fnr', 0.0)*100:.1f}%, indicating higher missed-case frequency.",
                "evidence": {"group": highest_fnr_grp.get("group"), "fnr": highest_fnr_grp.get("fnr", 0.0), "n": highest_fnr_grp.get("n", 0)},
                "recommendation": "Review clinical features and decision boundaries specifically affecting this subgroup."
            })

    # =========================================================================
    # 4. EXPLAINABILITY (SHAP) INSIGHTS
    # =========================================================================
    if top_shap_features:
        feat_names = [f.get("feature", "Feature") for f in top_shap_features[:3]]
        feat_str = ", ".join(feat_names)
        explainability_findings.append({
            "category": "Explainability",
            "title": "Primary Model-Influencing Predictors",
            "severity": SEVERITY_INFO,
            "explanation": f"The model's strongest overall feature influences were {feat_str}. These features contributed most to shifting prediction probabilities away from the cohort baseline.",
            "evidence": {"top_features": feat_names},
            "recommendation": "Review feature relevance with clinical subject matter experts to confirm clinical alignment with established disease pathways."
        })

        if is_sensitive_in_features:
            explainability_findings.append({
                "category": "Explainability",
                "title": "Direct Protected Attribute Inclusion",
                "severity": SEVERITY_WARNING,
                "explanation": f"The protected demographic attribute '{sensitive_col}' is directly present in the model's feature space. While SHAP attributions reflect mathematical reliance, direct inclusion warrants clinical justification.",
                "evidence": {"sensitive_attribute": sensitive_col, "included_in_features": True},
                "recommendation": "Assess whether omitting the sensitive attribute reduces direct disparate impact or if domain-specific calibration requires explicit demographic modeling."
            })
            recommendations.append({
                "id": "REC-XAI-SENSITIVE-FEATURE",
                "title": "Review Protected Attribute Inclusion Policy",
                "category": "Model Governance",
                "severity": SEVERITY_WARNING,
                "trigger": f"Sensitive attribute '{sensitive_col}' is used as a model feature.",
                "action": "Conduct feature ablation experiments comparing model performance and fairness with and without the protected attribute.",
                "investigate": ["Potential proxy feature redundancies", "Clinical necessity of demographic input", "Algorithmic fairness impact of attribute removal"]
            })
    elif shap_results and shap_results.get("status") == "unavailable":
        explainability_findings.append({
            "category": "Explainability",
            "title": "SHAP Unavailable — External Model Audit",
            "severity": SEVERITY_INFO,
            "explanation": "Model internals were not provided in this external audit. Feature-level SHAP attributions require access to underlying model architectures or estimators and are withheld to maintain scientific integrity.",
            "evidence": {"status": "unavailable", "reason": "model_internals_not_provided"},
            "recommendation": "If feature-level attribution is required, execute a localized white-box model audit with access to model internals."
        })
    else:
        explainability_findings.append({
            "category": "Explainability",
            "title": "Explainability Analysis Available",
            "severity": SEVERITY_INFO,
            "explanation": "SHAP feature attribution analysis provides game-theoretic feature impact breakdowns.",
            "evidence": {"status": "available"},
            "recommendation": "Inspect patient-level SHAP waterfalls in the Explainable AI section."
        })

    # =========================================================================
    # 5. OVERALL SEVERITY & PRIMARY FINDING SYNTHESIS
    # =========================================================================
    if has_insufficient_sample:
        overall_severity = SEVERITY_INSUFFICIENT
    elif not fnr_pass and fnr_val > 0.10:
        overall_severity = SEVERITY_HIGH_ATTENTION
    elif not dpd_pass or not dir_pass or not eopd_pass or not eod_pass or not fnr_pass or (prec - rec) >= 0.15:
        overall_severity = SEVERITY_WARNING
    else:
        overall_severity = SEVERITY_INFO

    if primary_finding is None:
        if fairness_findings:
            primary_finding = fairness_findings[0]
        else:
            primary_finding = {
                "title": "Model and Fairness Metrics Evaluated",
                "explanation": "Evaluation completed across model performance, fairness, and explainability standards.",
                "severity": SEVERITY_INFO,
                "recommendation": "Proceed with regular clinical AI governance and validation protocols."
            }

    # Default next action
    if recommendations:
        next_action_text = recommendations[0]["action"]
    else:
        next_action_text = "Proceed to export the audit-ready clinical summary report and log findings in analysis history."

    # Fairness summary statement
    if has_insufficient_sample:
        fairness_summary_text = "Fairness evaluation is inconclusive due to limited subgroup sample sizes (n < 30)."
    elif overall_severity == SEVERITY_INFO:
        fairness_summary_text = "All 5 core fairness screening metrics meet project thresholds."
    elif not fnr_pass:
        fairness_summary_text = f"Underdiagnosis disparity ({fnr_val:.3f}) requires clinical investigation."
    else:
        fairness_summary_text = f"Moderate demographic disparity detected (DPD = {dpd_val:.3f}, DIR = {dir_val:.2f})."

    # Explainability summary statement
    if top_shap_features:
        feat_names = [f.get("feature", "Feature") for f in top_shap_features[:3]]
        explain_summary_text = f"Top model predictors: {', '.join(feat_names)}."
    elif shap_results and shap_results.get("status") == "unavailable":
        explain_summary_text = "Feature-level SHAP unavailable — model internals were not provided."
    else:
        explain_summary_text = "Feature importances evaluated via SHAP Shapley values."

    is_shap_unavailable = bool(shap_results and shap_results.get("status") == "unavailable")

    # =========================================================================
    # 6. STRUCTURED OUTPUT BUNDLE
    # =========================================================================
    return {
        "status": "success",
        "overall_severity": overall_severity,
        "summary": {
            "model_name": model_name,
            "target_outcome": target_col,
            "protected_attribute": sensitive_col,
            "performance_summary": perf_summary_text,
            "fairness_summary": fairness_summary_text,
            "explainability_summary": explain_summary_text,
            "explainability_status": "unavailable" if is_shap_unavailable else ("available" if top_shap_features else "unavailable"),
            "explainability_pill": "External model internals not provided" if is_shap_unavailable else ("Attributions Ready" if top_shap_features else "Model internals not provided"),
            "primary_finding": primary_finding["title"],
            "primary_explanation": primary_finding["explanation"],
            "primary_severity": primary_finding["severity"],
            "recommended_next_action": next_action_text,
        },
        "key_finding": primary_finding,
        "performance_findings": performance_findings,
        "fairness_findings": fairness_findings,
        "subgroup_findings": subgroup_findings,
        "explainability_findings": explainability_findings,
        "recommendations": recommendations,
        "screening_thresholds": {
            "dpd": "≤ 0.05",
            "dir": "0.80 – 1.25 (80% Rule)",
            "eopd": "≤ 0.05",
            "eod": "≤ 0.05",
            "fnr_disparity": "≤ 0.05",
            "transparency_note": "These are project screening criteria and are not universal clinical or regulatory standards."
        },
        "disclaimers": {
            "clinical_safety": "Insights and recommendations reflect statistical evaluation of the provided dataset and model. They do not constitute medical advice or clinical diagnoses.",
            "fairness_interpretation": "SHAP explains model feature attributions; it does not establish demographic causation or prove discrimination.",
        }
    }

