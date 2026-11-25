"""
Asclepius FairAI - Audit Result Container
Phase 14: Structured Audit Results, Human-Readable Console Summaries, Machine-Readable Dictionaries,
PDF Report Generation, and SQLite History Persistence
"""

from typing import Dict, Any, Optional
import os

from src.core.report_generator import generate_fairness_report
from src.core.history import save_analysis


class AuditResult:
    """
    Encapsulates complete local evaluation outputs from a FairAIAuditor run.
    Provides concise summaries, report generation, history logging, and data export.
    """

    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle
        model_res = bundle.get("model_results", {})
        fair_res = bundle.get("fairness_results", {})
        shap_res = bundle.get("shap_results", {})
        ins_res = bundle.get("insights_results", {})
        mit_res = bundle.get("mitigation_results", {})
        ds_info = bundle.get("dataset_info", {})

        self.audit_id: str = bundle.get("audit_id", "ANL-AUDIT")
        self.model_name: str = model_res.get("model_name", "Clinical Classifier")
        self.model_version: str = model_res.get("model_version", "v1.0")
        self.model_type: str = model_res.get("model_type", "scikit-learn")
        self.target_name: str = model_res.get("target", "Outcome")
        self.sensitive_name: str = model_res.get("sensitive_attribute", "Protected Group")
        self.sample_count: int = ds_info.get("rows", model_res.get("test_samples", 0))

        self.performance: Dict[str, Any] = model_res
        self.fairness: Dict[str, Any] = fair_res
        self.shap: Dict[str, Any] = shap_res
        self.insights: Dict[str, Any] = ins_res
        self.mitigation: Dict[str, Any] = mit_res
        self.report_path: Optional[str] = bundle.get("report_path")

    def summary(self) -> str:
        """
        Produces a concise, professional developer-facing console summary.
        """
        lines = [
            "ASCLÉPIUS FAIRAI AUDIT",
            "----------------------",
            "",
            f"Model: {self.model_name}",
            f"Version: {self.model_version}",
            f"Samples: {self.sample_count}",
            "",
            "Performance",
            f"Accuracy: {self.performance.get('accuracy', 0.0):.2f}",
            f"Balanced Accuracy: {self.performance.get('balanced_accuracy', 0.0):.2f}",
        ]

        roc_auc = self.performance.get("roc_auc")
        if roc_auc is not None and isinstance(roc_auc, (int, float)):
            lines.append(f"ROC-AUC: {roc_auc:.2f}")
        else:
            lines.append("ROC-AUC: Unavailable (Continuous probabilities not supplied)")

        lines.append("")
        lines.append("Fairness")
        f_metrics = self.fairness.get("metrics", {})
        dpd = f_metrics.get("demographic_parity_difference", {})
        dir_m = f_metrics.get("disparate_impact_ratio", {})
        eopd = f_metrics.get("equal_opportunity_difference", {})
        eod = f_metrics.get("equalized_odds_difference", {})
        fnrd = f_metrics.get("fnr_disparity", {})

        dpd_val = dpd.get("value", 0.0)
        dpd_tag = "PASS" if dpd.get("passed", True) else "REVIEW"
        lines.append(f"DPD: {dpd_val:.2f} {dpd_tag}")

        dir_val = dir_m.get("value", 1.0)
        dir_tag = "PASS" if dir_m.get("passed", True) else "REVIEW"
        lines.append(f"DIR: {dir_val:.2f} {dir_tag}")

        eopd_val = eopd.get("value", 0.0)
        eopd_tag = "PASS" if eopd.get("passed", True) else "REVIEW"
        lines.append(f"EOpD: {eopd_val:.2f} {eopd_tag}")

        eod_val = eod.get("value", 0.0)
        eod_tag = "PASS" if eod.get("passed", True) else "REVIEW"
        lines.append(f"EOD: {eod_val:.2f} {eod_tag}")

        fnrd_val = fnrd.get("value", 0.0)
        fnrd_tag = "PASS" if fnrd.get("passed", True) else "REVIEW"
        lines.append(f"FNRD: {fnrd_val:.2f} {fnrd_tag}")

        lines.append("")
        lines.append("Reliability")
        rel_stat = self.fairness.get("overall_reliability", "Standard Evaluation")
        lines.append(rel_stat)

        lines.append("")
        lines.append("SHAP")
        shap_stat = self.shap.get("status")
        if shap_stat == "success" and len(self.shap.get("global_importance", [])) > 0:
            lines.append("Available")
        elif shap_stat == "unavailable":
            lines.append("Unavailable (Model internals withheld)")
        else:
            lines.append("Not Available")

        lines.append("")
        lines.append("Mitigation")
        mit_stat = self.mitigation.get("status")
        if mit_stat == "success":
            sel = self.mitigation.get("selection", {})
            cand = sel.get("selected_candidate", {})
            th = cand.get("threshold", 0.50)
            lines.append(f"Candidate threshold: {th:.2f}")
        elif self.mitigation.get("availability") == "unavailable" or not self.performance.get("has_probabilities", False):
            lines.append("Disabled (Requires continuous prediction probabilities)")
        else:
            lines.append("Not Evaluated")

        lines.append("")
        lines.append("Report")
        if self.report_path:
            lines.append(f"Generated: {os.path.basename(self.report_path)}")
        else:
            lines.append("Ready to generate via result.generate_report()")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns full machine-readable audit dictionary.
        """
        return self.bundle

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generates an audit-ready clinical PDF report reusing the core report engine.
        Returns the absolute filepath to the created PDF document.
        """
        rep_res = generate_fairness_report(
            analysis_results=self.bundle,
            output_path=output_path,
        )
        if rep_res.get("status") == "success":
            saved_path = rep_res.get("filepath", output_path)
            self.report_path = saved_path
            self.bundle["report_filename"] = os.path.basename(saved_path) if saved_path else None
            self.bundle["report_path"] = saved_path
            return saved_path
        else:
            raise RuntimeError(f"Report generation failed: {rep_res.get('error_message')}")

    def save_history(self, db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Persists summarized metadata to the SQLite analysis history database.
        Strictly zero PHI and zero raw evaluation vectors are saved.
        """
        res = save_analysis(self.bundle, db_path=db_path)
        return res

    def __repr__(self) -> str:
        return f"<AuditResult model='{self.model_name}' samples={self.sample_count} verdict='{self.fairness.get('verdict_title', 'Evaluated')}'>"
