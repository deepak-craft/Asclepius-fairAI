"""
Asclepius FairAI - Local Audit SDK Configuration
Phase 14: Local-First Developer SDK Configuration Tokens & Default Thresholds
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class AuditConfig:
    """
    Configuration options for local algorithmic fairness & model auditing.
    """
    # Core screening thresholds
    dpd_threshold: float = 0.05
    dir_min_threshold: float = 0.80
    dir_max_threshold: float = 1.25
    eopd_threshold: float = 0.05
    eod_threshold: float = 0.05
    fnr_threshold: float = 0.05

    # Subgroup reliability thresholds
    insufficient_sample_threshold: int = 30
    caution_sample_threshold: int = 50

    # SHAP performance settings
    max_shap_samples: int = 500
    explainability_enabled: bool = True

    # Mitigation settings
    mitigation_enabled: bool = True
    mitigation_objective: str = "balance_fairness_performance"
    mitigation_max_perf_drop: float = 0.05

    # Reporting settings
    default_report_title: str = "Healthcare AI Fairness Audit Report"

    # Privacy guarantee
    offline_mode: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dpd_threshold": self.dpd_threshold,
            "dir_range": [self.dir_min_threshold, self.dir_max_threshold],
            "eopd_threshold": self.eopd_threshold,
            "eod_threshold": self.eod_threshold,
            "fnr_threshold": self.fnr_threshold,
            "insufficient_sample_threshold": self.insufficient_sample_threshold,
            "caution_sample_threshold": self.caution_sample_threshold,
            "max_shap_samples": self.max_shap_samples,
            "mitigation_objective": self.mitigation_objective,
            "offline_mode": self.offline_mode,
        }
