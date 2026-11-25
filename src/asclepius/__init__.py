"""
Asclepius FairAI - Local Developer SDK
Phase 14: 'Before Healthcare AI Goes Live, Run Asclepius.'

Provides programmatic, local-first fairness auditing, subgroup reliability analysis,
SHAP explainability, deterministic clinical recommendations, and threshold mitigation
for healthcare machine learning models.
"""

from src.asclepius.config import AuditConfig
from src.asclepius.result import AuditResult
from src.asclepius.auditor import FairAIAuditor

__version__ = "1.0.0"
__all__ = [
    "FairAIAuditor",
    "AuditResult",
    "AuditConfig",
    "__version__",
]
