"""
Asclepius FairAI - Root-level Package Alias
Allows direct import: `from asclepius import FairAIAuditor, AuditResult, AuditConfig`
"""

import sys
import os

# Ensure src is in sys.path
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

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
