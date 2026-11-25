"""
Asclepius FairAI - Explainability Core Bridge
"""

from core.explainability import generate_shap_explanations


def explain_model(model):
    """Legacy compatibility bridge."""
    return {"status": "ready", "module": "explainability", "engine": "SHAP"}
