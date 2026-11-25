# Asclepius FairAI -- Presentation Talking Points & Defense Q&A

This document provides concise, rigorous, and defensible answers for presentation questions, technical defenses, and clinical discussions.

---

### 1. "What problem are you solving?"
**Answer**: Clinical machine learning models often exhibit unequal performance across demographic subgroups (e.g. race, gender, age, insurance tier), resulting in elevated false-negative rates (underdiagnosis) or unequal allocation of healthcare resources. Asclepius FairAI provides a standardized, transparent audit pipeline to quantify and evaluate these disparities prior to clinical translation.

---

### 2. "Where is the AI in this platform?"
**Answer**: AI and Machine Learning operate in four discrete layers:
1. **Supervised ML Models**: Scikit-Learn classifiers (Logistic Regression, Random Forest, Gradient Boosting) trained on clinical predictors to output risk predictions.
2. **Explainable AI (SHAP)**: Game-theoretic Shapley value attribution (`TreeExplainer` / `LinearExplainer`) quantifying feature importance.
3. **Deterministic Insights Engine**: Rule-based synthesis interpreting statistical patterns into clinical governance findings.
4. **Candidate Mitigation Search**: Algorithmic optimization evaluating candidate decision thresholds.
*Note: Fairness metrics themselves are mathematical statistical parity indicators, not machine learning models.*

---

### 3. "Why is fairness important in clinical AI?"
**Answer**: In medicine, false negatives can lead to delayed intervention, while false positives can lead to unnecessary invasive procedures. If a model exhibits an elevated False Negative Rate in an underserved cohort, it exacerbates systemic healthcare disparities. Ensuring demographic parity and error-rate equality is essential for clinical safety and equitable care.

---

### 4. "How is bias detected?"
**Answer**: The platform evaluates 5 standard screening metrics against project criteria:
- **Demographic Parity Difference (DPD <= 0.05)**
- **Disparate Impact Ratio (DIR 0.80 - 1.25)**
- **Equal Opportunity Difference (EOpD <= 0.05)**
- **Equalized Odds Difference (EOD <= 0.05)**
- **False Negative Rate Disparity (FNR <= 0.05)**

---

### 5. "What is SHAP doing?"
**Answer**: SHAP (SHapley Additive exPlanations) computes the marginal contribution of each feature to the model's output relative to the base expectation. This verifies whether the model relies on legitimate clinical predictors (e.g. ICU length of stay, comorbidity index) versus inappropriate demographic proxies.

---

### 6. "Why do you need sample-size reliability rules?"
**Answer**: Small subgroups suffer from high statistical variance. In cohorts with n < 30, a shift of just 1 or 2 patient outcomes drastically swings sensitivity and selection rates, producing misleading fairness alerts. Asclepius FairAI assigns reliability tiers (n < 30: *Insufficient Evidence*, 30-49: *Caution*, 50+: *Robust*) to prevent false alarms.

---

### 7. "Why don't you call a disparity proof of bias?"
**Answer**: A statistical disparity indicates an uneven distribution of predictions, but does not automatically prove algorithmic bias, discrimination, or clinical harm. Disparities can arise from baseline disease prevalence differences, sampling artifacts, or clinical co-factors. The tool serves as a screening instrument requiring expert clinical review.

---

### 8. "Why don't you claim HIPAA compliance?"
**Answer**: HIPAA compliance requires full administrative, physical, and technical organizational safeguards, business associate agreements (BAAs), and institutional certification. Asclepius FairAI is an academic and research software tool. To protect privacy by design, the platform enforces Zero-PHI storage and processes data in-memory.

---

### 9. "What does mitigation actually change?"
**Answer**: The mitigation engine tests post-processing *Decision Threshold Adjustment*. Instead of classifying patients at the default threshold (theta = 0.50), it identifies an operational threshold (e.g. theta = 0.65) that balances false-negative and false-positive rates across groups while constraining overall performance degradation to <= 5 percentage points.

---

### 10. "Why does ROC-AUC remain unchanged when only the threshold changes?"
**Answer**: ROC-AUC measures the model's ability to rank-order continuous predicted probabilities across all possible decision thresholds. Adjusting the classification decision cut-point changes binary predictions (affecting Accuracy, Sensitivity, Specificity, FNR, and DPD), but does not alter the underlying continuous probabilities or their rank order.

---

### 11. "What happens if the subgroup is too small?"
**Answer**: If any subgroup has n < 30, the fairness verdict automatically defaults to `INSUFFICIENT_EVIDENCE`. The system surfaces a clinical safety notice recommending larger sample collection before drawing conclusions.

---

### 12. "Is this a medical diagnostic system?"
**Answer**: No. Asclepius FairAI is strictly a model auditing, research, and educational platform. It does not provide medical diagnoses, treatment recommendations, or autonomous clinical decision-making.

---

### 13. "What is the project's future scope?"
**Answer**: Future development targets:
- In-processing adversarial debiasing and fair representation learning.
- Intersectional multi-attribute fairness auditing (e.g. race x gender x age).
- Integration with FHIR (Fast Healthcare Interoperability Resources) data pipelines.
- Longitudinal post-market bias monitoring over time.
