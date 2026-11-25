# Asclepius FairAI — Local Audit SDK Guide
**Version:** 1.0.0 (Phase 14)  
**Vision:** *"Before Healthcare AI Goes Live, Run Asclepius."*

---

## 1. Overview & Philosophy

Asclepius FairAI provides a local-first Python SDK and Command-Line Interface (CLI) enabling healthcare AI creators to audit models in their own compute environment without uploading model weights or sensitive patient cohorts to third-party cloud servers.

```
Healthcare AI Developer
        ↓
Creates & fits healthcare AI model (scikit-learn)
        ↓
Keeps model + evaluation data in their own environment
        ↓
Runs Asclepius locally (`FairAIAuditor`)
        ↓
Performance + Fairness + Explainability (SHAP) + Mitigation
        ↓
Audit Evidence & Findings
        ↓
Publication-Grade PDF Report & SQLite History
        ↓
Review / Calibrate / Deploy
```

---

## 2. Installation

### From Workspace / Source
```bash
pip install -e .
```

### Direct Import
Both import patterns are supported:
```python
from asclepius import FairAIAuditor, AuditResult, AuditConfig
# or
from src.asclepius import FairAIAuditor
```

---

## 3. Supported Model Types

The SDK accepts fitted scikit-learn estimators and pipelines directly:
- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- Custom `sklearn.pipeline.Pipeline` with transformers and estimators

> **Security Rule:** Asclepius does not accept or deserialize arbitrary `.pkl`, `.pickle`, or `.joblib` model binaries from untrusted sources. Pass your already instantiated and fitted model object directly in memory.

---

## 4. Quickstart Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from asclepius import FairAIAuditor

# 1. Load data and split
df = pd.read_csv("patient_cohort.csv")
X = df.drop(columns=["target"])
y = df["target"]
sensitive = df["sex"]

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive, test_size=0.25, random_state=42
)

# 2. Fit model locally
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Instantiate auditor
auditor = FairAIAuditor(
    model=model,
    X_test=X_test,
    y_test=y_test,
    sensitive_features=sens_test,
    target_name="30-Day Readmission",
    sensitive_name="Sex",
    model_name="Readmission Risk Classifier",
    model_version="v1.0"
)

# 4. Execute audit
result = auditor.run()

# 5. Review summary
print(result.summary())

# 6. Generate PDF audit report
result.generate_report("outputs/reports/audit_report.pdf")

# 7. Optionally save to local SQLite history (Zero PHI)
result.save_history()
```

---

## 5. Console Summary Output

Calling `result.summary()` prints the standardized developer evaluation output:

```text
ASCLÉPIUS FAIRAI AUDIT
----------------------

Model: Readmission Risk Classifier
Version: v1.0
Samples: 150

Performance
Accuracy: 0.87
Balanced Accuracy: 0.86
ROC-AUC: 0.91

Fairness
DPD: 0.04 PASS
DIR: 0.91 PASS
EOpD: 0.06 REVIEW
EOD: 0.05 PASS
FNRD: 0.03 PASS

Reliability
Standard Evaluation

SHAP
Available

Mitigation
Candidate threshold: 0.60

Report
Generated: audit_report.pdf
```

---

## 6. Algorithmic Fairness Engine

Asclepius reuses its locked clinical screening criteria:

| Metric | Project Screening Threshold | Clinical Significance |
|:---|:---:|:---|
| **Demographic Parity Diff (DPD)** | $\le 0.05$ | Overall positive selection rate parity |
| **Disparate Impact Ratio (DIR)** | $0.80 - 1.25$ | Four-Fifths / 80% Rule selection ratio |
| **Equal Opportunity Diff (EOpD)**| $\le 0.05$ | Sensitivity / True Positive Rate parity |
| **Equalized Odds Diff (EOD)** | $\le 0.05$ | Average of TPR and FPR disparities |
| **Underdiagnosis Disparity (FNR)** | $\le 0.05$ | False Negative Rate disparity (missed disease cases) |

### Subgroup Sample Reliability Tiers
- **$n < 30$**: `Insufficient Evidence` (Variance too high for definitive conclusions)
- **$30 \le n < 50$**: `Caution` (Moderate sample size)
- **$n \ge 50$**: `Standard Evaluation` (Statistically robust cohort size)

---

## 7. Explainable AI (SHAP)

Because the SDK operates in-memory with direct access to the fitted estimator:
- **Random Forest & Gradient Boosting**: Evaluated using SHAP `TreeExplainer`.
- **Logistic Regression**: Evaluated using SHAP `LinearExplainer`.
- **Sample Cap**: Evaluates up to `max_shap_samples=500` (configurable) for rapid local turnaround.
- **Scientific Caveat**: SHAP values explain game-theoretic feature attribution on model predictions; they do not establish biological or clinical causation.

---

## 8. Threshold Mitigation

When the model supports `.predict_proba()`:
- `FairAIAuditor` evaluates candidate decision thresholds $\theta \in [0.20, 0.80]$.
- Identifies optimal operating points that reduce demographic disparities (e.g., FNR or DPD) while bounding performance drops to $\le 5.0\%$.
- Leaves the original fitted baseline estimator intact.

If `.predict_proba()` is not implemented:
- Mitigation is defensively disabled with message:
  > *"Decision Threshold Calibration requires continuous probability scores (y_prob). Only discrete predictions were supplied."*

---

## 9. PDF Report Generation & SQLite History

### Report Generation
```python
# Save to default path (outputs/reports/asclepius_fairai_report_<timestamp>.pdf)
path = result.generate_report()

# Or specify a custom destination:
path = result.generate_report("my_audit.pdf")
```

### Persistent History Logging
```python
result.save_history()
```
- Logs record in `outputs/history.db` under `model_type = "scikit-learn"` or custom architecture.
- **Zero PHI Compliance**: Strictly persists only aggregated statistics, confusion matrix counts, and structured findings. Raw evaluation matrices (`X_test`, `y_test`, `y_pred`, `sensitive_features`) are never saved.

---

## 10. Command-Line Interface (CLI)

Asclepius provides a CLI for scripting and automation:

```bash
# Help menu
asclepius --help
asclepius audit --help

# System and architecture info
asclepius info

# Run built-in clinical demonstration audit
asclepius audit --demo

# Run audit on a local CSV dataset
asclepius audit --data clinical_test.csv --target readmitted --sensitive gender --report audit.pdf --save-history

# Output JSON machine-readable summary
asclepius audit --demo --format json
```

---

## 11. Privacy & Local-First Guarantees

- **No Remote Telemetry**: Zero analytics, telemetry, or network calls.
- **Complete Offline Operation**: Runs isolated in air-gapped environments.
- **Zero PHI Retention**: Evaluated tensors exist only ephemerally in RAM during execution.
- *Notice*: Designed for local model evaluation so sensitive evaluation data can remain within the model owner's environment. Does not constitute legal advice or guarantee regulatory compliance.

---

## 12. Future Product Roadmap

The following phases build on the local SDK:
- **Phase 15**: REST API Evaluation Gateway
- **Phase 16**: CI/CD Algorithmic Fairness Gate
- **Phase 17**: Local Standalone Audit Agent
- **Phase 18**: Prospective Clinical Model Surveillance
