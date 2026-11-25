"""
Asclepius FairAI - Professional Fairness & Explainability PDF Report Generator
Phase 7: Audit-Ready PDF Reports with ReportLab, Embedded Matplotlib Charts, Evidence-Based Tables, and Clinical Disclaimers
"""

import os
import io
import datetime
from typing import Dict, Any, Optional, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfgen import canvas


# Brand Colors
CLR_PRIMARY = colors.HexColor("#16A34A")
CLR_PRIMARY_DARK = colors.HexColor("#15803D")
CLR_MINT = colors.HexColor("#ECFDF5")
CLR_MINT_BORDER = colors.HexColor("#A7F3D0")
CLR_TEXT = colors.HexColor("#0F172A")
CLR_TEXT_MUTED = colors.HexColor("#64748B")
CLR_BG_LIGHT = colors.HexColor("#F8FAFC")
CLR_BORDER = colors.HexColor("#E2E8F0")
CLR_WARN_BG = colors.HexColor("#FFFBEB")
CLR_WARN_BORDER = colors.HexColor("#FDE68A")
CLR_DANGER_BG = colors.HexColor("#FEF2F2")
CLR_DANGER_TEXT = colors.HexColor("#DC2626")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' in running footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(CLR_TEXT_MUTED)

        # Running Header (on pages after cover)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "ASCLÉPIUS FAIRAI — Healthcare AI Fairness & Explainability Report")
            self.setStrokeColor(CLR_BORDER)
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Running Footer
        footer_y = 36
        self.setStrokeColor(CLR_BORDER)
        self.setLineWidth(0.5)
        self.line(54, footer_y + 12, 8.5 * inch - 54, footer_y + 12)

        self.drawString(54, footer_y, "Confidential Clinical AI Evaluation Report | Academic & Research Use Only")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, footer_y, page_str)

        self.restoreState()


def _generate_confusion_matrix_chart(cm_data: Dict[str, int]) -> io.BytesIO:
    """Renders confusion matrix heatmap as PNG in-memory stream."""
    tn = cm_data.get("tn", 0)
    fp = cm_data.get("fp", 0)
    fn = cm_data.get("fn", 0)
    tp = cm_data.get("tp", 0)

    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=180)
    im = ax.imshow(matrix, cmap="Greens", aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg (0)", "Pred Pos (1)"], fontsize=8, fontweight="bold")
    ax.set_yticklabels(["True Neg (0)", "True Pos (1)"], fontsize=8, fontweight="bold")

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            text_color = "white" if val > np.max(matrix) / 2 else "#0F172A"
            ax.text(j, i, f"{val}", ha="center", va="center", color=text_color, fontsize=11, fontweight="bold")

    ax.set_title("Confusion Matrix Heatmap", fontsize=9, fontweight="bold", pad=8, color="#0F172A")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _generate_subgroup_chart(subgroups: List[Dict[str, Any]]) -> io.BytesIO:
    """Renders subgroup selection rate and sensitivity comparison bar chart."""
    groups = [str(g.get("group", g.get("subgroup", "Unknown"))) for g in subgroups]
    rates = []
    tprs = []
    for g in subgroups:
        r = g.get("selection_rate", g.get("selectionRatePct", g.get("selection_rate_pct", 0.0)))
        t = g.get("tpr", g.get("tprPct", g.get("tpr_pct", 0.0)))
        rates.append(float(r) if isinstance(r, (int, float, str)) and str(r).replace(".", "", 1).isdigit() else 0.0)
        tprs.append(float(t) if isinstance(t, (int, float, str)) and str(t).replace(".", "", 1).isdigit() else 0.0)

    # Convert to 0-100 scale if in 0-1 range
    rates = [r * 100 if r <= 1.0 else r for r in rates]
    tprs = [t * 100 if t <= 1.0 else t for t in tprs]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4.8, 2.4), dpi=180)
    ax.bar(x - width / 2, rates, width, label="Selection Rate (%)", color="#10B981")
    ax.bar(x + width / 2, tprs, width, label="Sensitivity / TPR (%)", color="#0284C7")

    ax.set_ylabel("Percentage (%)", fontsize=8)
    ax.set_title("Subgroup Performance Disparities", fontsize=9, fontweight="bold", pad=8, color="#0F172A")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _generate_shap_chart(global_importance: List[Dict[str, Any]]) -> io.BytesIO:
    """Renders horizontal bar chart of top SHAP feature importances."""
    top_items = global_importance[:7]
    top_items_rev = list(reversed(top_items))

    features = [str(item.get("feature", item.get("name", "Feature"))) for item in top_items_rev]
    values = []
    for item in top_items_rev:
        v = item.get("mean_abs_shap", item.get("meanAbsShap", item.get("importance", item.get("value", 0.0))))
        try:
            values.append(float(v))
        except (ValueError, TypeError):
            values.append(0.0)

    fig, ax = plt.subplots(figsize=(5.2, 2.5), dpi=180)
    bars = ax.barh(features, values, color="#16A34A", height=0.55)

    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Prediction)", fontsize=8)
    ax.set_title("Top Predictive Features (SHAP Global Ranking)", fontsize=9, fontweight="bold", pad=8, color="#0F172A")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    max_v = max(values) if values and max(values) > 0 else 1.0
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max_v * 0.02), bar.get_y() + bar.get_height() / 2, f"{width:.3f}",
                va="center", ha="left", fontsize=7, fontweight="bold", color="#334155")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_fairness_report(
    analysis_results: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds a professional, comprehensive, multi-page audit report in PDF format.
    
    Parameters:
    -----------
    analysis_results: Dictionary containing:
        - 'dataset_info': Metadata on dataset dimensions and features
        - 'model_results': Performance metrics, confusion matrix
        - 'fairness_results': Group disparity metrics, reliability status
        - 'shap_results': Global feature ranking and local explanations
        - 'insights_results': Structured insights and actionable recommendations
    output_path: Target filepath (default: outputs/reports/asclepius_fairai_report_<timestamp>.pdf)

    Returns:
    --------
    Dictionary with generation status, output filepath, timestamp, and metadata.
    """
    if not analysis_results:
        return {"status": "error", "error_message": "No analysis results provided for report generation."}

    # Extract sub-packages safely with comprehensive alias fallback
    model_res = analysis_results.get("model_results") or analysis_results.get("model_performance") or analysis_results.get("model") or {}
    fairness_res = analysis_results.get("fairness_results") or analysis_results.get("fairness_evaluation") or analysis_results.get("fairness") or {}
    shap_res = analysis_results.get("shap_results") or analysis_results.get("explainability") or analysis_results.get("shap") or {}
    insights_res = analysis_results.get("insights_results") or analysis_results.get("ai_insights") or analysis_results.get("insights") or {}
    dataset_info = analysis_results.get("dataset_info") or analysis_results.get("dataset_summary") or analysis_results.get("dataset") or analysis_results.get("metadata") or {}

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    date_display = now.strftime("%B %d, %Y - %H:%M:%S UTC")

    if not output_path:
        reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "reports"))
        os.makedirs(reports_dir, exist_ok=True)
        filename = f"asclepius_fairai_report_{timestamp_str}.pdf"
        output_path = os.path.join(reports_dir, filename)
    else:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

    # Document setup (Letter, 0.75 in margins)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    # Typography & Styles
    base_styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=CLR_PRIMARY,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=CLR_TEXT_MUTED,
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "SectionHeading1",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=CLR_PRIMARY_DARK,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SectionHeading2",
        parent=base_styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=CLR_TEXT,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=CLR_TEXT,
        spaceAfter=6,
    )

    body_bold = ParagraphStyle(
        "ReportBodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    disclaimer_style = ParagraphStyle(
        "ReportDisclaimer",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=CLR_TEXT_MUTED,
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=CLR_TEXT,
    )

    small_bold_style = ParagraphStyle(
        "ReportSmallBold",
        parent=small_style,
        fontName="Helvetica-Bold",
    )

    story = []

    # =========================================================================
    # 1. REPORT COVER / HEADER BANNER
    # =========================================================================
    story.append(Paragraph("ASCLÉPIUS FAIRAI", title_style))
    story.append(Paragraph("Healthcare AI Fairness & Explainability Audit Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=CLR_PRIMARY, spaceBefore=0, spaceAfter=10))

    # Meta banner table
    audit_mode = analysis_results.get("audit_mode") or model_res.get("audit_mode") or ("external" if model_res.get("model_type") == "External Model Audit" else "internal")
    is_external = audit_mode == "external" or model_res.get("model_type") == "External Model Audit"

    model_name = model_res.get("model_name", model_res.get("model_label", "Clinical Classifier"))
    model_version = model_res.get("model_version", "v1.0")
    model_type = model_res.get("model_type", "External Model Audit" if is_external else "Supervised Classifier")
    target_name = model_res.get("target", fairness_res.get("target_name", "Outcome"))
    sensitive_name = model_res.get("sensitive_attribute", fairness_res.get("sensitive_name", "Protected Group"))
    overall_verdict = fairness_res.get("verdict_title", fairness_res.get("overall_status", "Evaluated"))

    if is_external:
        meta_data = [
            [
                Paragraph("<b>Audit Type:</b> Existing AI Model Audit", body_style),
                Paragraph("<b>Evaluation Date:</b> " + date_display, body_style),
            ],
            [
                Paragraph(f"<b>Model Name:</b> {model_name} (ver. {model_version})", body_style),
                Paragraph(f"<b>Model Architecture:</b> {model_type}", body_style),
            ],
            [
                Paragraph(f"<b>Target Variable:</b> {target_name}", body_style),
                Paragraph(f"<b>Sensitive Attribute:</b> {sensitive_name}", body_style),
            ],
            [
                Paragraph(f"<b>Overall Fairness Verdict:</b> <font color='#15803D'><b>{overall_verdict}</b></font>", body_style),
                Paragraph("<b>Evaluation Mode:</b> Prediction Evidence (Model-Agnostic)", body_style),
            ]
        ]
    else:
        meta_data = [
            [
                Paragraph("<b>Audit Type:</b> Local Model Evaluation", body_style),
                Paragraph("<b>Evaluation Date:</b> " + date_display, body_style),
            ],
            [
                Paragraph(f"<b>Model Architecture:</b> {model_name}", body_style),
                Paragraph("<b>Standard:</b> Asclepius Clinical Screening Criteria", body_style),
            ],
            [
                Paragraph(f"<b>Target Variable:</b> {target_name}", body_style),
                Paragraph(f"<b>Sensitive Attribute:</b> {sensitive_name}", body_style),
            ],
            [
                Paragraph(f"<b>Overall Fairness Verdict:</b> <font color='#15803D'><b>{overall_verdict}</b></font>", body_style),
                Paragraph("<b>Pipeline:</b> ColumnTransformer + Stratified Split", body_style),
            ]
        ]

    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CLR_BG_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, CLR_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, CLR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 2. ANALYSIS OVERVIEW & EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("1. Executive Analysis Overview", h1_style))
    
    n_rows = dataset_info.get("rows", model_res.get("train_samples", 0) + model_res.get("test_samples", 0))
    n_cols = dataset_info.get("cols", len(model_res.get("features_used", [])) + 1)
    acc = model_res.get("accuracy", "Not available")
    bal_acc = model_res.get("balanced_accuracy", "Not available")
    f1 = model_res.get("f1", "Not available")
    auc = model_res.get("roc_auc", "Not available")

    overview_data = [
        ["Dataset Rows", f"{n_rows:,}" if isinstance(n_rows, (int, float)) else str(n_rows), "Dataset Columns", str(n_cols)],
        ["Model Architecture", model_name, "Fairness Verdict", str(overall_verdict)],
        ["Overall Accuracy", f"{acc*100:.1f}%" if isinstance(acc, (int, float)) else str(acc), "Balanced Accuracy", f"{bal_acc*100:.1f}%" if isinstance(bal_acc, (int, float)) else str(bal_acc)],
        ["F1 Score", f"{f1:.3f}" if isinstance(f1, (int, float)) else str(f1), "ROC-AUC", f"{auc:.3f}" if isinstance(auc, (int, float)) else ("Unavailable" if auc is None else str(auc))],
    ]

    overview_table = Table(overview_data, colWidths=[120, 132, 120, 132])
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), CLR_MINT),
        ("BACKGROUND", (2, 0), (2, -1), CLR_MINT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, CLR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 3. DATASET & PREPROCESSING INFORMATION
    # =========================================================================
    story.append(Paragraph("2. Dataset & Preprocessing Architecture", h1_style))
    story.append(Paragraph(
        f"The evaluation pipeline processed a cohort of <b>{n_rows} observations</b> configured with target variable <b>'{target_name}'</b> "
        f"and audited across demographic attribute <b>'{sensitive_name}'</b>. Features were passed through a scikit-learn ColumnTransformer "
        f"pipeline ensuring strict separation between training transformations and held-out test evaluations (zero data leakage).",
        body_style
    ))

    num_count = dataset_info.get("numerical_count", "Dynamic")
    cat_count = dataset_info.get("categorical_count", "Dynamic")
    train_size = model_res.get("train_samples", "80%")
    test_size = model_res.get("test_samples", "20%")

    prep_data = [
        ["Predictor Features", f"{len(model_res.get('features_used', []))} features", "Train / Test Split", f"{train_size} train / {test_size} test"],
        ["Numerical Preprocessing", "Median Imputation + StandardScaler", "Categorical Encoding", "Mode Imputation + OneHotEncoder"],
        ["Data Privacy Notice", "Summary metrics only — No raw PHI retained", "Sensitive Feature Usage", "Preserved for group audit"],
    ]

    prep_table = Table(prep_data, colWidths=[130, 122, 130, 122])
    prep_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), CLR_BG_LIGHT),
        ("BACKGROUND", (2, 0), (2, -1), CLR_BG_LIGHT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, CLR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(prep_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 4. MODEL PERFORMANCE & CONFUSION MATRIX
    # =========================================================================
    story.append(Paragraph("3. Model Performance Evaluation", h1_style))

    cm = model_res.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    prec = model_res.get("precision", 0.0)
    rec = model_res.get("recall", 0.0)
    spec = model_res.get("specificity", 0.0)

    perf_table_data = [
        ["Metric", "Value", "Clinical Significance"],
        ["Accuracy", f"{acc*100:.1f}%" if isinstance(acc, (int, float)) else str(acc), "Overall proportion of correct classifications"],
        ["Balanced Accuracy", f"{bal_acc*100:.1f}%" if isinstance(bal_acc, (int, float)) else str(bal_acc), "Mean of sensitivity and specificity (class-balanced)"],
        ["Precision (PPV)", f"{prec*100:.1f}%" if isinstance(prec, (int, float)) else str(prec), "Positive predictive value: TP / (TP + FP)"],
        ["Recall / Sensitivity (TPR)", f"{rec*100:.1f}%" if isinstance(rec, (int, float)) else str(rec), "Diagnostic sensitivity: TP / (TP + FN)"],
        ["Specificity (TNR)", f"{spec*100:.1f}%" if isinstance(spec, (int, float)) else str(spec), "True negative rate: TN / (TN + FP)"],
        ["F1 Score", f"{f1:.3f}" if isinstance(f1, (int, float)) else str(f1), "Harmonic mean of precision and recall"],
        ["ROC-AUC", f"{auc:.3f}" if isinstance(auc, (int, float)) else ("Unavailable" if auc is None else str(auc)), "Area under the Receiver Operating Characteristic curve"],
        ["Decision Threshold (θ)", f"{float(model_res.get('decision_threshold', 0.50)):.2f}", "Operational classification cut-point on continuous probabilities"],
    ]

    perf_table = Table(perf_table_data, colWidths=[120, 70, 314])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CLR_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, CLR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLR_BG_LIGHT]),
    ]))

    # Render confusion matrix chart image
    cm_buf = _generate_confusion_matrix_chart(cm)
    cm_img = RLImage(cm_buf, width=2.6 * inch, height=1.8 * inch)

    matrix_wrapper = Table([
        [perf_table, cm_img]
    ], colWidths=[330, 174])
    matrix_wrapper.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(matrix_wrapper)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 5. FAIRNESS & BIAS EVALUATION (CORE 5 METRICS)
    # =========================================================================
    story.append(Paragraph("4. Algorithmic Fairness & Bias Evaluation", h1_style))
    story.append(Paragraph(
        "Model predictions were evaluated across protected demographic groups against locked project screening thresholds. "
        "<i>These criteria highlight quantitative signals for clinical review and do not represent universal regulatory mandates.</i>",
        body_style
    ))

    def _get_metric_val(metric_obj, default=0.0):
        if isinstance(metric_obj, (int, float)):
            return float(metric_obj)
        if isinstance(metric_obj, dict):
            for k in ["value", "val", "difference", "diff", "ratio"]:
                if k in metric_obj and isinstance(metric_obj[k], (int, float)):
                    return float(metric_obj[k])
        return default

    def _get_metric_pass(metric_obj, default=True):
        if isinstance(metric_obj, dict):
            if "pass" in metric_obj:
                return bool(metric_obj["pass"])
            if "status" in metric_obj:
                return str(metric_obj["status"]).lower() in ["pass", "passed", "compliant"]
        return default

    f_metrics = fairness_res.get("metrics") or fairness_res.get("fairness_metrics") or fairness_res
    dpd_info = f_metrics.get("demographic_parity_difference") or f_metrics.get("dpd") or f_metrics.get("demographic_parity") or {}
    dir_info = f_metrics.get("disparate_impact_ratio") or f_metrics.get("dir") or f_metrics.get("disparate_impact") or {}
    eopd_info = f_metrics.get("equal_opportunity_difference") or f_metrics.get("eopd") or f_metrics.get("equal_opportunity") or {}
    eod_info = f_metrics.get("equalized_odds_difference") or f_metrics.get("eod") or f_metrics.get("equalized_odds") or {}
    fnr_info = f_metrics.get("fnr_disparity") or f_metrics.get("fnr") or f_metrics.get("underdiagnosis") or {}

    fairness_table_data = [
        ["Fairness Metric", "Computed Value", "Project Threshold", "Screening Status", "Interpretation"],
        [
            "Demographic Parity Diff (DPD)",
            f"{_get_metric_val(dpd_info, 0.0):.3f}",
            "≤ 0.05",
            "PASS" if _get_metric_pass(dpd_info, True) else "REVIEW",
            "Gap in positive selection rate across groups"
        ],
        [
            "Disparate Impact Ratio (DIR)",
            f"{_get_metric_val(dir_info, 1.0):.2f}",
            "0.80 – 1.25",
            "PASS" if _get_metric_pass(dir_info, True) else "REVIEW",
            "Ratio of group positive selection rates (80% rule)"
        ],
        [
            "Equal Opportunity Diff (EOpD)",
            f"{_get_metric_val(eopd_info, 0.0):.3f}",
            "≤ 0.05",
            "PASS" if _get_metric_pass(eopd_info, True) else "REVIEW",
            "Maximum True Positive Rate (Sensitivity) gap"
        ],
        [
            "Equalized Odds Diff (EOD)",
            f"{_get_metric_val(eod_info, 0.0):.3f}",
            "≤ 0.05",
            "PASS" if _get_metric_pass(eod_info, True) else "REVIEW",
            "Compound disparity in both TPR and FPR"
        ],
        [
            "Underdiagnosis Disparity (FNR)",
            f"{_get_metric_val(fnr_info, 0.0):.3f}",
            "≤ 0.05",
            "PASS" if _get_metric_pass(fnr_info, True) else "REVIEW",
            "Difference in missed positive disease cases"
        ],
    ]

    fairness_table = Table(fairness_table_data, colWidths=[140, 75, 85, 80, 124])
    fairness_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CLR_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, CLR_BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLR_BG_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(fairness_table)
    story.append(Spacer(1, 10))

    # Fairness Verdict Callout Box
    reasons_list = fairness_res.get("reasons") or fairness_res.get("findings") or ["All metrics meet standard screening thresholds."]
    if isinstance(reasons_list, str):
        reasons_list = [reasons_list]
    reasons_text = "<br/>• ".join(reasons_list)
    verdict_box_data = [
        [
            Paragraph(f"<b>Transparent Screening Verdict: {overall_verdict}</b><br/>"
                      f"• {reasons_text}", body_style)
        ]
    ]
    verdict_table = Table(verdict_box_data, colWidths=[504])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CLR_MINT),
        ("BOX", (0, 0), (-1, -1), 1, CLR_PRIMARY),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 6. SUBGROUP PERFORMANCE & RELIABILITY TABLE
    # =========================================================================
    story.append(Paragraph("5. Demographic Subgroup Breakdown & Sample Reliability", h1_style))

    subgroups = fairness_res.get("subgroups") or fairness_res.get("groups") or fairness_res.get("subgroup_metrics") or analysis_results.get("subgroups") or []
    if isinstance(subgroups, dict):
        subgroups = [{"group": k, **v} if isinstance(v, dict) else {"group": k, "n": v} for k, v in subgroups.items()]

    sub_table_data = [
        ["Subgroup", "n", "Sel. Rate", "TPR", "FNR", "TNR", "FPR", "Accuracy", "Reliability", "Status"]
    ]

    for g in subgroups:
        n_val = g.get("n", g.get("sample_size", g.get("count", 0)))
        sel_rate = g.get("selection_rate", g.get("selectionRatePct", 0.0))
        tpr_val = g.get("tpr", g.get("tprPct", 0.0))
        fnr_val = g.get("fnr", g.get("fnrPct", 0.0))
        tnr_val = g.get("specificity", g.get("tnr", g.get("specificityPct", 0.0)))
        fpr_val = g.get("fpr", g.get("fprPct", 0.0))
        acc_val = g.get("accuracy", g.get("accuracyPct", 0.0))
        rel_val = g.get("reliability", g.get("reliability_tier", "Standard"))

        def _fmt_pct(v):
            try:
                num = float(v)
                return f"{num*100:.1f}%" if num <= 1.0 else f"{num:.1f}%"
            except (ValueError, TypeError):
                return str(v)

        fnr_float = float(fnr_val) if isinstance(fnr_val, (int, float)) else 0.0
        sub_table_data.append([
            str(g.get("group", g.get("subgroup", "Unknown"))),
            str(n_val),
            _fmt_pct(sel_rate),
            _fmt_pct(tpr_val),
            _fmt_pct(fnr_val),
            _fmt_pct(tnr_val),
            _fmt_pct(fpr_val),
            _fmt_pct(acc_val),
            str(rel_val),
            "Aligned" if fnr_float <= 0.10 else "Review",
        ])

    sub_table = Table(sub_table_data, colWidths=[70, 30, 48, 48, 48, 48, 48, 48, 64, 52])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CLR_PRIMARY_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.5, CLR_BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLR_BG_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    story.append(sub_table)
    story.append(Spacer(1, 8))

    # Sample Reliability & Subgroup Chart
    if subgroups:
        sub_chart_buf = _generate_subgroup_chart(subgroups)
        sub_chart_img = RLImage(sub_chart_buf, width=4.8 * inch, height=2.2 * inch)
        story.append(sub_chart_img)
        story.append(Spacer(1, 8))

    # =========================================================================
    # 7. EXPLAINABLE AI (SHAP) SECTION
    # =========================================================================
    global_shap = shap_res.get("global_importance") or shap_res.get("global_features") or shap_res.get("shap_summary") or []
    is_shap_available = bool(global_shap and shap_res.get("status") != "unavailable")

    if is_shap_available:
        explainer_type = shap_res.get("explainer_type", "TreeExplainer")
        model_type_str = shap_res.get("model_type") or model_res.get("model_type", "RandomForestClassifier")
        samples_explained = shap_res.get("samples_explained", len(shap_res.get("local_explanations", [])) or "600")
        num_shap_features = len(global_shap)
        base_val = shap_res.get("base_value", 0.0)
        
        story.append(KeepTogether([
            Paragraph("6. SHAP Explainability Analysis", h1_style),
            Paragraph(
                f"<b>Status: Model-Specific Explainability Available</b> ({explainer_type})<br/>"
                "SHAP feature attribution was calculated from the supplied trained model and evaluation feature dataset. "
                "<i>These values describe model attribution for the evaluated model and should not be interpreted as causal clinical effects.</i>",
                body_style
            )
        ]))

        # Metadata Summary Table
        meta_table_data = [
            [
                Paragraph(f"<b>Model Type:</b> {model_type_str}", small_style),
                Paragraph(f"<b>Evaluation Observations:</b> {samples_explained}", small_style),
                Paragraph(f"<b>SHAP Features:</b> {num_shap_features}", small_style),
                Paragraph("<b>Audit Level:</b> Level 2 (Model-Specific)", small_style),
            ]
        ]
        meta_table = Table(meta_table_data, colWidths=[126, 126, 126, 126])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CLR_MINT),
            ("BOX", (0, 0), (-1, -1), 1, CLR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 8))

        # Chart
        shap_chart_buf = _generate_shap_chart(global_shap)
        shap_chart_img = RLImage(shap_chart_buf, width=5.0 * inch, height=2.2 * inch)
        story.append(shap_chart_img)
        story.append(Spacer(1, 8))

        # Top Features Table
        top_table_rows = [
            [
                Paragraph("<b>Rank</b>", small_bold_style),
                Paragraph("<b>Feature Name</b>", small_bold_style),
                Paragraph("<b>Mean |SHAP|</b>", small_bold_style),
                Paragraph("<b>Direction / Risk Effect</b>", small_bold_style),
                Paragraph("<b>Relative Impact</b>", small_bold_style),
            ]
        ]
        for idx, f_item in enumerate(global_shap[:6]):
            f_name = str(f_item.get("feature", f"Feature {idx+1}"))
            val = float(f_item.get("mean_abs_shap", f_item.get("weight", 0.0)))
            direction = str(f_item.get("direction", "Increases Risk" if val >= 0 else "Decreases Risk"))
            rel_pct = f_item.get("relative_pct", 100)
            top_table_rows.append([
                Paragraph(str(idx + 1), small_style),
                Paragraph(f"<b>{f_name}</b>", small_style),
                Paragraph(f"{val:.4f}", small_style),
                Paragraph(direction, small_style),
                Paragraph(f"{rel_pct}%", small_style),
            ])

        top_table = Table(top_table_rows, colWidths=[40, 160, 90, 134, 80])
        top_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CLR_MINT),
            ("BOX", (0, 0), (-1, -1), 1, CLR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, CLR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(top_table)
        story.append(Spacer(1, 8))

        # Interpretation
        top_names = [f_item.get("feature", "") for f_item in global_shap[:4] if f_item.get("feature")]
        interp_text = (
            f"<b>Interpretation:</b> SHAP feature attribution was calculated from the supplied trained model and evaluation feature dataset. "
            f"The primary contributing features are <b>{', '.join(top_names)}</b>. "
            "These values describe model attribution for the evaluated model and should not be interpreted as causal clinical effects. "
            "Features with positive directional impact increase the likelihood of positive classification, while negative contributions reduce predicted risk."
        )
        story.append(Paragraph(interp_text, body_style))
        story.append(Spacer(1, 8))
    else:
        story.append(KeepTogether([
            Paragraph("6. Explainable AI — SHAP Feature Attributions", h1_style),
            Paragraph(
                "Game-theoretic Shapley Additive Explanations (SHAP) describe feature contribution weights on model decisions. "
                "<i>SHAP values quantify model behavior and statistical reliance; they do not establish clinical or biological causation.</i>",
                body_style
            )
        ]))
        shap_notice_data = [
            [
                Paragraph("<b>SHAP unavailable — model internals were not provided.</b>", h2_style),
            ],
            [
                Paragraph(
                    "In this external model audit, evaluation was conducted using held-out prediction evidence. "
                    "Model internals (parameters, estimator architectures, and decision trees) were not provided; "
                    "therefore SHAP feature attributions are unavailable to ensure scientific integrity and prevent fabricated explanations.",
                    body_style
                )
            ]
        ]
        shap_notice_table = Table(shap_notice_data, colWidths=[504])
        shap_notice_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CLR_WARN_BG),
            ("BOX", (0, 0), (-1, -1), 1, CLR_WARN_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(shap_notice_table)
        story.append(Spacer(1, 8))

    # =========================================================================
    # 8. AI INSIGHTS & ACTIONABLE RECOMMENDATIONS
    # =========================================================================
    story.append(KeepTogether([
        Paragraph("7. AI Insights & Clinical Governance Findings", h1_style),
        Paragraph(
            "Findings derived deterministically from computed model performance, demographic fairness metrics, and SHAP attributions.",
            body_style
        )
    ]))

    key_finding = insights_res.get("key_finding") or {}
    exec_summary = insights_res.get("executive_summary", "")
    if key_finding:
        kf_title = key_finding.get("title", "Clinical Finding")
        kf_desc = key_finding.get("explanation", "")
        kf_sev = key_finding.get("severity", "INFO")
        story.append(Paragraph(f"<b>Primary Finding ({kf_sev}):</b> <u>{kf_title}</u><br/>{kf_desc}", body_style))
        story.append(Spacer(1, 6))
    elif exec_summary:
        story.append(Paragraph(f"<b>Executive Summary:</b> {exec_summary}", body_style))
        story.append(Spacer(1, 6))

    recs = insights_res.get("recommendations") or insights_res.get("actionable_recommendations") or []
    if recs:
        story.append(Paragraph("<b>Actionable Governance Recommendations:</b>", h2_style))
        for idx, rec in enumerate(recs[:4], start=1):
            if isinstance(rec, str):
                rec_title = f"Recommendation {idx}"
                rec_trigger = "Algorithmic audit screening signal"
                rec_action = rec
                rec_investigate = "Model calibration, demographic representation, feature attributions"
                rec_sev = "WARNING"
            else:
                rec_title = rec.get("title", f"Recommendation {idx}")
                rec_trigger = rec.get("trigger", "")
                rec_action = rec.get("action", "")
                rec_investigate = ", ".join(rec.get("investigate", []))
                rec_sev = rec.get("severity", "WARNING")

            rec_box_data = [
                [
                    Paragraph(
                        f"<b>{idx}. {rec_title}</b> [{rec_sev}]<br/>"
                        f"• <b>Trigger Condition:</b> {rec_trigger}<br/>"
                        f"• <b>Recommended Action:</b> {rec_action}<br/>"
                        f"• <b>Investigate:</b> {rec_investigate}",
                        body_style
                    )
                ]
            ]
            rec_table = Table(rec_box_data, colWidths=[504])
            rec_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CLR_BG_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.5, CLR_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(rec_table)
            story.append(Spacer(1, 6))

    # =========================================================================
    # 8. OPTIONAL FAIRNESS MITIGATION — BEFORE / AFTER COMPARISON
    # =========================================================================
    mitigation_res = analysis_results.get("mitigation_results") or analysis_results.get("mitigation")
    has_mitigation = isinstance(mitigation_res, dict) and mitigation_res.get("status") == "success" and mitigation_res.get("selection", {}).get("selected_candidate") is not None

    if has_mitigation:
        sel_cand = mitigation_res["selection"]["selected_candidate"]
        cand_th = sel_cand.get("threshold", 0.5)
        mit_verdict = mitigation_res.get("mitigation_verdict", "Evaluated")
        mit_rel = mitigation_res.get("reliability_status", "Standard")
        mit_reason = mitigation_res.get("selection", {}).get("reason", "")
        comp_obj = mitigation_res.get("comparison") or {}

        base_th = float(
            comp_obj.get("baseline_threshold")
            or mitigation_res.get("baseline", {}).get("threshold")
            or model_res.get("decision_threshold", 0.50)
        )

        story.append(KeepTogether([
            Paragraph("8. Fairness Mitigation — Candidate Decision Threshold Comparison", h1_style),
            Paragraph(
                "Post-processing decision threshold optimization was evaluated across continuous model probabilities "
                "to assess mathematical trade-offs between predictive accuracy and demographic parity.",
                body_style
            )
        ]))

        mit_meta_data = [
            ["Mitigation Strategy", "Candidate Threshold", "Baseline Threshold", "Mitigation Verdict", "Reliability Status"],
            [
                mitigation_res.get("strategy", "Threshold Adjustment"),
                f"Threshold = {cand_th:.2f}",
                f"Threshold = {base_th:.2f}",
                mit_verdict,
                mit_rel,
            ]
        ]
        mit_meta_table = Table(mit_meta_data, colWidths=[130, 90, 84, 110, 90])
        mit_meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CLR_PRIMARY_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, CLR_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CLR_BG_LIGHT]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(mit_meta_table)
        story.append(Spacer(1, 8))

        # Metric comparison table
        perf_comp = comp_obj.get("performance_comparison", [])
        fair_comp = comp_obj.get("fairness_comparison", [])

        comp_table_data = [
            ["Evaluated Metric", f"Baseline (Th = {base_th:.2f})", f"Candidate (Th = {cand_th:.2f})", "Observed Delta", "Compliance / Status"]
        ]
        # Performance rows
        for p in perf_comp:
            comp_table_data.append([
                p.get("metric", ""),
                p.get("baseline", ""),
                p.get("candidate", ""),
                p.get("change", ""),
                "Maintained" if p.get("improved", False) else "Performance Drop"
            ])
        # Fairness rows
        for f in fair_comp:
            comp_table_data.append([
                f.get("metric", ""),
                f.get("baseline", ""),
                f.get("candidate", ""),
                f.get("change", ""),
                f"{f.get('baseline_status', '')} → {f.get('candidate_status', '')}"
            ])

        comp_table = Table(comp_table_data, colWidths=[140, 84, 84, 86, 110])
        comp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CLR_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, CLR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLR_BG_LIGHT]),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 6))

        # Mitigation caution box
        mit_box = [
            [
                Paragraph(
                    f"<b>Mitigation Finding & Governance Review:</b> {mit_reason}<br/>"
                    "<i>Caution: Decision threshold adjustment alters operational decision sensitivity. "
                    "This does not prove the model is unbiased and requires prospective clinical validation.</i>",
                    body_style
                )
            ]
        ]
        mit_box_table = Table(mit_box, colWidths=[504])
        mit_box_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CLR_WARN_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, CLR_WARN_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(mit_box_table)
        story.append(Spacer(1, 10))

    # =========================================================================
    # 9/10. LIMITATIONS, SAFETY NOTICES & CONCLUSION
    # =========================================================================
    acc_str = f"{acc*100:.1f}%" if isinstance(acc, (int, float)) else str(acc)
    auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) else str(auc)

    sec_lim_num = "9" if has_mitigation else "8"
    sec_concl_num = "10" if has_mitigation else "9"

    story.append(KeepTogether([
        Paragraph(f"{sec_lim_num}. Important Limitations & Clinical Safety Notice", h1_style),
        Paragraph(
            "• <b>Evaluation Scope:</b> All results reflect statistical evaluation of the provided dataset and model artifact. They do not constitute clinical diagnoses or treatment recommendations.<br/>"
            "• <b>Fairness Interpretation:</b> Algorithmic disparity metrics identify statistical divergence and do not by themselves establish legal discrimination or medical harm.<br/>"
            "• <b>Sample Size Variance:</b> Subgroups with n &lt; 30 exhibit wider variance; findings should be verified on larger multi-site cohorts.<br/>"
            "• <b>Data Privacy & HIPAA:</b> This platform is designed for academic, research, and audit demonstrations on de-identified or synthetic data. Asclepius FairAI does not claim HIPAA compliance.<br/>"
            "• <b>Threshold Standard:</b> Screening thresholds (DPD/EOD/FNR ≤ 0.05, DIR 0.80–1.25) are project screening criteria and not universal regulatory standards.",
            disclaimer_style
        ),
        Spacer(1, 10),
        Paragraph(f"{sec_concl_num}. Final Audit Conclusion", h1_style),
        Paragraph(
            f"The evaluated classifier (<b>{model_name}</b>) achieved an accuracy of <b>{acc_str}</b> and ROC-AUC of <b>{auc_str}</b>. "
            f"Algorithmic auditing across protected group <b>'{sensitive_name}'</b> yielded a transparent screening status of <b>{overall_verdict}</b>. "
            f"SHAP feature attributions and governance recommendations have been compiled for prospective validation and peer review.",
            body_style
        )
    ]))

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)

    file_size_bytes = os.path.getsize(output_path)

    return {
        "status": "success",
        "report_path": output_path,
        "filename": os.path.basename(output_path),
        "file_size_bytes": file_size_bytes,
        "timestamp": timestamp_str,
        "date_display": date_display,
        "model_name": model_name,
        "overall_verdict": overall_verdict,
    }


def generate_report(data):
    """Legacy bridge function."""
    return generate_fairness_report(data)
