"""
Asclepius FairAI - Analysis History & Persistent Results Module
Phase 8 & 9: SQLite-backed Persistent Audit Trail, Summarized Metrics Storage, Mitigation Linkage, Filtering, and Privacy Compliance (Zero PHI)
"""

import os
import sqlite3
import json
import datetime
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple


def get_default_db_path() -> str:
    """Returns absolute path to SQLite history database."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "history.db")


def initialize_history_db(db_path: Optional[str] = None) -> str:
    """
    Initializes SQLite database and creates `analysis_history` table with indexes.
    
    Safe & Idempotent: Does not alter existing rows or tables.
    """
    path = db_path or get_default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                analysis_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                dataset_row_count INTEGER,
                dataset_column_count INTEGER,
                target_name TEXT,
                sensitive_attribute TEXT,
                model_type TEXT,
                test_size REAL,
                random_state INTEGER,
                accuracy REAL,
                balanced_accuracy REAL,
                precision REAL,
                recall REAL,
                specificity REAL,
                f1 REAL,
                roc_auc REAL,
                demographic_parity_difference REAL,
                disparate_impact_ratio REAL,
                equal_opportunity_difference REAL,
                equalized_odds_difference REAL,
                fnr_disparity REAL,
                fairness_status TEXT,
                reliability_status TEXT,
                key_finding TEXT,
                findings_json TEXT,
                recommendations_json TEXT,
                subgroups_json TEXT,
                shap_summary_json TEXT,
                report_filename TEXT,
                mitigation_json TEXT,
                decision_threshold REAL DEFAULT 0.50
            );
        """)

        # Migration check: ensure mitigation_json and decision_threshold columns exist
        cursor.execute("PRAGMA table_info(analysis_history)")
        columns = [col[1] for col in cursor.fetchall()]
        if "mitigation_json" not in columns:
            cursor.execute("ALTER TABLE analysis_history ADD COLUMN mitigation_json TEXT;")
        if "decision_threshold" not in columns:
            cursor.execute("ALTER TABLE analysis_history ADD COLUMN decision_threshold REAL DEFAULT 0.50;")

        # Performance & filtering indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_created_at ON analysis_history (created_at DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_fairness_status ON analysis_history (fairness_status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_model_type ON analysis_history (model_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_dataset_name ON analysis_history (dataset_name);")

        conn.commit()
    finally:
        conn.close()

    return path


def _extract_val(obj: Any, keys: List[str], default: Any = None) -> Any:
    """Safely extracts value from dict or nested objects."""
    if not isinstance(obj, dict):
        return default
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def _extract_metric_num(metric_obj: Any, default: float = 0.0) -> float:
    """Extracts floating point value from dict or scalar."""
    if isinstance(metric_obj, (int, float)):
        return float(metric_obj)
    if isinstance(metric_obj, dict):
        for k in ["value", "val", "difference", "diff", "ratio"]:
            if k in metric_obj and isinstance(metric_obj[k], (int, float)):
                return float(metric_obj[k])
    return default


def save_analysis(analysis_bundle: Dict[str, Any], db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Saves or updates an analysis evaluation record in SQLite history.
    
    Privacy Compliance:
    - Stores only aggregated metadata, summary statistics, and structured JSON results.
    - NEVER stores raw patient-level rows, dataframes, or PHI.
    """
    if not isinstance(analysis_bundle, dict):
        return {
            "status": "error",
            "error_message": "Invalid analysis bundle provided.",
        }

    path = initialize_history_db(db_path)

    # 1. Extract sub-objects
    dataset_info = analysis_bundle.get("dataset_info") or analysis_bundle.get("dataset") or {}
    model_res = analysis_bundle.get("model_results") or analysis_bundle.get("model") or {}
    fairness_res = (
        analysis_bundle.get("fairness_results")
        or analysis_bundle.get("fairness_evaluation")
        or analysis_bundle.get("fairness")
        or {}
    )
    shap_res = (
        analysis_bundle.get("shap_results")
        or analysis_bundle.get("explainability")
        or analysis_bundle.get("shap")
        or {}
    )
    insights_res = (
        analysis_bundle.get("insights_results")
        or analysis_bundle.get("ai_insights")
        or analysis_bundle.get("insights")
        or {}
    )
    mitigation_res = (
        analysis_bundle.get("mitigation_results")
        or analysis_bundle.get("mitigation")
        or {}
    )

    # 2. Extract fields
    analysis_id = analysis_bundle.get("analysis_id") or f"ANL-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    created_at = analysis_bundle.get("created_at") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dataset_name = _extract_val(dataset_info, ["dataset_name", "filename", "name"], "Clinical_Cohort.csv")
    row_count = _extract_val(dataset_info, ["rows", "total_rows", "row_count", "samples"], model_res.get("train_samples", 0) + model_res.get("test_samples", 0))
    col_count = _extract_val(dataset_info, ["cols", "total_columns", "column_count", "features"], len(model_res.get("features_used", [])) + 1)

    target_name = _extract_val(model_res, ["target", "target_variable"], _extract_val(fairness_res, ["target_name"], "Outcome"))
    sensitive_attr = _extract_val(model_res, ["sensitive_attribute", "sensitive_col"], _extract_val(fairness_res, ["sensitive_name"], "Protected Group"))
    if model_res.get("audit_mode") == "external" or model_res.get("model_type") == "External Model Audit":
        model_type = "External Model Audit"
    else:
        model_type = _extract_val(model_res, ["model_label", "model_name", "model_type"], "Clinical Classifier")

    test_size = float(_extract_val(model_res, ["test_size"], 0.2))
    random_state = int(_extract_val(model_res, ["random_state"], 42))
    decision_threshold = float(_extract_val(model_res, ["decision_threshold", "threshold", "baseline_threshold"], 0.50))

    # Model metrics
    acc = float(_extract_val(model_res, ["accuracy", "acc"], 0.0))
    bal_acc = float(_extract_val(model_res, ["balanced_accuracy", "bal_acc"], acc))
    prec = float(_extract_val(model_res, ["precision", "prec"], 0.0))
    rec = float(_extract_val(model_res, ["recall", "rec", "sensitivity"], 0.0))
    spec = float(_extract_val(model_res, ["specificity", "spec"], 0.0))
    f1 = float(_extract_val(model_res, ["f1", "f1_score"], 0.0))
    raw_roc = _extract_val(model_res, ["roc_auc", "auc"], None)
    roc_auc = float(raw_roc) if raw_roc is not None else None

    # Fairness metrics
    f_metrics = fairness_res.get("metrics") or fairness_res.get("fairness_metrics") or fairness_res
    dpd_info = f_metrics.get("demographic_parity_difference") or f_metrics.get("dpd") or {}
    dir_info = f_metrics.get("disparate_impact_ratio") or f_metrics.get("dir") or {}
    eopd_info = f_metrics.get("equal_opportunity_difference") or f_metrics.get("eopd") or {}
    eod_info = f_metrics.get("equalized_odds_difference") or f_metrics.get("eod") or {}
    fnr_info = f_metrics.get("fnr_disparity") or f_metrics.get("fnr") or {}

    dpd_val = _extract_metric_num(dpd_info, 0.0)
    dir_val = _extract_metric_num(dir_info, 1.0)
    eopd_val = _extract_metric_num(eopd_info, 0.0)
    eod_val = _extract_metric_num(eod_info, 0.0)
    fnr_val = _extract_metric_num(fnr_info, 0.0)

    fairness_status = _extract_val(fairness_res, ["verdict_title", "overall_status", "fairness_status"], "Evaluated")
    reliability_status = _extract_val(fairness_res, ["reliability_status", "overall_reliability"], "Standard")

    # Insights & Findings
    key_finding_obj = insights_res.get("key_finding") or {}
    if isinstance(key_finding_obj, dict):
        key_finding = key_finding_obj.get("title", insights_res.get("executive_summary", "Evaluation complete."))
    else:
        key_finding = str(key_finding_obj)

    findings_list = insights_res.get("category_findings") or insights_res.get("findings") or []
    recs_list = insights_res.get("recommendations") or insights_res.get("actionable_recommendations") or []
    subgroups_list = fairness_res.get("subgroups") or fairness_res.get("groups") or []
    shap_list = shap_res.get("global_importance") or shap_res.get("global_features") or []

    # Clean subgroups and shap list (ensure no complex objects)
    subgroups_summary = []
    if isinstance(subgroups_list, list):
        for g in subgroups_list:
            if isinstance(g, dict):
                subgroups_summary.append({
                    "group": str(g.get("group", g.get("subgroup", "Unknown"))),
                    "n": int(g.get("n", g.get("sample_size", 0))),
                    "selection_rate": float(g.get("selection_rate", g.get("selectionRatePct", 0.0))),
                    "tpr": float(g.get("tpr", g.get("tprPct", 0.0))),
                    "fnr": float(g.get("fnr", g.get("fnrPct", 0.0))),
                    "fpr": float(g.get("fpr", g.get("fprPct", 0.0))),
                    "accuracy": float(g.get("accuracy", g.get("accuracyPct", 0.0))),
                    "reliability": str(g.get("reliability", "Standard")),
                })

    shap_summary = []
    if isinstance(shap_list, list):
        for s in shap_list[:10]:
            if isinstance(s, dict):
                shap_summary.append({
                    "feature": str(s.get("feature", s.get("name", "Unknown"))),
                    "mean_abs_shap": float(s.get("mean_abs_shap", s.get("abs_importance", s.get("importance", 0.0)))),
                    "relative_pct": float(s.get("relative_pct", 0.0)),
                })

    report_filename = analysis_bundle.get("report_filename") or analysis_bundle.get("report_path")
    if report_filename:
        report_filename = os.path.basename(report_filename)

    # Serialize JSON fields
    findings_json = json.dumps(findings_list)
    recs_json = json.dumps(recs_list)
    subgroups_json = json.dumps(subgroups_summary)
    shap_summary_json = json.dumps(shap_summary)
    mitigation_json = json.dumps(mitigation_res) if mitigation_res and mitigation_res.get("status") == "success" else None

    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()

        # Check duplicate by ID
        cursor.execute("SELECT analysis_id, report_filename, mitigation_json FROM analysis_history WHERE analysis_id = ?", (analysis_id,))
        existing = cursor.fetchone()

        if existing:
            # Update existing record
            final_report = report_filename or existing[1]
            final_mitigation = mitigation_json or existing[2]
            cursor.execute("""
                UPDATE analysis_history SET
                    dataset_name = ?,
                    dataset_row_count = ?,
                    dataset_column_count = ?,
                    target_name = ?,
                    sensitive_attribute = ?,
                    model_type = ?,
                    test_size = ?,
                    random_state = ?,
                    accuracy = ?,
                    balanced_accuracy = ?,
                    precision = ?,
                    recall = ?,
                    specificity = ?,
                    f1 = ?,
                    roc_auc = ?,
                    demographic_parity_difference = ?,
                    disparate_impact_ratio = ?,
                    equal_opportunity_difference = ?,
                    equalized_odds_difference = ?,
                    fnr_disparity = ?,
                    fairness_status = ?,
                    reliability_status = ?,
                    key_finding = ?,
                    findings_json = ?,
                    recommendations_json = ?,
                    subgroups_json = ?,
                    shap_summary_json = ?,
                    report_filename = ?,
                    mitigation_json = ?,
                    decision_threshold = ?
                WHERE analysis_id = ?
            """, (
                dataset_name, row_count, col_count, target_name, sensitive_attr, model_type,
                test_size, random_state, acc, bal_acc, prec, rec, spec, f1, roc_auc,
                dpd_val, dir_val, eopd_val, eod_val, fnr_val, fairness_status, reliability_status,
                key_finding, findings_json, recs_json, subgroups_json, shap_summary_json,
                final_report, final_mitigation, decision_threshold, analysis_id
            ))
            conn.commit()
            return {
                "status": "success",
                "action": "updated",
                "analysis_id": analysis_id,
                "created_at": created_at,
                "message": f"Analysis {analysis_id} updated successfully in history."
            }

        # Check if identical result without ID was already saved in last 5 minutes
        cursor.execute("""
            SELECT analysis_id, mitigation_json FROM analysis_history
            WHERE dataset_name = ? AND target_name = ? AND sensitive_attribute = ?
              AND model_type = ? AND ABS(accuracy - ?) < 0.0001 AND ABS(demographic_parity_difference - ?) < 0.0001
            ORDER BY created_at DESC LIMIT 1
        """, (dataset_name, target_name, sensitive_attr, model_type, acc, dpd_val))
        recent_match = cursor.fetchone()

        if recent_match:
            match_id = recent_match[0]
            updates = []
            params = []
            if report_filename:
                updates.append("report_filename = ?")
                params.append(report_filename)
            if mitigation_json:
                updates.append("mitigation_json = ?")
                params.append(mitigation_json)

            if updates:
                params.append(match_id)
                cursor.execute(f"UPDATE analysis_history SET {', '.join(updates)} WHERE analysis_id = ?", params)
                conn.commit()

            return {
                "status": "success",
                "action": "already_saved",
                "analysis_id": match_id,
                "created_at": created_at,
                "message": f"Analysis already exists in history (ID: {match_id})."
            }

        # Insert new analysis record
        cursor.execute("""
            INSERT INTO analysis_history (
                analysis_id, created_at, dataset_name, dataset_row_count, dataset_column_count,
                target_name, sensitive_attribute, model_type, test_size, random_state,
                accuracy, balanced_accuracy, precision, recall, specificity, f1, roc_auc,
                demographic_parity_difference, disparate_impact_ratio, equal_opportunity_difference,
                equalized_odds_difference, fnr_disparity, fairness_status, reliability_status,
                key_finding, findings_json, recommendations_json, subgroups_json,
                shap_summary_json, report_filename, mitigation_json, decision_threshold
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )
        """, (
            analysis_id, created_at, dataset_name, row_count, col_count,
            target_name, sensitive_attr, model_type, test_size, random_state,
            acc, bal_acc, prec, rec, spec, f1, roc_auc,
            dpd_val, dir_val, eopd_val,
            eod_val, fnr_val, fairness_status, reliability_status,
            key_finding, findings_json, recs_json, subgroups_json,
            shap_summary_json, report_filename, mitigation_json, decision_threshold
        ))
        conn.commit()

        return {
            "status": "success",
            "action": "created",
            "analysis_id": analysis_id,
            "created_at": created_at,
            "message": f"Analysis {analysis_id} saved to persistent history."
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to save analysis to history: {str(e)}"
        }
    finally:
        conn.close()


def get_analysis_history(
    db_path: Optional[str] = None,
    sort_by: str = "date_desc",
    model_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    search_query: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Retrieves filtered and sorted list of analysis history records.
    """
    path = initialize_history_db(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM analysis_history WHERE 1=1"
        params: List[Any] = []

        if model_filter and model_filter.lower() != "all":
            query += " AND LOWER(model_type) LIKE ?"
            params.append(f"%{model_filter.lower()}%")

        if status_filter and status_filter.lower() != "all":
            query += " AND LOWER(fairness_status) LIKE ?"
            params.append(f"%{status_filter.lower()}%")

        if search_query and search_query.strip():
            sq = f"%{search_query.strip().lower()}%"
            query += " AND (LOWER(dataset_name) LIKE ? OR LOWER(target_name) LIKE ? OR LOWER(sensitive_attribute) LIKE ? OR LOWER(analysis_id) LIKE ?)"
            params.extend([sq, sq, sq, sq])

        # Ordering
        if sort_by == "date_asc":
            query += " ORDER BY created_at ASC, rowid ASC"
        elif sort_by == "accuracy_desc":
            query += " ORDER BY accuracy DESC, rowid DESC"
        elif sort_by == "fairness_status":
            query += " ORDER BY fairness_status ASC, rowid DESC"
        else:
            query += " ORDER BY created_at DESC, rowid DESC"

        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            d = dict(r)
            # Check report file availability
            report_fn = d.get("report_filename")
            report_available = False
            if report_fn:
                reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "reports"))
                candidate_paths = [
                    report_fn,
                    os.path.join(reports_dir, report_fn),
                    os.path.join(reports_dir, os.path.basename(report_fn)),
                ]
                report_available = any(os.path.exists(p) for p in candidate_paths)
            d["report_available"] = report_available

            # Deserialize JSON
            for j_key in ["findings_json", "recommendations_json", "subgroups_json", "shap_summary_json", "mitigation_json"]:
                if d.get(j_key):
                    try:
                        d[j_key.replace("_json", "")] = json.loads(d[j_key])
                    except Exception:
                        d[j_key.replace("_json", "")] = None

            results.append(d)

        return results
    except Exception:
        return []
    finally:
        conn.close()


def get_analysis(analysis_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single summarized analysis record by its unique ID.
    """
    if not analysis_id:
        return None

    path = initialize_history_db(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_history WHERE analysis_id = ?", (analysis_id,))
        row = cursor.fetchone()
        if not row:
            return None

        d = dict(row)
        report_fn = d.get("report_filename")
        report_available = False
        report_abs_path = None
        if report_fn:
            reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "reports"))
            candidates = [
                report_fn,
                os.path.join(reports_dir, report_fn),
                os.path.join(reports_dir, os.path.basename(report_fn)),
            ]
            for p in candidates:
                if os.path.exists(p):
                    report_available = True
                    report_abs_path = p
                    break
        d["report_available"] = report_available
        d["report_path"] = report_abs_path

        for j_key in ["findings_json", "recommendations_json", "subgroups_json", "shap_summary_json", "mitigation_json"]:
            if d.get(j_key):
                try:
                    d[j_key.replace("_json", "")] = json.loads(d[j_key])
                except Exception:
                    d[j_key.replace("_json", "")] = None

        return d
    except Exception:
        return None
    finally:
        conn.close()


def delete_analysis(analysis_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Deletes an analysis history record from the database.
    """
    if not analysis_id:
        return {"status": "error", "error_message": "No analysis ID provided for deletion."}

    path = initialize_history_db(db_path)
    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT analysis_id FROM analysis_history WHERE analysis_id = ?", (analysis_id,))
        row = cursor.fetchone()
        if not row:
            return {"status": "error", "error_message": f"Analysis ID '{analysis_id}' not found."}

        cursor.execute("DELETE FROM analysis_history WHERE analysis_id = ?", (analysis_id,))
        conn.commit()

        return {
            "status": "success",
            "analysis_id": analysis_id,
            "message": f"Analysis '{analysis_id}' deleted from history."
        }
    except Exception as e:
        return {"status": "error", "error_message": f"Database error during deletion: {str(e)}"}
    finally:
        conn.close()


def get_history_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes real summary statistics across all persisted history records.
    """
    path = initialize_history_db(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analysis_history")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT
                SUM(CASE WHEN LOWER(fairness_status) LIKE '%fair%' OR LOWER(fairness_status) LIKE '%balanced%' OR LOWER(fairness_status) LIKE '%compliant%' THEN 1 ELSE 0 END),
                SUM(CASE WHEN LOWER(fairness_status) LIKE '%moderate%' THEN 1 ELSE 0 END),
                SUM(CASE WHEN LOWER(fairness_status) LIKE '%significant%' OR LOWER(fairness_status) LIKE '%high%' OR LOWER(fairness_status) LIKE '%failed%' THEN 1 ELSE 0 END),
                SUM(CASE WHEN LOWER(fairness_status) LIKE '%insufficient%' OR LOWER(fairness_status) LIKE '%small%' OR LOWER(fairness_status) LIKE '%unevaluated%' THEN 1 ELSE 0 END)
            FROM analysis_history
        """)
        row = cursor.fetchone()
        fair_count = int(row[0] or 0)
        mod_count = int(row[1] or 0)
        sig_count = int(row[2] or 0)
        insuf_count = int(row[3] or 0)

        # Retrieve latest record
        cursor.execute("SELECT analysis_id, created_at, dataset_name, model_type, accuracy, fairness_status FROM analysis_history ORDER BY created_at DESC, rowid DESC LIMIT 1")
        latest_row = cursor.fetchone()
        latest = dict(latest_row) if latest_row else None

        return {
            "status": "success",
            "total_analyses": total,
            "fair_balanced_count": fair_count,
            "moderate_disparity_count": mod_count,
            "significant_disparity_count": sig_count,
            "insufficient_evidence_count": insuf_count,
            "latest_analysis": latest,
        }
    except Exception as e:
        return {
            "status": "error",
            "total_analyses": 0,
            "fair_balanced_count": 0,
            "moderate_disparity_count": 0,
            "significant_disparity_count": 0,
            "insufficient_evidence_count": 0,
            "latest_analysis": None,
            "error_message": str(e)
        }
    finally:
        conn.close()
