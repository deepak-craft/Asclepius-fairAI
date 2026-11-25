"""
Asclepius FairAI - Command-Line Interface (CLI)
Phase 14: Local Developer Command-Line Entry Point for Algorithmic Healthcare AI Audits
"""

import sys
import os
import argparse
import json
import pandas as pd
from typing import Optional

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.asclepius.config import AuditConfig
from src.asclepius.result import AuditResult
from src.asclepius.auditor import FairAIAuditor
from src.core.train_model import train_clinical_model


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="asclepius",
        description="Asclepius FairAI — Local Healthcare AI Fairness & Model Auditing CLI.\n'Before Healthcare AI Goes Live, Run Asclepius.'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="asclepius 1.0.0 (Phase 14 Local Audit SDK)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: audit
    audit_parser = subparsers.add_parser(
        "audit",
        help="Run a local fairness & performance audit on a healthcare AI model.",
        description="Audit a model locally using evaluation data without network transmission.",
    )
    audit_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run audit against the synthetic clinical ICU demonstration model.",
    )
    audit_parser.add_argument(
        "--data",
        type=str,
        help="Path to CSV dataset containing test features, ground truth, and sensitive attributes.",
    )
    audit_parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Name of the ground truth binary outcome column (default: 'target').",
    )
    audit_parser.add_argument(
        "--sensitive",
        type=str,
        default="sex",
        help="Name of the protected demographic attribute column (default: 'sex').",
    )
    audit_parser.add_argument(
        "--model-name",
        type=str,
        default="Clinical Classifier",
        help="Human-readable model name for the audit report.",
    )
    audit_parser.add_argument(
        "--report",
        type=str,
        help="Generate and save an audit PDF report to the specified file path.",
    )
    audit_parser.add_argument(
        "--save-history",
        action="store_true",
        help="Persist summarized audit metadata to the local SQLite history database (Zero PHI).",
    )
    audit_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: 'text').",
    )

    # Command: info
    subparsers.add_parser(
        "info",
        help="Display Asclepius local audit system information and privacy policy.",
    )

    return parser


def run_demo_audit(output_format: str = "text", report_path: Optional[str] = None, save_history: bool = False) -> int:
    """Runs a quick end-to-end demo audit using synthetic clinical data."""
    demo_path = os.path.join(PROJECT_ROOT, "data", "demo", "synthetic_clinical_demo.csv")
    if not os.path.exists(demo_path):
        print(f"Error: Demo file not found at {demo_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(demo_path)
    target_col = "readmitted_30d"
    sensitive_col = "gender"

    # Train a local model in memory for demonstration
    train_res = train_clinical_model(
        df=df,
        target_col=target_col,
        sensitive_col=sensitive_col,
        model_type="random_forest",
        test_size=0.25,
        random_state=42,
    )
    if train_res["status"] != "success":
        print(f"Error: Failed to train demonstration model: {train_res.get('error_message')}", file=sys.stderr)
        return 1

    feature_cols = [c for c in df.columns if c not in [target_col]]
    n_test = len(train_res["y_test"])
    X_test = df[feature_cols].iloc[-n_test:].copy()
    y_test = train_res["y_test"]

    # Execute local audit via FairAIAuditor
    auditor = FairAIAuditor(
        model=train_res["pipeline"],
        X_test=X_test,
        y_test=y_test,
        sensitive_features=sensitive_col,
        target_name="30-Day ICU Readmission",
        sensitive_name="Gender",
        model_name="ICU Readmission Risk Predictor",
        model_version="v1.0-demo",
        model_type="Random Forest",
    )
    result = auditor.run()

    if report_path:
        out_p = result.generate_report(report_path)
        if output_format == "text":
            print(f"PDF Report Generated: {out_p}")

    if save_history:
        result.save_history()
        if output_format == "text":
            print("Audit record saved to local SQLite history (Zero PHI).")

    if output_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())

    return 0


def main(args: Optional[list] = None) -> int:
    """Main CLI execution routine."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 0

    if parsed.command == "info":
        print("ASCLÉPIUS FAIRAI — LOCAL AUDIT SDK")
        print("---------------------------------")
        print("Local-First Healthcare AI Evaluation Engine")
        print("Architecture: In-situ evaluation, Zero remote telemetry, Zero unpickling.")
        print("Core Criteria: DPD <= 0.05, DIR in [0.80, 1.25], EOpD <= 0.05, EOD <= 0.05, FNRD <= 0.05")
        print("Reliability Tiers: n < 30 (Insufficient Evidence), 30-49 (Caution), 50+ (Standard Evaluation)")
        return 0

    if parsed.command == "audit":
        if parsed.demo:
            return run_demo_audit(
                output_format=parsed.format,
                report_path=parsed.report,
                save_history=parsed.save_history,
            )

        if not parsed.data:
            print("Error: Either --demo or --data <path> must be specified for 'asclepius audit'.", file=sys.stderr)
            return 1

        if not os.path.exists(parsed.data):
            print(f"Error: Data file not found: {parsed.data}", file=sys.stderr)
            return 1

        try:
            df = pd.read_csv(parsed.data)
            if parsed.target not in df.columns:
                print(f"Error: Target column '{parsed.target}' not found in {parsed.data}.", file=sys.stderr)
                return 1
            if parsed.sensitive not in df.columns:
                print(f"Error: Sensitive attribute '{parsed.sensitive}' not found in {parsed.data}.", file=sys.stderr)
                return 1

            feature_cols = [c for c in df.columns if c not in [parsed.target]]
            X = df[feature_cols]
            y = df[parsed.target]

            # Fit baseline demonstration classifier if raw features provided
            train_res = train_clinical_model(
                df=df,
                target_col=parsed.target,
                sensitive_col=parsed.sensitive,
                model_type="random_forest",
                test_size=0.25,
                random_state=42,
            )
            n_test = len(train_res["y_test"])
            X_test = X.iloc[-n_test:]
            y_test = train_res["y_test"]

            auditor = FairAIAuditor(
                model=train_res["pipeline"],
                X_test=X_test,
                y_test=y_test,
                sensitive_features=parsed.sensitive,
                target_name=parsed.target,
                sensitive_name=parsed.sensitive,
                model_name=parsed.model_name,
            )
            result = auditor.run()

            if parsed.report:
                result.generate_report(parsed.report)
            if parsed.save_history:
                result.save_history()

            if parsed.format == "json":
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(result.summary())

            return 0
        except Exception as e:
            print(f"Audit Execution Failed: {str(e)}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
