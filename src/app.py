import streamlit as st
import streamlit.components.v1 as components

from config import (
    APP_ICON,
    APP_LAYOUT,
    APP_TITLE,
    SIDEBAR_STATE,
)

from ui.loader import (
    load_asset,
    load_page_template,
    load_image,
)


# ==========================================
# STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE,
)


# ==========================================
# HIDE STREAMLIT UI
# ==========================================

st.markdown(
    """
    <style>

        #MainMenu {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# LOAD TEMPLATES & ASSETS
# ==========================================

home_html = load_page_template("home.html")
hero_image = load_image("hero-illustration.png")
home_html = (
    home_html.replace("{{HERO_IMAGE}}", hero_image)
    .replace("{{ HERO_IMAGE }}", hero_image)
    .replace("{{hero_image}}", hero_image)
    .replace("{{ hero_image }}", hero_image)
)

dashboard_html = load_page_template("dashboard.html")

import os
import json
from pathlib import Path
from src.core.external_audit import load_demo_audit_csv, get_demo_audit_csv_path, get_shap_demo_paths

# Canonical demo audit evidence CSV from data/demo/external_model_audit_demo.csv
demo_csv_path = Path(get_demo_audit_csv_path())
if not demo_csv_path.exists():
    load_demo_audit_csv()
demo_csv_text = demo_csv_path.read_text(encoding="utf-8")

# Level 2 SHAP Demonstration artifacts from data/demo/shap/
shap_paths = get_shap_demo_paths()
shap_features_text = ""
shap_predictions_text = ""
shap_attributions_json = {}
if os.path.exists(shap_paths["features"]):
    shap_features_text = Path(shap_paths["features"]).read_text(encoding="utf-8")
if os.path.exists(shap_paths["predictions"]):
    shap_predictions_text = Path(shap_paths["predictions"]).read_text(encoding="utf-8")
attributions_file = os.path.join(shap_paths["dir"], "asclepius_shap_demo_attributions.json")
if os.path.exists(attributions_file):
    try:
        with open(attributions_file, "r", encoding="utf-8") as f:
            shap_attributions_json = json.load(f)
    except Exception:
        shap_attributions_json = {}

css = load_asset("css", "style.css")
js = load_asset("js", "script.js")

current_page = st.query_params.get("page", "home").lower()
home_display = "none" if current_page == "dashboard" else "block"
dashboard_display = "block" if current_page == "dashboard" else "none"

page_content = f"""
<div id="view-home" style="display: {home_display};">
    {home_html}
</div>
<div id="view-dashboard" style="display: {dashboard_display};">
    {dashboard_html}
</div>
"""


# ==========================================
# BUILD FINAL PAGE
# ==========================================

page = f"""
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>{APP_TITLE}</title>

    <link
        rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css"
    >

    <style>
        {css}
    </style>

</head>

<body>

    {page_content}

    <script>
        window.AsclepiusDemoAuditCSV = {json.dumps(demo_csv_text)};
        window.AsclepiusShapDemoFeaturesCSV = {json.dumps(shap_features_text)};
        window.AsclepiusShapDemoPredictionsCSV = {json.dumps(shap_predictions_text)};
        window.AsclepiusShapDemoAttributions = {json.dumps(shap_attributions_json)};
        window.AsclepiusShapDemoModelMeta = {{
            name: "asclepius_shap_demo_model.pkl",
            type: "RandomForestClassifier",
            features: ["age", "bmi", "blood_pressure", "cholesterol", "glucose", "heart_rate", "smoking", "activity_level"],
            sensitive: "sex",
            records: 600,
            n_estimators: 45,
            max_depth: 5,
            random_state: 42
        }};
    </script>
    <script>
        {js}
    </script>

</body>

</html>
"""


# ==========================================
# DISPLAY PAGE
# ==========================================

components.html(
    page,
    height=1400,
    scrolling=True,
)
# Reload with External Audit SHAP limitation panel and isolated explainability view - 2026-10-08T02:21:20