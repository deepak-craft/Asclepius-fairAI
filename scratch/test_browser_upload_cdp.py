"""
Asclepius FairAI - Chrome DevTools Protocol (CDP) Browser Verification
Verifies the complete, real NORMAL CSV UPLOAD flow via #ext-audit-file-input,
change event, FileReader, column auto-mapping, strict semantic validation,
audit execution against uploaded data, and separate demo and Cleveland paths.
"""

import asyncio
import json
import os
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DEMO_CSV = os.path.abspath(r"data\demo\external_model_audit_demo.csv")
CLEVELAND_CSV = os.path.abspath(r"data\demo\heart_disease_test.csv")
TEST_600_CSV = os.path.abspath(r"data\demo\external_model_audit_test_600.csv")

USER_TEST_CSV = """id,y_true,y_pred,y_prob,sex
1,1,1,0.91,F
2,0,0,0.18,M
3,1,0,0.42,F"""

async def run_test():
    print("======================================================================")
    print("ASCLEPIUS FAIRAI — BROWSER CDP NORMAL CSV UPLOAD VERIFICATION")
    print("======================================================================\n")

    import tempfile
    user_data_dir = tempfile.mkdtemp()
    port = 9224
    print(f"Launching isolated headless Chrome on port {port}...")
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "http://localhost:8501"
    ])
    await asyncio.sleep(4)
    
    try:
        targets_raw = urllib.request.urlopen(f"http://localhost:{port}/json").read()
        targets = json.loads(targets_raw.decode("utf-8"))
        asclepius_targets = [t for t in targets if t.get("type") == "page" and "localhost:8501" in t.get("url", "")]
        if asclepius_targets:
            page_target = asclepius_targets[0]
        else:
            page_target = [t for t in targets if t.get("type") == "page"][0]
        ws_url = page_target["webSocketDebuggerUrl"]
        print(f"Connected to page target: {page_target.get('title')} ({ws_url})")
        
        async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
            msg_id = 1
            contexts = {}

            async def send_cmd(method, params=None):
                nonlocal msg_id
                cid = msg_id
                msg_id += 1
                payload = {"id": cid, "method": method}
                if params:
                    payload["params"] = params
                await ws.send(json.dumps(payload))
                while True:
                    resp = await ws.recv()
                    data = json.loads(resp)
                    if "method" in data and data["method"] == "Runtime.executionContextCreated":
                        ctx = data["params"]["context"]
                        contexts[ctx["id"]] = ctx
                    if data.get("id") == cid:
                        return data

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")
            await send_cmd("DOM.enable")
            await send_cmd("Page.reload")

            # Find execution context where Asclepius dashboard runs
            print("Searching for Asclepius application execution context...")
            target_context_id = None
            for _ in range(25):
                await asyncio.sleep(1)
                try:
                    await send_cmd("Runtime.evaluate", {"expression": "1+1"})
                except Exception:
                    pass
                for cid in list(contexts.keys()):
                    try:
                        res = await send_cmd("Runtime.evaluate", {
                            "expression": "typeof window.AsclepiusState",
                            "contextId": cid
                        })
                        if res.get("result", {}).get("result", {}).get("value") == "object":
                            target_context_id = cid
                            break
                    except Exception:
                        pass
                if target_context_id is not None:
                    break
            
            if target_context_id is None:
                raise RuntimeError("Failed to locate Asclepius execution context inside browser.")
            print(f"Target execution context found: contextId={target_context_id}")

            # -----------------------------------------------------------------
            # 1. Switch to Existing AI Model Audit view
            # -----------------------------------------------------------------
            print("\n[Step 1] Navigating to Existing AI Model Audit...")
            switch_code = """
            (function() {
                window.switchView('dashboard');
                window.switchDashSection('external-audit');
                const fileInput = document.getElementById('ext-audit-file-input');
                const dropzone = document.getElementById('ext-dropzone-container');
                return {
                    auditMode: window.AsclepiusState.auditMode,
                    currentSection: window.AsclepiusState.currentSection,
                    hasFileInput: !!fileInput,
                    fileInputTag: fileInput ? fileInput.tagName : null,
                    fileInputAccept: fileInput ? fileInput.accept : null,
                    hasDropzone: !!dropzone
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": switch_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            print("Step 1 raw res:", res)
            step1 = res.get("result", {}).get("result", {}).get("value")
            print("Step 1 Result:", json.dumps(step1, indent=2))
            assert step1 is not None and step1.get("auditMode") == "external", f"Audit mode should be 'external', got {step1}"
            assert step1["hasFileInput"] is True, "File input #ext-audit-file-input must exist"
            assert step1["fileInputAccept"] == ".csv", "File input must accept .csv"

            # -----------------------------------------------------------------
            # 2. Perform Real Normal Upload via File + DataTransfer + change event
            # -----------------------------------------------------------------
            print("\n[Step 2] Simulating Normal User CSV File Selection via #ext-audit-file-input...")
            normal_upload_code = f"""
            (function() {{
                const fileInput = document.getElementById('ext-audit-file-input');
                if (!fileInput) return {{ error: 'File input element not found' }};

                const csvText = {json.dumps(USER_TEST_CSV)};
                const fileName = 'user_model_evaluation.csv';

                // Construct a real HTML5 File object
                const file = new File([csvText], fileName, {{ type: 'text/csv' }});

                // Attach to file input via DataTransfer API
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;

                // Fire the real change event (bubbles to handleExternalFileSelect)
                fileInput.dispatchEvent(new Event('change', {{ bubbles: true }}));

                return {{
                    fileAttached: fileInput.files.length > 0,
                    fileName: fileInput.files[0].name,
                    fileSize: fileInput.files[0].size
                }};
            }})()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": normal_upload_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step2 = res.get("result", {}).get("result", {}).get("value")
            print("Step 2 File Input Dispatch:", json.dumps(step2, indent=2))
            assert step2["fileAttached"] is True, "File must be attached to file input"

            # -----------------------------------------------------------------
            # 3. Wait for asynchronous FileReader and inspect parsed state
            # -----------------------------------------------------------------
            print("\n[Step 3] Waiting for asynchronous FileReader.readAsText to complete...")
            await asyncio.sleep(1) # Allow FileReader onload callback to fire

            check_upload_state = """
            (function() {
                const metaBanner = document.getElementById('ext-loaded-file-meta-banner');
                const statusBox = document.getElementById('ext-upload-status-box');
                const statusText = document.getElementById('ext-upload-status-text');
                const badge = document.getElementById('ext-validation-badge');
                const runBtn = document.getElementById('btn-run-external-audit');
                const issuesList = document.getElementById('ext-validation-issues-list');
                const warningsList = document.getElementById('ext-validation-warnings-list');

                const yTrue = document.getElementById('ext-map-ytrue')?.value;
                const yPred = document.getElementById('ext-map-ypred')?.value;
                const yProb = document.getElementById('ext-map-yprob')?.value;
                const sens = document.getElementById('ext-map-sensitive')?.value;
                const idVal = document.getElementById('ext-map-id')?.value;

                const dataset = window.AsclepiusState.externalAudit?.dataset;

                return {
                    hasDatasetInState: !!dataset,
                    filename: dataset?.filename,
                    headers: dataset?.headers,
                    rowCount: dataset?.rows?.length,
                    metaBannerDisplay: metaBanner?.style?.display,
                    metaBannerText: metaBanner?.innerText?.replace(/\\s+/g, ' ').trim(),
                    statusBoxText: statusText?.innerText?.replace(/\\s+/g, ' ').trim(),
                    badgeText: badge?.innerText?.trim(),
                    badgeClass: badge?.className,
                    runBtnDisabled: runBtn?.hasAttribute('disabled') || runBtn?.disabled || false,
                    issuesCount: issuesList ? issuesList.children.length : 0,
                    warningsCount: warningsList ? warningsList.children.length : 0,
                    mapping: {
                        y_true: yTrue,
                        y_pred: yPred,
                        y_prob: yProb,
                        sensitive: sens,
                        id: idVal
                    }
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": check_upload_state,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step3 = res.get("result", {}).get("result", {}).get("value")
            print("Step 3 Parsed & Validated State:", json.dumps(step3, indent=2))

            assert step3["hasDatasetInState"] is True, "Dataset should be stored in window.AsclepiusState"
            assert step3["filename"] == "user_model_evaluation.csv", "Filename must match uploaded file"
            assert step3["headers"] == ["id", "y_true", "y_pred", "y_prob", "sex"], "Headers must match CSV"
            assert step3["rowCount"] == 3, "Record count must be exactly 3"
            assert step3["mapping"]["y_true"] == "y_true", "y_true auto-mapped"
            assert step3["mapping"]["y_pred"] == "y_pred", "y_pred auto-mapped"
            assert step3["mapping"]["y_prob"] == "y_prob", "y_prob auto-mapped"
            assert step3["mapping"]["sensitive"] == "sex", "sensitive auto-mapped to sex"
            assert step3["mapping"]["id"] == "id", "id auto-mapped to id"
            assert step3["badgeText"] == "✓ Audit Input Ready", "Validation badge must be '✓ Audit Input Ready'"
            assert step3["runBtnDisabled"] is False, "Run button must be enabled"

            # -----------------------------------------------------------------
            # 4. Click Run Model Audit & verify execution on uploaded data
            # -----------------------------------------------------------------
            print("\n[Step 4] Clicking 'Run Model Audit & Fairness Evaluation'...")
            run_audit_code = """
            (function() {
                const runBtn = document.getElementById('btn-run-external-audit');
                runBtn.click();

                const state = window.AsclepiusState;
                const tm = state.trainedModel;
                const fr = state.fairnessResults;
                const ins = state.insightsResults;
                const errCard = document.getElementById('ext-audit-error-card');
                const errMsg = document.getElementById('ext-audit-error-msg');

                return {
                    activeSection: state.currentSection,
                    hasTrainedModel: !!tm,
                    modelType: tm?.modelType,
                    nTest: tm?.nTest,
                    accuracy: tm?.metrics?.accuracy,
                    balancedAccuracy: tm?.metrics?.balancedAccuracy,
                    rocAuc: tm?.metrics?.rocAuc,
                    yTest: tm?.yTest,
                    yPred: tm?.yPred,
                    hasFairness: !!fr,
                    subgroups: fr?.subgroups?.map(s => ({ group: s.group, n: s.n })),
                    dpd: fr?.dpd,
                    hasInsights: !!ins,
                    insightsStatus: ins?.status,
                    isErrorVisible: errCard && errCard.style.display !== 'none',
                    errMsgText: errMsg?.innerText
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": run_audit_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step4 = res.get("result", {}).get("result", {}).get("value")
            print("Step 4 Execution Result:", json.dumps(step4, indent=2))

            assert step4["isErrorVisible"] is False, f"Audit error card should not be visible, got: {step4.get('errMsgText')}"
            assert step4["hasTrainedModel"] is True, "Audit execution created trainedModel"
            assert step4["nTest"] == 3, "Execution must evaluate the 3 uploaded records"
            # 2 out of 3 correct: accuracy = 0.6667
            assert step4["accuracy"] == "0.6667", f"Expected accuracy 0.6667, got {step4['accuracy']}"
            assert step4["hasFairness"] is True, "Fairness results computed"
            assert len(step4["subgroups"]) == 2, "Fairness evaluated 2 subgroups (F and M)"
            assert step4["hasInsights"] is True, "Deterministic insights generated"

            # -----------------------------------------------------------------
            # 5. Verify Report Generation for Uploaded Model
            # -----------------------------------------------------------------
            print("\n[Step 5] Triggering PDF Report Generation for Uploaded Audit...")
            report_code = """
            (function() {
                window.triggerReportGeneration();
                const banner = document.getElementById('generated-report-banner');
                const readyMeta = document.getElementById('report-ready-meta');
                return {
                    bannerDisplay: banner?.style?.display,
                    readyMetaText: readyMeta?.innerText,
                    hasReportData: !!window.AsclepiusState.reportData,
                    filename: window.AsclepiusState.reportData?.filename
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": report_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step5 = res.get("result", {}).get("result", {}).get("value")
            print("Step 5 Report Result:", json.dumps(step5, indent=2))
            assert step5["bannerDisplay"] == "block", "Report banner displayed"
            assert step5["hasReportData"] is True, "Report data compiled in state"

            # -----------------------------------------------------------------
            # 6. Test Separate "Load Demo Audit Evidence" Button
            # -----------------------------------------------------------------
            print("\n[Step 6] Testing 'Load Demo Audit Evidence' Button...")
            demo_code = """
            (function() {
                window.switchDashSection('external-audit');
                const demoBtn = document.getElementById('btn-load-demo-audit');
                if (demoBtn) demoBtn.click();
                else window.loadDemoExternalAuditData();

                const ds = window.AsclepiusState.externalAudit?.dataset;
                const badge = document.getElementById('ext-validation-badge');
                const runBtn = document.getElementById('btn-run-external-audit');

                return {
                    datasetLoaded: !!ds,
                    filename: ds?.filename,
                    headers: ds?.headers,
                    rowCount: ds?.rows?.length,
                    badgeText: badge?.innerText?.trim(),
                    runBtnDisabled: runBtn?.hasAttribute('disabled') || runBtn?.disabled || false
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": demo_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step6 = res.get("result", {}).get("result", {}).get("value")
            print("Step 6 Demo Loader Result:", json.dumps(step6, indent=2))
            assert step6["datasetLoaded"] is True, "Demo evidence must be loaded"
            assert step6["filename"] == "external_model_audit_demo.csv", "Demo filename must match"
            assert step6["rowCount"] >= 50, f"Demo evidence should have ~600 rows, got {step6['rowCount']}"
            assert step6["badgeText"] == "✓ Audit Input Ready", "Demo badge must be ready"
            assert step6["runBtnDisabled"] is False, "Run button must be enabled for demo"

            # Run demo audit to verify independence from previous 3-row audit
            res = await send_cmd("Runtime.evaluate", {
                "expression": run_audit_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step6_exec = res.get("result", {}).get("result", {}).get("value")
            print(f"Step 6 Demo Execution Record Count: {step6_exec['nTest']} (Accuracy: {step6_exec['accuracy']})")
            assert step6_exec["nTest"] == step6["rowCount"], "Demo execution must evaluate demo rows, not user rows"
            assert step6_exec["nTest"] != 3, "Demo execution must be distinct from user upload"

            # -----------------------------------------------------------------
            # 7. Test Real 600-Row Evidence Normal Upload & Full Execution
            # -----------------------------------------------------------------
            print("\n[Step 7] Testing Real 600-Row External Audit Evidence Upload via #ext-audit-file-input...")
            with open(TEST_600_CSV, "r", encoding="utf-8") as f:
                csv_600_raw = f.read()

            upload_600_code = f"""
            (function() {{
                window.switchDashSection('external-audit');
                const fileInput = document.getElementById('ext-audit-file-input');
                const file = new File([{json.dumps(csv_600_raw)}], 'external_model_audit_test_600.csv', {{ type: 'text/csv' }});
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
                fileInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return {{ fileAttached: true }};
            }})()
            """
            await send_cmd("Runtime.evaluate", {
                "expression": upload_600_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            await asyncio.sleep(1) # Wait for FileReader

            check_600_state = """
            (function() {
                const badge = document.getElementById('ext-validation-badge');
                const runBtn = document.getElementById('btn-run-external-audit');
                const ds = window.AsclepiusState.externalAudit?.dataset;
                const mapping = window.AsclepiusState.externalAudit?.mapping;

                return {
                    filename: ds?.filename,
                    rowCount: ds?.rows?.length,
                    headers: ds?.headers,
                    badgeText: badge?.innerText?.trim(),
                    runBtnDisabled: runBtn?.hasAttribute('disabled') || runBtn?.disabled || false,
                    mapping: {
                        y_true: mapping?.yTrue,
                        y_pred: mapping?.yPred,
                        y_prob: mapping?.yProb,
                        sensitive: mapping?.sensitive,
                        id: mapping?.id
                    }
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": check_600_state,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step7_state = res.get("result", {}).get("result", {}).get("value")
            print("Step 7 600-Row Upload & Validation State:", json.dumps(step7_state, indent=2))

            assert step7_state["filename"] == "external_model_audit_test_600.csv"
            assert step7_state["rowCount"] == 600
            assert step7_state["mapping"]["y_true"] == "y_true"
            assert step7_state["mapping"]["y_pred"] == "y_pred"
            assert step7_state["mapping"]["y_prob"] == "y_prob"
            assert step7_state["mapping"]["sensitive"] == "sex"
            assert step7_state["badgeText"] == "✓ Audit Input Ready"
            assert step7_state["runBtnDisabled"] is False

            # Click Run Model Audit for 600-Row Dataset
            res = await send_cmd("Runtime.evaluate", {
                "expression": run_audit_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step7_exec = res.get("result", {}).get("result", {}).get("value")
            print("Step 7 600-Row Execution Result:", json.dumps(step7_exec, indent=2))

            assert step7_exec["isErrorVisible"] is False, f"600-row audit failed with: {step7_exec.get('errMsgText')}"
            assert step7_exec["hasTrainedModel"] is True
            assert step7_exec["nTest"] == 600, f"Expected nTest=600, got {step7_exec['nTest']}"
            assert float(step7_exec["accuracy"]) > 0.85
            assert step7_exec["hasFairness"] is True
            assert len(step7_exec["subgroups"]) == 2
            subgroup_counts = {s["group"]: s["n"] for s in step7_exec["subgroups"]}
            assert subgroup_counts.get("F") == 300
            assert subgroup_counts.get("M") == 300
            assert step7_exec["hasInsights"] is True

            # -----------------------------------------------------------------
            # 8. Test Cleveland Clinical Dataset Rejection via Normal Upload
            # -----------------------------------------------------------------
            print("\n[Step 8] Testing Rejection of Cleveland Feature Dataset via Normal Upload...")
            with open(CLEVELAND_CSV, "r", encoding="utf-8") as f:
                cleveland_raw = f.read()

            cleveland_upload_code = f"""
            (function() {{
                window.switchDashSection('external-audit');
                const fileInput = document.getElementById('ext-audit-file-input');
                const file = new File([{json.dumps(cleveland_raw)}], 'heart_disease_test.csv', {{ type: 'text/csv' }});
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
                fileInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return {{ fileAttached: true }};
            }})()
            """
            await send_cmd("Runtime.evaluate", {
                "expression": cleveland_upload_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            await asyncio.sleep(1) # Wait for FileReader

            cleveland_check_code = r"""
            (function() {
                const badge = document.getElementById('ext-validation-badge');
                const summaryText = document.getElementById('ext-validation-summary-text');
                const issuesList = document.getElementById('ext-validation-issues-list');
                const runBtn = document.getElementById('btn-run-external-audit');
                const statusBox = document.getElementById('ext-upload-status-box');

                const yTrue = document.getElementById('ext-map-ytrue')?.value;
                const yPred = document.getElementById('ext-map-ypred')?.value;
                const yProb = document.getElementById('ext-map-yprob')?.value;
                const sens = document.getElementById('ext-map-sensitive')?.value;
                const idVal = document.getElementById('ext-map-id')?.value;

                return {
                    statusBoxText: statusBox?.innerText?.replace(/\s+/g, ' ').trim(),
                    badgeText: badge?.innerText?.trim(),
                    summaryText: summaryText?.innerText?.replace(/\s+/g, ' ').trim(),
                    issuesText: issuesList?.innerText?.replace(/\s+/g, ' ').trim(),
                    runBtnDisabled: runBtn?.hasAttribute('disabled') || runBtn?.disabled || false,
                    mapping: {
                        y_true: yTrue,
                        y_pred: yPred,
                        y_prob: yProb,
                        sensitive: sens,
                        id: idVal
                    }
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": cleveland_check_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step8 = res.get("result", {}).get("result", {}).get("value")
            print("Step 8 Cleveland Rejection Result:", json.dumps(step8, indent=2))

            # Validate auto-mapping conservative behavior
            assert step8["mapping"]["y_pred"] != "sex", "y_pred must NOT be auto-mapped to sex"
            assert step8["mapping"]["y_pred"] != "heart_disease", "y_pred must NOT be auto-mapped to heart_disease"
            assert step8["mapping"]["y_pred"] == "", "y_pred must remain unselected ('') for raw clinical data"
            assert step8["mapping"]["y_prob"] != "heart_disease", "y_prob must NOT be auto-mapped to heart_disease"
            assert step8["mapping"]["y_prob"] == "", "y_prob must remain unselected ('') for raw clinical data"
            assert step8["mapping"]["sensitive"] == "sex", "sensitive may map to sex"

            assert step8["badgeText"] == "⚠ Audit Input Needs Correction", "Cleveland dataset must be rejected"
            assert "ground truth and model predictions" in step8["summaryText"], "Must display explicit rejection reason"
            assert "Cleveland clinical training dataset detected" in step8["issuesText"], "Must explain Cleveland feature detection"
            assert step8["runBtnDisabled"] is True, "Run button must be disabled for Cleveland dataset"

            # -----------------------------------------------------------------
            # 9. Verify External Audit Explainable AI (SHAP) View & Limitation Panel
            # -----------------------------------------------------------------
            print("\n[Step 9] Verifying External Audit Explainable AI (SHAP) Limitation UI...")
            # Re-upload the 600-row external audit file and run audit
            re_upload_600_code = f"""
            (function() {{
                window.setAuditMode('external');
                window.switchDashSection('external-audit');
                const fileInput = document.getElementById('ext-audit-file-input');
                const file = new File([{json.dumps(csv_600_raw)}], 'external_model_audit_test_600.csv', {{ type: 'text/csv' }});
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
                fileInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return {{ reloaded: true }};
            }})()
            """
            await send_cmd("Runtime.evaluate", {
                "expression": re_upload_600_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            await asyncio.sleep(1)
            # Execute audit
            await send_cmd("Runtime.evaluate", {
                "expression": run_audit_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            await asyncio.sleep(1)

            check_ext_shap_code = r"""
            (function() {
                window.switchDashSection('explain');
                const banner = document.getElementById('ext-shap-unavailable-banner');
                const limitationPanel = document.getElementById('ext-shap-limitation-panel');
                const emptyState = document.getElementById('explain-empty-state');
                const contentArea = document.getElementById('explain-content-area');

                const explainerEl = document.getElementById('shap-explainer-type');
                const baseValEl = document.getElementById('shap-base-val');
                const globalBars = document.getElementById('shap-global-bars-container');
                const localSelect = document.getElementById('local-patient-select');

                return {
                    bannerDisplay: banner ? window.getComputedStyle(banner).display : null,
                    limitationDisplay: limitationPanel ? window.getComputedStyle(limitationPanel).display : null,
                    emptyDisplay: emptyState ? window.getComputedStyle(emptyState).display : null,
                    contentDisplay: contentArea ? window.getComputedStyle(contentArea).display : null,
                    bannerText: banner ? banner.innerText : '',
                    limitationText: limitationPanel ? limitationPanel.innerText : '',
                    explainerText: explainerEl ? explainerEl.innerText : '',
                    baseValText: baseValEl ? baseValEl.innerText : '',
                    globalBarsCount: globalBars ? globalBars.children.length : -1,
                    localSelectCount: localSelect ? localSelect.children.length : -1,
                    shapStatus: window.AsclepiusState.shapResults?.status
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": check_ext_shap_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step9 = res.get("result", {}).get("result", {}).get("value")
            print("Step 9 External SHAP Result:", json.dumps(step9, indent=2))

            assert step9["bannerDisplay"] != "none", "ext-shap-unavailable-banner must be visible in external audit mode"
            assert "SHAP Unavailable — Model Internals Were Not Provided" in step9["bannerText"]
            assert step9["limitationDisplay"] != "none", "ext-shap-limitation-panel must be visible in external audit mode"
            assert "Feature-Level Explainability Unavailable" in step9["limitationText"]
            assert "Available Evidence" in step9["limitationText"]
            assert "Ground-truth labels (y_true)" in step9["limitationText"]
            assert "Unavailable" in step9["limitationText"]
            assert "Original model object" in step9["limitationText"]
            assert step9["contentDisplay"] == "none", "explain-content-area MUST be hidden in external audit mode"
            assert step9["shapStatus"] == "unavailable"
            assert step9["explainerText"] != "TreeExplainer", "TreeExplainer must NOT be displayed as executed"
            assert step9["baseValText"] != "0.484", "Base value 0.484 must NOT be displayed"
            assert step9["globalBarsCount"] == 0, "Global feature importance bars must NOT be rendered"
            assert step9["localSelectCount"] == 0, "Local patient selector must NOT be rendered"

            # -----------------------------------------------------------------
            # 9B. Verify External Audit Recommendations Page Truthful SHAP Status
            # -----------------------------------------------------------------
            print("\n[Step 9B] Testing External Audit Recommendations Page Explainability Status...")
            check_rec_shap_code = r"""
            (function() {
                window.switchDashSection('recommendations');
                const sumExplainText = document.getElementById('sum-explain-text');
                const sumExplainPill = document.getElementById('sum-explain-pill');
                const expList = document.getElementById('explainability-findings-list');
                const recContainer = document.getElementById('recommendations-container');

                return {
                    sumExplainText: sumExplainText ? sumExplainText.innerText : '',
                    sumExplainPill: sumExplainPill ? sumExplainPill.innerText : '',
                    expListHtml: expList ? expList.innerText : '',
                    recContainerHtml: recContainer ? recContainer.innerText : '',
                    insightsSummary: window.AsclepiusState.insightsResults?.summary
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": check_rec_shap_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step9b = res.get("result", {}).get("result", {}).get("value")
            print("Step 9B Recommendations SHAP Result:", json.dumps(step9b, indent=2))

            assert "Shapley feature attributions evaluated" not in step9b["sumExplainText"], "Must NOT claim Shapley feature attributions evaluated"
            assert "Attributions Ready" not in step9b["sumExplainPill"], "Must NOT claim Attributions Ready"
            assert "Feature-level SHAP unavailable" in step9b["sumExplainText"], "sum-explain-text must state Feature-level SHAP unavailable"
            assert "model internals not provided" in step9b["sumExplainPill"].lower(), "sum-explain-pill must state model internals not provided"
            assert "SHAP Unavailable" in step9b["expListHtml"] or "Feature-level SHAP unavailable" in step9b["expListHtml"], "Findings list must state SHAP is unavailable"
            assert len(step9b["recContainerHtml"]) > 0, "Fairness recommendations must be visible"

            # -----------------------------------------------------------------
            # 10. Verify Local Model Mode SHAP Pipeline Execution & Rendering
            # -----------------------------------------------------------------
            print("\n[Step 10] Testing Local Model Mode SHAP Execution...")
            local_train_and_shap_code = r"""
            (function() {
                window.setAuditMode('local');
                window.switchDashSection('dataset');
                const mockModelRes = {
                    status: "success",
                    modelType: "random_forest",
                    modelLabel: "Random Forest Classifier",
                    nTrain: 48,
                    nTest: 12,
                    yTest: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                    yPred: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                    yProb: [0.85, 0.15, 0.88, 0.22, 0.79, 0.18, 0.91, 0.12, 0.77, 0.25, 0.82, 0.19],
                    sensitiveCol: "gender",
                    sensitiveTest: ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
                    featureImportance: [
                        { rank: 1, feature: "bmi", weight: "0.25", absWeight: "0.25" },
                        { rank: 2, feature: "systolic_bp", weight: "0.18", absWeight: "0.18" },
                        { rank: 3, feature: "cholesterol", weight: "0.12", absWeight: "0.12" }
                    ],
                    metrics: { accuracy: "1.0000", balancedAccuracy: "1.0000", precision: "1.0000", recall: "1.0000", specificity: "1.0000", f1: "1.0000", rocAuc: 1.0 }
                };
                window.AsclepiusState.trainedModel = mockModelRes;
                window.runFairnessEvaluation(mockModelRes);
                window.runExplainabilityAnalysis(mockModelRes);

                window.switchDashSection('explain');

                const banner = document.getElementById('ext-shap-unavailable-banner');
                const limitationPanel = document.getElementById('ext-shap-limitation-panel');
                const emptyState = document.getElementById('explain-empty-state');
                const contentArea = document.getElementById('explain-content-area');
                const explainerEl = document.getElementById('shap-explainer-type');
                const globalBars = document.getElementById('shap-global-bars-container');

                return {
                    bannerDisplay: banner ? window.getComputedStyle(banner).display : null,
                    limitationDisplay: limitationPanel ? window.getComputedStyle(limitationPanel).display : null,
                    emptyDisplay: emptyState ? window.getComputedStyle(emptyState).display : null,
                    contentDisplay: contentArea ? window.getComputedStyle(contentArea).display : null,
                    explainerText: explainerEl ? explainerEl.innerText : '',
                    globalBarsCount: globalBars ? globalBars.children.length : -1,
                    shapStatus: window.AsclepiusState.shapResults?.status
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": local_train_and_shap_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step10 = res.get("result", {}).get("result", {}).get("value")
            print("Step 10 Local SHAP Result:", json.dumps(step10, indent=2))

            assert step10["bannerDisplay"] == "none", "ext-shap-unavailable-banner must be hidden in local mode"
            assert step10["limitationDisplay"] == "none", "ext-shap-limitation-panel must be hidden in local mode"
            assert step10["emptyDisplay"] == "none"
            assert step10["contentDisplay"] != "none", "explain-content-area MUST be visible for trained local model"
            assert step10["explainerText"] == "TreeExplainer", "TreeExplainer must be shown for trained local random forest"
            assert step10["globalBarsCount"] == 3, "Global SHAP bars must be rendered for local model features"
            assert step10["shapStatus"] == "success"

            # -----------------------------------------------------------------
            # 11. Verify Stale State Isolation: Switching to External Audit Purges SHAP
            # -----------------------------------------------------------------
            print("\n[Step 11] Testing Stale State Isolation when Switching to External Audit...")
            stale_isolation_code = r"""
            (function() {
                window.setAuditMode('external');
                window.switchDashSection('explain');

                const banner = document.getElementById('ext-shap-unavailable-banner');
                const limitationPanel = document.getElementById('ext-shap-limitation-panel');
                const contentArea = document.getElementById('explain-content-area');
                const explainerEl = document.getElementById('shap-explainer-type');
                const baseValEl = document.getElementById('shap-base-val');
                const globalBars = document.getElementById('shap-global-bars-container');

                return {
                    bannerDisplay: banner ? window.getComputedStyle(banner).display : null,
                    limitationDisplay: limitationPanel ? window.getComputedStyle(limitationPanel).display : null,
                    contentDisplay: contentArea ? window.getComputedStyle(contentArea).display : null,
                    explainerText: explainerEl ? explainerEl.innerText : '',
                    baseValText: baseValEl ? baseValEl.innerText : '',
                    globalBarsCount: globalBars ? globalBars.children.length : -1,
                    shapStatus: window.AsclepiusState.shapResults?.status,
                    globalImportanceLen: window.AsclepiusState.shapResults?.global_importance?.length || 0
                };
            })()
            """
            res = await send_cmd("Runtime.evaluate", {
                "expression": stale_isolation_code,
                "contextId": target_context_id,
                "returnByValue": True
            })
            step11 = res.get("result", {}).get("result", {}).get("value")
            print("Step 11 Stale State Isolation Result:", json.dumps(step11, indent=2))

            assert step11["bannerDisplay"] != "none", "Banner must appear when switched to external"
            assert step11["limitationDisplay"] != "none", "Limitation panel must appear when switched to external"
            assert step11["contentDisplay"] == "none", "Local explain content area must be hidden in external mode"
            assert step11["shapStatus"] == "unavailable", "State shapResults status must be unavailable"
            assert step11["globalImportanceLen"] == 0, "No stale global importance allowed"
            assert step11["explainerText"] != "TreeExplainer", "No stale TreeExplainer allowed"
            assert step11["baseValText"] != "0.484", "No stale base value allowed"
            assert step11["globalBarsCount"] == 0, "Global bars must be cleared"

            print("\n======================================================================")
            print("ALL BROWSER VERIFICATION CHECKS PASSED SUCCESSFULLY (100% VERIFIED)")
            print("======================================================================")
            return True

    finally:
        if proc:
            proc.terminate()
            proc.wait()

if __name__ == "__main__":
    success = asyncio.run(run_test())
    exit(0 if success else 1)
