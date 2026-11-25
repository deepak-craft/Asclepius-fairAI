"""
Asclepius FairAI - Chrome DevTools Protocol (CDP) Browser Verification
Verifies end-to-end in the actual running browser:
1. Level 1 prediction-only audit:
   - Loads demo predictions
   - Shows Model-Agnostic Audit
   - Executes audit
   - Displays 'SHAP Unavailable — Model Internals Were Not Provided'
2. Level 2 model-specific explainability audit:
   - Loads SHAP demo model + features
   - Validates explainability inputs
   - Displays 'Model-Specific Explainability Available'
   - Executes audit
   - Displays real SHAP feature importance with top features
   - Verifies no SHAP unavailable warning
3. Reset and re-verify Level 1 isolation:
   - Model-Agnostic Audit restored
"""

import asyncio
import json
import os
import sys
import subprocess
import tempfile
import urllib.request
import websockets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SHAP_DEMO_DIR = os.path.abspath(r"data\demo\shap")
MODEL_PKL = os.path.join(SHAP_DEMO_DIR, "asclepius_shap_demo_model.pkl")
FEATURES_CSV = os.path.join(SHAP_DEMO_DIR, "asclepius_shap_demo_features.csv")
PREDICTIONS_CSV = os.path.join(SHAP_DEMO_DIR, "asclepius_shap_demo_predictions.csv")


async def run_browser_verification():
    print("=" * 80)
    print("ASCLEPIUS FAIRAI — BROWSER CDP LEVEL 2 SHAP DEMO VERIFICATION")
    print("=" * 80)

    user_data_dir = tempfile.mkdtemp()
    port = 9225
    print(f"Launching isolated Chrome on port {port}...")
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "http://localhost:8501/?page=dashboard"
    ])
    await asyncio.sleep(4)

    try:
        targets_raw = urllib.request.urlopen(f"http://localhost:{port}/json").read()
        targets = json.loads(targets_raw.decode("utf-8"))
        page_targets = [t for t in targets if t.get("type") == "page"]
        assert page_targets, "No page target found"
        ws_url = page_targets[0]["webSocketDebuggerUrl"]
        print(f"Connected to page WebSocket: {ws_url}")

        async with websockets.connect(ws_url, max_size=25 * 1024 * 1024) as ws:
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

            async def eval_js(expr):
                r = await send_cmd("Runtime.evaluate", {
                    "expression": expr,
                    "contextId": target_context_id,
                    "returnByValue": True,
                    "awaitPromise": True
                })
                return r.get("result", {}).get("result", {}).get("value")

            # -----------------------------------------------------------------
            # TEST STEP 1: Verify Initial Setup & Switch to External Audit
            # -----------------------------------------------------------------
            print("\n[STEP 1] Switching to Existing AI Model Audit Mode...")
            res1 = await eval_js("""
                (() => {
                    setAuditMode('external');
                    switchDashSection('dataset');
                    return {
                        mode: window.AsclepiusState.auditMode,
                        demoAuditAvailable: typeof window.loadDemoExternalAuditData === 'function',
                        demoShapAvailable: typeof window.loadDemoShapAuditData === 'function'
                    };
                })()
            """)
            print(f" -> Mode: {res1['mode']}, DemoLoaders: Level1={res1['demoAuditAvailable']}, Level2={res1['demoShapAvailable']}")
            assert res1['mode'] == 'external'
            assert res1['demoAuditAvailable']
            assert res1['demoShapAvailable']

            # -----------------------------------------------------------------
            # TEST STEP 2: Verify Level 1 Prediction-Only Audit
            # -----------------------------------------------------------------
            print("\n[STEP 2] Testing Level 1 Prediction-Only Audit...")
            res2 = await eval_js("""
                (() => {
                    loadDemoExternalAuditData();
                    const pill = document.getElementById('ext-xai-status-pill')?.innerText || '';
                    const valBadge = document.getElementById('ext-validation-badge')?.innerText || '';
                    return {
                        pill: pill,
                        valBadge: valBadge,
                        shapAvail: !!(window.AsclepiusState.externalAudit && window.AsclepiusState.externalAudit.shapAvailable)
                    };
                })()
            """)
            print(f" -> XAI Pill: '{res2.get('pill')}' | Validation: '{res2.get('valBadge')}' | shapAvailable: {res2.get('shapAvail')}")
            assert "Model-Agnostic" in res2.get('pill', '')
            assert res2.get('shapAvail') == False

            # Run Level 1 Audit
            print(" -> Running Level 1 External Audit...")
            res2_run = await eval_js("""
                (() => {
                    runExternalModelAudit();
                    const shapStatus = window.AsclepiusState.shapResults?.status || '';
                    const shapMsg = window.AsclepiusState.shapResults?.message || '';
                    return { shapStatus, shapMsg };
                })()
            """)
            print(f" -> Level 1 SHAP Status: '{res2_run['shapStatus']}' | Message: '{res2_run['shapMsg']}'")
            assert res2_run['shapStatus'] == 'unavailable'
            assert "model internals were not provided" in res2_run['shapMsg'].lower()

            # -----------------------------------------------------------------
            # TEST STEP 3: Reset and Test Level 2 Model-Specific Audit
            # -----------------------------------------------------------------
            print("\n[STEP 3] Testing Level 2 Explainability Audit (Real SHAP Demo)...")
            res3_reset = await eval_js("""
                (() => {
                    resetExternalAuditForm();
                    switchDashSection('dataset');
                    return {
                        shapAvail: !!(window.AsclepiusState.externalAudit && window.AsclepiusState.externalAudit.shapAvailable),
                        model: window.AsclepiusState.externalAudit?.model || null
                    };
                })()
            """)
            print(f" -> Form Reset: shapAvailable={res3_reset['shapAvail']}, model={res3_reset['model']}")

            # Load Level 2 SHAP Demo
            res3_load = await eval_js("""
                (() => {
                    loadDemoShapAuditData();
                    const pill = document.getElementById('ext-xai-status-pill')?.innerText || '';
                    const msg = document.getElementById('ext-xai-validation-message')?.innerText || '';
                    const modelLabel = document.getElementById('ext-model-file-name')?.innerText || '';
                    const featLabel = document.getElementById('ext-features-file-name')?.innerText || '';
                    return {
                        pill: pill,
                        msg: msg,
                        modelLabel: modelLabel,
                        featLabel: featLabel,
                        shapAvail: !!(window.AsclepiusState.externalAudit && window.AsclepiusState.externalAudit.shapAvailable),
                        validCols: window.AsclepiusState.externalAudit?.validFeatureCols || []
                    };
                })()
            """)
            print(f" -> Model Label: '{res3_load['modelLabel']}'")
            print(f" -> Features Label: '{res3_load['featLabel']}'")
            print(f" -> Status Pill: '{res3_load['pill']}'")
            print(f" -> Validation Message: '{res3_load['msg']}'")
            print(f" -> Valid Feature Columns ({len(res3_load['validCols'])}): {res3_load['validCols']}")

            assert "Model-Specific Explainability Available" in res3_load['pill']
            assert res3_load['shapAvail'] == True
            assert len(res3_load['validCols']) == 8
            assert "sex" not in res3_load['validCols'], "Sensitive attribute sex must NOT be in SHAP features"

            # Execute Level 2 Audit
            print("\n[STEP 4] Executing Level 2 Model Audit & Fairness Evaluation...")
            res4_run = await eval_js("""
                (() => {
                    runExternalModelAudit();
                    const shapRes = window.AsclepiusState.shapResults || {};
                    const trainedModel = window.AsclepiusState.trainedModel || {};
                    const fairness = window.AsclepiusState.fairnessResults || {};
                    return {
                        shapStatus: shapRes.status,
                        explainerType: shapRes.explainerType,
                        modelType: shapRes.modelType,
                        baseValue: shapRes.baseValue,
                        samplesExplained: shapRes.samplesExplained,
                        globalImportanceCount: (shapRes.globalImportance || []).length,
                        topFeatures: (shapRes.globalImportance || []).slice(0, 4),
                        featureImportanceInModel: (trainedModel.featureImportance || []).length,
                        fairnessEvaluated: !!fairness.subgroups
                    };
                })()
            """)
            print(f" -> SHAP Status: '{res4_run['shapStatus']}' ({res4_run['explainerType']})")
            print(f" -> Model Type: '{res4_run['modelType']}' | Base Value: {res4_run['baseValue']}")
            print(f" -> Samples Explained: {res4_run['samplesExplained']}")
            print(f" -> Global Features Count: {res4_run['globalImportanceCount']}")
            print(f" -> Top Features: {res4_run['topFeatures']}")
            print(f" -> Fairness Evaluated: {res4_run['fairnessEvaluated']}")

            assert res4_run['shapStatus'] == 'success'
            assert res4_run['explainerType'] == 'TreeExplainer'
            assert res4_run['globalImportanceCount'] == 8
            assert res4_run['fairnessEvaluated'] == True

            # -----------------------------------------------------------------
            # TEST STEP 5: Reset and Re-verify Level 1 Isolation
            # -----------------------------------------------------------------
            print("\n[STEP 5] Re-verifying Level 1 Isolation After Level 2 Execution...")
            res5 = await eval_js("""
                (() => {
                    resetExternalAuditForm();
                    loadDemoExternalAuditData();
                    runExternalModelAudit();
                    return {
                        pill: document.getElementById('ext-xai-status-pill')?.innerText || '',
                        shapStatus: window.AsclepiusState.shapResults?.status,
                        shapMsg: window.AsclepiusState.shapResults?.message
                    };
                })()
            """)
            print(f" -> XAI Pill: '{res5['pill']}' | Status: '{res5['shapStatus']}'")
            print(f" -> Message: '{res5['shapMsg']}'")
            assert "Model-Agnostic" in res5['pill']
            assert res5['shapStatus'] == 'unavailable'
            assert "model internals were not provided" in res5['shapMsg'].lower()

            print("\n" + "=" * 80)
            print("BROWSER CDP VERIFICATION COMPLETED SUCCESSFULLY: ALL 5 STEPS PASSED!")
            print("=" * 80 + "\n")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(run_browser_verification())
