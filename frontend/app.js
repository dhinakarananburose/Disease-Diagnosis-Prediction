/**
 * Disease Diagnosis Prediction — Client-Side Application Logic (Phase 21)
 * Consumes existing FastAPI Backend endpoints (GET /health, POST /predict, POST /predict/batch)
 * Does NOT modify backend logic, ML pipeline, or probability calculations.
 */

// 1. API BASE URL CONFIGURATION
const API_BASE_URL = "http://localhost:8000";

// DOM Elements
const elements = {
    // Status Indicator
    statusDot: document.getElementById("statusDot"),
    statusText: document.getElementById("statusText"),
    statusSubtext: document.getElementById("statusSubtext"),

    // Tabs
    tabButtons: document.querySelectorAll(".tab-btn"),
    tabPanes: document.querySelectorAll(".tab-pane"),

    // Single Prediction Form & Buttons
    predictionForm: document.getElementById("predictionForm"),
    btnPredict: document.getElementById("btnPredict"),
    btnReset: document.getElementById("btnReset"),

    // Result Card Views
    resultPlaceholder: document.getElementById("resultPlaceholder"),
    resultLoading: document.getElementById("resultLoading"),
    resultError: document.getElementById("resultError"),
    resultContent: document.getElementById("resultContent"),
    btnRetry: document.getElementById("btnRetry"),
    errorTitle: document.getElementById("errorTitle"),
    errorMsg: document.getElementById("errorMsg"),

    // Result Card Content Elements
    resModelName: document.getElementById("resModelName"),
    predictionStatusBox: document.getElementById("predictionStatusBox"),
    statusBoxIcon: document.getElementById("statusBoxIcon"),
    resPredictedClassTitle: document.getElementById("resPredictedClassTitle"),
    resPredictedClassCode: document.getElementById("resPredictedClassCode"),
    resProbabilityValue: document.getElementById("resProbabilityValue"),
    resGaugeBar: document.getElementById("resGaugeBar"),
    summaryChips: document.getElementById("summaryChips"),

    // Batch Prediction Elements
    batchJsonInput: document.getElementById("batchJsonInput"),
    btnRunBatch: document.getElementById("btnRunBatch"),
    btnLoadSampleBatch: document.getElementById("btnLoadSampleBatch"),
    btnClearBatch: document.getElementById("btnClearBatch"),
    batchResultsWrapper: document.getElementById("batchResultsWrapper"),
    batchTableBody: document.getElementById("batchTableBody"),
};

// Application State
let apiOnline = false;

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initForm();
    initBatch();
    checkApiHealth();
    
    // Poll API Health status every 15 seconds
    setInterval(checkApiHealth, 15000);
});

/* ==========================================================================
   1. API HEALTH CHECK
   ========================================================================== */
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: "GET",
            headers: { "Accept": "application/json" }
        });

        if (response.ok) {
            const data = await response.json();
            if (data.status === "healthy" && data.model_loaded) {
                setApiStatus(true, "API: Online", "Model Artifact Loaded");
            } else {
                setApiStatus(false, "API: Warning", "Model Not Loaded");
            }
        } else {
            setApiStatus(false, "API: Offline", `HTTP Error ${response.status}`);
        }
    } catch (err) {
        setApiStatus(false, "API: Unreachable", "Backend Server Offline");
    }
}

function setApiStatus(isHealthy, text, subtext) {
    apiOnline = isHealthy;
    if (elements.statusDot) {
        elements.statusDot.className = "status-dot " + (isHealthy ? "online" : "offline");
    }
    if (elements.statusText) elements.statusText.textContent = text;
    if (elements.statusSubtext) elements.statusSubtext.textContent = subtext;
}

/* ==========================================================================
   2. TAB NAVIGATION
   ========================================================================== */
function initTabs() {
    elements.tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");

            elements.tabButtons.forEach(b => {
                b.classList.remove("active");
                b.setAttribute("aria-selected", "false");
            });
            elements.tabPanes.forEach(p => p.classList.add("hidden"));

            btn.classList.add("active");
            btn.setAttribute("aria-selected", "true");
            const activePane = document.getElementById(targetTab);
            if (activePane) activePane.classList.remove("hidden");
        });
    });
}

/* ==========================================================================
   3. SINGLE PREDICTION FORM HANDLING & CLIENT-SIDE VALIDATION
   ========================================================================== */
function initForm() {
    if (elements.predictionForm) {
        elements.predictionForm.addEventListener("submit", handleSinglePredictSubmit);
    }
    if (elements.btnReset) {
        elements.btnReset.addEventListener("click", handleReset);
    }
    if (elements.btnRetry) {
        elements.btnRetry.addEventListener("click", handleSinglePredictSubmit);
    }
}

function clearFormErrors() {
    const errorSpans = document.querySelectorAll(".error-msg");
    errorSpans.forEach(span => span.textContent = "");
    const groups = document.querySelectorAll(".input-group");
    groups.forEach(group => group.classList.remove("has-error"));
}

function setFieldError(fieldId, message) {
    const errSpan = document.getElementById(`err-${fieldId}`);
    if (errSpan) errSpan.textContent = message;
    const inputElem = document.getElementById(fieldId);
    if (inputElem && inputElem.closest(".input-group")) {
        inputElem.closest(".input-group").classList.add("has-error");
    }
}

function getFormData() {
    const age = parseFloat(document.getElementById("age").value);
    const sex = document.getElementById("sex").value;
    const cp = document.getElementById("cp").value;
    const trestbps = parseFloat(document.getElementById("trestbps").value);
    const chol = parseFloat(document.getElementById("chol").value);
    const fbsStr = document.getElementById("fbs").value;
    const fbs = (fbsStr === "true" || fbsStr === "True" || fbsStr === "1");
    const restecg = document.getElementById("restecg").value;
    const thalch = parseFloat(document.getElementById("thalch").value);
    const exangStr = document.getElementById("exang").value;
    const exang = (exangStr === "true" || exangStr === "True" || exangStr === "1");
    const oldpeak = parseFloat(document.getElementById("oldpeak").value);

    return { age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak };
}

function validateFormData(payload) {
    clearFormErrors();
    let isValid = true;

    // age: 1 <= age <= 120
    if (isNaN(payload.age) || payload.age < 1 || payload.age > 120) {
        setFieldError("age", "Age must be a number between 1 and 120.");
        isValid = false;
    }

    // trestbps: 0 <= trestbps <= 300
    if (isNaN(payload.trestbps) || payload.trestbps < 0 || payload.trestbps > 300) {
        setFieldError("trestbps", "Resting BP must be between 0 and 300 mm Hg.");
        isValid = false;
    }

    // chol: 0 <= chol <= 1500 (chol = 0 is allowed and treated as missing in preprocessing)
    if (isNaN(payload.chol) || payload.chol < 0 || payload.chol > 1500) {
        setFieldError("chol", "Cholesterol must be between 0 and 1500 mg/dl.");
        isValid = false;
    }

    // thalch: 1 <= thalch <= 250
    if (isNaN(payload.thalch) || payload.thalch < 1 || payload.thalch > 250) {
        setFieldError("thalch", "Max heart rate must be between 1 and 250 bpm.");
        isValid = false;
    }

    // oldpeak: -10.0 <= oldpeak <= 15.0
    if (isNaN(payload.oldpeak) || payload.oldpeak < -10.0 || payload.oldpeak > 15.0) {
        setFieldError("oldpeak", "ST depression must be between -10.0 and 15.0 mm.");
        isValid = false;
    }

    // cp canonical values
    const validCp = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"];
    if (!validCp.includes(payload.cp)) {
        setFieldError("cp", "Invalid chest pain type selected.");
        isValid = false;
    }

    // restecg canonical values
    const validRestEcg = ["normal", "st-t abnormality", "lv hypertrophy"];
    if (!validRestEcg.includes(payload.restecg)) {
        setFieldError("restecg", "Invalid resting ECG value selected.");
        isValid = false;
    }

    return isValid;
}

async function handleSinglePredictSubmit(e) {
    if (e && e.preventDefault) e.preventDefault();

    const payload = getFormData();
    if (!validateFormData(payload)) {
        return;
    }

    // Set UI to loading state
    showResultState("loading");
    setPredictButtonState(false, "Generating prediction...");

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const resultData = await response.json();
            renderPredictionResult(resultData, payload);
            showResultState("content");
        } else {
            let errorDetail = `HTTP Error ${response.status}`;
            try {
                const errJson = await response.json();
                if (errJson.detail) {
                    errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
                }
            } catch (_) {}

            displayError("Input Validation / API Error", errorDetail);
        }
    } catch (err) {
        displayError("API Connection Unavailable", `Failed to connect to backend server at ${API_BASE_URL}. Ensure the FastAPI server is running.`);
    } finally {
        setPredictButtonState(true);
    }
}

function setPredictButtonState(enabled, label = "PREDICT RISK") {
    if (!elements.btnPredict) return;
    elements.btnPredict.disabled = !enabled;
    const labelSpan = elements.btnPredict.querySelector(".btn-label");
    if (labelSpan) labelSpan.textContent = label;
}

function showResultState(state) {
    elements.resultPlaceholder.classList.add("hidden");
    elements.resultLoading.classList.add("hidden");
    elements.resultError.classList.add("hidden");
    elements.resultContent.classList.add("hidden");

    if (state === "placeholder") elements.resultPlaceholder.classList.remove("hidden");
    if (state === "loading") elements.resultLoading.classList.remove("hidden");
    if (state === "error") elements.resultError.classList.remove("hidden");
    if (state === "content") elements.resultContent.classList.remove("hidden");
}

function displayError(title, message) {
    if (elements.errorTitle) elements.errorTitle.textContent = title;
    if (elements.errorMsg) elements.errorMsg.textContent = message;
    showResultState("error");
}

/* ==========================================================================
   4. RENDER PREDICTION RESULT
   ========================================================================== */
function renderPredictionResult(result, inputPayload) {
    // 1. Model Name
    if (elements.resModelName) {
        elements.resModelName.textContent = result.model_name || "Tuned Support Vector Machine";
    }

    // 2. Predicted Class (0 or 1)
    const predClass = Number(result.predicted_class);
    const predProb = Number(result.predicted_probability);

    // Format probability percentage safely
    const probPctStr = (isNaN(predProb) ? 0 : (predProb * 100)).toFixed(1) + "%";
    
    if (elements.resProbabilityValue) {
        elements.resProbabilityValue.textContent = probPctStr;
    }
    if (elements.resGaugeBar) {
        elements.resGaugeBar.style.width = isNaN(predProb) ? "0%" : `${Math.min(100, Math.max(0, predProb * 100))}%`;
        if (predProb >= 0.5) {
            elements.resGaugeBar.classList.add("high-risk");
        } else {
            elements.resGaugeBar.classList.remove("high-risk");
        }
    }

    // Classification Result Status Box
    if (elements.predictionStatusBox) {
        if (predClass === 1) {
            elements.predictionStatusBox.className = "prediction-status-box disease-present";
            elements.statusBoxIcon.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                </svg>`;
            elements.resPredictedClassTitle.textContent = "Heart Disease Predicted";
            elements.resPredictedClassCode.textContent = "Predicted Class Label: 1 (Positive Risk)";
        } else {
            elements.predictionStatusBox.className = "prediction-status-box no-disease";
            elements.statusBoxIcon.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>`;
            elements.resPredictedClassTitle.textContent = "No Heart Disease Predicted";
            elements.resPredictedClassCode.textContent = "Predicted Class Label: 0 (Negative Risk)";
        }
    }

    // Render Observation Chips
    if (elements.summaryChips) {
        elements.summaryChips.innerHTML = `
            <span class="chip">Age: ${inputPayload.age} y/o</span>
            <span class="chip">Sex: ${inputPayload.sex}</span>
            <span class="chip">CP: ${inputPayload.cp}</span>
            <span class="chip">BP: ${inputPayload.trestbps} mm Hg</span>
            <span class="chip">Chol: ${inputPayload.chol === 0 ? "0 (Unrecorded)" : inputPayload.chol + " mg/dl"}</span>
            <span class="chip">FBS > 120: ${inputPayload.fbs ? "True" : "False"}</span>
            <span class="chip">ECG: ${inputPayload.restecg}</span>
            <span class="chip">Max HR: ${inputPayload.thalch} bpm</span>
            <span class="chip">ExAngina: ${inputPayload.exang ? "True" : "False"}</span>
            <span class="chip">Oldpeak: ${inputPayload.oldpeak}</span>
        `;
    }
}

function handleReset() {
    clearFormErrors();
    
    // Reset Form fields to defaults
    document.getElementById("age").value = 55;
    document.getElementById("sex").value = "Male";
    document.getElementById("cp").value = "asymptomatic";
    document.getElementById("trestbps").value = 140;
    document.getElementById("chol").value = 250;
    document.getElementById("fbs").value = "false";
    document.getElementById("restecg").value = "normal";
    document.getElementById("thalch").value = 150;
    document.getElementById("exang").value = "false";
    document.getElementById("oldpeak").value = 1.2;

    showResultState("placeholder");
    setPredictButtonState(true);
}

/* ==========================================================================
   5. BATCH PREDICTION INTERFACE
   ========================================================================== */
function initBatch() {
    if (elements.btnRunBatch) {
        elements.btnRunBatch.addEventListener("click", handleBatchPredict);
    }
    if (elements.btnLoadSampleBatch) {
        elements.btnLoadSampleBatch.addEventListener("click", loadSampleBatchJson);
    }
    if (elements.btnClearBatch) {
        elements.btnClearBatch.addEventListener("click", () => {
            elements.batchJsonInput.value = "";
            elements.batchResultsWrapper.classList.add("hidden");
        });
    }
}

const SAMPLE_BATCH_DATA = {
    "records": [
        {
            "age": 55, "sex": "Male", "cp": "asymptomatic", "trestbps": 140, "chol": 250,
            "fbs": false, "restecg": "normal", "thalch": 150, "exang": false, "oldpeak": 1.2
        },
        {
            "age": 41, "sex": "Female", "cp": "atypical angina", "trestbps": 120, "chol": 157,
            "fbs": false, "restecg": "normal", "thalch": 182, "exang": false, "oldpeak": 0.0
        },
        {
            "age": 67, "sex": "Male", "cp": "asymptomatic", "trestbps": 160, "chol": 286,
            "fbs": true, "restecg": "lv hypertrophy", "thalch": 108, "exang": true, "oldpeak": 1.5
        }
    ]
};

function loadSampleBatchJson() {
    if (elements.batchJsonInput) {
        elements.batchJsonInput.value = JSON.stringify(SAMPLE_BATCH_DATA, null, 2);
    }
}

async function handleBatchPredict() {
    const jsonText = elements.batchJsonInput.value.trim();
    if (!jsonText) {
        alert("Please enter or load a valid batch JSON payload.");
        return;
    }

    let parsedPayload;
    try {
        parsedPayload = JSON.parse(jsonText);
    } catch (e) {
        alert("Invalid JSON format. Please verify JSON syntax.");
        return;
    }

    if (!parsedPayload.records || !Array.isArray(parsedPayload.records) || parsedPayload.records.length === 0) {
        alert("Batch JSON must contain a non-empty 'records' array.");
        return;
    }

    elements.btnRunBatch.disabled = true;
    elements.btnRunBatch.querySelector(".btn-label").textContent = "PROCESSING BATCH...";

    try {
        const response = await fetch(`${API_BASE_URL}/predict/batch`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(parsedPayload)
        });

        if (response.ok) {
            const data = await response.json();
            renderBatchTable(data.predictions, parsedPayload.records);
            elements.batchResultsWrapper.classList.remove("hidden");
        } else {
            const errJson = await response.json().catch(() => ({}));
            alert(`Batch Prediction Error (${response.status}): ` + (errJson.detail || "Server validation error."));
        }
    } catch (err) {
        alert("Failed to connect to API backend for batch prediction.");
    } finally {
        elements.btnRunBatch.disabled = false;
        elements.btnRunBatch.querySelector(".btn-label").textContent = "RUN BATCH INFERENCE";
    }
}

function renderBatchTable(predictions, records) {
    if (!elements.batchTableBody) return;
    elements.batchTableBody.innerHTML = "";

    predictions.forEach((pred, idx) => {
        const rec = records[idx] || {};
        const tr = document.createElement("tr");

        const predClass = Number(pred.predicted_class);
        const predProb = Number(pred.predicted_probability);
        const probPctStr = (isNaN(predProb) ? 0 : (predProb * 100)).toFixed(1) + "%";

        const badgeClass = predClass === 1 ? "danger-text" : "success-text";
        const classLabel = predClass === 1 ? "1 (Heart Disease)" : "0 (No Disease)";

        tr.innerHTML = `
            <td>#${idx + 1}</td>
            <td>${rec.age || "-"} / ${rec.sex || "-"}</td>
            <td>${rec.cp || "-"}</td>
            <td>${rec.trestbps || "-"} / ${rec.chol !== undefined ? rec.chol : "-"}</td>
            <td class="${badgeClass}" style="font-weight: 600;">${classLabel}</td>
            <td style="font-weight: 600;">${probPctStr}</td>
            <td style="font-size: 0.78rem; color: #94a3b8;">${pred.model_name || "SVM"}</td>
        `;
        elements.batchTableBody.appendChild(tr);
    });
}
